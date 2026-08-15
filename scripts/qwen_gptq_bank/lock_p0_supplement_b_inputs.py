from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from moe_lab.reporting import ROOT


MODEL = ROOT / "models/qwen3-30b-a3b-base"
HERA = ROOT / "data/corpora/hera_moe_p0"
PREREG = ROOT / "reports/qwen_gptq_bank/P0_SUPPLEMENT_B_PREREGISTRATION.md"
OUTPUT = ROOT / "reports/qwen_gptq_bank/p0_supplement_b_input_lock.json"
ARTIFACT = ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_b_input_ids.safetensors"
CONTEXT = 1_024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def window(tokenizer, text: str, start: int, count: int, label: str) -> torch.Tensor:
    values = tokenizer.encode(
        text, add_special_tokens=False, truncation=True, max_length=start + count
    )[start : start + count]
    if len(values) != count:
        raise RuntimeError(f"{label} produced {len(values)} of {count} tokens at offset {start}")
    return torch.tensor(values, dtype=torch.int64).reshape(-1, CONTEXT)


if __name__ == "__main__":
    if OUTPUT.exists() or ARTIFACT.exists():
        raise FileExistsError("refusing to overwrite supplement B input lock")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)

    math_path = HERA / "math/main/train-00000-of-00001.parquet"
    math_table = pq.read_table(math_path, columns=["question", "answer"])
    math_text = "\n\n".join(
        f"Question: {question}\nAnswer: {answer}"
        for question, answer in zip(
            math_table["question"].to_pylist(), math_table["answer"].to_pylist()
        )
    )
    math_ids = window(tokenizer, math_text, 163_840, 262_144, "math_b")

    instruction_path = HERA / "instruction/databricks-dolly-15k.jsonl"
    instruction_rows = []
    with instruction_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            instruction_rows.append(
                f"Instruction: {row['instruction']}\nContext: {row['context']}\nResponse: {row['response']}"
            )
    instruction_ids = window(
        tokenizer, "\n\n".join(instruction_rows), 196_608, 524_288, "instruction_b"
    )
    tensors = {"math": math_ids, "instruction": instruction_ids}
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, ARTIFACT, metadata={
        "kind": "qwen_gptq_bank_p0_supplement_b_input_lock",
        "model_revision": "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
        "routing_opened": "false",
    })
    payload = {
        "kind": "qwen_gptq_bank_p0_supplement_b_input_lock",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256_file(PREREG),
        "model_revision": "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
        "artifact": str(ARTIFACT.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256_file(ARTIFACT),
        "context_tokens": CONTEXT,
        "domains": ["math", "instruction"],
        "contexts": {key: int(value.shape[0]) for key, value in tensors.items()},
        "tokens": {key: int(value.numel()) for key, value in tensors.items()},
        "total_tokens": sum(int(value.numel()) for value in tensors.values()),
        "input_ids_sha256": {key: sha256_tensor(value) for key, value in tensors.items()},
        "sources": {
            "math": {
                "path": str(math_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(math_path), "offset": 163_840, "tokens": 262_144,
            },
            "instruction": {
                "path": str(instruction_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(instruction_path), "offset": 196_608, "tokens": 524_288,
            },
        },
        "design_evidence": {
            "supplement_a_layer_13_expert_99_rows": 89,
            "supplement_a_layer_43_expert_95_rows": 113,
        },
        "supplement_b_routes_opened": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact": payload["artifact"], "artifact_sha256": payload["artifact_sha256"],
        "contexts": payload["contexts"], "total_tokens": payload["total_tokens"],
    }, indent=2))
