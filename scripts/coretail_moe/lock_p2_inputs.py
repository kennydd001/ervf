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
CORPUS = ROOT / "data/corpora/hera_moe_p0"
PREREG = ROOT / "reports/coretail_moe/P2_MODEL_QUALITY_PREREGISTRATION.md"
P1_VERIFY = ROOT / "reports/coretail_moe/p1_full_benchmark_verification.json"
ARTIFACT = ROOT / "reports/runs/coretail_moe/p2_heldout_input_ids.safetensors"
OUTPUT = ROOT / "reports/coretail_moe/p2_input_lock.json"
CONTEXT = 128
TOKENS = 256
LANGUAGES = ("arb_Arab", "zho_Hans", "hin_Deva", "rus_Cyrl", "spa_Latn", "swh_Latn", "jpn_Jpan", "nld_Latn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def end_split(values: list[int], label: str, per_split: int = TOKENS):
    if len(values) < 2 * per_split:
        raise RuntimeError(f"{label} has {len(values)} tokens, fewer than {2 * per_split}")
    start = len(values) - 2 * per_split
    return (
        values[start : start + per_split],
        values[start + per_split : start + 2 * per_split],
        {"total_tokens": len(values), "validation_offset": start, "test_offset": start + per_split, "tokens_per_split": per_split},
    )


if __name__ == "__main__":
    if OUTPUT.exists() or ARTIFACT.exists():
        raise FileExistsError("refusing to overwrite P2 input lock")
    p1 = json.loads(P1_VERIFY.read_text(encoding="utf-8"))
    if p1.get("status") != "p1_verification_pass":
        raise ValueError("independent P1 pass required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, object] = {}

    wiki_root = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1"
    for split in ("validation", "test"):
        path = wiki_root / f"{split}-00000-of-00001.parquet"
        rows = pq.read_table(path, columns=["text"])["text"].to_pylist()
        values = encode(tokenizer, "\n\n".join(x for x in rows if x and x.strip()))
        if len(values) < TOKENS:
            raise RuntimeError(f"WikiText {split} too short")
        tensors[f"{split}_general"] = torch.tensor(values[:TOKENS], dtype=torch.int64).reshape(-1, CONTEXT)
        manifest[f"{split}_general"] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "offset": 0, "total_tokens": len(values)}

    code_by_split = {"validation": [], "test": []}
    code_manifest = {}
    for language in ("python", "java"):
        path = CORPUS / f"code/{language}/train-00000-of-00001.parquet"
        rows = pq.read_table(path, columns=["input"])["input"].to_pylist()
        validation, test, selection = end_split(encode(tokenizer, "\n\n".join(x for x in rows if x)), f"code_{language}", TOKENS // 2)
        code_by_split["validation"].extend(validation); code_by_split["test"].extend(test)
        code_manifest[language] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), **selection}
    for split in ("validation", "test"):
        tensors[f"{split}_code"] = torch.tensor(code_by_split[split], dtype=torch.int64).reshape(-1, CONTEXT)
        manifest[f"{split}_code"] = code_manifest

    math_path = CORPUS / "math/main/train-00000-of-00001.parquet"
    math_table = pq.read_table(math_path, columns=["question", "answer"])
    math_text = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in zip(math_table["question"].to_pylist(), math_table["answer"].to_pylist()))
    validation, test, selection = end_split(encode(tokenizer, math_text), "math")
    for split, values in (("validation", validation), ("test", test)):
        tensors[f"{split}_math"] = torch.tensor(values, dtype=torch.int64).reshape(-1, CONTEXT)
        manifest[f"{split}_math"] = {"path": str(math_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(math_path), **selection}

    instruction_path = CORPUS / "instruction/databricks-dolly-15k.jsonl"
    instruction_rows = []
    with instruction_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            instruction_rows.append(f"Instruction: {row['instruction']}\nContext: {row['context']}\nResponse: {row['response']}")
    validation, test, selection = end_split(encode(tokenizer, "\n\n".join(instruction_rows)), "instruction")
    for split, values in (("validation", validation), ("test", test)):
        tensors[f"{split}_instruction"] = torch.tensor(values, dtype=torch.int64).reshape(-1, CONTEXT)
        manifest[f"{split}_instruction"] = {"path": str(instruction_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(instruction_path), **selection}

    multilingual_path = CORPUS / "multilingual/dev.parquet"
    multilingual_table = pq.read_table(multilingual_path, columns=list(LANGUAGES))
    multi_by_split = {"validation": [], "test": []}
    multi_manifest = {}
    for language in LANGUAGES:
        values = encode(tokenizer, "\n\n".join(x for x in multilingual_table[language].to_pylist() if x))
        validation, test, selection = end_split(values, f"multilingual_{language}", TOKENS // len(LANGUAGES))
        multi_by_split["validation"].extend(validation); multi_by_split["test"].extend(test)
        multi_manifest[language] = selection
    for split in ("validation", "test"):
        tensors[f"{split}_multilingual"] = torch.tensor(multi_by_split[split], dtype=torch.int64).reshape(-1, CONTEXT)
        manifest[f"{split}_multilingual"] = {"path": str(multilingual_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(multilingual_path), "languages": multi_manifest}

    expected = {f"{split}_{domain}" for split in ("validation", "test") for domain in ("general", "code", "math", "multilingual", "instruction")}
    if set(tensors) != expected or any(tuple(value.shape) != (2, CONTEXT) for value in tensors.values()):
        raise RuntimeError("P2 tensor contract failed")
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, ARTIFACT, metadata={"kind": "coretail_moe_p2_heldout_inputs", "outputs_opened": "false"})
    payload = {
        "kind": "coretail_moe_p2_input_lock",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG),
        "p1_verification_sha256": sha256(P1_VERIFY),
        "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
        "artifact": str(ARTIFACT.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256(ARTIFACT),
        "splits": ["validation", "test"], "domains": ["general", "code", "math", "multilingual", "instruction"],
        "contexts_per_domain": 2, "context_tokens": CONTEXT, "tokens_per_domain": TOKENS,
        "input_ids_sha256": {name: tensor_sha(value) for name, value in tensors.items()},
        "source_manifest": manifest,
        "outputs_opened": False,
        "variants": ["bf16_teacher", "gptq_experts_bf16_trunk", "bf16_experts_int4_trunk", "gptq_experts_int4_trunk", "gptq_experts_int8_trunk"],
        "trunk_quantization": {"group_size": 128, "int4_codes": [-7, 7], "int8_codes": [-127, 127], "rounding": "torch_round_nearest_even", "dequant_dtype": "bfloat16", "rank1_norms": "bf16"},
        "primary_gate": {"variant": "gptq_experts_int4_trunk", "validation_relative_ce_max": 0.02, "test_relative_ce_max": 0.02},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "locked", "artifact": payload["artifact"], "sha256": payload["artifact_sha256"], "shapes": {key: list(value.shape) for key, value in tensors.items()}, "source_manifest": manifest}, indent=2))
