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


TOKENS = 131_072
CONTEXT = 1_024
LANGUAGES = (
    "arb_Arab", "zho_Hans", "hin_Deva", "rus_Cyrl",
    "spa_Latn", "swh_Latn", "jpn_Jpan", "nld_Latn",
)
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
MODEL = ROOT / "models/qwen3-30b-a3b-base"
HERA = ROOT / "data/corpora/hera_moe_p0"
OUTPUT = ROOT / "reports/qwen_gptq_bank/p0_input_lock.json"
ARTIFACT = ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_input_ids.safetensors"


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
    return torch.tensor(values, dtype=torch.int64)


def joined_column(path: Path, column: str) -> str:
    values = pq.read_table(path, columns=[column])[column].to_pylist()
    return "\n\n".join(value for value in values if value and value.strip())


if __name__ == "__main__":
    if OUTPUT.exists() or ARTIFACT.exists():
        raise FileExistsError("refusing to overwrite the full-bank supplement input lock")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    sources: dict[str, list[dict[str, object]]] = {domain: [] for domain in DOMAINS}

    wiki = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet"
    general = window(tokenizer, joined_column(wiki, "text"), 32_768, TOKENS, "general")
    sources["general"].append({"path": wiki, "offset": 32_768, "tokens": TOKENS})

    code_parts = []
    for language in ("python", "java"):
        path = HERA / f"code/{language}/train-00000-of-00001.parquet"
        count = TOKENS // 2
        code_parts.append(window(tokenizer, joined_column(path, "input"), 32_768, count, f"code_{language}"))
        sources["code"].append({"path": path, "offset": 32_768, "tokens": count})
    code = torch.cat(code_parts)

    math_path = HERA / "math/main/train-00000-of-00001.parquet"
    math_table = pq.read_table(math_path, columns=["question", "answer"])
    math_text = "\n\n".join(
        f"Question: {question}\nAnswer: {answer}"
        for question, answer in zip(
            math_table["question"].to_pylist(), math_table["answer"].to_pylist()
        )
    )
    math_ids = window(tokenizer, math_text, 32_768, TOKENS, "math")
    sources["math"].append({"path": math_path, "offset": 32_768, "tokens": TOKENS})

    multilingual_path = HERA / "multilingual/dev.parquet"
    multilingual_table = pq.read_table(multilingual_path, columns=list(LANGUAGES))
    multilingual_parts = []
    for language in LANGUAGES:
        count = TOKENS // len(LANGUAGES)
        text = "\n\n".join(value for value in multilingual_table[language].to_pylist() if value)
        multilingual_parts.append(window(tokenizer, text, 4_096, count, f"multilingual_{language}"))
        sources["multilingual"].append(
            {"path": multilingual_path, "column": language, "offset": 4_096, "tokens": count}
        )
    multilingual = torch.cat(multilingual_parts)

    instruction_path = HERA / "instruction/databricks-dolly-15k.jsonl"
    instruction_rows = []
    with instruction_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            instruction_rows.append(
                f"Instruction: {row['instruction']}\nContext: {row['context']}\nResponse: {row['response']}"
            )
    instruction = window(
        tokenizer, "\n\n".join(instruction_rows), 65_536, TOKENS, "instruction"
    )
    sources["instruction"].append(
        {"path": instruction_path, "offset": 65_536, "tokens": TOKENS}
    )

    tensors = {
        "general": general.reshape(-1, CONTEXT),
        "code": code.reshape(-1, CONTEXT),
        "math": math_ids.reshape(-1, CONTEXT),
        "multilingual": multilingual.reshape(-1, CONTEXT),
        "instruction": instruction.reshape(-1, CONTEXT),
    }
    if tuple(tensors) != DOMAINS or any(tuple(value.shape) != (128, 1_024) for value in tensors.values()):
        raise RuntimeError("supplement tensor contract failed")

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        ARTIFACT,
        metadata={
            "kind": "qwen_gptq_bank_p0_supplement_input_lock",
            "model_revision": "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
            "routing_opened": "false",
        },
    )
    serialized_sources = {}
    for domain, rows in sources.items():
        serialized_sources[domain] = []
        for row in rows:
            path = row["path"]
            serialized_sources[domain].append({
                **{key: value for key, value in row.items() if key != "path"},
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
            })
    payload = {
        "kind": "qwen_gptq_bank_p0_input_lock",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "model_revision": "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
        "artifact": str(ARTIFACT.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256_file(ARTIFACT),
        "domains": list(DOMAINS),
        "contexts_per_domain": 128,
        "context_tokens": CONTEXT,
        "tokens_per_domain": TOKENS,
        "total_supplement_tokens": TOKENS * len(DOMAINS),
        "input_ids_sha256": {key: sha256_tensor(value) for key, value in tensors.items()},
        "sources": serialized_sources,
        "base_route_corpora": [
            "reports/runs/hera_moe/p0_input_ids.safetensors",
            "reports/runs/dhera_moe/p0_input_ids.safetensors",
        ],
        "selection_disclosure": (
            "Supplement size and source windows were fixed after inspecting aggregate HERA+DHERA "
            "coverage (6/6144 pairs below 128), but before opening any supplement routes."
        ),
        "supplement_routes_opened": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact": payload["artifact"],
        "sha256": payload["artifact_sha256"],
        "domains": payload["domains"],
        "total_supplement_tokens": payload["total_supplement_tokens"],
    }, indent=2))
