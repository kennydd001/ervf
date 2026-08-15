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


TOKENS = 32768
CONTEXT = 1024
LANGUAGES = ("arb_Arab", "zho_Hans", "hin_Deva", "rus_Cyrl", "spa_Latn", "swh_Latn", "jpn_Jpan", "nld_Latn")
SOURCE_MANIFEST = ROOT / "reports/dhera_moe/p0_validation_source_acquisition.json"
ARTIFACT = ROOT / "reports/runs/dhera_moe/p0_input_ids.safetensors"
OUTPUT = ROOT / "reports/dhera_moe/p0_input_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def window(tokenizer, text: str, start: int, count: int) -> torch.Tensor:
    values = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=start + count)
    values = values[start:start + count]
    if len(values) != count:
        raise RuntimeError(f"source produced {len(values)} of {count} tokens for offset {start}")
    return torch.tensor(values, dtype=torch.int64)


if __name__ == "__main__":
    if OUTPUT.exists() or ARTIFACT.exists():
        raise FileExistsError("refusing to overwrite DHERA P0 input lock")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/qwen3-30b-a3b-base", local_files_only=True, use_fast=True)
    hera = ROOT / "data/corpora/hera_moe_p0"
    dhera = ROOT / "data/corpora/dhera_moe_p0"

    wiki = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1/validation-00000-of-00001.parquet"
    rows = pq.read_table(wiki, columns=["text"])["text"].to_pylist()
    general = window(tokenizer, "\n\n".join(x for x in rows if x and x.strip()), 0, TOKENS)

    code_parts = []
    for language in ("python", "java"):
        table = pq.read_table(hera / f"code/{language}/train-00000-of-00001.parquet", columns=["input"])
        code_parts.append(window(tokenizer, "\n\n".join(x for x in table["input"].to_pylist() if x), TOKENS // 2, TOKENS // 2))
    code = torch.cat(code_parts)

    math_table = pq.read_table(dhera / "math/main/test-00000-of-00001.parquet", columns=["question", "answer"])
    math_text = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in zip(math_table["question"].to_pylist(), math_table["answer"].to_pylist()))
    math_ids = window(tokenizer, math_text, 0, TOKENS)

    multilingual_table = pq.read_table(dhera / "multilingual/devtest.parquet", columns=list(LANGUAGES))
    multilingual = torch.cat([window(tokenizer, "\n\n".join(x for x in multilingual_table[lang].to_pylist() if x), 0, TOKENS // len(LANGUAGES)) for lang in LANGUAGES])

    instruction_rows = []
    with (hera / "instruction/databricks-dolly-15k.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            instruction_rows.append(f"Instruction: {row['instruction']}\nContext: {row['context']}\nResponse: {row['response']}")
    instruction = window(tokenizer, "\n\n".join(instruction_rows), TOKENS, TOKENS)

    tensors = {"general": general.reshape(-1, CONTEXT), "code": code.reshape(-1, CONTEXT), "math": math_ids.reshape(-1, CONTEXT), "multilingual": multilingual.reshape(-1, CONTEXT), "instruction": instruction.reshape(-1, CONTEXT)}
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, ARTIFACT, metadata={"kind": "dhera_moe_p0_validation_inputs", "source_manifest_sha256": sha256(SOURCE_MANIFEST)})
    payload = {
        "kind": "dhera_moe_p0_input_lock", "locked_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST), "artifact": str(ARTIFACT.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256(ARTIFACT), "domains": list(tensors), "contexts_per_domain": 32,
        "context_tokens": CONTEXT, "tokens_per_domain": TOKENS, "multilingual_languages": list(LANGUAGES),
        "input_ids_sha256": {name: tensor_sha(value) for name, value in tensors.items()}, "routing_opened": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
