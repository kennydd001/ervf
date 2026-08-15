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
SOURCE_MANIFEST = ROOT / "reports/hera_moe/p0_source_acquisition.json"
ARTIFACT = ROOT / "reports/runs/hera_moe/p0_input_ids.safetensors"
OUTPUT = ROOT / "reports/hera_moe/p0_input_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def tokens(tokenizer, text: str, count: int) -> torch.Tensor:
    values = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=count)
    if len(values) != count:
        raise RuntimeError(f"source produced {len(values)} of {count} required tokens")
    return torch.tensor(values, dtype=torch.int64)


if __name__ == "__main__":
    if OUTPUT.exists() or ARTIFACT.exists():
        raise FileExistsError("refusing to overwrite HERA P0 input lock")
    model = ROOT / "models/qwen3-30b-a3b-base"
    corpus = ROOT / "data/corpora/hera_moe_p0"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)

    wiki = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet"
    rows = pq.read_table(wiki, columns=["text"])["text"].to_pylist()
    general = tokens(tokenizer, "\n\n".join(x for x in rows if x and x.strip()), TOKENS)

    code_parts = []
    for language in ("python", "java"):
        table = pq.read_table(corpus / f"code/{language}/train-00000-of-00001.parquet", columns=["input"])
        code_parts.append(tokens(tokenizer, "\n\n".join(x for x in table["input"].to_pylist() if x), TOKENS // 2))
    code = torch.cat(code_parts)

    math_table = pq.read_table(corpus / "math/main/train-00000-of-00001.parquet", columns=["question", "answer"])
    math_text = "\n\n".join(
        f"Question: {q}\nAnswer: {a}" for q, a in zip(math_table["question"].to_pylist(), math_table["answer"].to_pylist())
    )
    math_ids = tokens(tokenizer, math_text, TOKENS)

    multilingual_table = pq.read_table(corpus / "multilingual/dev.parquet", columns=list(LANGUAGES))
    multilingual_parts = [
        tokens(tokenizer, "\n\n".join(x for x in multilingual_table[lang].to_pylist() if x), TOKENS // len(LANGUAGES))
        for lang in LANGUAGES
    ]
    multilingual = torch.cat(multilingual_parts)

    instruction_rows = []
    with (corpus / "instruction/databricks-dolly-15k.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            instruction_rows.append(
                f"Instruction: {row['instruction']}\nContext: {row['context']}\nResponse: {row['response']}"
            )
    instruction = tokens(tokenizer, "\n\n".join(instruction_rows), TOKENS)

    tensors = {
        "general": general.reshape(-1, CONTEXT), "code": code.reshape(-1, CONTEXT),
        "math": math_ids.reshape(-1, CONTEXT), "multilingual": multilingual.reshape(-1, CONTEXT),
        "instruction": instruction.reshape(-1, CONTEXT),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, ARTIFACT, metadata={
        "kind": "hera_moe_p0_input_lock", "model_revision": "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
    })
    payload = {
        "kind": "hera_moe_p0_input_lock", "locked_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "artifact": str(ARTIFACT.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256(ARTIFACT), "domains": list(tensors),
        "contexts_per_domain": 32, "context_tokens": CONTEXT, "tokens_per_domain": TOKENS,
        "multilingual_languages": list(LANGUAGES),
        "input_ids_sha256": {name: tensor_sha(value) for name, value in tensors.items()},
        "tier_threshold_rows": 128, "tier_rule": "union of per-domain count>=128 sets",
        "routing_opened": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
