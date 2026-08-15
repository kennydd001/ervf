from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from moe_lab.reporting import ROOT


TOKENS = 32_768
CONTEXT = 1_024
OUTPUT = ROOT / "reports/e2gq_moe/p0_input_lock.json"


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    parquet = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet"
    model = ROOT / "models/qwen3-30b-a3b-base"
    texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    joined = "\n\n".join(text for text in texts if text and text.strip())
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
    ids = tokenizer.encode(joined, add_special_tokens=False)[:TOKENS]
    if len(ids) != TOKENS:
        raise RuntimeError(f"expected {TOKENS} tokens, got {len(ids)}")
    tensor = torch.tensor(ids, dtype=torch.int64).reshape(TOKENS // CONTEXT, CONTEXT)
    payload = {
        "kind": "e2gq_p0_input_lock", "locked_utc": datetime.now(timezone.utc).isoformat(),
        "source_parquet": str(parquet.relative_to(ROOT)).replace("\\", "/"),
        "source_parquet_sha256": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
        "model_revision": "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
        "selection_rule": "first 32768 tokens after joining nonempty train texts with two newlines",
        "contexts": 32, "context_tokens": 1024, "tokens": TOKENS,
        "input_ids_sha256": tensor_sha256(tensor),
        "minimum_routed_rows_per_expert": 128,
        "router_counts_opened": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
