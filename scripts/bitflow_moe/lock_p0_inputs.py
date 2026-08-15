from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

from moe_lab.reporting import ROOT


MODEL = ROOT / "models/deepseek-v2-lite"
CORPUS = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1"
PREREG = ROOT / "reports/bitflow_moe/P0_C1_Q4_PREREGISTRATION.md"
ARTIFACT = ROOT / "reports/runs/bitflow_moe/p0_input_ids.safetensors"
OUTPUT = ROOT / "reports/bitflow_moe/p0_input_lock.json"
SPLITS = {"train": 1024, "validation": 256, "test": 256}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if ARTIFACT.exists() or OUTPUT.exists():
        raise FileExistsError("refusing to overwrite BITFLOW input lock")
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    tensors, sources = {}, {}
    for split, tokens in SPLITS.items():
        path = CORPUS / f"{split}-00000-of-00001.parquet"
        texts = pq.read_table(path, columns=["text"])["text"].to_pylist()
        joined = "\n\n".join(text for text in texts if text and text.strip())
        ids = tokenizer.encode(joined).ids[:tokens]
        if len(ids) != tokens or tokens % 128:
            raise ValueError(f"invalid token range for {split}")
        tensors[split] = torch.tensor(ids, dtype=torch.int32).view(-1, 128)
        sources[split] = {
            "source_sha256": sha256(path),
            "tokens": tokens,
            "token_start": 0,
            "token_end": tokens,
            "token_ids_sha256": hashlib.sha256(
                tensors[split].numpy().tobytes()
            ).hexdigest(),
        }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, ARTIFACT, metadata={"kind": "bitflow_p0_input_ids"})
    payload = {
        "kind": "bitflow_moe_p0_input_lock",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "model_config_sha256": sha256(MODEL / "config.json"),
        "tokenizer_sha256": sha256(MODEL / "tokenizer.json"),
        "preregistration_sha256": sha256(PREREG),
        "artifact": str(ARTIFACT.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256(ARTIFACT),
        "splits": sources,
        "validation_metrics_opened": False,
        "test_metrics_opened": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
