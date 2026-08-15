from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

from moe_lab.reporting import ROOT, envelope, write_json


DATASET_ID = "Salesforce/wikitext"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CONFIG = "wikitext-2-raw-v1"
FILES = {
    split: f"{CONFIG}/{split}-00000-of-00001.parquet"
    for split in ("train", "validation", "test")
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    destination = ROOT / "data" / "corpora" / "wikitext"
    records = []
    for split, filename in FILES.items():
        path = Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=filename,
                revision=REVISION,
                local_dir=destination,
            )
        )
        records.append(
            {
                "split": split,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    report = {
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "config": CONFIG,
        "files": records,
    }
    path = write_json("wikitext_corpus.json", envelope("corpus_manifest", report))
    print(path)
    for record in records:
        print(record)
