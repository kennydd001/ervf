from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

from moe_lab.reporting import ROOT


OUTPUT = ROOT / "reports/hera_moe/p0_source_acquisition.json"
DESTINATION = ROOT / "data/corpora/hera_moe_p0"
SOURCES = [
    {
        "domain": "code", "repo": "google/code_x_glue_cc_code_completion_line",
        "revision": "d480ae0bde7b9f18677131dce01a03d6d028e964",
        "files": ["python/train-00000-of-00001.parquet", "java/train-00000-of-00001.parquet"],
    },
    {
        "domain": "math", "repo": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "files": ["main/train-00000-of-00001.parquet"],
    },
    {
        "domain": "multilingual", "repo": "yash9439/flores200",
        "revision": "3c6628a4571f383d029d6e897a89ac953ae756d3",
        "files": ["dev.parquet"],
    },
    {
        "domain": "instruction", "repo": "databricks/databricks-dolly-15k",
        "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "files": ["databricks-dolly-15k.jsonl"],
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    files = []
    for source in SOURCES:
        for filename in source["files"]:
            cached = Path(hf_hub_download(
                repo_id=source["repo"], repo_type="dataset", filename=filename,
                revision=source["revision"],
            ))
            target = DESTINATION / source["domain"] / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"refusing to overwrite {target}")
            shutil.copyfile(cached, target)
            files.append({
                "domain": source["domain"], "repo": source["repo"],
                "revision": source["revision"], "source_file": filename,
                "local_file": str(target.relative_to(ROOT)).replace("\\", "/"),
                "bytes": target.stat().st_size, "sha256": sha256(target),
            })
            print(json.dumps(files[-1]), flush=True)
    general = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet"
    files.insert(0, {
        "domain": "general", "repo": "Salesforce/wikitext",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "source_file": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "local_file": str(general.relative_to(ROOT)).replace("\\", "/"),
        "bytes": general.stat().st_size, "sha256": sha256(general),
    })
    payload = {
        "kind": "hera_moe_p0_source_acquisition", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "files": files,
        "claim_boundary": "Source acquisition only; no token selection, routing, tier or quality result.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "files": len(files), "bytes": sum(x["bytes"] for x in files)}, indent=2))
