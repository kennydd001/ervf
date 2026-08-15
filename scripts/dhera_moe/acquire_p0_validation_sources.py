from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

from moe_lab.reporting import ROOT


OUTPUT = ROOT / "reports/dhera_moe/p0_validation_source_acquisition.json"
DEST = ROOT / "data/corpora/dhera_moe_p0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    downloads = [
        ("math", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet"),
        ("multilingual", "yash9439/flores200", "3c6628a4571f383d029d6e897a89ac953ae756d3", "devtest.parquet"),
    ]
    files = []
    for domain, repo, revision, filename in downloads:
        cached = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename, revision=revision))
        target = DEST / domain / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        shutil.copyfile(cached, target)
        files.append({"domain": domain, "repo": repo, "revision": revision, "source_file": filename, "local_file": str(target.relative_to(ROOT)).replace("\\", "/"), "bytes": target.stat().st_size, "sha256": sha256(target)})
    reused = [
        ("general", ROOT / "data/corpora/wikitext/wikitext-2-raw-v1/validation-00000-of-00001.parquet"),
        ("code_python", ROOT / "data/corpora/hera_moe_p0/code/python/train-00000-of-00001.parquet"),
        ("code_java", ROOT / "data/corpora/hera_moe_p0/code/java/train-00000-of-00001.parquet"),
        ("instruction", ROOT / "data/corpora/hera_moe_p0/instruction/databricks-dolly-15k.jsonl"),
    ]
    for domain, path in reused:
        files.append({"domain": domain, "local_file": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path), "reused_pinned_source": True})
    payload = {"kind": "dhera_moe_p0_validation_sources", "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "complete", "files": files, "routing_opened": False}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
