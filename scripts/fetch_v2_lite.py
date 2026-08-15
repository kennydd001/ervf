from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from moe_lab.reporting import ROOT, envelope, write_json


MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite"
METADATA_PATTERNS = [
    "*.json",
    "*.py",
    "*.md",
    "*.txt",
    "LICENSE*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pin V2-Lite and fetch metadata; weights require explicit consent."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metadata-only", action="store_true")
    group.add_argument("--include-weights", action="store_true")
    parser.add_argument("--revision", help="Optional commit SHA; defaults to current main SHA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    info = api.model_info(MODEL_ID, revision=args.revision, files_metadata=True)
    revision = info.sha
    files = [
        {"path": sibling.rfilename, "size_bytes": sibling.size}
        for sibling in info.siblings
    ]
    weight_bytes = sum(
        item["size_bytes"] or 0
        for item in files
        if item["path"].endswith((".safetensors", ".bin"))
    )
    local_dir = ROOT / "models" / "deepseek-v2-lite"
    payload: dict[str, Any] = {
        "model_id": MODEL_ID,
        "revision": revision,
        "metadata_only": not args.include_weights,
        "snapshot_path": None,
        "download_status": "manifest_resolved",
        "declared_weight_bytes": weight_bytes,
        "declared_weight_gib": round(weight_bytes / 2**30, 3),
        "files": files,
    }
    path = write_json("model.json", envelope("model_snapshot", payload))
    allow_patterns = None if args.include_weights else METADATA_PATTERNS
    try:
        snapshot_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            max_workers=1,
        )
    except Exception as exc:
        payload["download_status"] = "failed"
        payload["download_error"] = f"{type(exc).__name__}: {exc}"
        write_json("model.json", envelope("model_snapshot", payload))
        raise
    payload["snapshot_path"] = str(Path(snapshot_path).resolve())
    payload["download_status"] = "complete"
    path = write_json("model.json", envelope("model_snapshot", payload))
    print(path)
    print(json.dumps({k: payload[k] for k in payload if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
