from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

REPO_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
# NVIDIA NemoClaw documents physical single-DGX-Spark validation of this
# immutable public revision. Callers may explicitly override it.
DEFAULT_REVISION = "0dcd680e5585c791728c83342b311d0a0026dbeb"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Defaults to HF_TOKEN.")
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Download config/code/index/tokenizer but not weight shards.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Install huggingface_hub in the selected Python environment."
        ) from exc

    api = HfApi(token=args.token)
    info = api.model_info(REPO_ID, revision=args.revision)
    resolved = info.sha
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    allow_patterns = None
    if args.metadata_only:
        allow_patterns = ["*.json", "*.md", "*.py", "*.txt", "*.yaml", "*.yml", "*.jinja"]

    snapshot_download(
        repo_id=REPO_ID, revision=resolved, local_dir=str(destination),
        token=args.token, allow_patterns=allow_patterns,
    )

    manifest = {
        "kind": "s100_lightning_acquisition_provenance",
        "repo_id": REPO_ID,
        "requested_revision": args.revision,
        "resolved_revision": resolved,
        "metadata_only": bool(args.metadata_only),
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "destination": str(destination),
    }
    (destination / "ACQUISITION_PROVENANCE.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
