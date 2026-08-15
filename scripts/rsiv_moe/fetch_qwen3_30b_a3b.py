from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


MODEL_ID = "Qwen/Qwen3-30B-A3B-Base"
REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
EXPECTED_SHARDS = 16
EXPECTED_WEIGHT_BYTES = 61_066_575_648
MIN_FREE_BYTES = 90 * 2**30
ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = ROOT / "models" / "qwen3-30b-a3b-base"
REPORT_PATH = ROOT / "reports" / "rsiv_moe" / "qwen_checkpoint_acquisition.json"
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "LICENSE",
    "README.md",
    "model-*-of-00016.safetensors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire the exact preregistered Qwen3-30B-A3B BF16 checkpoint."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest-only", action="store_true")
    group.add_argument("--include-weights", action="store_true")
    parser.add_argument("--skip-local-sha256", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_report(payload: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 2**20):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError(f"revision mismatch: {info.sha} != {REVISION}")

    files = []
    for sibling in info.siblings:
        lfs_sha = sibling.lfs.sha256 if sibling.lfs is not None else None
        files.append(
            {
                "path": sibling.rfilename,
                "size_bytes": sibling.size,
                "lfs_sha256": lfs_sha,
            }
        )
    shards = sorted(
        (
            item
            for item in files
            if item["path"].startswith("model-")
            and item["path"].endswith("-of-00016.safetensors")
        ),
        key=lambda item: item["path"],
    )
    if len(shards) != EXPECTED_SHARDS:
        raise RuntimeError(f"expected {EXPECTED_SHARDS} shards, found {len(shards)}")
    shard_bytes = sum(int(item["size_bytes"] or 0) for item in shards)
    if shard_bytes != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            f"weight-byte mismatch: {shard_bytes} != {EXPECTED_WEIGHT_BYTES}"
        )
    if any(not item["lfs_sha256"] for item in shards):
        raise RuntimeError("one or more weight shards lack an official LFS SHA-256")
    return files, shards


def main() -> None:
    args = parse_args()
    files, shards = resolve_manifest()
    free_before = shutil.disk_usage(ROOT).free
    payload: dict[str, Any] = {
        "kind": "rsiv_moe_qwen_checkpoint_acquisition",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "local_dir": str(LOCAL_DIR.resolve()),
        "expected_weight_bytes": EXPECTED_WEIGHT_BYTES,
        "expected_weight_gib": EXPECTED_WEIGHT_BYTES / 2**30,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "free_bytes_before": free_before,
        "manifest_resolved_at_utc": utc_now(),
        "files": files,
        "weight_shards": shards,
        "status": "manifest_verified",
        "transport": {
            "max_workers": args.max_workers,
            "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET"),
            "hf_xet_high_performance": os.environ.get("HF_XET_HIGH_PERFORMANCE"),
        },
    }
    write_report(payload)
    if args.manifest_only:
        print(json.dumps({k: v for k, v in payload.items() if k != "files"}, indent=2))
        return
    if free_before < MIN_FREE_BYTES:
        payload["status"] = "blocked_insufficient_disk"
        write_report(payload)
        raise RuntimeError(
            f"only {free_before / 2**30:.3f} GiB free; {MIN_FREE_BYTES / 2**30:.0f} required"
        )
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")

    payload["status"] = "downloading"
    payload["download_started_at_utc"] = utc_now()
    write_report(payload)
    try:
        snapshot_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            local_dir=LOCAL_DIR,
            allow_patterns=ALLOW_PATTERNS,
            max_workers=args.max_workers,
        )
    except BaseException as exc:
        payload["status"] = "download_interrupted"
        payload["download_error"] = f"{type(exc).__name__}: {exc}"
        payload["download_interrupted_at_utc"] = utc_now()
        write_report(payload)
        raise

    payload["snapshot_path"] = str(Path(snapshot_path).resolve())
    local_shards = []
    for expected in shards:
        path = LOCAL_DIR / expected["path"]
        if not path.is_file():
            raise RuntimeError(f"missing downloaded shard: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected["size_bytes"]:
            raise RuntimeError(
                f"size mismatch for {path.name}: {actual_size} != {expected['size_bytes']}"
            )
        actual_sha = None if args.skip_local_sha256 else sha256_file(path)
        if actual_sha is not None and actual_sha != expected["lfs_sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {path.name}: {actual_sha} != {expected['lfs_sha256']}"
            )
        local_shards.append(
            {
                "path": expected["path"],
                "size_bytes": actual_size,
                "sha256": actual_sha,
                "official_lfs_sha256": expected["lfs_sha256"],
            }
        )

    payload["local_weight_shards"] = local_shards
    payload["local_weight_bytes"] = sum(item["size_bytes"] for item in local_shards)
    payload["local_sha256_verified"] = not args.skip_local_sha256
    payload["free_bytes_after"] = shutil.disk_usage(ROOT).free
    payload["download_completed_at_utc"] = utc_now()
    payload["status"] = "complete_verified"
    write_report(payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "revision": REVISION,
                "local_weight_bytes": payload["local_weight_bytes"],
                "local_sha256_verified": payload["local_sha256_verified"],
                "report": str(REPORT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
