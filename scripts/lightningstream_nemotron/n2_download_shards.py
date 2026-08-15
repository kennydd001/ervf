"""N2 step 1-3: download and verify the five official NVFP4 shards.

Executes the frozen preregistration
``N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS_PREREGISTRATION_2026-08-14.md`` §3.1-3.3.

Shard digests are the ones frozen by N0R.  A mismatch permits exactly one
redownload of that shard, recorded as such; a second mismatch is a hard stop.
Nothing outside the Nemotron allowlist is written.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
CACHE_DIR = REPO_ROOT / ".cache" / "nemotron_3_5_lightning"
OUT = REPO_ROOT / "reports" / "lightningstream_nemotron" / "n2_payload_manifest.json"

REPO_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
REVISION = "ce1b118ae66ec705d02c241525192832eb045fd3"

# Frozen by N0R from Hugging Face LFS OIDs.
EXPECTED = {
    "model-00001-of-00005.safetensors": (3_998_838_864, "2fdac76b3e4906ce0fb0dd33ab51f011372a5473e0d6c5bb479b6f10d3f29fdb"),
    "model-00002-of-00005.safetensors": (4_000_414_120, "559806ee0cb6edcfc01805e24bac9182cb2611bad3993e0da05487d7a79b4f38"),
    "model-00003-of-00005.safetensors": (3_999_641_680, "d820849788701123d041501fb8ac88e4ade24a28a63cd663118797cfae910be2"),
    "model-00004-of-00005.safetensors": (4_000_413_336, "f5ccb7cfa7870ab2d099134c3f771ad4a158e0421b3bf7b2a0da53311a09cb14"),
    "model-00005-of-00005.safetensors": (3_343_488_520, "c9dd9142839367ad274019a7683bc84993217c8a63e70dd8e18656de0c4050eb"),
}

# Small companions kept beside the shards so the model directory is self-contained.
COMPANIONS = (
    "config.json",
    "generation_config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "chat_template.jinja",
    "configuration_nemotron_h.py",
    "modeling_nemotron_h.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> dict:
    """Read the leading u64 header length and the JSON header itself."""
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        (header_len,) = struct.unpack("<Q", raw_len)
        header_bytes = handle.read(header_len)
    return {
        "header_len": header_len,
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "tensor_count": len(
            [k for k in json.loads(header_bytes.decode("utf-8")) if k != "__metadata__"]
        ),
    }


def free_bytes() -> int:
    return shutil.disk_usage(str(REPO_ROOT)).free


def fetch(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        revision=REVISION,
        cache_dir=str(CACHE_DIR),
    )
    return Path(local)


def place(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.copy2(src, dest)


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    disk_before = free_bytes()

    rows = []
    hard_stop = False

    for name, (want_bytes, want_sha) in EXPECTED.items():
        dest = MODEL_DIR / name
        attempts = []
        ok = False

        for attempt in (1, 2):  # one redownload permitted
            if dest.exists() and dest.stat().st_size == want_bytes and attempt == 1:
                got = sha256_path(dest)
                attempts.append({"attempt": attempt, "source": "existing_local",
                                 "bytes": dest.stat().st_size, "sha256": got,
                                 "match": got == want_sha})
                if got == want_sha:
                    ok = True
                    break
                continue

            try:
                src = fetch(name)
            except Exception as exc:
                attempts.append({"attempt": attempt, "source": "download",
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue

            place(src, dest)
            got = sha256_path(dest)
            attempts.append({"attempt": attempt, "source": "download",
                             "bytes": dest.stat().st_size, "sha256": got,
                             "match": got == want_sha})
            if got == want_sha:
                ok = True
                break

        header = read_safetensors_header(dest) if ok else None
        rows.append({
            "shard": name,
            "expected_bytes": want_bytes,
            "expected_sha256": want_sha,
            "verified": ok,
            "attempts": attempts,
            "header": header,
        })
        print(f"{name}: verified={ok} attempts={len(attempts)}")
        if not ok:
            hard_stop = True
            print(f"  HARD STOP: {name} failed verification after {len(attempts)} attempts")
            break

    companions = {}
    if not hard_stop:
        for name in COMPANIONS:
            try:
                src = fetch(name)
            except Exception as exc:
                companions[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                continue
            dest = MODEL_DIR / name
            place(src, dest)
            companions[name] = {"ok": True, "bytes": dest.stat().st_size,
                                "sha256": sha256_path(dest)}

    disk_after = free_bytes()
    total_bytes = sum(r["expected_bytes"] for r in rows if r["verified"])

    manifest = {
        "kind": "lightningstream_nemotron_n2_payload_manifest",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS",
        "step": "download_and_verify",
        "started_utc": started,
        "completed_utc": utc_now(),
        "repo_id": REPO_ID,
        "revision": REVISION,
        "model_dir": str(MODEL_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
        "runner_sha256": sha256_path(Path(__file__)),
        "shards": rows,
        "companions": companions,
        "shards_verified": sum(1 for r in rows if r["verified"]),
        "shards_expected": len(EXPECTED),
        "verified_payload_bytes": total_bytes,
        "disk_free_before": disk_before,
        "disk_free_after": disk_after,
        "disk_consumed": disk_before - disk_after,
        "artifact_gib": round(total_bytes / (1024 ** 3), 6),
        "artifact_gate_25gib_pass": total_bytes <= 25 * (1024 ** 3),
        "hard_stop": hard_stop,
        "gates": {
            "all_five_shards_verified": (not hard_stop) and len(rows) == 5
                                        and all(r["verified"] for r in rows),
            "all_headers_parsed": (not hard_stop)
                                  and all(r["header"] is not None for r in rows),
            "artifact_under_25gib": total_bytes <= 25 * (1024 ** 3),
        },
    }
    manifest["gates_all_pass"] = all(manifest["gates"].values())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"shards verified : {manifest['shards_verified']}/{manifest['shards_expected']}")
    print(f"payload bytes   : {total_bytes:,} ({manifest['artifact_gib']} GiB)")
    print(f"disk consumed   : {manifest['disk_consumed']:,}")
    print(f"gates all pass  : {manifest['gates_all_pass']}")
    print(f"written         : {OUT}")
    return 0 if manifest["gates_all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
