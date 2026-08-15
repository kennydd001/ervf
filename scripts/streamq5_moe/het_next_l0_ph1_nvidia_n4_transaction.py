#!/usr/bin/env python3
"""Create-new bundle and bounded failure lifecycle for PH1 NVIDIA N4."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

MAX_ARTIFACT_BYTES = 16 * 2**20


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fsync_file(path: Path):
    mode = "r+b" if os.name == "nt" else "rb"
    with Path(path).open(mode) as handle:
        os.fsync(handle.fileno())


def atomic_create(path: Path, data: bytes):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".inprogress." + uuid.uuid4().hex)
    if path.exists():
        raise FileExistsError(path)
    try:
        with temp.open("xb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path); fsync_file(path); temp.unlink()
    except Exception:
        try:
            if temp.exists(): temp.unlink()
        finally:
            raise


def bundle_bytes(result, kind: str, extras=None):
    extras = dict(extras or {})
    if any(name in {"result.json", "manifest.json", "commit.json"} or Path(name).name != name for name in extras):
        raise ValueError("extra_name")
    result_bytes = canonical(result)
    payload = {"result.json": result_bytes, **extras}
    manifest = {"kind": kind + "_manifest", "files": [{"name": name, "bytes": len(data), "sha256": sha(data)} for name, data in sorted(payload.items())]}
    manifest_bytes = canonical(manifest)
    commit = {"kind": kind + "_commit", "result_sha256": sha(result_bytes), "manifest_sha256": sha(manifest_bytes)}
    commit_bytes = canonical(commit)
    if sum(map(len, payload.values())) + len(manifest_bytes) + len(commit_bytes) > MAX_ARTIFACT_BYTES:
        raise RuntimeError("artifact_cap")
    return {**payload, "manifest.json": manifest_bytes, "commit.json": commit_bytes}


def verify_bundle(directory: Path, kind: str):
    directory = Path(directory)
    if not directory.is_dir():
        return False
    names = {p.name for p in directory.iterdir() if p.is_file()}
    if not {"result.json", "manifest.json", "commit.json"} <= names or any(not p.is_file() for p in directory.iterdir()):
        return False
    raw = {name: (directory / name).read_bytes() for name in names}
    if sum(map(len, raw.values())) > MAX_ARTIFACT_BYTES:
        return False
    try:
        manifest, commit = json.loads(raw["manifest.json"]), json.loads(raw["commit.json"])
    except Exception:
        return False
    payload_names = names - {"manifest.json", "commit.json"}
    expected_manifest = {"kind": kind + "_manifest", "files": [{"name": name, "bytes": len(raw[name]), "sha256": sha(raw[name])} for name in sorted(payload_names)]}
    return manifest == expected_manifest and commit == {"kind": kind + "_commit", "result_sha256": sha(raw["result.json"]), "manifest_sha256": sha(raw["manifest.json"])}


def publish_bundle(output: Path, result, kind: str, verifier, extras=None, quarantine=None):
    output = Path(output); parent = output.parent; temp = parent / (output.name + ".inprogress")
    if output.exists() or temp.exists():
        raise FileExistsError("nonclean_bundle")
    files = bundle_bytes(result, kind, extras); temp.mkdir(parents=False)
    try:
        for name in [n for n in sorted(files) if n not in {"manifest.json", "commit.json"}] + ["manifest.json", "commit.json"]:
            atomic_create(temp / name, files[name])
        if not verify_bundle(temp, kind) or not verifier(temp):
            raise RuntimeError("precommit_verifier")
        output.mkdir(parents=False)
        for name in [n for n in sorted(files) if n not in {"manifest.json", "commit.json"}] + ["manifest.json", "commit.json"]:
            os.link(temp / name, output / name); fsync_file(output / name)
        if not verify_bundle(output, kind):
            raise RuntimeError("postcommit_verify")
    except Exception:
        if output.exists() and not verify_bundle(output, kind):
            qroot = Path(quarantine) if quarantine is not None else output.parent / (output.name + "_quarantine")
            qroot.mkdir(parents=True, exist_ok=True); target = qroot / f"{output.name}.partial.{time.time_ns()}.{uuid.uuid4().hex}"
            output.rename(target)
        raise
    finally:
        if temp.exists(): shutil.rmtree(temp)


def clean_or_quarantine(output: Path, failure_root: Path, quarantine: Path, kind: str):
    """Return already_complete, clean, or quarantine stale state then abort."""
    output, failure_root, quarantine = map(Path, (output, failure_root, quarantine)); temp = output.with_name(output.name + ".inprogress")
    if output.exists() and verify_bundle(output, kind):
        return "already_complete"
    if failure_root.exists() and any(failure_root.iterdir()):
        raise RuntimeError("previous_failure_no_retry")
    if quarantine.exists() and any(quarantine.iterdir()):
        raise RuntimeError("prior_quarantine_no_retry")
    stale = [path for path in (output, temp) if path.exists()] + list(output.parent.glob(output.name + ".inprogress.*"))
    if stale:
        quarantine.mkdir(parents=True, exist_ok=True)
        for path in stale:
            target = quarantine / f"{path.name}.{time.time_ns()}.{uuid.uuid4().hex}"
            if target.exists(): raise FileExistsError(target)
            path.rename(target)
        raise RuntimeError("stale_quarantined_no_retry")
    return "clean"


def atomic_failure(failure_root: Path, payload):
    failure_root = Path(failure_root); failure_root.mkdir(parents=True, exist_ok=True)
    attempt = failure_root / ("attempt_" + str(time.time_ns()) + "_" + uuid.uuid4().hex)
    attempt.mkdir()
    reduced = payload
    raw = canonical(reduced)
    if len(raw) > MAX_ARTIFACT_BYTES:
        reduced = {"kind": payload.get("kind", "ph1_nvidia_n4_failure"), "stage": payload.get("stage"), "error": payload.get("error"), "error_type": payload.get("error_type"), "device_opened": payload.get("device_opened"), "oversized_original_bytes": len(raw), "original_sha256": sha(raw), "disposition": "bounded_summary"}
        raw = canonical(reduced)
    atomic_create(attempt / "failure.json", raw)
    return attempt / "failure.json"




