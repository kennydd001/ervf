"""Expand the SHA-256 checked PRO research source payload in-place."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import lzma
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

PRO = Path(__file__).resolve().parent
PARTS = PRO / "payload"
EXPECTED_ARCHIVE_SHA256 = "f39a14ae6ca34d6bb1f953a28cdf944ac081be90addd41115e3109a289cffbb0"
EXPECTED_REQUIRED = {
    "common.py", "graph_e1f22.py", "ervf_dense.py", "epoch_graph.py",
    "run_all.py", "verify_results.py", "build_report.py", "INSTALL_AND_RUN.ps1",
    "PRO_HYPOTHESES.md", "EXPERIMENT_REGISTRY.yaml", "SOURCE_MANIFEST_SHA256.json",
}


def safe_member(name: str) -> bool:
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts and bool(p.parts)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    part_files = sorted(PARTS.glob("part_*.txt"))
    if not part_files:
        raise FileNotFoundError(f"no payload parts in {PARTS}")
    encoded = "".join("".join(p.read_text(encoding="ascii").split()) for p in part_files)
    archive = base64.b85decode(encoded.encode("ascii"))
    digest = hashlib.sha256(archive).hexdigest()
    if digest != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"payload SHA-256 mismatch: {digest} != {EXPECTED_ARCHIVE_SHA256}")
    tar_bytes = lzma.decompress(archive)

    with tempfile.TemporaryDirectory(prefix="ervf_pro_") as td:
        unpacked = Path(td) / "unpacked"
        unpacked.mkdir()
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
            names = [m.name for m in tf.getmembers() if m.isfile()]
            bad = [name for name in names if not safe_member(name)]
            if bad:
                raise RuntimeError(f"unsafe archive paths: {bad}")
            tf.extractall(unpacked, filter="data")

        found = {str(p.relative_to(unpacked)).replace("\\", "/") for p in unpacked.rglob("*") if p.is_file()}
        missing = EXPECTED_REQUIRED - found
        if missing:
            raise RuntimeError(f"payload missing required files: {sorted(missing)}")

        manifest = json.loads((unpacked / "SOURCE_MANIFEST_SHA256.json").read_text(encoding="utf-8"))
        mismatches = {}
        for rel, expected in manifest["files"].items():
            actual = sha256(unpacked / rel)
            if actual != expected:
                mismatches[rel] = {"expected": expected, "actual": actual}
        if mismatches:
            raise RuntimeError(f"source manifest mismatch: {mismatches}")

        for source in sorted(unpacked.rglob("*")):
            if not source.is_file():
                continue
            rel = source.relative_to(unpacked)
            target = PRO / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    print(f"Installed {len(found)} verified PRO source files in {PRO}")
    print(f"Payload SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
