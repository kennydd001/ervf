"""Protected-path integrity manifest for the LIGHTNINGSTREAM_NEMOTRON line.

The Qwen3-Coder-Next / PORT80B / STREAMQ5 research line is owned by another
agent and is still active.  This tool freezes a byte-level fingerprint of every
pre-existing artifact in the repository so that any accidental modification by
the Nemotron line is detected as a hard stop.

Scope model
-----------
Everything in the repository is protected EXCEPT the Nemotron write allowlist.
That is deliberately broader than the enumerated protected paths in the
assignment: it automatically covers CRAFT, RSIV, FLEQ, E2GQ, HERA/DCHERA/
LDHERA/ADHERA, CORETAIL, BITFLOW, STREAMQ4, STREAMQ5, ERVF, P13, PORT80B and
TierFlow reports and registries, plus the DeepSeek-V2-Lite baseline, without
depending on a name pattern staying in sync.

Fingerprint tiers
-----------------
``full``     files <= FULL_HASH_LIMIT bytes: complete SHA-256.
``partial``  files >  FULL_HASH_LIMIT bytes: size, mtime_ns, and SHA-256 over
             the leading and trailing EDGE_BYTES.  Reading 174 GiB of frozen
             model shards and capture tensors in full after every phase is not
             affordable; size + mtime + both edges detects any in-place write,
             truncation, append or replacement that a research runtime could
             plausibly cause.
``listing``  virtual-environment and cache trees: no content is read at all.
             A digest over the sorted (relpath, size, mtime_ns) triples detects
             any package install, removal or rewrite.

The tier of every entry is recorded, so a report can never silently present a
partial check as a full one.

Usage
-----
    python protected_manifest.py build  --out <manifest.json> [--label TEXT]
    python protected_manifest.py verify --baseline <manifest.json>
                                        --out <diff.json> [--label TEXT]

``verify`` exits 0 when the protected set is unchanged and 2 when it is not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FULL_HASH_LIMIT = 32 * 1024 * 1024
EDGE_BYTES = 4 * 1024 * 1024
CHUNK = 1024 * 1024

# Paths the Nemotron line is allowed to write.  Relative to REPO_ROOT, POSIX
# separators, matched as directory prefixes or exact file paths.
WRITE_ALLOWLIST = (
    "reports/lightningstream_nemotron",
    "reports/runs/lightningstream_nemotron",
    "scripts/lightningstream_nemotron",
    "src/moe_lab/lightningstream_nemotron",
    "tests/lightningstream_nemotron",
    "models/nemotron_3_5_lightning",
    ".cache/nemotron_3_5_lightning",
    ".venv-nemotron",
    "docs/LIGHTNINGSTREAM_NEMOTRON_RESEARCH_LOG.md",
)

# Trees fingerprinted by listing only.  ``third_party`` is a vendored external
# checkout (llama.cpp, including a node_modules build tree) rather than research
# output; a listing digest still detects any change to it at a fraction of the
# cost of hashing several hundred thousand dependency files.
LISTING_ROOTS = (
    ".venv",
    ".venv-next-ref",
    ".pytest_cache",
    "third_party",
)

# Trees excluded entirely.  ``.git`` is tracked separately via its HEAD state;
# ``__pycache__`` is a derived byproduct of merely importing protected code.
EXCLUDED_ROOTS = (
    ".git",
)
EXCLUDED_DIR_NAMES = frozenset({"__pycache__"})


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _has_prefix(rel: str, prefixes: tuple[str, ...]) -> bool:
    for prefix in prefixes:
        if rel == prefix or rel.startswith(prefix + "/"):
            return True
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_range(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining > 0:
            block = handle.read(min(CHUNK, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict:
    try:
        stat = path.stat()
    except OSError as exc:
        # Reparse points and broken dependency shims inside vendored trees are
        # recorded rather than skipped, so their appearance or disappearance is
        # still visible to verification.
        return {"tier": "unreadable", "bytes": 0, "error": f"{type(exc).__name__}:{exc.errno}"}
    size = stat.st_size
    try:
        if size <= FULL_HASH_LIMIT:
            return {
                "tier": "full",
                "bytes": size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256_file(path),
            }
        edge = min(EDGE_BYTES, size)
        return {
            "tier": "partial",
            "bytes": size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256_head": _sha256_range(path, 0, edge),
            "sha256_tail": _sha256_range(path, size - edge, edge),
            "edge_bytes": edge,
        }
    except OSError as exc:
        return {"tier": "unreadable", "bytes": size,
                "error": f"{type(exc).__name__}:{exc.errno}"}


def _walk(root: Path):
    """Yield protected files, skipping allowlist, listing and excluded trees."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda e: e.name)
        except (PermissionError, FileNotFoundError):
            continue
        for entry in entries:
            path = Path(entry.path)
            rel = _rel(path)
            if entry.is_dir(follow_symlinks=False):
                if entry.name in EXCLUDED_DIR_NAMES:
                    continue
                if _has_prefix(rel, EXCLUDED_ROOTS):
                    continue
                if _has_prefix(rel, LISTING_ROOTS):
                    continue
                if _has_prefix(rel, WRITE_ALLOWLIST):
                    continue
                stack.append(path)
            elif entry.is_file(follow_symlinks=False):
                if _has_prefix(rel, WRITE_ALLOWLIST):
                    continue
                yield path, rel


def _listing_digest(root: Path) -> dict:
    """Digest a tree by (relpath, size, mtime_ns) without reading content."""
    rows: list[str] = []
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda e: e.name)
        except (PermissionError, FileNotFoundError):
            continue
        for entry in entries:
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                stack.append(path)
            elif entry.is_file(follow_symlinks=False):
                try:
                    stat = entry.stat()
                except OSError as exc:
                    rows.append(f"{_rel(path)}\0unreadable\0{type(exc).__name__}:{exc.errno}")
                    continue
                rows.append(f"{_rel(path)}\0{stat.st_size}\0{stat.st_mtime_ns}")
                total += stat.st_size
    rows.sort()
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return {"tier": "listing", "files": len(rows), "bytes": total, "sha256": digest}


def _git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "status_short_lines": len((run("status", "--short") or "").splitlines()),
    }


def build(label: str) -> dict:
    files: dict[str, dict] = {}
    full_count = partial_count = 0
    total_bytes = 0
    for path, rel in _walk(REPO_ROOT):
        entry = _fingerprint(path)
        files[rel] = entry
        total_bytes += entry["bytes"]
        if entry["tier"] == "full":
            full_count += 1
        else:
            partial_count += 1

    listings: dict[str, dict] = {}
    for name in LISTING_ROOTS:
        root = REPO_ROOT / name
        if root.is_dir():
            listings[name] = _listing_digest(root)

    ordered = {rel: files[rel] for rel in sorted(files)}
    root_material = json.dumps(
        {"files": ordered, "listings": listings},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return {
        "kind": "lightningstream_nemotron_protected_manifest",
        "version": 1,
        "label": label,
        "repo_root": str(REPO_ROOT),
        "policy": {
            "full_hash_limit_bytes": FULL_HASH_LIMIT,
            "edge_bytes": EDGE_BYTES,
            "write_allowlist": list(WRITE_ALLOWLIST),
            "listing_roots": list(LISTING_ROOTS),
            "excluded_roots": list(EXCLUDED_ROOTS),
            "excluded_dir_names": sorted(EXCLUDED_DIR_NAMES),
        },
        "git": _git_state(),
        "summary": {
            "protected_files": len(ordered),
            "full_hashed_files": full_count,
            "partial_hashed_files": partial_count,
            "protected_bytes": total_bytes,
            "listing_trees": len(listings),
        },
        "root_digest": hashlib.sha256(root_material).hexdigest(),
        "files": ordered,
        "listings": listings,
    }


def verify(baseline: dict, label: str) -> dict:
    current = build(label)
    old_files = baseline["files"]
    new_files = current["files"]

    removed = sorted(set(old_files) - set(new_files))
    added = sorted(set(new_files) - set(old_files))
    modified: list[dict] = []
    touched: list[dict] = []

    for rel in sorted(set(old_files) & set(new_files)):
        before, after = old_files[rel], new_files[rel]
        if before["tier"] != after["tier"]:
            modified.append({"path": rel, "reason": "tier_changed",
                             "before": before, "after": after})
            continue
        if before["tier"] == "unreadable":
            if before.get("error") != after.get("error") or before["bytes"] != after["bytes"]:
                modified.append({"path": rel, "reason": "unreadable_entry_changed",
                                 "before": before, "after": after})
        elif before["tier"] == "full":
            if before["sha256"] != after["sha256"] or before["bytes"] != after["bytes"]:
                modified.append({"path": rel, "reason": "content_changed",
                                 "before": before, "after": after})
            elif before["mtime_ns"] != after["mtime_ns"]:
                touched.append({"path": rel, "reason": "mtime_only",
                                "before_mtime_ns": before["mtime_ns"],
                                "after_mtime_ns": after["mtime_ns"]})
        else:
            content_changed = (
                before["bytes"] != after["bytes"]
                or before["sha256_head"] != after["sha256_head"]
                or before["sha256_tail"] != after["sha256_tail"]
            )
            if content_changed:
                modified.append({"path": rel, "reason": "content_changed",
                                 "before": before, "after": after})
            elif before["mtime_ns"] != after["mtime_ns"]:
                # For a partial entry an mtime change is NOT provably benign:
                # an interior write leaves both edges intact.  Escalate.
                modified.append({"path": rel, "reason": "partial_mtime_changed",
                                 "before": before, "after": after})

    listing_changed: list[dict] = []
    for name, before in baseline.get("listings", {}).items():
        after = current["listings"].get(name)
        if after is None:
            listing_changed.append({"tree": name, "reason": "tree_missing"})
        elif before["sha256"] != after["sha256"]:
            listing_changed.append({"tree": name, "reason": "listing_changed",
                                    "before": before, "after": after})

    # `added` is informational: a protected directory gaining a file is not a
    # modification of the 80B agent's bytes, but it must still be visible.
    hard_stop = bool(removed or modified or listing_changed)

    return {
        "kind": "lightningstream_nemotron_protected_verification",
        "version": 1,
        "label": label,
        "baseline_label": baseline.get("label"),
        "baseline_root_digest": baseline["root_digest"],
        "current_root_digest": current["root_digest"],
        "root_digest_match": baseline["root_digest"] == current["root_digest"],
        "git_before": baseline.get("git"),
        "git_now": current["git"],
        "summary_before": baseline["summary"],
        "summary_now": current["summary"],
        "counts": {
            "removed": len(removed),
            "modified": len(modified),
            "touched_mtime_only": len(touched),
            "added": len(added),
            "listing_changed": len(listing_changed),
        },
        "removed": removed,
        "modified": modified,
        "touched_mtime_only": touched,
        "added": added,
        "listing_changed": listing_changed,
        "hard_stop": hard_stop,
        "verdict": "PROTECTED_80B_INTACT" if not hard_stop else "PROTECTED_80B_VIOLATION",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--label", default="unlabeled")

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--baseline", required=True)
    p_verify.add_argument("--out", required=True)
    p_verify.add_argument("--label", default="unlabeled")

    args = parser.parse_args()

    if args.mode == "build":
        payload = build(args.label)
        _write_json(Path(args.out), payload)
        print(f"protected files : {payload['summary']['protected_files']}")
        print(f"  full-hashed   : {payload['summary']['full_hashed_files']}")
        print(f"  partial-hashed: {payload['summary']['partial_hashed_files']}")
        print(f"protected bytes : {payload['summary']['protected_bytes']}")
        print(f"listing trees   : {payload['summary']['listing_trees']}")
        print(f"root digest     : {payload['root_digest']}")
        return 0

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    report = verify(baseline, args.label)
    _write_json(Path(args.out), report)
    print(f"verdict        : {report['verdict']}")
    print(f"root digest ok : {report['root_digest_match']}")
    for key, value in report["counts"].items():
        print(f"  {key:<20}: {value}")
    if report["hard_stop"]:
        for item in report["modified"][:20]:
            print(f"  !! {item['reason']}: {item['path']}")
        for item in report["removed"][:20]:
            print(f"  !! removed: {item}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
