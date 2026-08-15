"""Write a phase input lock: hashes of every source artifact a phase depended on.

A lock binds the preregistration, runner(s), verifier(s) and produced results by
byte count and SHA-256 so a later phase can prove which exact sources it
inherited.  Files are read from the working tree at call time; the lock is the
record of that moment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(paths: list[str], role: str) -> list[dict]:
    rows = []
    for rel in paths:
        path = REPO_ROOT / rel
        if not path.is_file():
            rows.append({"path": rel, "role": role, "present": False})
            continue
        rows.append({
            "path": rel,
            "role": role,
            "present": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preregistration", nargs="*", default=[])
    parser.add_argument("--runner", nargs="*", default=[])
    parser.add_argument("--verifier", nargs="*", default=[])
    parser.add_argument("--result", nargs="*", default=[])
    parser.add_argument("--inherited", nargs="*", default=[])
    args = parser.parse_args()

    entries = (
        bind(args.preregistration, "preregistration")
        + bind(args.runner, "runner")
        + bind(args.verifier, "verifier")
        + bind(args.result, "result")
        + bind(args.inherited, "inherited_readonly")
    )

    payload = {
        "kind": "lightningstream_nemotron_input_lock",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": args.phase,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(entries),
        "all_present": all(e["present"] for e in entries),
        "entries": entries,
    }

    out = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"phase       : {args.phase}")
    print(f"entries     : {len(entries)}")
    print(f"all present : {payload['all_present']}")
    for entry in entries:
        mark = entry.get("sha256", "MISSING")[:16]
        print(f"  [{entry['role']:<18}] {mark}  {entry['path']}")
    return 0 if payload["all_present"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
