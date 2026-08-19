from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-archive", action="store_true")
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry.get("storage") == "sharded":
        rows = []
        for shard in registry.get("shards") or []:
            value = json.loads(
                (args.registry.parent / shard["path"]).read_text(encoding="utf-8")
            )
            rows.extend(value.get("experiments") or [])
    else:
        rows = registry["experiments"]
    if not args.include_archive:
        rows = [x for x in rows if x["migration_decision"] not in {"ARCHIVE"}]
    rows.sort(key=lambda x: (int(x["wave"]), x["id"]))

    waves = defaultdict(list)
    for row in rows:
        waves[str(row["wave"])].append(row)

    result = {
        "kind": "s100_lightning_retest_plan",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_count": len(rows),
        "waves": dict(waves),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "RETEST_PLAN.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Generated Lightning retest plan", "",
        f"Generated: {result['created_utc']}", "",
        "The order is dependency-first. A failed technical step remains "
        "`INCOMPLETE`; it never closes the hypothesis.", "",
    ]
    for wave in sorted(waves, key=lambda x: int(x)):
        lines += [f"## Wave {wave}", ""]
        for row in waves[wave]:
            lines += [
                f"### {row['id']} — {row['title']}", "",
                f"- Decision: `{row['migration_decision']}`",
                f"- Transfer class: `{row['transfer_class']}`",
                f"- Dependencies: {', '.join(row['dependencies']) or 'none'}",
                f"- Test: {row['lightning_test']}",
                f"- Gate: {row['success_gate']}", "",
            ]
    md_path = args.output_dir / "RETEST_PLAN.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "experiments": len(rows),
        "waves": {k: len(v) for k, v in waves.items()},
        "json": str(json_path),
        "markdown": str(md_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
