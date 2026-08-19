from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Iterable

NANO = re.compile(
    r"(NVIDIA-Nemotron-3-Nano|Nemotron[-_ ]?3[-_ ]?Nano|"
    r"nemotron_3_5_lightning(?!_v35))", re.I,
)
LIGHTNING = re.compile(
    r"(NVIDIA-Nemotron-3\.5-Lightning|nemotron_3_5_lightning_v35)", re.I,
)
DEEPSEEK = re.compile(r"DeepSeek[-_ ]?V2[-_ ]?Lite", re.I)


def strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, (str, int, float, bool)):
        yield str(value)


def classify(blob: str) -> str:
    n = bool(NANO.search(blob)); l = bool(LIGHTNING.search(blob)); d = bool(DEEPSEEK.search(blob))
    if n and l: return "mixed_or_conflicting"
    if n: return "nano_evidence"
    if l: return "lightning_claim_unverified"
    if d: return "different_model_deepseek"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    candidates = []
    for base in (repo / "pro_research" / "results", repo / "reports", repo / "agents"):
        if not base.exists(): continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".md", ".txt"} and path.stat().st_size <= 8 << 20:
                candidates.append(path)

    rows = []
    for path in sorted(candidates):
        text = path.read_text(encoding="utf-8", errors="replace")
        obj = None; kind = status = model_dir = None
        if path.suffix.lower() == ".json":
            try: obj = json.loads(text)
            except json.JSONDecodeError: pass
        if isinstance(obj, dict):
            kind = obj.get("kind"); status = obj.get("status")
            if isinstance(obj.get("environment"), dict):
                model_dir = obj["environment"].get("model_dir")
            model_dir = model_dir or obj.get("model_dir")
            blob = "\n".join(strings(obj))
        else:
            blob = text
        rows.append({
            "path": str(path.relative_to(repo)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "kind": kind, "status": status, "model_dir": model_dir,
            "provenance": classify(blob),
            "technical_incomplete": bool(
                status == "technical_failure" or
                re.search(r"instrumentation complete:\s*false", blob, re.I)
            ),
        })

    result = {
        "kind": "s100_current_tree_result_provenance_audit",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo": str(repo), "file_count": len(rows), "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    j = args.output_dir / "CURRENT_TREE_PROVENANCE_AUDIT.json"
    j.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    c = args.output_dir / "CURRENT_TREE_PROVENANCE_AUDIT.csv"
    with c.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "path", "bytes", "kind", "status", "model_dir", "provenance", "technical_incomplete"
        ])
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"files": len(rows), "json": str(j), "csv": str(c)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
