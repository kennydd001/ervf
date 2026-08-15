"""S10-A input lock: freeze the text this phase is allowed to see.

Run with a python that has pyarrow (`.venv`), NOT the runtime venv -- the point
is that the corpus is fixed on disk before the measurement venv ever starts, so
runner and verifier read identical bytes and neither can reselect text after
seeing a result.

Selection is deterministic and stated here, not tuned: WikiText-2 raw
validation, rows in file order, keeping paragraphs of at least 800 characters
that are not section headers.  The first four go to the A1 wiring resolution;
later rows, disjoint from those four, are concatenated for the long-context
secondary arm.  The three gate prompts are hand-written and live here so that
the runner cannot vary them.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET = REPO_ROOT / "data/corpora/wikitext/wikitext-2-raw-v1/validation-00000-of-00001.parquet"
OUT = REPO_ROOT / "reports/lightningstream_nemotron/s10a_corpus.json"

# Three domains. Written before any measurement; frozen by this file's sha256.
GATE_PROMPTS = [
    {
        "id": "expository",
        "text": "The development of the steam engine transformed manufacturing in Britain. "
                "Early designs were inefficient, and it took several decades before",
    },
    {
        "id": "narrative",
        "text": "The lighthouse keeper had not spoken to anyone for eleven days. "
                "When the boat finally appeared on the horizon, she",
    },
    {
        "id": "code",
        "text": "def merge_intervals(intervals):\n"
                "    \"\"\"Merge overlapping intervals and return them sorted by start.\"\"\"\n"
                "    if not intervals:\n"
                "        return []\n",
    },
]


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> int:
    import pyarrow.parquet as pq

    table = pq.read_table(PARQUET)
    rows = table.column("text").to_pylist()

    kept: list[tuple[int, str]] = []
    for i, text in enumerate(rows):
        t = (text or "").strip()
        if len(t) < 800:
            continue
        if t.startswith("="):
            continue
        kept.append((i, t))

    a1 = kept[:4]
    long_rows = kept[4:]

    long_text_parts, total = [], 0
    for _, t in long_rows:
        long_text_parts.append(t)
        total += len(t)
        if total >= 22000:          # ~4-6k tokens; the runner truncates by token
            break

    payload = {
        "kind": "lightningstream_nemotron_s10a_corpus",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(PARQUET.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": sha256_path(PARQUET),
            "rows_total": len(rows),
            "rows_kept": len(kept),
        },
        "selection_rule": (
            "validation split, file order, strip(), keep len>=800 chars and not "
            "startswith('='); first 4 kept rows -> A1, rows 5.. -> long-context arm"),
        "a1_passages": [{"row": i, "text": t} for i, t in a1],
        "a1_row_indices": [i for i, _ in a1],
        "long_ctx_row_indices": [i for i, _ in long_rows[:len(long_text_parts)]],
        "long_ctx_text": "\n\n".join(long_text_parts),
        "gate_prompts": GATE_PROMPTS,
        "extractor_sha256": sha256_path(Path(__file__)),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"rows kept {len(kept)}; A1 rows {payload['a1_row_indices']}; "
          f"long-ctx chars {len(payload['long_ctx_text'])}")
    print(f"written {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
