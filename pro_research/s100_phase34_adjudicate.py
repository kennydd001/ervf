from __future__ import annotations

from collections import Counter
import json

import numpy as np

from common import utc_now, write_json_atomic
from s100_phase34_panel_reuse import OUT, PANEL_BYTES, RESULTS, TRACE


def main() -> int:
    diagnostic = json.loads(OUT.read_text(encoding="utf-8"))
    with np.load(TRACE) as trace:
        keys = list(
            zip(
                trace["layer"].tolist(),
                trace["expert"].tolist(),
                trace["panel"].tolist(),
            )
        )
    counts = Counter(keys)
    total = len(keys)
    oracle = {}
    for mib in (32, 64, 96, 128):
        entries = mib * 2**20 // PANEL_BYTES
        hottest = counts.most_common(entries)
        repeat_hits = sum(max(count - 1, 0) for _, count in hottest)
        oracle[str(mib)] = {
            "capacity_entries": entries,
            "repeat_hits": repeat_hits,
            "repeat_hit_rate": repeat_hits / max(total, 1),
            "host_bytes_avoided": repeat_hits * PANEL_BYTES,
        }
    best = oracle["128"]
    payload = {
        "kind": "s100_phase34_adjudication",
        "status": "measured",
        "created_utc": utc_now(),
        "tokens_exact": diagnostic.get("all_tokens_exact"),
        "lru": diagnostic.get("lru"),
        "static_frequency_oracle": oracle,
        "gates": {
            "measured_lru_128_hit_rate_ge_20pct": float(
                diagnostic["lru"]["128"]["steady_state_hit_rate"]
            ) >= 0.20,
            "oracle_128_repeat_hit_rate_ge_20pct": float(
                best["repeat_hit_rate"]
            ) >= 0.20,
            "tokens_exact": bool(diagnostic.get("all_tokens_exact")),
        },
        "PERSISTENT_PANEL_CACHE_IMPLEMENTATION_OPEN": False,
        "NEXT_ROUTE": "NATIVE_FP4_REAL_ACTIVATION_C3B_C3C",
        "claim_boundary": "trace and cache oracle only; no runtime cache or speed claim",
    }
    write_json_atomic(
        RESULTS / "S100_PHASE34_ADJUDICATION.json", payload, archive=True
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
