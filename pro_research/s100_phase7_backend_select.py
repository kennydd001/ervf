
from __future__ import annotations
import json
from common import REPO, utc_now, write_json_atomic

OUT = (
    REPO / "pro_research" / "results"
    / "S100_PHASE7_BACKEND_SELECT.json"
)


def main() -> int:
    p = (
        REPO / "pro_research" / "results"
        / "S100_PHASE7_PACKED_COMPARE_FULL.json"
    )
    result = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    selected = (
        "packed"
        if result.get("status") == "exact_backend_candidate"
        else "legacy"
    )
    payload = {
        "kind": "s100_phase7_backend_select",
        "created_utc": utc_now(),
        "selected_backend": selected,
        "minimum_gain_ms": 0.15,
        "packed_result": result,
    }
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
