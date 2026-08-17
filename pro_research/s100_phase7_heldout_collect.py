
"""Collect fresh-process heldout results without rerunning a GPU."""
from __future__ import annotations
import json
from common import REPO, utc_now, write_json_atomic
from s100_phase7_common import (
    EXPECTED,
    load_frozen_candidates,
    public_spec,
)

OUT = (
    REPO / "pro_research" / "results" / "S100_PHASE7_HELDOUT.json"
)


def main() -> int:
    candidates = load_frozen_candidates()
    results = {}
    for name in EXPECTED:
        path = (
            REPO / "pro_research" / "results"
            / f"S100_PHASE7_HELDOUT_{name.upper()}.json"
        )
        if not path.exists():
            results[name] = {
                "status": "missing",
                "spec": public_spec(candidates[name]),
            }
            continue
        results[name] = json.loads(path.read_text(encoding="utf-8"))

    payload = {
        "kind": "s100_phase7_heldout",
        "status": "complete",
        "created_utc": utc_now(),
        "frozen_candidates": {
            name: public_spec(candidates[name]) for name in EXPECTED
        },
        "results": results,
        "green": [
            name
            for name, rec in results.items()
            if rec.get("status") == "v18_fidelity_candidate"
        ],
        "technical_failures": [
            name
            for name, rec in results.items()
            if rec.get("status") in {"technical_failure", "missing"}
        ],
    }
    write_json_atomic(OUT, payload, archive=True)
    print(
        json.dumps(
            {
                "green": payload["green"],
                "technical_failures": payload["technical_failures"],
                "output": str(OUT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
