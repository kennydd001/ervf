"""Close Phase41 when overlap and serial pointer-slice arms diverge."""
from __future__ import annotations

import json

from common import REPO, utc_now, write_json_atomic

RESULTS = REPO / "pro_research" / "results" / "s100_phase41"


def load(arm: str) -> dict:
    return json.loads(
        (RESULTS / f"S100_PHASE41_{arm}_CTX1024.json").read_text(encoding="utf-8")
    )


def main() -> int:
    compile_result = load("COMPILE")
    base = load("BASE_A")
    overlap = load("FULL_PIPELINE_B3")
    serial = load("SERIAL_CONTROL")
    resources = compile_result["kernel_resources"]
    zero_local = all(
        int(row.get("local_size_bytes") or 0) == 0 for row in resources.values()
    )
    payload = {
        "kind": "s100_phase41_adjudication",
        "status": "correctness_failed",
        "created_utc": utc_now(),
        "base_a": {
            "status": base.get("status"),
            "median_ms": (base.get("summary") or {}).get("median_ms"),
            "tokens_exact": base.get("tokens_exact"),
        },
        "overlap": {
            "status": overlap.get("status"),
            "error": (overlap.get("error") or {}).get("message"),
        },
        "serial_control": {
            "status": serial.get("status"),
            "error": (serial.get("error") or {}).get("message"),
        },
        "gates": {
            "G41_R1_zero_local_memory": zero_local,
            "G41_C1_overlap_exact": False,
            "G41_C2_serial_control_exact": False,
            "G41_DIAG_failure_is_not_cross_stream": True,
        },
        "root_cause": (
            "Pointer-sliced group metadata changed effective group-index semantics. "
            "Phase42 replaced slicing with explicit global g0 indexing and passed."
        ),
        "claim_boundary": "correctness closure; no Phase41 speed claim",
    }
    write_json_atomic(RESULTS / "S100_PHASE41_ADJUDICATION.json", payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

