from __future__ import annotations

import json

from common import utc_now, write_json_atomic
from s100_phase33_common import ARMS, RESULTS


def main() -> int:
    control = json.loads(
        (RESULTS / "S100_PHASE33_SCREEN_PHASE32_CONTROL_CTX1024.json").read_text(
            encoding="utf-8"
        )
    )
    control_ms = float(control["summary"]["median_ms"])
    rows = []
    for arm in ARMS:
        result = json.loads(
            (RESULTS / f"S100_PHASE33_SCREEN_{arm.upper()}_CTX1024.json").read_text(
                encoding="utf-8"
            )
        )
        median = float(result["summary"]["median_ms"])
        rows.append(
            {
                "arm": arm,
                "median_ms": median,
                "target_only_tok_s": 8000.0 / median,
                "gain_vs_phase32_control": (control_ms - median) / control_ms,
                "all_tokens_exact": bool(result["summary"]["all_token_exact"]),
            }
        )
    compile_result = json.loads(
        (RESULTS / "S100_PHASE33_COMPILE_COMPILE_CTX1024.json").read_text(
            encoding="utf-8"
        )
    )
    resource = compile_result["kernel_resources"]["nvfp4_m8_warp32_direct_l2"]
    best = min(rows, key=lambda row: row["median_ms"])
    promotion = bool(
        best["gain_vs_phase32_control"] >= 0.03
        and best["all_tokens_exact"]
        and int(resource.get("local_size_bytes") or 0) == 0
    )
    payload = {
        "kind": "s100_phase33_adjudication",
        "status": "measured",
        "created_utc": utc_now(),
        "control_median_ms": control_ms,
        "kernel_resource": resource,
        "arms": rows,
        "selected": best,
        "RUN_STATE_THERMAL_PROMOTION": promotion,
        "PHASE33_ADOPTED": False,
        "S100_SINGLE_ACHIEVED": False,
        "NEXT_ROUTE": "MEASURE_TEMPORAL_SPARSE_DOWN_PANEL_REUSE",
        "claim_boundary": "exact target-only H8 component screen",
    }
    write_json_atomic(RESULTS / "S100_PHASE33_ADJUDICATION.json", payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
