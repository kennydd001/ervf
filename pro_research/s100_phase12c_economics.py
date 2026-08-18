from __future__ import annotations

import json
from common import REPO, write_json_atomic, utc_now

R = REPO / "pro_research" / "results"
P12 = R / "s100_phase12"
P12C = R / "s100_phase12c"

def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    dense=load(P12C/"S100_PHASE12C_DENSE.json") or {}
    grouped=load(P12C/"S100_PHASE12C_GROUPED_MOE.json") or {}
    block=load(P12/"S100_PHASE12A_BLOCK_VERIFIER.json") or {}

    dense_b4=bool(dense.get("dense_b4_gate_pass"))
    grouped_b4=bool(grouped.get("grouped_b4_gate_pass"))
    opens=bool(dense_b4 and grouped_b4)

    projection=None
    try:
        floor=float(block["per_B"]["4"]["cycle_ms_median"])
        d4=dense["per_m"]["4"]
        dense_saved=(
            float(d4["baseline_independent_m1"]["median_ms"])
            - float(d4["candidate_ervfm"]["median_ms"])
        )
        # Grouped JSON reports a weighted ratio over the exact Phase-12B
        # distribution, but its absolute units pool the full census. Use only
        # a conservative ratio-derived fraction of the 12A cycle.
        gs=float(grouped["weighted"]["4"]["weighted_speedup"])
        route_reduction=float(
            json.loads(
                (P12/"S100_PHASE12B_CENSUS.json").read_text(encoding="utf-8")
            )["per_B"]["4"]["device_read_reduction_median"]
        )
        # Conservative cap: routed sharing may remove at most the measured
        # route-read fraction from 35% of the block floor.
        grouped_saved_upper=floor*0.35*route_reduction
        grouped_saved_conservative=grouped_saved_upper*max(
            0.0,min(1.0,(gs-1.0)/0.20)
        )
        direct=floor-dense_saved-grouped_saved_conservative
        projection={
            "claim_boundary":"component substitution projection, not measured verifier",
            "phase12a_b4_floor_ms":floor,
            "dense_microbench_saved_ms":dense_saved,
            "grouped_conservative_saved_ms":grouped_saved_conservative,
            "projected_b4_cycle_ms":direct,
            "projected_perfect_draft_tok_s":4000.0/direct if direct>0 else None,
            "b4_gate_ms":28.0,
            "projection_gate_pass":bool(direct<=28.0),
        }
    except Exception as exc:
        projection={"error":f"{type(exc).__name__}: {exc}"}

    out={
        "kind":"s100_phase12c_economics",
        "created_utc":utc_now(),
        "dense_status":dense.get("status"),
        "grouped_status":grouped.get("status"),
        "dense_b4_gate_pass":dense_b4,
        "grouped_b4_gate_pass":grouped_b4,
        "INTEGRATED_VERIFIER_BUILD_OPEN":opens,
        "projection":projection,
        "decision":(
            "OPEN_PHASE12C_INTEGRATED_BLOCK_VERIFIER"
            if opens else
            "CLOSE_OR_REDESIGN_BLOCK_ERVF_MICROKERNELS"
        ),
    }
    P12C.mkdir(parents=True,exist_ok=True)
    write_json_atomic(P12C/"S100_PHASE12C_ECONOMICS.json",out,archive=True)
    print(json.dumps(out,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
