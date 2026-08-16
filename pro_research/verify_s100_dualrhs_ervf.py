"""Independent CPU verifier for PRO_S100_DUALRHS_ERVF.json."""
from __future__ import annotations

import json
from pathlib import Path

from common import REPO, utc_now

SRC = REPO / "pro_research" / "results" / "s100_dualrhs" / "PRO_S100_DUALRHS_ERVF.json"
OUT = REPO / "pro_research" / "results" / "s100_dualrhs" / "PRO_S100_DUALRHS_ERVF_VERIFICATION.json"


def main() -> int:
    errors = []
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    p = json.loads(SRC.read_text(encoding="utf-8"))
    recs = p.get("records") or []
    names = {r.get("name") for r in recs}
    required_names = {
        "attn_q_bf16", "attn_k_bf16", "attn_v_bf16", "attn_o_bf16",
        "mamba_in_fp8", "mamba_out_fp8", "router_f32",
        "shared_up_nvfp4", "shared_down_nvfp4", "lm_head_nvfp4",
    }
    all_exact = bool(recs) and all(bool(r.get("all_exact")) for r in recs)
    all_families = names == required_names
    try:
        wr = sum(float(r["reference_mid_ms_per_pair"]) * int(r["calls_per_token"]) for r in recs)
        wc = sum(float(r["candidate_mid_ms_per_pair"]) * int(r["calls_per_token"]) for r in recs)
        speed = wr / wc
        saving = wr - wc
        max_drift = max(float(r["reference_drift_fraction"]) for r in recs)
        min_speed = min(float(r["speedup"]) for r in recs)
        mamba_in = next(float(r["speedup"]) for r in recs if str(r["name"]).startswith("mamba_in_"))
        lm_head = next(float(r["speedup"]) for r in recs if str(r["name"]).startswith("lm_head_"))
    except Exception as exc:
        errors.append(f"summary recompute failed: {type(exc).__name__}: {exc}")
        wr = wc = speed = saving = max_drift = min_speed = mamba_in = lm_head = 0.0

    full = p.get("mode") == "full"
    gates = {
        "G1_G4_all_outputs_bitexact_deterministic_finite": all_exact,
        "G5_all_10_families_present": all_families,
        "P1_weighted_common_projection_speedup_ge_1_50x": speed >= 1.50,
        "P2_projected_common_projection_saving_ge_6_0ms_per_K2_block": saving >= 6.0,
        "P3_mamba_in_speedup_ge_1_40x": mamba_in >= 1.40,
        "P4_lm_head_speedup_ge_1_35x": lm_head >= 1.35,
        "P5_no_registered_family_below_0_90x": min_speed >= 0.90,
        "D1_max_reference_drift_le_0_07": (max_drift <= 0.07) if full else None,
    }
    integration_open = bool(
        all_exact and all_families and
        gates["P1_weighted_common_projection_speedup_ge_1_50x"] and
        gates["P2_projected_common_projection_saving_ge_6_0ms_per_K2_block"] and
        gates["P5_no_registered_family_below_0_90x"] and
        (not full or gates["D1_max_reference_drift_le_0_07"])
    )
    gates["integration_open"] = integration_open

    stored_summary = p.get("summary") or {}
    checks = {
        "weighted_reference_ms_per_K2_common_projection_subset": wr,
        "weighted_candidate_ms_per_K2_common_projection_subset": wc,
        "weighted_speedup": speed,
        "projected_common_projection_saving_ms_per_K2_block": saving,
        "max_reference_drift_fraction": max_drift,
        "min_family_speedup": min_speed,
        "mamba_in_speedup": mamba_in,
        "lm_head_speedup": lm_head,
    }
    for k, v in checks.items():
        if k not in stored_summary:
            errors.append(f"missing summary key {k}")
        elif abs(float(stored_summary[k]) - float(v)) > max(1e-9, abs(float(v)) * 1e-9):
            errors.append(f"summary mismatch {k}: stored={stored_summary[k]} recomputed={v}")
    stored_gates = p.get("gates") or {}
    for k, v in gates.items():
        if stored_gates.get(k) != v:
            errors.append(f"gate mismatch {k}: stored={stored_gates.get(k)!r} recomputed={v!r}")

    if not all_exact or not all_families:
        status = "correctness_failed"
    elif not full:
        status = "smoke_pass"
    elif integration_open:
        status = "dualrhs_integration_candidate"
    elif saving < 3.0:
        status = "micro_null_close_geometry"
    else:
        status = "micro_below_integration_gate"
    if p.get("status") != status:
        errors.append(f"status mismatch stored={p.get('status')} recomputed={status}")

    result = {
        "kind": "s100_dualrhs_ervf_independent_verification",
        "created_utc": utc_now(),
        "source": str(SRC.relative_to(REPO)),
        "source_status": p.get("status"),
        "recomputed_status": status,
        "recomputed_summary": checks,
        "recomputed_gates": gates,
        "errors": errors,
        "passed": not errors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
