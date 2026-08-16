"""Independent CPU verifier for E100-MRHS256 V3 adopted-baseline evidence."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pro_research" / "results" / "e100_mrhs256" / "PRO_E100_MRHS256.json"
OUT = ROOT / "pro_research" / "results" / "e100_mrhs256" / "PRO_E100_MRHS256_VERIFICATION.json"


def _families(names):
    return {
        "attention_q": "attn_q_bf16" in names,
        "attention_o": "attn_o_bf16" in names,
        "router": "router_f32" in names,
        "mamba_in": any(x.startswith("mamba_in_") for x in names),
        "mamba_out": any(x.startswith("mamba_out_") for x in names),
        "shared_up": "shared_up_nvfp4" in names,
        "shared_down": "shared_down_nvfp4" in names,
        "lm_head": "lm_head_nvfp4" in names,
    }


def _summary(recs, n):
    rr = [r for r in recs if int(r.get("n_rhs", 0)) == n]
    refw = sum(float(r["reference_mid_ms"]) * int(r["calls_per_token"]) for r in rr)
    candw = sum(float(r["candidate_ms"]) * int(r["calls_per_token"]) for r in rr)
    by = {r["name"]: r for r in rr}
    mi = next((r for r in rr if str(r["name"]).startswith("mamba_in_")), None)
    mo = next((r for r in rr if str(r["name"]).startswith("mamba_out_")), None)
    return {
        "weighted": refw / candw if candw else 0.0,
        "minimum": min((float(r["aggregate_speedup"]) for r in rr), default=0.0),
        "max_drift": max((float(r["reference_drift_fraction"]) for r in rr), default=math.inf),
        "lm": float(by.get("lm_head_nvfp4", {}).get("aggregate_speedup", 0.0)),
        "mi": 0.0 if mi is None else float(mi["aggregate_speedup"]),
        "mo": 0.0 if mo is None else float(mo["aggregate_speedup"]),
    }


def main() -> int:
    if not SRC.exists():
        print(f"missing result: {SRC}")
        return 2
    p = json.loads(SRC.read_text(encoding="utf-8"))
    recs = p.get("records") or []
    errors = []
    all_exact = bool(recs)

    if p.get("runner_revision") != "v3_adopted_v6_baseline":
        errors.append("result was not produced by MRHS256 V3")
    if p.get("baseline_revision") != "adopted_v6_selective_single_rhs":
        errors.append("unexpected performance baseline")

    for r in recs:
        key = f"N{r.get('n_rhs')}:{r.get('name')}"
        bs = r.get("correctness_batches") or []
        exact = bool(bs) and all(
            int(b.get("production_vs_adopted_mismatch_count", -1)) == 0
            and int(b.get("adopted_vs_candidate_mismatch_count", -1)) == 0
            and int(b.get("candidate_repeat_mismatch_count", -1)) == 0
            and bool(b.get("finite")) for b in bs
        )
        all_exact = all_exact and exact
        if bool(r.get("all_exact")) != exact:
            errors.append(f"{key}: all_exact summary mismatch")
        if r.get("reference_name") != "adopted_v6_selective_single_rhs":
            errors.append(f"{key}: stale/non-adopted reference")
        ra = float(r.get("reference_a_ms", math.nan))
        rb = float(r.get("reference_b_ms", math.nan))
        cm = float(r.get("candidate_ms", math.nan))
        if not all(math.isfinite(x) and x > 0 for x in (ra, rb, cm)):
            errors.append(f"{key}: invalid timing")
            continue
        mid = 0.5 * (ra + rb); speed = mid / cm; drift = abs(ra - rb) / mid
        if abs(float(r.get("reference_mid_ms", math.nan)) - mid) > max(1e-7, mid * 1e-6): errors.append(f"{key}: midpoint mismatch")
        if abs(float(r.get("aggregate_speedup", math.nan)) - speed) > max(1e-7, speed * 1e-6): errors.append(f"{key}: speedup mismatch")
        if abs(float(r.get("reference_drift_fraction", math.nan)) - drift) > max(1e-7, drift * 1e-6): errors.append(f"{key}: drift mismatch")

    names = set(p.get("supported_cases") or [])
    fam = _families(names)
    all_families = all(fam.values())
    if (p.get("family_support") or {}) != fam:
        errors.append("family support evidence mismatch")
    dispatch = p.get("adopted_dispatch_counters") or {}
    has_fp8 = any(str(r.get("kind")) == "fp8" for r in recs)
    selected_exercised = int(dispatch.get("bf16_ervf", 0)) > 0 and (
        int(dispatch.get("fp8_ervf", 0)) > 0 if has_fp8 else True
    )

    n4 = _summary(recs, 4)
    mode = p.get("mode")
    if mode == "full":
        n16 = _summary(recs, 16)
        perf = {
            "n4_weighted_ge_1_50": n4["weighted"] >= 1.50,
            "n16_weighted_ge_3_0": n16["weighted"] >= 3.0,
            "n16_lm_head_ge_3_0": n16["lm"] >= 3.0,
            "n16_mamba_in_ge_2_5": n16["mi"] >= 2.5,
            "n16_mamba_out_ge_2_5": n16["mo"] >= 2.5,
            "n16_no_case_below_0_95": n16["minimum"] >= 0.95,
            "n16_all_ref_drift_le_7pct": n16["max_drift"] <= 0.07,
        }
        perf_pass = all(perf.values())
    else:
        n16 = None
        perf = {"n4_weighted_ge_1_50": n4["weighted"] >= 1.50}
        perf_pass = None

    if not all_exact or not all_families or not selected_exercised:
        expected = "correctness_failed"
    elif mode == "smoke":
        expected = "smoke_pass"
    elif perf_pass:
        expected = "mrhs256_candidate"
    else:
        expected = "micro_null"
    if p.get("status") != expected:
        errors.append(f"status mismatch file={p.get('status')} recomputed={expected}")

    out = {
        "kind": "pro_e100_mrhs256_v3_independent_verification",
        "source": str(SRC.relative_to(ROOT)),
        "source_status": p.get("status"),
        "recomputed_status": expected,
        "all_production_adopted_candidate_exact": all_exact,
        "family_support": fam,
        "all_eight_families": all_families,
        "adopted_selected_dense_dispatch_exercised": selected_exercised,
        "n4": n4,
        "n16": n16,
        "performance_gates": perf,
        "performance_pass": perf_pass,
        "errors": errors,
        "passed": not errors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(json.dumps(out, indent=2))
    return 0 if out["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
