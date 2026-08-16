"""Independent CPU verifier for E100-NVFP4-SMEM-MRHS evidence."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pro_research" / "results" / "e100_nvfp4_smem_mrhs" / "PRO_E100_NVFP4_SMEM_MRHS.json"
OUT = ROOT / "pro_research" / "results" / "e100_nvfp4_smem_mrhs" / "PRO_E100_NVFP4_SMEM_MRHS_VERIFICATION.json"
MANDATORY = {"shared_up_nvfp4", "shared_down_nvfp4", "lm_head_nvfp4"}


def main() -> int:
    if not SRC.exists():
        print(f"missing result: {SRC}")
        return 2
    p = json.loads(SRC.read_text(encoding="utf-8"))
    errors = []
    recs = p.get("records") or []
    names = {str(r.get("name")) for r in recs}
    support = MANDATORY.issubset(names)
    all_exact = bool(recs)

    for r in recs:
        ev = r.get("correctness_batches") or []
        exact = bool(ev) and all(
            int(e.get("production_vs_adopted_mismatch_count", -1)) == 0
            and int(e.get("adopted_vs_candidate_mismatch_count", -1)) == 0
            and int(e.get("candidate_repeat_mismatch_count", -1)) == 0
            and bool(e.get("finite"))
            for e in ev
        )
        all_exact = all_exact and exact
        ra = float(r.get("reference_a_ms", math.nan))
        rb = float(r.get("reference_b_ms", math.nan))
        cm = float(r.get("candidate_ms", math.nan))
        if not all(math.isfinite(x) and x > 0 for x in (ra, rb, cm)):
            errors.append(f"{r.get('name')} N={r.get('n_rhs')}: invalid timing")
            continue
        mid = 0.5 * (ra + rb)
        sp = mid / cm
        drift = abs(ra - rb) / mid
        if abs(float(r.get("reference_mid_ms", math.nan)) - mid) > max(1e-7, mid * 1e-6):
            errors.append(f"{r.get('name')} N={r.get('n_rhs')}: midpoint mismatch")
        if abs(float(r.get("aggregate_speedup", math.nan)) - sp) > max(1e-7, sp * 1e-6):
            errors.append(f"{r.get('name')} N={r.get('n_rhs')}: speedup mismatch")
        if abs(float(r.get("reference_drift_fraction", math.nan)) - drift) > max(1e-7, max(drift, 1e-9) * 1e-6):
            errors.append(f"{r.get('name')} N={r.get('n_rhs')}: drift mismatch")

    if bool((p.get("gates") or {}).get("all_production_adopted_candidate_exact")) != all_exact:
        errors.append("global exactness gate mismatch")
    if bool((p.get("gates") or {}).get("all_three_nvfp4_families_supported")) != support:
        errors.append("support gate mismatch")

    n16 = [r for r in recs if int(r.get("n_rhs", -1)) == 16]
    if n16:
        refw = sum(float(r["reference_mid_ms"]) * int(r["calls_per_token"]) for r in n16)
        candw = sum(float(r["candidate_ms"]) * int(r["calls_per_token"]) for r in n16)
        by = {r["name"]: r for r in n16}
        perf = {
            "n16_weighted_ge_2_0": (refw / candw if candw > 0 else 0.0) >= 2.0,
            "n16_lm_head_ge_2_0": float(by.get("lm_head_nvfp4", {}).get("aggregate_speedup", 0.0)) >= 2.0,
            "n16_no_case_below_1_20": min((float(r["aggregate_speedup"]) for r in n16), default=0.0) >= 1.20,
            "n16_all_ref_drift_le_7pct": max((float(r["reference_drift_fraction"]) for r in n16), default=math.inf) <= 0.07,
        }
    else:
        perf = {k: None for k in (
            "n16_weighted_ge_2_0", "n16_lm_head_ge_2_0",
            "n16_no_case_below_1_20", "n16_all_ref_drift_le_7pct")}

    mode = p.get("mode")
    full_perf = all(v is True for v in perf.values()) if mode == "full" else None
    if not support or not all_exact:
        expected = "correctness_failed"
    elif mode == "smoke":
        expected = "smoke_pass"
    elif full_perf:
        expected = "smem_mrhs_candidate"
    else:
        expected = "micro_null"
    if p.get("status") != expected:
        errors.append(f"status mismatch: file={p.get('status')} recomputed={expected}")

    out = {
        "kind": "pro_e100_nvfp4_smem_mrhs_independent_verification",
        "source": str(SRC.relative_to(ROOT)),
        "source_status": p.get("status"),
        "recomputed_status": expected,
        "all_exact": all_exact,
        "support_pass": support,
        "n16_performance_gates": perf,
        "full_performance_pass": full_perf,
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
