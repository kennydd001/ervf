"""Independent verifier for E100-NVFP4-TILED-MRHS V5 evidence."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pro_research" / "results" / "e100_nvfp4_tiled_mrhs" / "PRO_E100_NVFP4_TILED_MRHS.json"
OUT = ROOT / "pro_research" / "results" / "e100_nvfp4_tiled_mrhs" / "PRO_E100_NVFP4_TILED_MRHS_VERIFICATION.json"
MANDATORY = ("shared_up_nvfp4", "shared_down_nvfp4", "lm_head_nvfp4")


def main() -> int:
    if not SRC.exists():
        print(f"missing result: {SRC}")
        return 2
    p = json.loads(SRC.read_text(encoding="utf-8"))
    errors = []
    recs = p.get("records") or []
    names = {str(r.get("name")) for r in recs}
    support = all(x in names for x in MANDATORY)
    all_exact = bool(recs)

    for r in recs:
        batches = r.get("correctness_batches") or []
        exact = bool(batches) and all(
            bool(b.get("production_vs_adopted_bit_equal"))
            and int(b.get("production_vs_adopted_mismatch_count", -1)) == 0
            and bool(b.get("adopted_vs_candidate_bit_equal"))
            and int(b.get("adopted_vs_candidate_mismatch_count", -1)) == 0
            and bool(b.get("candidate_deterministic"))
            and int(b.get("candidate_repeat_mismatch_count", -1)) == 0
            and bool(b.get("finite"))
            for b in batches
        )
        all_exact = all_exact and exact
        ra = float(r.get("reference_a_ms", math.nan))
        rb = float(r.get("reference_b_ms", math.nan))
        cm = float(r.get("candidate_ms", math.nan))
        if not all(math.isfinite(v) and v > 0 for v in (ra, rb, cm)):
            errors.append(f"{r.get('name')}: invalid timing")
            continue
        mid = 0.5 * (ra + rb)
        speed = mid / cm
        drift = abs(ra - rb) / mid
        if abs(float(r.get("reference_mid_ms", math.nan)) - mid) > max(1e-7, mid * 1e-6):
            errors.append(f"{r.get('name')}: midpoint arithmetic mismatch")
        if abs(float(r.get("aggregate_speedup", math.nan)) - speed) > max(1e-7, speed * 1e-6):
            errors.append(f"{r.get('name')}: speedup arithmetic mismatch")
        if abs(float(r.get("reference_drift_fraction", math.nan)) - drift) > max(1e-7, drift * 1e-6):
            errors.append(f"{r.get('name')}: drift arithmetic mismatch")

    summaries = p.get("summary_by_n") or {}
    n16 = summaries.get("16") or {}
    mode = p.get("mode")
    perf = {
        "n16_weighted_ge_2_0": ((n16.get("weighted_aggregate_speedup") or 0.0) >= 2.0) if mode == "full" else None,
        "n16_lm_head_ge_2_0": ((n16.get("lm_head_speedup") or 0.0) >= 2.0) if mode == "full" else None,
        "n16_shared_up_ge_1_20": ((n16.get("shared_up_speedup") or 0.0) >= 1.20) if mode == "full" else None,
        "n16_shared_down_ge_1_20": ((n16.get("shared_down_speedup") or 0.0) >= 1.20) if mode == "full" else None,
        "n16_no_case_below_1_20": ((n16.get("min_case_speedup") or 0.0) >= 1.20) if mode == "full" else None,
        "n16_all_ref_drift_le_7pct": ((n16.get("max_reference_drift_fraction") or float("inf")) <= 0.07) if mode == "full" else None,
    }
    full_perf = all(v is True for v in perf.values()) if mode == "full" else None
    if not support or not all_exact:
        expected = "correctness_failed"
    elif mode == "smoke":
        expected = "smoke_pass"
    elif full_perf:
        expected = "tiled_mrhs_candidate"
    else:
        expected = "micro_null"
    if p.get("status") != expected:
        errors.append(f"status mismatch: file={p.get('status')} recomputed={expected}")

    gates = p.get("gates") or {}
    if bool(gates.get("all_production_adopted_candidate_exact")) != bool(all_exact):
        errors.append("global exactness gate mismatch")
    if bool(gates.get("all_three_nvfp4_families_supported")) != bool(support):
        errors.append("support gate mismatch")

    out = {
        "kind": "pro_e100_nvfp4_tiled_mrhs_independent_verification",
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
