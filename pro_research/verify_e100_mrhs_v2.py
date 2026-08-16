"""Independent verifier for E100-MRHS v2 family semantics."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pro_research" / "results" / "e100_mrhs" / "PRO_E100_MRHS.json"
OUT = ROOT / "pro_research" / "results" / "e100_mrhs" / "PRO_E100_MRHS_VERIFICATION.json"


def families(names: set[str]) -> dict[str, bool]:
    return {
        "attention_q": "attn_q_bf16" in names,
        "attention_o": "attn_o_bf16" in names,
        "router": "router_f32" in names,
        "mamba_in": any(x.startswith("mamba_in_") for x in names),
        "shared_up": "shared_up_nvfp4" in names,
        "shared_down": "shared_down_nvfp4" in names,
        "lm_head": "lm_head_nvfp4" in names,
    }


def main() -> int:
    if not SRC.exists():
        print(f"missing result: {SRC}")
        return 2
    p = json.loads(SRC.read_text(encoding="utf-8"))
    recs = p.get("records") or []
    errors = []
    all_exact = bool(recs)
    for r in recs:
        key = f"N{r.get('n_rhs')}:{r.get('name')}"
        bs = r.get("correctness_batches") or []
        exact = bool(bs) and all(
            int(b.get("mismatch_count", -1)) == 0
            and int(b.get("determinism_mismatch_count", -1)) == 0
            and bool(b.get("finite")) for b in bs
        )
        all_exact = all_exact and exact
        ra, rb, cm = (float(r.get(x, math.nan)) for x in ("reference_a_ms", "reference_b_ms", "mrhs_ms"))
        if not all(math.isfinite(x) and x > 0 for x in (ra, rb, cm)):
            errors.append(f"{key}: invalid timing")
            continue
        mid = 0.5 * (ra + rb)
        speed = mid / cm
        drift = abs(ra - rb) / mid
        if abs(float(r.get("reference_mid_ms", math.nan)) - mid) > max(1e-7, mid * 1e-6):
            errors.append(f"{key}: midpoint mismatch")
        if abs(float(r.get("aggregate_speedup", math.nan)) - speed) > max(1e-7, speed * 1e-6):
            errors.append(f"{key}: speedup mismatch")
        if abs(float(r.get("reference_drift_fraction", math.nan)) - drift) > max(1e-7, drift * 1e-6):
            errors.append(f"{key}: drift mismatch")

    names = set(p.get("supported_cases") or [])
    fam = families(names)
    family_count = sum(1 for v in fam.values() if v)
    six = family_count >= 6
    mandatory = fam["lm_head"] and fam["mamba_in"]
    n4 = [r for r in recs if int(r.get("n_rhs", 0)) == 4]
    refw = sum(float(r["reference_mid_ms"]) * int(r["calls_per_token"]) for r in n4)
    candw = sum(float(r["mrhs_ms"]) * int(r["calls_per_token"]) for r in n4)
    weighted = refw / candw if candw else 0.0
    minimum = min((float(r["aggregate_speedup"]) for r in n4), default=0.0)
    lm = next((float(r["aggregate_speedup"]) for r in n4 if r["name"] == "lm_head_nvfp4"), 0.0)
    mi = next((float(r["aggregate_speedup"]) for r in n4 if str(r["name"]).startswith("mamba_in_")), 0.0)
    drift_ok = bool(n4) and all(float(r["reference_drift_fraction"]) <= 0.07 for r in n4)
    perf = {
        "weighted_speedup_ge_1_75": weighted >= 1.75,
        "lm_head_speedup_ge_1_50": lm >= 1.50,
        "mamba_in_speedup_ge_1_50": mi >= 1.50,
        "no_n4_case_regression_gt_5pct": minimum >= 0.95,
        "all_n4_reference_drift_le_7pct": drift_ok,
    }
    mode = p.get("mode")
    if not all_exact or not mandatory or not six:
        expected = "correctness_failed"
    elif mode == "smoke":
        expected = "smoke_pass"
    elif all(perf.values()):
        expected = "mrhs_candidate"
    else:
        expected = "micro_null"
    if p.get("status") != expected:
        errors.append(f"status mismatch file={p.get('status')} recomputed={expected}")
    if p.get("runner_revision") != "v2_family_gate":
        errors.append("result was not produced by v2_family_gate runner")
    file_fam = p.get("family_support") or {}
    if file_fam != fam:
        errors.append("family_support evidence mismatch")

    out = {
        "kind": "pro_e100_mrhs_v2_independent_verification",
        "source": str(SRC.relative_to(ROOT)),
        "source_status": p.get("status"),
        "recomputed_status": expected,
        "all_exact_deterministic_finite": all_exact,
        "family_support": fam,
        "family_support_count": family_count,
        "mandatory_supported": mandatory,
        "n4_weighted_speedup": weighted,
        "n4_min_speedup": minimum,
        "n4_lm_head_speedup": lm,
        "n4_mamba_in_speedup": mi,
        "n4_performance_gates": perf,
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
