"""Independent CPU verifier for PRO_E100_MRHS256.json."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pro_research" / "results" / "e100_mrhs256" / "PRO_E100_MRHS256.json"
OUT = ROOT / "pro_research" / "results" / "e100_mrhs256" / "PRO_E100_MRHS256_VERIFICATION.json"


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
        batches = r.get("correctness_batches") or []
        exact = bool(batches) and all(
            int(b.get("mismatch_count", -1)) == 0
            and int(b.get("determinism_mismatch_count", -1)) == 0
            and bool(b.get("finite")) for b in batches
        )
        all_exact = all_exact and exact
        ra = float(r.get("reference_a_ms", math.nan))
        rb = float(r.get("reference_b_ms", math.nan))
        cm = float(r.get("mrhs256_ms", math.nan))
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
    mandatory = (
        "lm_head_nvfp4" in names
        and any(x.startswith("mamba_in_") for x in names)
        and any(x.startswith("mamba_out_") for x in names)
    )
    enough = len(names) >= 7

    def summary(n):
        rr = [r for r in recs if int(r.get("n_rhs", 0)) == n]
        refw = sum(float(r["reference_mid_ms"]) * int(r["calls_per_token"]) for r in rr)
        candw = sum(float(r["mrhs256_ms"]) * int(r["calls_per_token"]) for r in rr)
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

    n4 = summary(4)
    mode = p.get("mode")
    if mode == "full":
        n16 = summary(16)
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

    if not all_exact or not mandatory or not enough:
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
        "kind": "pro_e100_mrhs256_independent_verification",
        "source": str(SRC.relative_to(ROOT)),
        "source_status": p.get("status"),
        "recomputed_status": expected,
        "all_exact_deterministic_finite": all_exact,
        "mandatory_cases_supported": mandatory,
        "at_least_seven_cases": enough,
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
