"""Independent CPU verifier for PRO_E100_MRHS.json.

Deliberately does not import e100_mrhs.py or mrhs_exact_kernels.py.  It
recomputes correctness and performance gates only from the emitted evidence.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pro_research" / "results" / "e100_mrhs" / "PRO_E100_MRHS.json"
OUT = ROOT / "pro_research" / "results" / "e100_mrhs" / "PRO_E100_MRHS_VERIFICATION.json"


def _finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def main() -> int:
    if not SRC.exists():
        print(f"missing result: {SRC}")
        return 2
    p = json.loads(SRC.read_text(encoding="utf-8"))
    records = p.get("records") or []
    errors: list[str] = []

    if p.get("kind") != "pro_e100_exact_mrhs":
        errors.append("unexpected kind")
    if not records:
        errors.append("no records")

    recomputed: dict[str, Any] = {}
    all_exact = True
    for rec in records:
        key = f"N{rec.get('n_rhs')}:{rec.get('name')}"
        batches = rec.get("correctness_batches") or []
        exact = bool(batches) and all(
            int(b.get("mismatch_count", -1)) == 0
            and int(b.get("determinism_mismatch_count", -1)) == 0
            and bool(b.get("finite"))
            for b in batches
        )
        all_exact = all_exact and exact
        for fld in ("reference_a_ms", "reference_b_ms", "reference_mid_ms", "mrhs_ms", "aggregate_speedup", "reference_drift_fraction"):
            if not _finite_number(rec.get(fld)):
                errors.append(f"{key}: non-finite {fld}")
        ra = float(rec.get("reference_a_ms", math.nan))
        rb = float(rec.get("reference_b_ms", math.nan))
        mid = 0.5 * (ra + rb)
        cm = float(rec.get("mrhs_ms", math.nan))
        speed = mid / cm if cm > 0 else math.inf
        drift = abs(ra - rb) / mid if mid > 0 else math.inf
        if _finite_number(rec.get("reference_mid_ms")) and abs(float(rec["reference_mid_ms"]) - mid) > max(1e-7, abs(mid) * 1e-6):
            errors.append(f"{key}: reference midpoint mismatch")
        if _finite_number(rec.get("aggregate_speedup")) and abs(float(rec["aggregate_speedup"]) - speed) > max(1e-7, abs(speed) * 1e-6):
            errors.append(f"{key}: speedup arithmetic mismatch")
        if _finite_number(rec.get("reference_drift_fraction")) and abs(float(rec["reference_drift_fraction"]) - drift) > max(1e-7, abs(drift) * 1e-6):
            errors.append(f"{key}: drift arithmetic mismatch")
        if bool(rec.get("all_bit_equal")) != exact:
            # all_bit_equal excludes finite/deterministic in the runner, so only
            # enforce false->false on actual bit mismatch below.
            bit_only = bool(batches) and all(int(b.get("mismatch_count", -1)) == 0 for b in batches)
            if bool(rec.get("all_bit_equal")) != bit_only:
                errors.append(f"{key}: all_bit_equal summary mismatch")
        recomputed[key] = {"exact_deterministic_finite": exact, "speedup": speed, "drift": drift}

    names = {str(r.get("name")) for r in records}
    mandatory = "lm_head_nvfp4" in names and any(n.startswith("mamba_in_") for n in names)
    case_names = {str(r.get("name")) for r in records if int(r.get("n_rhs", 0)) == min((int(x.get("n_rhs", 0)) for x in records), default=0)}
    six = len(case_names) >= 6

    n4 = [r for r in records if int(r.get("n_rhs", 0)) == 4]
    refw = sum(float(r["reference_mid_ms"]) * int(r["calls_per_token"]) for r in n4) if n4 else 0.0
    candw = sum(float(r["mrhs_ms"]) * int(r["calls_per_token"]) for r in n4) if n4 else 0.0
    weighted = refw / candw if candw > 0 else 0.0
    min_speed = min((float(r["aggregate_speedup"]) for r in n4), default=0.0)
    lm = next((float(r["aggregate_speedup"]) for r in n4 if r.get("name") == "lm_head_nvfp4"), 0.0)
    mamba = next((float(r["aggregate_speedup"]) for r in n4 if str(r.get("name", "")).startswith("mamba_in_")), 0.0)
    drift_ok = bool(n4) and all(float(r["reference_drift_fraction"]) <= 0.07 for r in n4)
    perf = {
        "weighted_speedup_ge_1_75": weighted >= 1.75,
        "lm_head_speedup_ge_1_50": lm >= 1.50,
        "mamba_in_speedup_ge_1_50": mamba >= 1.50,
        "no_n4_case_regression_gt_5pct": min_speed >= 0.95,
        "all_n4_reference_drift_le_7pct": drift_ok,
    }

    mode = p.get("mode")
    if not all_exact or not mandatory or not six:
        expected_status = "correctness_failed"
    elif mode == "smoke":
        expected_status = "smoke_pass"
    elif all(perf.values()):
        expected_status = "mrhs_candidate"
    else:
        expected_status = "micro_null"

    if p.get("status") != expected_status:
        errors.append(f"status mismatch: file={p.get('status')} recomputed={expected_status}")

    out = {
        "kind": "pro_e100_mrhs_independent_verification",
        "source": str(SRC.relative_to(ROOT)),
        "source_status": p.get("status"),
        "recomputed_status": expected_status,
        "record_count": len(records),
        "all_exact_deterministic_finite": all_exact,
        "mandatory_cases_supported": mandatory,
        "at_least_six_case_families": six,
        "n4_weighted_registered_speedup": weighted,
        "n4_min_case_speedup": min_speed,
        "n4_lm_head_speedup": lm,
        "n4_mamba_in_speedup": mamba,
        "n4_performance_gates": perf,
        "errors": errors,
        "passed": not errors,
        "recomputed_records": recomputed,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(json.dumps(out, indent=2))
    return 0 if out["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
