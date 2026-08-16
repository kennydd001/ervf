"""Independent CPU verifier for E100-PAIRBATCH result evidence."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pro_research" / "results" / "e100_pairbatch" / "PRO_E100_PAIRBATCH.json"
OUT = ROOT / "pro_research" / "results" / "e100_pairbatch" / "PRO_E100_PAIRBATCH_VERIFICATION.json"


def main() -> int:
    if not SRC.exists():
        print(f"missing result: {SRC}")
        return 2
    p = json.loads(SRC.read_text(encoding="utf-8"))
    errors: list[str] = []
    recs = p.get("records") or []
    by = {r.get("map"): r for r in recs}
    for need in ("unique24", "n4_typical22", "repeat6"):
        if need not in by:
            errors.append(f"missing map {need}")

    all_correct = bool(recs)
    for r in recs:
        name = str(r.get("map"))
        batches = r.get("correctness_batches") or []
        exact = bool(batches) and all(
            int(b.get("mismatch_count", -1)) == 0
            and int(b.get("determinism_mismatch_count", -1)) == 0
            and bool(b.get("finite"))
            for b in batches
        )
        prod = r.get("production_reference_spot") or {}
        exact = exact and int(prod.get("mismatch_count", -1)) == 0 and bool(prod.get("bit_equal"))
        all_correct = all_correct and exact

        ra = float(r.get("reference_a_ms", math.nan))
        rb = float(r.get("reference_b_ms", math.nan))
        pm = float(r.get("pair_ms", math.nan))
        if not all(math.isfinite(v) and v > 0 for v in (ra, rb, pm)):
            errors.append(f"{name}: non-finite/nonpositive timing")
            continue
        mid = 0.5 * (ra + rb)
        speed = mid / pm
        drift = abs(ra - rb) / mid
        if abs(float(r.get("reference_mid_ms", math.nan)) - mid) > max(1e-7, mid * 1e-6):
            errors.append(f"{name}: midpoint arithmetic mismatch")
        if abs(float(r.get("pair_speedup", math.nan)) - speed) > max(1e-7, speed * 1e-6):
            errors.append(f"{name}: speedup arithmetic mismatch")
        if abs(float(r.get("reference_drift_fraction", math.nan)) - drift) > max(1e-7, drift * 1e-6):
            errors.append(f"{name}: drift arithmetic mismatch")
        if bool(r.get("all_bit_equal")) != (bool(batches) and all(int(b.get("mismatch_count", -1)) == 0 for b in batches)):
            errors.append(f"{name}: bit-equal summary mismatch")

    if all_correct != bool((p.get("gates") or {}).get("all_maps_exact_deterministic_finite_and_reference_spot_exact")):
        errors.append("global correctness gate mismatch")

    if all(k in by for k in ("unique24", "n4_typical22")):
        primary = by["unique24"]
        typical = by["n4_typical22"]
        perf = {
            "unique24_speedup_ge_1_08": float(primary["pair_speedup"]) >= 1.08,
            "unique24_reference_drift_le_7pct": float(primary["reference_drift_fraction"]) <= 0.07,
            "typical22_no_regression": float(typical["pair_speedup"]) >= 0.98,
        }
    else:
        perf = {
            "unique24_speedup_ge_1_08": False,
            "unique24_reference_drift_le_7pct": False,
            "typical22_no_regression": False,
        }
    perf_pass = all(perf.values())
    mode = p.get("mode")
    if not all_correct:
        expected = "correctness_failed"
    elif mode == "smoke":
        expected = "smoke_pass"
    elif perf_pass:
        expected = "pairbatch_candidate"
    else:
        expected = "pairbatch_null"
    if p.get("status") != expected:
        errors.append(f"status mismatch: file={p.get('status')} recomputed={expected}")

    out = {
        "kind": "pro_e100_pairbatch_independent_verification",
        "source": str(SRC.relative_to(ROOT)),
        "source_status": p.get("status"),
        "recomputed_status": expected,
        "all_correct": all_correct,
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
