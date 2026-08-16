"""Shared fail-closed benchmark core for E100 MRHS V3 runners."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from e100_adopted_baseline import AdoptedSingleRHS, production_call
from e100_mrhs import _time_cuda


def bench_case_adopted(
    rt,
    adopted: AdoptedSingleRHS,
    case,
    n: int,
    candidate_call: Callable,
    candidate_label: str,
    correctness_batches: int,
    repeats: int,
    rounds: int,
) -> dict:
    """Compare production, adopted V6 baseline and one exact MRHS candidate."""
    import cupy as cp

    evidence = []
    first_X = first_ref = first_cand = None
    for b in range(correctness_batches):
        seed = (0xE1030000 + 1009 * n + 9176 * b + sum(ord(c) for c in case.name)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        X = cp.asarray(rng.standard_normal((n, case.cols)).astype(np.float32))
        prod = cp.empty((n, case.rows), dtype=cp.float32)
        ref = cp.empty((n, case.rows), dtype=cp.float32)
        cand = cp.empty((n, case.rows), dtype=cp.float32)
        cand2 = cp.empty((n, case.rows), dtype=cp.float32)

        for r in range(n):
            production_call(rt, case, prod[r], X[r])
            adopted.call(case, ref[r], X[r])
        candidate_call(n, case, cand, X)
        candidate_call(n, case, cand2, X)
        cp.cuda.get_current_stream().synchronize()

        pp = cp.asnumpy(prod)
        rr = cp.asnumpy(ref)
        cc = cp.asnumpy(cand)
        dd = cp.asnumpy(cand2)
        pb, rb, cb, db = (x.view(np.uint32) for x in (pp, rr, cc, dd))
        pa = int(np.count_nonzero(pb != rb))
        ac = int(np.count_nonzero(rb != cb))
        cd = int(np.count_nonzero(cb != db))
        finite = bool(np.isfinite(pp).all() and np.isfinite(rr).all() and np.isfinite(cc).all() and np.isfinite(dd).all())
        evidence.append({
            "seed": int(seed),
            "production_vs_adopted_bit_equal": pa == 0,
            "production_vs_adopted_mismatch_count": pa,
            "adopted_vs_candidate_bit_equal": ac == 0,
            "adopted_vs_candidate_mismatch_count": ac,
            "candidate_deterministic": cd == 0,
            "candidate_repeat_mismatch_count": cd,
            "finite": finite,
        })
        if b == 0:
            first_X, first_ref, first_cand = X, ref, cand

    assert first_X is not None and first_ref is not None and first_cand is not None

    def ref_call() -> None:
        for r in range(n):
            adopted.call(case, first_ref[r], first_X[r])

    def cand_call() -> None:
        candidate_call(n, case, first_cand, first_X)

    # Candidate and adopted modules are already compiled before measurement;
    # warm both arms once, then frozen REF/CAND/CAND/REF ordering.
    ref_call(); cand_call(); cp.cuda.get_current_stream().synchronize()
    ref_a = _time_cuda(ref_call, repeats, rounds)
    cand_a = _time_cuda(cand_call, repeats, rounds)
    cand_b = _time_cuda(cand_call, repeats, rounds)
    ref_b = _time_cuda(ref_call, repeats, rounds)

    ra = float(np.median(np.asarray(ref_a, dtype=np.float64)))
    rb = float(np.median(np.asarray(ref_b, dtype=np.float64)))
    ref_mid = 0.5 * (ra + rb)
    cand_ms = float(np.median(np.asarray(cand_a + cand_b, dtype=np.float64)))
    drift = abs(ra - rb) / ref_mid if ref_mid > 0 else math.inf
    speedup = ref_mid / cand_ms if cand_ms > 0 else math.inf
    all_exact = all(
        e["production_vs_adopted_bit_equal"]
        and e["adopted_vs_candidate_bit_equal"]
        and e["candidate_deterministic"]
        and e["finite"]
        for e in evidence
    )

    return {
        "name": case.name,
        "kind": case.kind,
        "n_rhs": int(n),
        "rows": int(case.rows),
        "cols": int(case.cols),
        "calls_per_token": int(case.calls_per_token),
        "matrix_weight_bytes": int(case.weight_bytes),
        "correctness_batches": evidence,
        "production_equals_adopted": all(e["production_vs_adopted_bit_equal"] for e in evidence),
        "adopted_equals_candidate": all(e["adopted_vs_candidate_bit_equal"] for e in evidence),
        "candidate_deterministic": all(e["candidate_deterministic"] for e in evidence),
        "all_finite": all(e["finite"] for e in evidence),
        "all_exact": all_exact,
        "reference_name": "adopted_v6_selective_single_rhs",
        "candidate_name": candidate_label,
        "reference_a_ms": ra,
        "reference_b_ms": rb,
        "reference_mid_ms": ref_mid,
        "candidate_ms": cand_ms,
        "candidate_ms_per_rhs": cand_ms / n,
        "aggregate_speedup": float(speedup),
        "reference_drift_fraction": float(drift),
        "raw_reference_a_ms": ref_a,
        "raw_candidate_a_ms": cand_a,
        "raw_candidate_b_ms": cand_b,
        "raw_reference_b_ms": ref_b,
    }


def summarize_adopted(records: list[dict], n: int) -> dict:
    rr = [x for x in records if int(x["n_rhs"]) == int(n)]
    refw = sum(float(x["reference_mid_ms"]) * int(x["calls_per_token"]) for x in rr)
    candw = sum(float(x["candidate_ms"]) * int(x["calls_per_token"]) for x in rr)
    by = {x["name"]: x for x in rr}
    mi = next((x for x in rr if str(x["name"]).startswith("mamba_in_")), None)
    mo = next((x for x in rr if str(x["name"]).startswith("mamba_out_")), None)
    return {
        "n_rhs": int(n),
        "case_count": len(rr),
        "all_exact": bool(rr) and all(bool(x["all_exact"]) for x in rr),
        "weighted_registered_reference_ms": refw,
        "weighted_registered_candidate_ms": candw,
        "weighted_registered_aggregate_speedup": refw / candw if candw > 0 else None,
        "min_case_speedup": min((float(x["aggregate_speedup"]) for x in rr), default=None),
        "max_reference_drift_fraction": max((float(x["reference_drift_fraction"]) for x in rr), default=None),
        "lm_head_speedup": by.get("lm_head_nvfp4", {}).get("aggregate_speedup"),
        "mamba_in_speedup": None if mi is None else mi["aggregate_speedup"],
        "mamba_out_speedup": None if mo is None else mo["aggregate_speedup"],
    }
