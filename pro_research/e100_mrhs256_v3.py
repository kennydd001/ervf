"""E100-MRHS256 V3: full-warp MRHS through N16 vs adopted V6 baseline."""
from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

from common import REPO, environment_snapshot, require_gpu_free, utc_now
from e100_adopted_baseline import AdoptedSingleRHS
from e100_mrhs256 import _collect_cases, _new_runtime
from e100_mrhs_adopted_bench import bench_case_adopted, summarize_adopted
from mrhs256_exact_kernels import ExactMRHS256

RESULT_DIR = REPO / "pro_research" / "results" / "e100_mrhs256"
OUT = RESULT_DIR / "PRO_E100_MRHS256.json"
PREREG = REPO / "pro_research" / "E100_MRHS256_PREREGISTRATION.md"
ADDENDUM = REPO / "pro_research" / "E100_MRHS_V3_ADOPTED_BASELINE.md"


def _write(payload):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


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


def _candidate(k, rt, n, case, out, X):
    if case.kind == "bf16":
        return k.bf16(n, out, case.W, X, case.rows, case.cols)
    if case.kind == "f32":
        return k.f32(n, out, case.W, X, case.rows, case.cols)
    if case.kind == "fp8":
        return k.fp8(n, out, case.W, X, case.scale, case.rows, case.cols)
    if case.kind == "nvfp4":
        return k.nvfp4(n, out, case.codes, case.scales, rt.fused.e2m1, rt.fused.e4m3,
                       X, case.scale, case.rows, case.cols,
                       apply_relu2=case.apply_relu2, out_scale=case.out_scale)
    raise ValueError(case.kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload = {
        "kind": "pro_e100_mrhs256",
        "runner_revision": "v3_adopted_v6_baseline",
        "baseline_revision": "adopted_v6_selective_single_rhs",
        "mode": args.mode,
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "preregistration_addendum": str(ADDENDUM.relative_to(REPO)),
        "claim_boundary": "full-warp exact common-weight component through N16; not a full-model E100 claim",
    }
    try:
        require_gpu_free()
        payload["environment_start"] = environment_snapshot((
            Path(__file__), ADDENDUM,
            REPO / "pro_research" / "e100_adopted_baseline.py",
            REPO / "pro_research" / "e100_mrhs_adopted_bench.py",
            REPO / "pro_research" / "mrhs256_exact_kernels.py",
            REPO / "pro_research" / "selective_ervf_v3.py",
        ))
        rt = _new_runtime()
        cases, unsupported = _collect_cases(rt)
        names = {c.name for c in cases}
        families = _families(names)
        all_families = all(families.values())
        ns = (4,) if args.mode == "smoke" else (4, 8, 16)

        # Compile both candidate and adopted DenseERVF before measurement.
        k = ExactMRHS256(ns)
        adopted = AdoptedSingleRHS(rt)
        correctness_batches = 1 if args.mode == "smoke" else 3
        repeats = 2 if args.mode == "smoke" else 10
        rounds = 2 if args.mode == "smoke" else 4

        records = []
        for n in ns:
            for case in cases:
                r = bench_case_adopted(
                    rt, adopted, case, n,
                    lambda nn, cc, out, X: _candidate(k, rt, nn, cc, out, X),
                    "mrhs256_exact", correctness_batches, repeats, rounds,
                )
                records.append(r)
                print(
                    f"MRHS256-V3 N={n:>2} {case.name:<22} exact={r['all_exact']} "
                    f"ref={r['reference_mid_ms']:.4f}ms cand={r['candidate_ms']:.4f}ms "
                    f"speedup={r['aggregate_speedup']:.3f}x drift={100*r['reference_drift_fraction']:.2f}%",
                    flush=True,
                )

        summaries = {str(n): summarize_adopted(records, n) for n in ns}
        all_exact = bool(records) and all(r["all_exact"] for r in records)
        dispatch = dict(adopted.counters)
        selected_dense_exercised = dispatch["bf16_ervf"] > 0 and (
            dispatch["fp8_ervf"] > 0 if any(c.kind == "fp8" for c in cases) else True
        )
        n4 = summaries.get("4", {})
        n16 = summaries.get("16", {})
        perf = {
            "n4_weighted_ge_1_50": (n4.get("weighted_registered_aggregate_speedup") or 0.0) >= 1.50,
            "n16_weighted_ge_3_0": ((n16.get("weighted_registered_aggregate_speedup") or 0.0) >= 3.0) if args.mode == "full" else None,
            "n16_lm_head_ge_3_0": ((n16.get("lm_head_speedup") or 0.0) >= 3.0) if args.mode == "full" else None,
            "n16_mamba_in_ge_2_5": ((n16.get("mamba_in_speedup") or 0.0) >= 2.5) if args.mode == "full" else None,
            "n16_mamba_out_ge_2_5": ((n16.get("mamba_out_speedup") or 0.0) >= 2.5) if args.mode == "full" else None,
            "n16_no_case_below_0_95": ((n16.get("min_case_speedup") or 0.0) >= 0.95) if args.mode == "full" else None,
            "n16_all_ref_drift_le_7pct": ((n16.get("max_reference_drift_fraction") or float("inf")) <= 0.07) if args.mode == "full" else None,
        }
        full_perf = None
        if args.mode == "full":
            full_perf = bool(perf["n4_weighted_ge_1_50"] and all(v is True for key, v in perf.items() if key.startswith("n16_")))

        if not all_exact or not all_families or not selected_dense_exercised:
            status = "correctness_failed"
        elif args.mode == "smoke":
            status = "smoke_pass"
        elif full_perf:
            status = "mrhs256_candidate"
        else:
            status = "micro_null"

        payload.update({
            "config": {"rhs_values": list(ns), "correctness_batches": correctness_batches,
                       "timing_repeats": repeats, "timing_rounds_per_ABBA_arm": rounds},
            "adopted_baseline_policy": adopted.policy,
            "adopted_dispatch_counters": dispatch,
            "supported_cases": sorted(names),
            "unsupported_cases": unsupported,
            "family_support": families,
            "records": records,
            "summary_by_n": summaries,
            "gates": {
                "production_equals_adopted_equals_candidate": all_exact,
                "all_eight_frozen_families_supported": all_families,
                "adopted_selected_dense_dispatch_exercised": selected_dense_exercised,
                "performance": perf,
                "full_performance_pass": full_perf,
            },
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })
        del rt, k, adopted
        gc.collect()
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({"status": "technical_failure",
                        "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                        "completed_utc": utc_now()})
    _write(payload)
    print(json.dumps({"status": payload.get("status"), "output": str(OUT),
                      "baseline_revision": payload.get("baseline_revision"),
                      "summary_by_n": payload.get("summary_by_n"), "gates": payload.get("gates")}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
