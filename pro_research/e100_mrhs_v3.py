"""E100-MRHS V3: exact common-weight reuse against the adopted V6 baseline.

V3 supersedes V1/V2 for performance interpretation.  Correctness is stronger:
original production == adopted selective single-RHS == MRHS == MRHS repeat.
"""
from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, require_gpu_free, utc_now
from e100_adopted_baseline import AdoptedSingleRHS
from e100_mrhs import Case, _collect_cases, _new_runtime, cpu_mapping_selftest
from e100_mrhs_adopted_bench import bench_case_adopted, summarize_adopted
from mrhs_exact_kernels import ExactMRHS

RESULT_DIR = REPO / "pro_research" / "results" / "e100_mrhs"
OUT = RESULT_DIR / "PRO_E100_MRHS.json"
PREREG = REPO / "pro_research" / "E100_MRHS_PREREGISTRATION.md"
ADDENDUM = REPO / "pro_research" / "E100_MRHS_V3_ADOPTED_BASELINE.md"


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _append_mamba_out(rt, cases: list[Case], unsupported: list[dict[str, str]]) -> None:
    if not rt.mamba_layers:
        unsupported.append({"name": "mamba_out", "reason": "checkpoint has no Mamba layers"})
        return
    d = rt.layer[rt.mamba_layers[0]]
    calls = len(rt.mamba_layers)
    kind = d.get("out_k")
    if kind == "nvfp4" and all(k in d for k in ("out_codes", "out_scales", "out_g")):
        cases.append(Case("mamba_out_nvfp4", "nvfp4", rt.hidden, rt.d_inner, calls,
                          codes=d["out_codes"], scales=d["out_scales"], scale=float(d["out_g"])))
    elif kind == "fp8_tensor" and all(k in d for k in ("out_w8", "out_s")):
        cases.append(Case("mamba_out_fp8", "fp8", rt.hidden, rt.d_inner, calls,
                          W=d["out_w8"], scale=float(d["out_s"])))
    elif kind == "bf16" and "out_w" in d:
        cases.append(Case("mamba_out_bf16", "bf16", rt.hidden, rt.d_inner, calls, W=d["out_w"]))
    else:
        unsupported.append({"name": "mamba_out", "reason": f"unsupported/missing storage kind {kind!r}"})


def _families(names: set[str]) -> dict[str, bool]:
    # Frozen seven-family support rule from the original preregistration.
    # Mamba-out is measured in V3 but cannot substitute for a missing family.
    return {
        "attention_q": "attn_q_bf16" in names,
        "attention_o": "attn_o_bf16" in names,
        "router": "router_f32" in names,
        "mamba_in": any(x.startswith("mamba_in_") for x in names),
        "shared_up": "shared_up_nvfp4" in names,
        "shared_down": "shared_down_nvfp4" in names,
        "lm_head": "lm_head_nvfp4" in names,
    }


def _candidate(mrhs, rt, n, case, out, X):
    if case.kind == "bf16":
        return mrhs.bf16(n, out, case.W, X, case.rows, case.cols)
    if case.kind == "f32":
        return mrhs.f32(n, out, case.W, X, case.rows, case.cols)
    if case.kind == "fp8":
        return mrhs.fp8(n, out, case.W, X, case.scale, case.rows, case.cols)
    if case.kind == "nvfp4":
        return mrhs.nvfp4(n, out, case.codes, case.scales, rt.fused.e2m1, rt.fused.e4m3,
                          X, case.scale, case.rows, case.cols,
                          apply_relu2=case.apply_relu2, out_scale=case.out_scale)
    raise ValueError(case.kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    cpu = cpu_mapping_selftest()
    if args.selftest:
        print(json.dumps(cpu, indent=2))
        return 0 if cpu["passed"] else 2

    payload: dict[str, Any] = {
        "kind": "pro_e100_exact_mrhs",
        "runner_revision": "v3_adopted_v6_baseline",
        "baseline_revision": "adopted_v6_selective_single_rhs",
        "mode": args.mode,
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "preregistration_addendum": str(ADDENDUM.relative_to(REPO)),
        "claim_boundary": "component/oracle exact aggregate common-weight reuse; not a full-model or single-stream E100 claim",
        "cpu_mapping_selftest": cpu,
    }
    try:
        if not cpu["passed"]:
            raise RuntimeError("width-32 virtual reduction CPU selftest failed")
        require_gpu_free()
        payload["environment_start"] = environment_snapshot((
            Path(__file__), ADDENDUM,
            REPO / "pro_research" / "e100_adopted_baseline.py",
            REPO / "pro_research" / "e100_mrhs_adopted_bench.py",
            REPO / "pro_research" / "mrhs_exact_kernels.py",
            REPO / "pro_research" / "selective_ervf_v3.py",
            REPO / "pro_research" / "ervf_dense.py",
        ))
        rt = _new_runtime()
        cases, unsupported = _collect_cases(rt)
        _append_mamba_out(rt, cases, unsupported)
        rhs_values = (2, 4) if args.mode == "smoke" else (2, 4, 8)

        # Compile both candidate and adopted DenseERVF modules before any timing.
        mrhs = ExactMRHS(rhs_values)
        adopted = AdoptedSingleRHS(rt)
        correctness_batches = 1 if args.mode == "smoke" else 3
        repeats = 2 if args.mode == "smoke" else 10
        rounds = 2 if args.mode == "smoke" else 4

        records = []
        for n in rhs_values:
            for case in cases:
                rec = bench_case_adopted(
                    rt, adopted, case, n,
                    lambda nn, cc, out, X: _candidate(mrhs, rt, nn, cc, out, X),
                    "mrhs32_exact", correctness_batches, repeats, rounds,
                )
                records.append(rec)
                print(
                    f"MRHS-V3 N={n:>2} {case.name:<22} exact={rec['all_exact']} "
                    f"ref={rec['reference_mid_ms']:.4f}ms cand={rec['candidate_ms']:.4f}ms "
                    f"speedup={rec['aggregate_speedup']:.3f}x drift={100*rec['reference_drift_fraction']:.2f}%",
                    flush=True,
                )

        summaries = {str(n): summarize_adopted(records, n) for n in rhs_values}
        names = {c.name for c in cases}
        families = _families(names)
        family_count = sum(1 for v in families.values() if v)
        support_pass = family_count >= 6 and families["lm_head"] and families["mamba_in"]
        all_exact = bool(records) and all(bool(r["all_exact"]) for r in records)
        n4 = summaries.get("4", {})
        perf = {
            "weighted_speedup_ge_1_75": (n4.get("weighted_registered_aggregate_speedup") or 0.0) >= 1.75,
            "lm_head_speedup_ge_1_50": (n4.get("lm_head_speedup") or 0.0) >= 1.50,
            "mamba_in_speedup_ge_1_50": (n4.get("mamba_in_speedup") or 0.0) >= 1.50,
            "no_n4_case_regression_gt_5pct": (n4.get("min_case_speedup") or 0.0) >= 0.95,
            "all_n4_reference_drift_le_7pct": (n4.get("max_reference_drift_fraction") or float("inf")) <= 0.07,
        }
        perf_pass = all(perf.values())
        dispatch = dict(adopted.counters)
        selected_dense_exercised = dispatch["bf16_ervf"] > 0 and (
            dispatch["fp8_ervf"] > 0 if any(c.kind == "fp8" for c in cases) else True
        )

        if not all_exact or not support_pass or not selected_dense_exercised:
            status = "correctness_failed"
        elif args.mode == "smoke":
            status = "smoke_pass"
        elif perf_pass:
            status = "mrhs_candidate"
        else:
            status = "micro_null"

        payload.update({
            "config": {"rhs_values": list(rhs_values), "correctness_batches": correctness_batches,
                       "timing_repeats": repeats, "timing_rounds_per_ABBA_arm": rounds},
            "adopted_baseline_policy": adopted.policy,
            "adopted_dispatch_counters": dispatch,
            "supported_cases": sorted(names),
            "unsupported_cases": unsupported,
            "family_support": families,
            "family_support_count": family_count,
            "records": records,
            "summary_by_n": summaries,
            "gates": {
                "production_equals_adopted_equals_candidate": all_exact,
                "at_least_six_of_seven_frozen_families_with_mandatory": support_pass,
                "adopted_selected_dense_dispatch_exercised": selected_dense_exercised,
                "n4_performance": perf,
                "full_n4_performance_pass": perf_pass if args.mode == "full" else None,
            },
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })
        del rt, mrhs, adopted
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
