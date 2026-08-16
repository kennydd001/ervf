"""E100-MRHS v2 runner.

Mechanics/timing are imported unchanged from e100_mrhs.py.  V2 fixes only the
pre-run audit issue in the support gate: Q/K/V/O are cases inside the attention
family, not four substitutes for missing frozen families.  No target result
existed before this correction.
"""
from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, require_gpu_free, utc_now
from e100_mrhs import (
    ExactMRHS,
    _bench_case,
    _collect_cases,
    _new_runtime,
    _summarize_n,
    cpu_mapping_selftest,
)

RESULT_DIR = REPO / "pro_research" / "results" / "e100_mrhs"
OUT = RESULT_DIR / "PRO_E100_MRHS.json"
PREREG = REPO / "pro_research" / "E100_MRHS_PREREGISTRATION.md"


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _family_support(names: set[str]) -> dict[str, bool]:
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
        "runner_revision": "v2_family_gate",
        "mode": args.mode,
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "component/oracle test for exact aggregate multi-sequence common-weight reuse; not a full-model or single-stream E100 claim",
        "cpu_mapping_selftest": cpu,
    }
    try:
        if not cpu["passed"]:
            raise RuntimeError("width-32 virtual reduction CPU selftest failed")
        require_gpu_free()
        payload["environment_start"] = environment_snapshot((
            Path(__file__), REPO / "pro_research" / "e100_mrhs.py",
            REPO / "pro_research" / "mrhs_exact_kernels.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
        ))
        rt = _new_runtime()
        cases, unsupported = _collect_cases(rt)
        rhs_values = (2, 4) if args.mode == "smoke" else (2, 4, 8)
        mrhs = ExactMRHS(rhs_values)
        correctness_batches = 1 if args.mode == "smoke" else 3
        repeats = 2 if args.mode == "smoke" else 10
        rounds = 2 if args.mode == "smoke" else 4

        records = []
        for n in rhs_values:
            for case in cases:
                rec = _bench_case(rt, mrhs, case, n, correctness_batches, repeats, rounds)
                records.append(rec)
                print(
                    f"MRHS N={n:>2} {case.name:<22} exact={rec['all_bit_equal']} "
                    f"det={rec['all_deterministic']} ref={rec['reference_mid_ms']:.4f}ms "
                    f"mrhs={rec['mrhs_ms']:.4f}ms speedup={rec['aggregate_speedup']:.3f}x "
                    f"drift={100.0*rec['reference_drift_fraction']:.2f}%",
                    flush=True,
                )

        summaries = {str(n): _summarize_n(records, n) for n in rhs_values}
        all_exact = all(r["all_bit_equal"] and r["all_deterministic"] and r["all_finite"] for r in records)
        names = {x.name for x in cases}
        families = _family_support(names)
        family_count = sum(1 for v in families.values() if v)
        six_of_seven = family_count >= 6
        mandatory = families["lm_head"] and families["mamba_in"]

        n4 = summaries.get("4", {})
        perf = {
            "weighted_speedup_ge_1_75": bool((n4.get("weighted_registered_aggregate_speedup") or 0.0) >= 1.75),
            "lm_head_speedup_ge_1_50": bool((n4.get("lm_head_speedup") or 0.0) >= 1.50),
            "mamba_in_speedup_ge_1_50": bool((n4.get("mamba_in_speedup") or 0.0) >= 1.50),
            "no_n4_case_regression_gt_5pct": bool((n4.get("min_case_speedup") or 0.0) >= 0.95),
            "all_n4_reference_drift_le_7pct": bool(n4.get("all_reference_drift_le_7pct", False)),
        }
        perf_pass = all(perf.values())

        if not all_exact or not mandatory or not six_of_seven:
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
            "supported_cases": sorted(names),
            "unsupported_cases": unsupported,
            "family_support": families,
            "family_support_count": family_count,
            "records": records,
            "summary_by_n": summaries,
            "gates": {
                "all_real_outputs_bit_exact_deterministic_finite": all_exact,
                "at_least_six_of_seven_frozen_families_supported": six_of_seven,
                "lm_head_and_mamba_in_supported": mandatory,
                "n4_performance": perf,
                "full_n4_performance_pass": perf_pass if args.mode == "full" else None,
            },
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })
        del rt, mrhs
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
                      "family_support": payload.get("family_support"),
                      "summary_by_n": payload.get("summary_by_n"), "gates": payload.get("gates")}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
