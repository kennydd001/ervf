"""Audited E100-MRHS256 runner with explicit frozen-family support gates."""
from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

from common import REPO, environment_snapshot, require_gpu_free, utc_now
from e100_mrhs256 import _bench, _collect_cases, _new_runtime, _summary
from mrhs256_exact_kernels import ExactMRHS256

RESULT_DIR = REPO / "pro_research" / "results" / "e100_mrhs256"
OUT = RESULT_DIR / "PRO_E100_MRHS256.json"
PREREG = REPO / "pro_research" / "E100_MRHS256_PREREGISTRATION.md"


def _write(payload):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _families(names: set[str]):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload = {
        "kind": "pro_e100_mrhs256",
        "runner_revision": "v2_explicit_families",
        "mode": args.mode,
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "full-warp exact common-weight component through N16; not a full-model E100 claim",
    }
    try:
        require_gpu_free()
        payload["environment_start"] = environment_snapshot((
            Path(__file__), REPO / "pro_research" / "e100_mrhs256.py",
            REPO / "pro_research" / "mrhs256_exact_kernels.py",
        ))
        rt = _new_runtime()
        cases, unsupported = _collect_cases(rt)
        names = {x.name for x in cases}
        fam = _families(names)
        all_families = all(fam.values())
        ns = (4,) if args.mode == "smoke" else (4, 8, 16)
        k = ExactMRHS256(ns)
        correctness_batches = 1 if args.mode == "smoke" else 3
        repeats = 2 if args.mode == "smoke" else 10
        rounds = 2 if args.mode == "smoke" else 4

        records = []
        for n in ns:
            for case in cases:
                r = _bench(rt, k, n, case, correctness_batches, repeats, rounds)
                records.append(r)
                print(
                    f"MRHS256 N={n:>2} {case.name:<22} exact={r['all_bit_equal']} "
                    f"ref={r['reference_mid_ms']:.4f}ms cand={r['mrhs256_ms']:.4f}ms "
                    f"speedup={r['aggregate_speedup']:.3f}x drift={100*r['reference_drift_fraction']:.2f}%",
                    flush=True,
                )
        summaries = {str(n): _summary(records, n) for n in ns}
        all_exact = all(r["all_bit_equal"] and r["all_deterministic"] and r["all_finite"] for r in records)
        n4 = summaries.get("4", {})
        n16 = summaries.get("16", {})
        perf = {
            "n4_weighted_ge_1_50": (n4.get("weighted_speedup") or 0.0) >= 1.50,
            "n16_weighted_ge_3_0": ((n16.get("weighted_speedup") or 0.0) >= 3.0) if args.mode == "full" else None,
            "n16_lm_head_ge_3_0": ((n16.get("lm_head_speedup") or 0.0) >= 3.0) if args.mode == "full" else None,
            "n16_mamba_in_ge_2_5": ((n16.get("mamba_in_speedup") or 0.0) >= 2.5) if args.mode == "full" else None,
            "n16_mamba_out_ge_2_5": ((n16.get("mamba_out_speedup") or 0.0) >= 2.5) if args.mode == "full" else None,
            "n16_no_case_below_0_95": ((n16.get("min_case_speedup") or 0.0) >= 0.95) if args.mode == "full" else None,
            "n16_all_ref_drift_le_7pct": ((n16.get("max_reference_drift_fraction") or float('inf')) <= 0.07) if args.mode == "full" else None,
        }
        full_perf = None
        if args.mode == "full":
            full_perf = bool(perf["n4_weighted_ge_1_50"] and all(v is True for k2, v in perf.items() if k2.startswith("n16_")))

        if not all_exact or not all_families:
            status = "correctness_failed"
        elif args.mode == "smoke":
            status = "smoke_pass"
        elif full_perf:
            status = "mrhs256_candidate"
        else:
            status = "micro_null"

        payload.update({
            "config": {"rhs_values": list(ns), "correctness_batches": correctness_batches,
                       "timing_repeats": repeats, "timing_rounds_per_arm": rounds},
            "supported_cases": sorted(names),
            "unsupported_cases": unsupported,
            "family_support": fam,
            "records": records,
            "summary_by_n": summaries,
            "gates": {"all_exact": all_exact, "all_eight_frozen_families_supported": all_families,
                      "performance": perf, "full_performance_pass": full_perf},
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })
        del rt, k
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
