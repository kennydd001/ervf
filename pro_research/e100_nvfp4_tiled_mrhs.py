"""E100-NVFP4-TILED-MRHS V5 component benchmark."""
from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

from common import REPO, environment_snapshot, require_gpu_free, utc_now
from e100_adopted_baseline import AdoptedSingleRHS
from e100_mrhs256 import _collect_cases, _new_runtime
from e100_mrhs_adopted_bench import bench_case_adopted
from nvfp4_tiled_mrhs_kernels import ExactNVFP4TiledMRHS

RESULT_DIR = REPO / "pro_research" / "results" / "e100_nvfp4_tiled_mrhs"
OUT = RESULT_DIR / "PRO_E100_NVFP4_TILED_MRHS.json"
PREREG = REPO / "pro_research" / "E100_NVFP4_TILED_MRHS_PREREGISTRATION.md"
MANDATORY = ("shared_up_nvfp4", "shared_down_nvfp4", "lm_head_nvfp4")


def _write(payload):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _summary(records, n):
    rr = [r for r in records if int(r["n_rhs"]) == int(n)]
    refw = sum(float(r["reference_mid_ms"]) * int(r["calls_per_token"]) for r in rr)
    candw = sum(float(r["candidate_ms"]) * int(r["calls_per_token"]) for r in rr)
    by = {r["name"]: r for r in rr}
    return {
        "n_rhs": int(n),
        "case_count": len(rr),
        "all_exact": bool(rr) and all(bool(r["all_exact"]) for r in rr),
        "weighted_reference_ms": refw,
        "weighted_candidate_ms": candw,
        "weighted_aggregate_speedup": refw / candw if candw > 0 else None,
        "min_case_speedup": min((float(r["aggregate_speedup"]) for r in rr), default=None),
        "max_reference_drift_fraction": max((float(r["reference_drift_fraction"]) for r in rr), default=None),
        "shared_up_speedup": by.get("shared_up_nvfp4", {}).get("aggregate_speedup"),
        "shared_down_speedup": by.get("shared_down_nvfp4", {}).get("aggregate_speedup"),
        "lm_head_speedup": by.get("lm_head_nvfp4", {}).get("aggregate_speedup"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload = {
        "kind": "pro_e100_nvfp4_tiled_mrhs",
        "runner_revision": "v5_tiled_shared_decode",
        "mode": args.mode,
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "exact tiled NVFP4 shared-decode MRHS component; not a full-model E100 claim",
    }
    rt = k = adopted = None
    try:
        require_gpu_free()
        payload["environment_start"] = environment_snapshot((
            Path(__file__), PREREG,
            REPO / "pro_research" / "nvfp4_tiled_mrhs_kernels.py",
            REPO / "pro_research" / "e100_mrhs_adopted_bench.py",
            REPO / "pro_research" / "e100_adopted_baseline.py",
        ))
        rt = _new_runtime()
        all_cases, unsupported = _collect_cases(rt)
        cases = [c for c in all_cases if c.kind == "nvfp4" and c.name in MANDATORY]
        names = {c.name for c in cases}
        support = all(x in names for x in MANDATORY)
        ns = (4,) if args.mode == "smoke" else (4, 8, 16)
        k = ExactNVFP4TiledMRHS(ns)
        adopted = AdoptedSingleRHS(rt)
        correctness_batches = 1 if args.mode == "smoke" else 3
        repeats = 2 if args.mode == "smoke" else 10
        rounds = 2 if args.mode == "smoke" else 4

        records = []
        for n in ns:
            for case in cases:
                def candidate(nn, cc, out, X):
                    return k.run(
                        nn, out, cc.codes, cc.scales, rt.fused.e2m1, rt.fused.e4m3,
                        X, cc.scale, cc.rows, cc.cols,
                        apply_relu2=cc.apply_relu2, out_scale=cc.out_scale,
                    )
                r = bench_case_adopted(
                    rt, adopted, case, n, candidate,
                    "nvfp4_tiled_shared_decode_mrhs", correctness_batches, repeats, rounds,
                )
                records.append(r)
                print(
                    f"TILED-MRHS N={n:>2} {case.name:<22} exact={r['all_exact']} "
                    f"ref={r['reference_mid_ms']:.4f}ms cand={r['candidate_ms']:.4f}ms "
                    f"speedup={r['aggregate_speedup']:.3f}x drift={100*r['reference_drift_fraction']:.2f}%",
                    flush=True,
                )

        summaries = {str(n): _summary(records, n) for n in ns}
        all_exact = bool(records) and all(bool(r["all_exact"]) for r in records)
        n16 = summaries.get("16", {})
        perf = {
            "n16_weighted_ge_2_0": ((n16.get("weighted_aggregate_speedup") or 0.0) >= 2.0) if args.mode == "full" else None,
            "n16_lm_head_ge_2_0": ((n16.get("lm_head_speedup") or 0.0) >= 2.0) if args.mode == "full" else None,
            "n16_shared_up_ge_1_20": ((n16.get("shared_up_speedup") or 0.0) >= 1.20) if args.mode == "full" else None,
            "n16_shared_down_ge_1_20": ((n16.get("shared_down_speedup") or 0.0) >= 1.20) if args.mode == "full" else None,
            "n16_no_case_below_1_20": ((n16.get("min_case_speedup") or 0.0) >= 1.20) if args.mode == "full" else None,
            "n16_all_ref_drift_le_7pct": ((n16.get("max_reference_drift_fraction") or float("inf")) <= 0.07) if args.mode == "full" else None,
        }
        full_perf = all(v is True for v in perf.values()) if args.mode == "full" else None

        if not support or not all_exact:
            status = "correctness_failed"
        elif args.mode == "smoke":
            status = "smoke_pass"
        elif full_perf:
            status = "tiled_mrhs_candidate"
        else:
            status = "micro_null"

        payload.update({
            "config": {
                "rhs_values": list(ns),
                "mandatory_cases": list(MANDATORY),
                "correctness_batches": correctness_batches,
                "timing_repeats": repeats,
                "timing_rounds_per_ABBA_arm": rounds,
                "tile_packed_vectors": 256,
                "tile_weights_per_row": 2048,
                "max_dynamic_smem_bytes": {"4": 32768, "8": 16384, "16": 8192},
            },
            "supported_cases": sorted(names),
            "unsupported_cases_from_collector": unsupported,
            "mandatory_support_pass": support,
            "records": records,
            "summary_by_n": summaries,
            "gates": {
                "all_production_adopted_candidate_exact": all_exact,
                "all_three_nvfp4_families_supported": support,
                "performance": perf,
                "full_performance_pass": full_perf,
            },
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })
    finally:
        try:
            del rt, k, adopted
            gc.collect()
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as cleanup_exc:
            payload["cleanup_warning"] = {"type": type(cleanup_exc).__name__, "message": str(cleanup_exc)}

    _write(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "output": str(OUT),
        "summary_by_n": payload.get("summary_by_n"),
        "gates": payload.get("gates"),
        "error": payload.get("error"),
        "cleanup_warning": payload.get("cleanup_warning"),
    }, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
