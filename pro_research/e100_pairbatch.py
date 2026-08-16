"""E100-PAIRBATCH: flatten N*top_k routed up-projections into one launch.

This is a structural component experiment only.  It uses real routed-expert
checkpoint bytes and compares against the already verified six-slot batched
up-projection kernel, plus a direct spot check against the adopted production
indirect ERVF path.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import REPO, environment_snapshot, require_gpu_free, require_model_dir, utc_now
from up_proj_batch_kernels import UpProjBatchKernels
from up_proj_pair_batch_kernels import UpProjPairBatchKernels

RESULT_DIR = REPO / "pro_research" / "results" / "e100_pairbatch"
OUT = RESULT_DIR / "PRO_E100_PAIRBATCH.json"
PREREG = REPO / "pro_research" / "E100_PAIRBATCH_PREREGISTRATION.md"
N = 4
TOP_K = 6
P = N * TOP_K

PAIR_MAPS = {
    "unique24": np.arange(24, dtype=np.int32).reshape(N, TOP_K),
    "n4_typical22": np.asarray([
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 0, 6],
    ], dtype=np.int32),
    "repeat6": np.tile(np.arange(6, dtype=np.int32), (N, 1)),
}


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _time_cuda(call: Callable[[], None], repeats: int, rounds: int) -> list[float]:
    import cupy as cp

    samples: list[float] = []
    for _ in range(rounds):
        st, en = cp.cuda.Event(), cp.cuda.Event()
        st.record()
        for _ in range(repeats):
            call()
        en.record()
        en.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(st, en)) / repeats)
    return samples


def _build_cache(cp, bank, ids_np, up_code: int, up_scale: int):
    unique = sorted(set(int(x) for x in ids_np.reshape(-1)))
    slot_of = {e: s for s, e in enumerate(unique)}
    cache_c = cp.empty(len(unique) * up_code, dtype=cp.uint8)
    cache_s = cp.empty(len(unique) * up_scale, dtype=cp.uint8)
    for s, e in enumerate(unique):
        cache_c[s * up_code:(s + 1) * up_code].set(bank["up_code_view"][e])
        cache_s[s * up_scale:(s + 1) * up_scale].set(bank["up_scale_view"][e])
    slots_np = np.asarray([slot_of[int(e)] for e in ids_np.reshape(-1)], dtype=np.int32)
    return unique, cache_c, cache_s, cp.asarray(slots_np), cp.asarray(ids_np.reshape(-1))


def _run_one_map(rt, refk, pairk, layer: int, name: str, ids_np: np.ndarray,
                 correctness_batches: int, repeats: int, rounds: int) -> dict[str, Any]:
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import UP_CODE, UP_SCALE

    rows, cols = rt.moe_inter, rt.hidden
    bank = rt.bank[layer]
    globals_dev = cp.asarray(bank["globals"])
    unique, cache_c, cache_s, slots, ids = _build_cache(cp, bank, ids_np, UP_CODE, UP_SCALE)

    correctness: list[dict[str, Any]] = []
    first_X = first_ref = first_pair = None
    production_spot = None

    for b in range(correctness_batches):
        seed = 0xE100B000 + 10007 * b + sum(ord(c) for c in name)
        rng = np.random.default_rng(seed)
        X = cp.asarray(rng.standard_normal((N, cols)).astype(np.float32))
        ref = cp.empty(P * rows, dtype=cp.float32)
        pair = cp.empty(P * rows, dtype=cp.float32)
        pair2 = cp.empty(P * rows, dtype=cp.float32)

        for s in range(N):
            a = s * TOP_K
            o0 = a * rows
            refk.run_batched(
                ref[o0:o0 + TOP_K * rows], cache_c, cache_s,
                slots[a:a + TOP_K], ids[a:a + TOP_K], globals_dev, 1,
                rt.fused.e2m1, rt.fused.e4m3, X[s], rows, cols, True,
                UP_CODE, UP_SCALE, TOP_K,
            )
        pairk.run(
            pair, cache_c, cache_s, slots, ids, globals_dev, 1,
            rt.fused.e2m1, rt.fused.e4m3, X, rows, cols, True,
            UP_CODE, UP_SCALE, P, TOP_K,
        )
        pairk.run(
            pair2, cache_c, cache_s, slots, ids, globals_dev, 1,
            rt.fused.e2m1, rt.fused.e4m3, X, rows, cols, True,
            UP_CODE, UP_SCALE, P, TOP_K,
        )
        cp.cuda.get_current_stream().synchronize()

        rr = cp.asnumpy(ref).view(np.uint32)
        pp = cp.asnumpy(pair).view(np.uint32)
        qq = cp.asnumpy(pair2).view(np.uint32)
        mismatch = int(np.count_nonzero(rr != pp))
        det_mismatch = int(np.count_nonzero(pp != qq))
        finite = bool(np.isfinite(rr.view(np.float32)).all() and np.isfinite(pp.view(np.float32)).all())

        # Direct reference-self-check against production adopted indirect ERVF
        # for sequence zero, first correctness batch only.
        if b == 0:
            prod = cp.empty(TOP_K * rows, dtype=cp.float32)
            dev = {"slots": slots[:TOP_K], "ids": ids[:TOP_K]}
            for j in range(TOP_K):
                rt.fused.gemv_ervf_indirect(
                    prod[j * rows:(j + 1) * rows], cache_c, cache_s, dev, j,
                    globals_dev, 1, X[0], rows, cols, True, UP_CODE, UP_SCALE,
                )
            cp.cuda.get_current_stream().synchronize()
            prod_bits = cp.asnumpy(prod).view(np.uint32)
            ref0_bits = cp.asnumpy(ref[:TOP_K * rows]).view(np.uint32)
            production_spot = {
                "bit_equal": bool(np.array_equal(prod_bits, ref0_bits)),
                "mismatch_count": int(np.count_nonzero(prod_bits != ref0_bits)),
            }

        correctness.append({
            "seed": int(seed),
            "bit_equal": mismatch == 0,
            "mismatch_count": mismatch,
            "deterministic": det_mismatch == 0,
            "determinism_mismatch_count": det_mismatch,
            "finite": finite,
        })
        if b == 0:
            first_X, first_ref, first_pair = X, ref, pair

    assert first_X is not None and first_ref is not None and first_pair is not None

    def ref_call() -> None:
        for s in range(N):
            a = s * TOP_K
            o0 = a * rows
            refk.run_batched(
                first_ref[o0:o0 + TOP_K * rows], cache_c, cache_s,
                slots[a:a + TOP_K], ids[a:a + TOP_K], globals_dev, 1,
                rt.fused.e2m1, rt.fused.e4m3, first_X[s], rows, cols, True,
                UP_CODE, UP_SCALE, TOP_K,
            )

    def pair_call() -> None:
        pairk.run(
            first_pair, cache_c, cache_s, slots, ids, globals_dev, 1,
            rt.fused.e2m1, rt.fused.e4m3, first_X, rows, cols, True,
            UP_CODE, UP_SCALE, P, TOP_K,
        )

    ref_call(); pair_call(); cp.cuda.get_current_stream().synchronize()
    ref_a = _time_cuda(ref_call, repeats, rounds)
    pair_a = _time_cuda(pair_call, repeats, rounds)
    pair_b = _time_cuda(pair_call, repeats, rounds)
    ref_b = _time_cuda(ref_call, repeats, rounds)

    ra = float(np.median(np.asarray(ref_a, dtype=np.float64)))
    rb = float(np.median(np.asarray(ref_b, dtype=np.float64)))
    rmid = 0.5 * (ra + rb)
    pmid = float(np.median(np.asarray(pair_a + pair_b, dtype=np.float64)))
    drift = abs(ra - rb) / rmid if rmid > 0 else math.inf
    speedup = rmid / pmid if pmid > 0 else math.inf

    return {
        "map": name,
        "n_sequences": N,
        "top_k": TOP_K,
        "pair_count": P,
        "unique_experts": len(unique),
        "expert_ids": ids_np.tolist(),
        "correctness_batches": correctness,
        "all_bit_equal": all(x["bit_equal"] for x in correctness),
        "all_deterministic": all(x["deterministic"] for x in correctness),
        "all_finite": all(x["finite"] for x in correctness),
        "production_reference_spot": production_spot,
        "reference_a_ms": ra,
        "reference_b_ms": rb,
        "reference_mid_ms": rmid,
        "pair_ms": pmid,
        "pair_speedup": speedup,
        "reference_drift_fraction": drift,
        "raw_reference_a_ms": ref_a,
        "raw_pair_a_ms": pair_a,
        "raw_pair_b_ms": pair_b,
        "raw_reference_b_ms": ref_b,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_e100_pairbatch",
        "mode": args.mode,
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "routed up-projection launch-flattening primitive; not matrix-byte sharing and not a full-model E100 claim",
    }

    try:
        require_gpu_free()
        sys.path.insert(0, str(REPO / "src"))
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

        payload["environment_start"] = environment_snapshot((
            Path(__file__),
            REPO / "pro_research" / "up_proj_pair_batch_kernels.py",
            REPO / "pro_research" / "up_proj_batch_kernels.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "fused_nvfp4.py",
        ))

        rt = LightningRuntime(require_model_dir(), contexts_max=4096,
                              embed_on_host=True, fp8_kv=True, verbose=False)
        layer = rt.moe_layers[len(rt.moe_layers) // 2]
        rt.load_routed_bank([layer])
        refk = UpProjBatchKernels()
        pairk = UpProjPairBatchKernels()

        correctness_batches = 1 if args.mode == "smoke" else 3
        repeats = 2 if args.mode == "smoke" else 10
        rounds = 2 if args.mode == "smoke" else 4
        records = [
            _run_one_map(rt, refk, pairk, layer, name, ids_np,
                         correctness_batches, repeats, rounds)
            for name, ids_np in PAIR_MAPS.items()
        ]

        for r in records:
            print(
                f"PAIR {r['map']:<14} U={r['unique_experts']:>2}/24 "
                f"exact={r['all_bit_equal']} prodref={r['production_reference_spot']['bit_equal']} "
                f"ref={r['reference_mid_ms']:.4f}ms pair={r['pair_ms']:.4f}ms "
                f"speedup={r['pair_speedup']:.3f}x drift={100*r['reference_drift_fraction']:.2f}%",
                flush=True,
            )

        by = {r["map"]: r for r in records}
        all_correct = all(
            r["all_bit_equal"] and r["all_deterministic"] and r["all_finite"]
            and bool(r["production_reference_spot"]["bit_equal"])
            for r in records
        )
        primary = by["unique24"]
        typical = by["n4_typical22"]
        perf = {
            "unique24_speedup_ge_1_08": float(primary["pair_speedup"]) >= 1.08,
            "unique24_reference_drift_le_7pct": float(primary["reference_drift_fraction"]) <= 0.07,
            "typical22_no_regression": float(typical["pair_speedup"]) >= 0.98,
        }
        perf_pass = all(perf.values())
        if not all_correct:
            status = "correctness_failed"
        elif args.mode == "smoke":
            status = "smoke_pass"
        elif perf_pass:
            status = "pairbatch_candidate"
        else:
            status = "pairbatch_null"

        payload.update({
            "config": {
                "layer": int(layer),
                "rows": int(rt.moe_inter),
                "cols": int(rt.hidden),
                "n_sequences": N,
                "top_k": TOP_K,
                "pairs": P,
                "correctness_batches": correctness_batches,
                "timing_repeats": repeats,
                "timing_rounds_per_arm": rounds,
            },
            "records": records,
            "gates": {
                "all_maps_exact_deterministic_finite_and_reference_spot_exact": all_correct,
                "full_performance": perf,
                "full_performance_pass": perf_pass if args.mode == "full" else None,
            },
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })

        del rt, refk, pairk
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })

    _write(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "output": str(OUT),
        "gates": payload.get("gates"),
    }, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
