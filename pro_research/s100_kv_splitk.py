"""S100 K/V paired split-warp: micro exactness plus decisive V18 graph A/B/A."""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import REPO, environment_snapshot, first_divergence, percentiles, require_gpu_free, utc_now
from diag_component_marginals_graph import _prefill, _recapture, _reset_exact_state, _run
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _load_prompt_set, _new_runtime
from kv_splitk_exact import ExactKVSplitK, install_kv_pair_dispatch
from layer_capacity import apply_nonuniform_capacity
from moe_dev_batched import install_batched_moe_dev
from moe_dev_combined import install_combined_moe_dev
from moe_dev_scale_resident import planned_plane_bytes
from scale_resident_kernels import ScaleResidentKernels
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

RESULT_DIR = REPO / "pro_research" / "results" / "s100_kv_splitk"
OUT = RESULT_DIR / "PRO_S100_KV_SPLITK.json"
PREREG = REPO / "pro_research" / "S100_KV_SPLITK_PREREGISTRATION.md"


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _time_cuda(cp, fn: Callable[[], None], repeats: int) -> float:
    fn()
    cp.cuda.get_current_stream().synchronize()
    a, b = cp.cuda.Event(), cp.cuda.Event()
    a.record()
    for _ in range(repeats):
        fn()
    b.record()
    b.synchronize()
    return float(cp.cuda.get_elapsed_time(a, b)) / repeats


def _micro(rt, candidate: ExactKVSplitK, mode: str) -> dict[str, Any]:
    cp = rt.cp
    orig = rt.k.mv_bf16
    rng = np.random.default_rng(0x5100C0DE)
    repeats = 8 if mode == "smoke" else 40
    records = []

    for layer in rt.attn_layers:
        d = rt.layer[layer]
        x = cp.asarray(rng.standard_normal(rt.hidden).astype(np.float32))
        rk = cp.empty(rt.kv_dim, dtype=cp.float32)
        rv = cp.empty(rt.kv_dim, dtype=cp.float32)
        ck = cp.empty_like(rk)
        cv = cp.empty_like(rv)
        ck2 = cp.empty_like(rk)
        cv2 = cp.empty_like(rv)

        orig(rk, d["k_proj"], x, rt.kv_dim, rt.hidden)
        orig(rv, d["v_proj"], x, rt.kv_dim, rt.hidden)
        candidate.run_pair(ck, cv, d["k_proj"], d["v_proj"], x)
        candidate.run_pair(ck2, cv2, d["k_proj"], d["v_proj"], x)
        cp.cuda.get_current_stream().synchronize()

        rkb, rvb = cp.asnumpy(rk).view(np.uint32), cp.asnumpy(rv).view(np.uint32)
        ckb, cvb = cp.asnumpy(ck).view(np.uint32), cp.asnumpy(cv).view(np.uint32)
        c2kb, c2vb = cp.asnumpy(ck2).view(np.uint32), cp.asnumpy(cv2).view(np.uint32)
        km = int(np.count_nonzero(rkb != ckb))
        vm = int(np.count_nonzero(rvb != cvb))
        kd = int(np.count_nonzero(ckb != c2kb))
        vd = int(np.count_nonzero(cvb != c2vb))
        finite = bool(np.isfinite(ckb.view(np.float32)).all() and np.isfinite(cvb.view(np.float32)).all())

        def ref_call():
            orig(rk, d["k_proj"], x, rt.kv_dim, rt.hidden)
            orig(rv, d["v_proj"], x, rt.kv_dim, rt.hidden)

        def cand_call():
            candidate.run_pair(ck, cv, d["k_proj"], d["v_proj"], x)

        ra = _time_cuda(cp, ref_call, repeats)
        ca = _time_cuda(cp, cand_call, repeats)
        cb = _time_cuda(cp, cand_call, repeats)
        rb = _time_cuda(cp, ref_call, repeats)
        ref_mid = 0.5 * (ra + rb)
        cand_mid = 0.5 * (ca + cb)
        drift = abs(ra - rb) / ref_mid if ref_mid > 0 else float("inf")
        records.append({
            "layer": int(layer),
            "k_mismatch_count": km,
            "v_mismatch_count": vm,
            "k_repeat_mismatch_count": kd,
            "v_repeat_mismatch_count": vd,
            "finite": finite,
            "exact": km == 0 and vm == 0 and kd == 0 and vd == 0 and finite,
            "ref_a_ms_per_kv_pair": ra,
            "ref_b_ms_per_kv_pair": rb,
            "candidate_a_ms_per_kv_pair": ca,
            "candidate_b_ms_per_kv_pair": cb,
            "reference_mid_ms_per_kv_pair": ref_mid,
            "candidate_mid_ms_per_kv_pair": cand_mid,
            "speedup": ref_mid / cand_mid if cand_mid > 0 else None,
            "reference_drift_fraction": drift,
        })
        del x, rk, rv, ck, cv, ck2, cv2

    return {
        "records": records,
        "all_exact": bool(records) and all(bool(r["exact"]) for r in records),
        "median_speedup": float(np.median([r["speedup"] for r in records])) if records else None,
        "max_reference_drift_fraction": max((float(r["reference_drift_fraction"]) for r in records), default=None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload: dict[str, Any] = {
        "kind": "pro_s100_kv_splitk",
        "mode": args.mode,
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "base_revision": "pro-research@43720efbb202c115b49413e13157dad4867093bf",
        "baseline": "V18 = V6 + H-SCALE + B3",
    }

    restore_sel = restore_combined = restore_kv = None
    try:
        require_gpu_free()
        import cupy as cp

        prompts, _expected, n, capacity = _load_prompt_set(args.mode)
        n = min(n, 32) if args.mode == "smoke" else max(n, 256)
        preheat = 32 if args.mode == "smoke" else 128
        payload["config"] = {
            "tokens_per_prompt": int(n),
            "prompt_count": len(prompts),
            "preheat_tokens": preheat,
            "kv_shape": [256, 2688],
            "reference_warps": 8,
            "warps_per_partial_block": 4,
        }
        payload["environment"] = environment_snapshot((
            Path(__file__), PREREG,
            REPO / "pro_research" / "kv_splitk_exact.py",
            REPO / "pro_research" / "combined_v18.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
        ))

        rt = _new_runtime(capacity)
        dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
        split = ExactKVSplitK(rt.kv_dim, rt.hidden)
        rt.enable_cache(capacity)
        apply_nonuniform_capacity(rt)
        rt.device_cache = True
        rt.deterministic_accum = True
        restore_sel, selective_counters = _install_selective(rt, dense)
        install_batched_moe_dev(rt, down, up)

        # Compile/exercise the candidate and establish direct bit-exactness
        # before any graph performance claim.
        payload["micro"] = _micro(rt, split, args.mode)

        rt.setup_graph()
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        planned = planned_plane_bytes(rt)
        free_before = int(cp.cuda.Device(0).mem_info[0])
        payload["vram"] = {
            "planned_v18_plane_mib": planned / 1024 / 1024,
            "free_before_mib": free_before / 1024 / 1024,
            "fits": planned <= free_before,
            "split_partial_kib": int(split.partials.nbytes) / 1024,
        }
        if planned > free_before:
            payload.update({"status": "vram_gate_failed", "completed_utc": utc_now()})
            _write(payload)
            print(json.dumps(payload["vram"], indent=2))
            return 2

        sres = ScaleResidentKernels()
        restore_combined = install_combined_moe_dev(rt, down, up, sres)
        _recapture(rt)  # V18 is now the baseline graph.

        _reset_exact_state(rt)
        _prefill(rt, prompts[0]["prompt_ids"])
        for _ in range(preheat):
            rt.step_graph(None)
        rt._graph_stream.synchronize()

        def arm(label: str) -> dict[str, Any]:
            ids_by: dict[str, list[int]] = {}
            raw_ms: list[float] = []
            for p in prompts:
                ids, ms = _run(rt, p["prompt_ids"], n)
                ids_by[p["prompt"]] = [int(x) for x in ids]
                raw_ms.extend(float(x) for x in ms)
            pc = percentiles(raw_ms)
            return {
                "label": label,
                "token_ids": ids_by,
                "raw_ms": raw_ms,
                "percentiles": pc,
                "tok_s": 1000.0 / float(pc["p50"]),
            }

        base_a = arm("V18_BASE_A")
        restore_kv = install_kv_pair_dispatch(rt, split)
        _recapture(rt)
        cand = arm("V18_PLUS_KV_SPLIT")
        restore_kv()
        restore_kv = None
        _recapture(rt)
        base_b = arm("V18_BASE_B")

        a = float(base_a["percentiles"]["p50"])
        c = float(cand["percentiles"]["p50"])
        b = float(base_b["percentiles"]["p50"])
        mid = 0.5 * (a + b)
        drift = abs(a - b)
        gain = mid - c
        div_c = {name: first_divergence(base_a["token_ids"][name], cand["token_ids"][name])
                 for name in base_a["token_ids"]}
        div_b = {name: first_divergence(base_a["token_ids"][name], base_b["token_ids"][name])
                 for name in base_a["token_ids"]}

        gates = {
            "G1_candidate_token_parity": all(v is None for v in div_c.values()),
            "G2_baseline_token_parity": all(v is None for v in div_b.values()),
            "G3_micro_all_exact": bool(payload["micro"]["all_exact"]),
            "G4_baseline_drift_le_1ms": drift <= 1.0,
            "G5_gain_ge_0_20ms": gain >= 0.20,
            "G6_candidate_below_20ms_report_only": c < 20.0,
        }
        if not (gates["G1_candidate_token_parity"] and gates["G2_baseline_token_parity"] and gates["G3_micro_all_exact"]):
            status = "correctness_failed"
        elif not gates["G4_baseline_drift_le_1ms"]:
            status = "measurement_unstable"
        elif gates["G5_gain_ge_0_20ms"]:
            status = "adoption_candidate"
        else:
            status = "gate_failed"

        payload.update({
            "selective_dispatch_counters": selective_counters,
            "arms": {"BASE_A": base_a, "CAND": cand, "BASE_B": base_b},
            "baseline_midpoint_ms": mid,
            "baseline_drift_ms": drift,
            "candidate_p50_ms": c,
            "candidate_tok_s": 1000.0 / c,
            "gain_ms_per_token": gain,
            "first_divergence_candidate": div_c,
            "first_divergence_base_b": div_b,
            "gates": gates,
            "status": status,
            "completed_utc": utc_now(),
        })

        restore_combined()
        restore_combined = None
        restore_sel()
        restore_sel = None
        del rt, dense, down, up, split, sres
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })
    finally:
        # Restores are best effort only; they must never overwrite the scientific status.
        for fn in (restore_kv, restore_combined, restore_sel):
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass

    _write(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "micro": payload.get("micro"),
        "baseline_midpoint_ms": payload.get("baseline_midpoint_ms"),
        "baseline_drift_ms": payload.get("baseline_drift_ms"),
        "candidate_p50_ms": payload.get("candidate_p50_ms"),
        "candidate_tok_s": payload.get("candidate_tok_s"),
        "gain_ms_per_token": payload.get("gain_ms_per_token"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
    }, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed", "vram_gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
