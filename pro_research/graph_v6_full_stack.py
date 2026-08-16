"""PRO V6: integrate all three verified mechanisms into one CUDA graph
capture -- V3-G0S (graph-safe residency) + V3-G1B/V4 (selective ERVF on
rt.k.mv_bf16/mv_fp8_tensor) + V5 (batched panel_scan/reduce_partials via a
replaced _moe_dev).

_install_selective (selective_ervf_v3.py) patches rt.k.mv_bf16/mv_fp8_tensor/
mv_f32, used inside _attention (Q/K/V/O) and _mamba (in_proj/out_proj).
install_batched_moe_dev (moe_dev_batched.py) replaces rt._moe_dev entirely,
the MoE-layer dispatch used only for routed-expert up/down projections.
These are disjoint call sites -- both patches can be installed simultaneously
before setup_graph() captures, and each should be recorded into the same
graph independently, exactly as V4 did for the first pair (device routing +
graph-safe was already inside the graph; selective ERVF was layered on top).

Not preregistered as its own document -- this follows directly from
PRO_V4_PREREGISTRATION.md's and PRO_V5_PREREGISTRATION.md's own gates,
combined with no new policy decision (same four frozen ERVF shapes, same
batched-kernel scope, same safe prompt-staging). Gates below mirror V4's
exactly, plus V5's bitexact/control checks reused at the integration level.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from typing import Any

import numpy as np

from common import (
    REPO,
    environment_snapshot,
    first_divergence,
    percentiles,
    require_gpu_free,
    result_path,
    utc_now,
    write_json_atomic,
)
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _load_prompt_set, _new_runtime, _run_eager_timed
from moe_dev_batched import install_batched_moe_dev
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

OUT = result_path("PRO_V6_FULL_STACK.json")
G0S = result_path("PRO_V3_G0S_GRAPH_SAFE.json")
G1B = result_path("PRO_V3_G1B_SELECTIVE_ERVF.json")
V4 = result_path("PRO_V4_GRAPH_SELECTIVE.json")
V5 = result_path("PRO_V5_BATCHED_DOWNPROJ_AB.json")


def _run_graph_safe(rt, prompt_ids: list[int], n: int) -> tuple[list[int], list[float]]:
    rt.reset()
    start = int(rt._ring_i)
    for token in prompt_ids:
        rt.step_graph(int(token))
        rt._graph_stream.synchronize()
    first_slot = (start + len(prompt_ids) - 1) % int(rt._ring_size)
    cur = int(rt.ring_harvest(first_slot, 1)[0])
    ids = [cur]
    samples: list[float] = []
    for _ in range(n - 1):
        slot = int(rt._ring_i)
        t0 = time.perf_counter_ns()
        rt.step_graph(None)
        cur = int(rt.ring_harvest(slot, 1)[0])
        samples.append((time.perf_counter_ns() - t0) / 1e6)
        ids.append(cur)
    return ids, samples


def _run_graph_safe_collect(rt, prompt_ids: list[int], n: int) -> list[int]:
    ids, _ = _run_graph_safe(rt, prompt_ids, n)
    return ids


def _argmax_probe(rt) -> dict[str, Any]:
    import cupy as cp

    x = cp.full(rt.vocab, -11.0, dtype=cp.float32)
    lo, hi = 123, min(987, rt.vocab - 1)
    x[lo] = cp.float32(7.0)
    x[hi] = cp.float32(7.0)
    rt._tok_dev.fill(-1)
    rt.k.argmax_logits(rt._tok_dev, x, rt.vocab, rt._am_max, rt._am_idx)
    cp.cuda.Device(0).synchronize()
    got = int(cp.asnumpy(rt._tok_dev)[0])
    cupy_argmax = int(cp.asnumpy(cp.argmax(x)))
    return {"expected_low_index": lo, "kernel": got, "cupy": cupy_argmax, "passed": got == lo == cupy_argmax}


def _dot_probe(rt) -> dict[str, Any]:
    try:
        text = rt._graph.debug_dot_str()
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        low = text.lower()
        return {
            "available": True,
            "length": len(text),
            "contains_bf16_ervf": "pro_gemv_bf16_ervf16" in low,
            "contains_fp8_ervf": "pro_gemv_fp8_tensor_ervf16" in low,
            "contains_panel_scan_batched": "panel_scan_batched" in low,
            "contains_reduce_partials_batched": "reduce_partials_batched" in low,
            "contains_accumulate_batched": "weighted_accumulate_ind_batched" in low,
            "contains_up_proj_batched": "gemv_nvfp4_ervf_ind_batched" in low,
            "contains_gather_batched": "gather_down_sparse_ind_batched" in low,
            "contains_down_masked_batched": "gemv_down_masked_partial_ind_batched" in low,
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _load_prior(path, key_path):
    if not path.exists():
        return None
    from common import load_json

    d = load_json(path)
    cur = d
    for k in key_path:
        if cur is None:
            return None
        cur = cur.get(k)
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_v6_full_stack",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "mechanism_under_test": (
            "install selective-ERVF dispatch (rt.k.mv_bf16/mv_fp8_tensor) AND "
            "batched-_moe_dev (panel_scan/reduce_partials) BEFORE setup_graph(), "
            "so both get captured into the same graph alongside device routing "
            "and graph-safe prompt staging -- three verified mechanisms in one arm"
        ),
    }

    try:
        require_gpu_free()
        prompts, expected, n, capacity = _load_prompt_set(args.mode)
        payload["config"] = {"tokens_per_prompt": n, "capacity": capacity, "prompt_count": len(prompts),
                             "prompt_sync_excluded_from_decode_timing": True}
        payload["environment"] = environment_snapshot((
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "fused_nvfp4.py",
            REPO / "pro_research" / "graph_v6_full_stack.py",
            REPO / "pro_research" / "moe_dev_batched.py",
            REPO / "pro_research" / "selective_ervf_v3.py",
        ))
        payload["prior_results_informative"] = {
            "v4_p50_ms": _load_prior(V4, ["summary", "graph_selective_p50_ms"]),
            "v4_egr_p50_ms": _load_prior(V4, ["summary", "eager_p50_ms"]),
            "v5_batched_p50_ms_eager": _load_prior(V5, ["summary", "batched_p50_ms"]),
            "v5_base_mid_p50_ms_eager": _load_prior(V5, ["summary", "baseline_mid_p50_ms"]),
        }

        rt = _new_runtime(capacity)
        dense = DenseERVF()
        batch_kernels = DownProjBatchKernels()
        up_kernels = UpProjBatchKernels()
        # DownGatherBatchKernels (gather_down_sparse_ind + gemv_down_masked_
        # partial_ind batching) intentionally not constructed/installed here
        # -- see the comment at install_batched_moe_dev below.

        # EGR: production kernels, device-cache eager, no graph, no patches --
        # same-session fresh comparison point, matching V4's convention.
        eager_ids: dict[str, list[int]] = {}
        eager_ms: list[float] = []
        for p in prompts:
            ids, ms = _run_eager_timed(rt, p["prompt_ids"], n)
            eager_ids[p["prompt"]] = ids
            eager_ms.extend(ms)

        rt.enable_cache(capacity)
        rt.device_cache = True
        rt.deterministic_accum = True
        restore_selective, ervf_counters = _install_selective(rt, dense)
        # gather_kernels deliberately NOT passed here: bit-exact and a real
        # +0.68 ms/2.6% gain in isolation (v_gather_batched_ab.py), but once
        # combined with everything else in this graph the marginal benefit
        # vanished (47.3644 vs 47.3669 tok/s, within noise) while costing
        # ~387 MiB (top_k x 23 layers x 2.68 MB mirrors) against a 64 MiB
        # budget -- not worth adopting. See RESEARCH_NOTEBOOK.md 2026-08-16.
        restore_moe = install_batched_moe_dev(rt, batch_kernels, up_kernels)
        import cupy as cp

        free0 = int(cp.cuda.Device(0).mem_info[0])
        rt.setup_graph()
        free1 = int(cp.cuda.Device(0).mem_info[0])
        extra_vram = int(getattr(rt, "graph_extra_vram_bytes", free0 - free1))

        payload["argmax_probe"] = _argmax_probe(rt)
        payload["graph_dot_probe"] = _dot_probe(rt)
        payload["capture_dispatch_counters"] = dict(ervf_counters)

        v6_ids: dict[str, list[int]] = {}
        v6_ms: list[float] = []
        for p in prompts:
            ids, ms = _run_graph_safe(rt, p["prompt_ids"], n)
            v6_ids[p["prompt"]] = ids
            v6_ms.extend(ms)

        det_n = n if args.mode == "full" else min(n, 16)
        det: dict[str, Any] = {}
        for p in prompts:
            a = _run_graph_safe_collect(rt, p["prompt_ids"], det_n)
            b = _run_graph_safe_collect(rt, p["prompt_ids"], det_n)
            det[p["prompt"]] = {"identical": a == b, "first_divergence": first_divergence(a, b)}

        rt._bad_pick = 1
        rt.enable_cache(capacity)
        rt.device_cache = True
        rt.setup_graph()
        ctl_n = min(n, 64)
        ctl: dict[str, Any] = {}
        for p in prompts:
            ids = _run_graph_safe_collect(rt, p["prompt_ids"], ctl_n)
            ref = eager_ids[p["prompt"]][:ctl_n]
            ctl[p["prompt"]] = {"identical": ids == ref, "first_divergence": first_divergence(ids, ref)}
        rt._bad_pick = 0
        restore_selective()
        restore_moe()

        eager_p = percentiles(eager_ms)
        v6_p = percentiles(v6_ms)

        per_prompt: dict[str, Any] = {}
        for p in prompts:
            name = p["prompt"]
            eids, gids = eager_ids[name], v6_ids[name]
            per_prompt[name] = {
                "kind": p["kind"],
                "identical": eids == gids,
                "first_divergence": first_divergence(eids, gids),
            }

        gain = None
        if eager_p["p50"] is not None and v6_p["p50"] is not None:
            gain = float(eager_p["p50"] - v6_p["p50"])

        control_diverged = any(not v["identical"] for v in ctl.values())
        sample_ok = len(v6_ms) >= 500 if args.mode == "full" else None
        gates = {
            "argmax_direct_tie": bool(payload["argmax_probe"]["passed"]),
            "graph_dot_contains_all_mechanisms": bool(
                payload["graph_dot_probe"].get("contains_bf16_ervf")
                and payload["graph_dot_probe"].get("contains_fp8_ervf")
                and payload["graph_dot_probe"].get("contains_panel_scan_batched")
                and payload["graph_dot_probe"].get("contains_reduce_partials_batched")
                and payload["graph_dot_probe"].get("contains_accumulate_batched")
                and payload["graph_dot_probe"].get("contains_up_proj_batched")
                # contains_gather_batched/contains_down_masked_batched are NOT
                # required here: gather_kernels is deliberately not installed
                # below (see the comment there) -- probed for information only.
            ),
            "v6_equals_egr": all(v["identical"] for v in per_prompt.values()),
            "v6_deterministic": all(v["identical"] for v in det.values()),
            "bad_pick_control_diverges": control_diverged,
            "extra_vram_lt_64MiB": extra_vram < 64 * 1024 * 1024,
            "full_speed_gain_ge_2_5ms": bool(gain is not None and gain >= 2.5),
            "full_samples_ge_500": sample_ok,
        }
        correctness = all(gates[k] for k in (
            "argmax_direct_tie", "graph_dot_contains_all_mechanisms", "v6_equals_egr",
            "v6_deterministic", "bad_pick_control_diverges", "extra_vram_lt_64MiB",
        ))
        passed = (correctness and gates["full_speed_gain_ge_2_5ms"] and gates["full_samples_ge_500"]) if args.mode == "full" else correctness

        prior = payload["prior_results_informative"]
        composition_note = None
        if v6_p["p50"] is not None and prior.get("v4_p50_ms"):
            composition_note = {
                "v6_p50_ms": v6_p["p50"],
                "v4_p50_ms": prior["v4_p50_ms"],
                "v6_beats_v4": v6_p["p50"] <= prior["v4_p50_ms"],
                "caveat": "cross-session comparison, informative only, not a gate",
            }

        payload.update({
            "arms": {"EGR": {"timing_ms": eager_p}, "V6": {"timing_ms": v6_p, "extra_vram_bytes": extra_vram}},
            "per_prompt": per_prompt,
            "determinism": det,
            "control": ctl,
            "composition_vs_v4": composition_note,
            "gates": gates,
            "summary": {
                "eager_p50_ms": eager_p["p50"],
                "v6_p50_ms": v6_p["p50"],
                "gain_ms": gain,
                "eager_tok_s": None if not eager_p["p50"] else 1000.0 / float(eager_p["p50"]),
                "v6_tok_s": None if not v6_p["p50"] else 1000.0 / float(v6_p["p50"]),
            },
            "status": "pass" if passed else "gate_failed",
            "completed_utc": utc_now(),
        })

        del rt, dense, batch_kernels, up_kernels
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload["status"] = "technical_failure"
        payload["completed_utc"] = utc_now()
        payload["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}

    write_json_atomic(OUT, payload)
    print({
        "status": payload.get("status"),
        "output": str(OUT),
        "summary": payload.get("summary"),
        "gates": payload.get("gates"),
        "composition": payload.get("composition_vs_v4"),
    })
    return 0 if payload.get("status") in {"pass", "gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
