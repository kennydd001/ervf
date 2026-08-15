"""PRO V4: physically integrate graph-safe residency and selective ERVF.

Neither V3-G0S (graph residency alone) nor V3-G1B (selective ERVF alone) is
reinterpreted or overwritten. This runner tests whether installing the
selective-ERVF dense dispatch on ``rt.k`` BEFORE ``rt.setup_graph()`` captures
the ERVF kernels into the replayed CUDA graph, so both measured mechanisms run
in one physical arm instead of being added arithmetically.

See PRO_V4_PREREGISTRATION.md for the frozen gates. Written before this file
was run on target hardware.
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
from ervf_dense import DenseERVF
from graph_e1f22 import _load_prompt_set, _new_runtime, _run_eager_timed
from selective_ervf_v3 import _install_selective

OUT = result_path("PRO_V4_GRAPH_SELECTIVE.json")
G0S = result_path("PRO_V3_G0S_GRAPH_SAFE.json")
G1B = result_path("PRO_V3_G1B_SELECTIVE_ERVF.json")


def _run_graph_safe(rt, prompt_ids: list[int], n: int) -> tuple[list[int], list[float]]:
    """Identical staging discipline to V3-G0S: sync after each prompt token,
    decode hot path untouched."""
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
    return {
        "expected_low_index": lo,
        "kernel": got,
        "cupy": cupy_argmax,
        "passed": got == lo == cupy_argmax,
    }


def _dot_probe(rt) -> dict[str, Any]:
    try:
        text = rt._graph.debug_dot_str()
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        low = text.lower()
        return {
            "available": True,
            "length": len(text),
            "contains_argmax_part": "argmax_part" in low,
            "contains_embed_gather": "embed_gather" in low,
            "contains_bf16_ervf": "pro_gemv_bf16_ervf16" in low,
            "contains_fp8_ervf": "pro_gemv_fp8_tensor_ervf16" in low,
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
        "kind": "pro_v4_graph_selective",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "preregistration": "pro_research/PRO_V4_PREREGISTRATION.md",
        "mechanism_under_test": (
            "install selective ERVF dispatch on rt.k before setup_graph() so "
            "the CUDA graph captures the ERVF kernels for the four frozen "
            "shapes, combining V3-G0S graph residency and V3-G1B selective "
            "ERVF in one physical arm"
        ),
    }

    try:
        require_gpu_free()
        prompts, expected, n, capacity = _load_prompt_set(args.mode)
        payload["config"] = {
            "tokens_per_prompt": n,
            "capacity": capacity,
            "prompt_count": len(prompts),
            "prompt_sync_excluded_from_decode_timing": True,
        }
        payload["environment"] = environment_snapshot((
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "gpu_kernels.py",
            REPO / "pro_research" / "graph_selective_v4.py",
            REPO / "pro_research" / "selective_ervf_v3.py",
            REPO / "pro_research" / "ervf_dense.py",
            REPO / "pro_research" / "PRO_V4_PREREGISTRATION.md",
        ))
        payload["prior_results_informative"] = {
            "g0s_graph_safe_p50_ms": _load_prior(G0S, ["summary", "graph_safe_p50_ms"]),
            "g1b_selective_p50_ms": _load_prior(G1B, ["summary", "selective_p50_ms"]),
            "g0s_egr_p50_ms": _load_prior(G0S, ["summary", "eager_p50_ms"]),
            "g1b_base_mid_p50_ms": _load_prior(G1B, ["summary", "baseline_mid_p50_ms"]),
        }

        rt = _new_runtime(capacity)
        dense = DenseERVF()  # compile before any timed arm

        # EGR: production kernels, device-cache eager, no graph. Same
        # construction as G0S/G1B for a fresh same-session comparison point.
        eager_ids: dict[str, list[int]] = {}
        eager_ms: list[float] = []
        for p in prompts:
            ids, ms = _run_eager_timed(rt, p["prompt_ids"], n)
            eager_ids[p["prompt"]] = ids
            eager_ms.extend(ms)

        # Rebuild cache state, install selective dispatch, THEN capture so the
        # ERVF kernels for the four frozen shapes are what gets recorded.
        rt.enable_cache(capacity)
        rt.device_cache = True
        rt.deterministic_accum = True
        restore, counters = _install_selective(rt, dense)
        import cupy as cp

        free0 = int(cp.cuda.Device(0).mem_info[0])
        rt.setup_graph()
        free1 = int(cp.cuda.Device(0).mem_info[0])
        extra_vram = int(getattr(rt, "graph_extra_vram_bytes", free0 - free1))

        payload["argmax_probe"] = _argmax_probe(rt)
        payload["graph_dot_probe"] = _dot_probe(rt)
        payload["capture_dispatch_counters"] = dict(counters)

        gs_ids: dict[str, list[int]] = {}
        gs_ms: list[float] = []
        for p in prompts:
            ids, ms = _run_graph_safe(rt, p["prompt_ids"], n)
            gs_ids[p["prompt"]] = ids
            gs_ms.extend(ms)

        det_n = n if args.mode == "full" else min(n, 16)
        det: dict[str, Any] = {}
        for p in prompts:
            a = _run_graph_safe_collect(rt, p["prompt_ids"], det_n)
            b = _run_graph_safe_collect(rt, p["prompt_ids"], det_n)
            det[p["prompt"]] = {"identical": a == b, "first_divergence": first_divergence(a, b)}

        # CTL: recapture with the existing bad_pick sabotage baked into the
        # same selective-dispatch graph. Must diverge (rule 8).
        rt._bad_pick = 1
        rt.enable_cache(capacity)
        rt.device_cache = True
        rt.setup_graph()
        ctl_n = min(n, 64)
        ctl: dict[str, Any] = {}
        for p in prompts:
            ids = _run_graph_safe_collect(rt, p["prompt_ids"], ctl_n)
            ref = eager_ids[p["prompt"]][:ctl_n]
            ctl[p["prompt"]] = {
                "ids": ids,
                "reference_ids": ref,
                "identical": ids == ref,
                "first_divergence": first_divergence(ids, ref),
            }
        rt._bad_pick = 0
        restore()

        eager_p = percentiles(eager_ms)
        gs_p = percentiles(gs_ms)

        per_prompt: dict[str, Any] = {}
        anchor_info: dict[str, Any] = {}
        for p in prompts:
            name = p["prompt"]
            eids, gids = eager_ids[name], gs_ids[name]
            per_prompt[name] = {
                "kind": p["kind"],
                "identical": eids == gids,
                "first_divergence": first_divergence(eids, gids),
            }
            if p["kind"] == "anchor" and name in expected:
                m = min(len(gids), len(expected[name]))
                anchor_info[name] = {
                    "compared": m,
                    "identical_prefix": gids[:m] == expected[name][:m],
                    "first_divergence": first_divergence(gids[:m], expected[name][:m]),
                }

        gain = None
        if eager_p["p50"] is not None and gs_p["p50"] is not None:
            gain = float(eager_p["p50"] - gs_p["p50"])

        control_diverged = any(not v["identical"] for v in ctl.values())
        sample_ok = len(gs_ms) >= 500 if args.mode == "full" else None
        gates = {
            "argmax_direct_tie": bool(payload["argmax_probe"]["passed"]),
            "graph_dot_contains_ervf": bool(
                payload["graph_dot_probe"].get("contains_bf16_ervf")
                and payload["graph_dot_probe"].get("contains_fp8_ervf")
            ),
            "graph_selective_equals_egr": all(v["identical"] for v in per_prompt.values()),
            "graph_selective_deterministic": all(v["identical"] for v in det.values()),
            "bad_pick_control_diverges": control_diverged,
            "extra_vram_lt_64MiB": extra_vram < 64 * 1024 * 1024,
            "full_speed_gain_ge_2_5ms": bool(gain is not None and gain >= 2.5),
            "full_samples_ge_500": sample_ok,
        }
        correctness = all(gates[k] for k in (
            "argmax_direct_tie",
            "graph_dot_contains_ervf",
            "graph_selective_equals_egr",
            "graph_selective_deterministic",
            "bad_pick_control_diverges",
            "extra_vram_lt_64MiB",
        ))
        if args.mode == "full":
            passed = correctness and gates["full_speed_gain_ge_2_5ms"] and gates["full_samples_ge_500"]
        else:
            passed = correctness

        prior = payload["prior_results_informative"]
        composition_note = None
        if gs_p["p50"] is not None and prior.get("g0s_graph_safe_p50_ms") and prior.get("g1b_selective_p50_ms"):
            best_prior = min(prior["g0s_graph_safe_p50_ms"], prior["g1b_selective_p50_ms"])
            composition_note = {
                "v4_p50_ms": gs_p["p50"],
                "best_single_mechanism_p50_ms": best_prior,
                "v4_at_or_below_best_single": gs_p["p50"] <= best_prior,
                "caveat": "cross-session comparison, informative only, not a gate",
            }

        payload.update({
            "arms": {
                "EGR": {"timing_ms": eager_p},
                "GRAPH_SELECTIVE": {"timing_ms": gs_p, "extra_vram_bytes": extra_vram},
            },
            "per_prompt": per_prompt,
            "determinism": det,
            "control": ctl,
            "external_anchor_informative": anchor_info,
            "composition_vs_prior_single_mechanisms": composition_note,
            "gates": gates,
            "summary": {
                "eager_p50_ms": eager_p["p50"],
                "graph_selective_p50_ms": gs_p["p50"],
                "gain_ms": gain,
                "eager_tok_s": None if not eager_p["p50"] else 1000.0 / float(eager_p["p50"]),
                "graph_selective_tok_s": None if not gs_p["p50"] else 1000.0 / float(gs_p["p50"]),
            },
            "status": "pass" if passed else "gate_failed",
            "completed_utc": utc_now(),
        })

        del rt, dense
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload["status"] = "technical_failure"
        payload["completed_utc"] = utc_now()
        payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    write_json_atomic(OUT, payload)
    print({
        "status": payload.get("status"),
        "output": str(OUT),
        "summary": payload.get("summary"),
        "gates": payload.get("gates"),
        "composition": payload.get("composition_vs_prior_single_mechanisms"),
    })
    return 0 if payload.get("status") in {"pass", "gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
