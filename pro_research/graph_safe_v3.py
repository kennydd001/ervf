"""PRO V3-G0S: test the narrow prompt-staging repair for E1F22 CUDA graph.

The existing G0 result is not overwritten or reinterpreted.  This runner writes
PRO_V3_G0S_GRAPH_SAFE.json and synchronizes only prompt-token graph replays so a
single pinned 4-byte H2D source cannot be overwritten before CUDA consumes it.
Decode replays remain the original runtime path.
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
    load_json,
    percentiles,
    require_gpu_free,
    result_path,
    utc_now,
    write_json_atomic,
)
from graph_e1f22 import _load_prompt_set, _new_runtime, _run_eager_timed

OUT = result_path("PRO_V3_G0S_GRAPH_SAFE.json")
TS200 = REPO / "reports" / "treesweep200"


def _run_graph_safe(rt, prompt_ids: list[int], n: int) -> tuple[list[int], list[float]]:
    """Graph decode with safe prompt staging; decode hot path is unchanged."""
    rt.reset()
    start = int(rt._ring_i)

    # The original runtime reuses one pinned 4-byte host word for every async
    # prompt H2D.  Reusing that word before the copy has consumed it is a legal
    # host-side race.  A prompt-only stream sync is deliberately conservative:
    # it changes TTFT/prefill only, never the timed decode loop below.
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
    """Independent direct test of the existing two-pass argmax kernel."""
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
            "contains_argmax_final": "argmax_final" in low,
            "contains_embed_gather": "embed_gather" in low,
            "contains_pos_inc": "pos_inc" in low,
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_v3_g0s_graph_safe",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "preregistration": "pro_research/PRO_V3_PREREGISTRATION.md",
        "repair_under_test": "prompt-only stream synchronization before reusing pinned H2D staging word",
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
            REPO / "pro_research" / "graph_safe_v3.py",
            REPO / "pro_research" / "PRO_V3_PREREGISTRATION.md",
        ))

        rt = _new_runtime(capacity)
        eager_ids: dict[str, list[int]] = {}
        eager_ms: list[float] = []
        for p in prompts:
            ids, ms = _run_eager_timed(rt, p["prompt_ids"], n)
            eager_ids[p["prompt"]] = ids
            eager_ms.extend(ms)

        # Clean cache state before capture.  This preserves the tested E1F21
        # semantics and binds the graph to fresh device-cache tables.
        rt.enable_cache(capacity)
        rt.device_cache = True
        rt.deterministic_accum = True
        import cupy as cp

        free0 = int(cp.cuda.Device(0).mem_info[0])
        rt.setup_graph()
        free1 = int(cp.cuda.Device(0).mem_info[0])
        extra_vram = int(getattr(rt, "graph_extra_vram_bytes", free0 - free1))

        payload["argmax_probe"] = _argmax_probe(rt)
        payload["graph_dot_probe"] = _dot_probe(rt)

        graph_ids: dict[str, list[int]] = {}
        graph_ms: list[float] = []
        for p in prompts:
            ids, ms = _run_graph_safe(rt, p["prompt_ids"], n)
            graph_ids[p["prompt"]] = ids
            graph_ms.extend(ms)

        det_n = n if args.mode == "full" else min(n, 16)
        det: dict[str, Any] = {}
        for p in prompts:
            a = _run_graph_safe_collect(rt, p["prompt_ids"], det_n)
            b = _run_graph_safe_collect(rt, p["prompt_ids"], det_n)
            det[p["prompt"]] = {
                "identical": a == b,
                "first_divergence": first_divergence(a, b),
            }

        eager_p = percentiles(eager_ms)
        graph_p = percentiles(graph_ms)
        per_prompt: dict[str, Any] = {}
        anchor_info: dict[str, Any] = {}
        repeat_pathology = []
        for p in prompts:
            name = p["prompt"]
            eids, gids = eager_ids[name], graph_ids[name]
            last_prompt = int(p["prompt_ids"][-1])
            graph_repeats_last = bool(gids and all(x == last_prompt for x in gids))
            eager_repeats_last = bool(eids and all(x == last_prompt for x in eids))
            repeat_pathology.append(graph_repeats_last and not eager_repeats_last)
            per_prompt[name] = {
                "kind": p["kind"],
                "eager_ids": eids,
                "graph_safe_ids": gids,
                "identical": eids == gids,
                "first_divergence": first_divergence(eids, gids),
                "last_prompt_token": last_prompt,
                "graph_repeats_last_prompt_token": graph_repeats_last,
                "eager_repeats_last_prompt_token": eager_repeats_last,
            }
            if p["kind"] == "anchor" and name in expected:
                m = min(len(gids), len(expected[name]))
                anchor_info[name] = {
                    "compared": m,
                    "identical_prefix": gids[:m] == expected[name][:m],
                    "first_divergence": first_divergence(gids[:m], expected[name][:m]),
                }

        gain = None
        if eager_p["p50"] is not None and graph_p["p50"] is not None:
            gain = float(eager_p["p50"] - graph_p["p50"])

        sample_ok = len(graph_ms) >= 500 if args.mode == "full" else None
        gates = {
            "argmax_direct_tie": bool(payload["argmax_probe"]["passed"]),
            "graph_safe_equals_eager": all(v["identical"] for v in per_prompt.values()),
            "graph_safe_deterministic": all(v["identical"] for v in det.values()),
            "repeated_last_prompt_pathology_absent": not any(repeat_pathology),
            "extra_vram_lt_64MiB": extra_vram < 64 * 1024 * 1024,
            "full_speed_gain_ge_2_5ms": bool(gain is not None and gain >= 2.5),
            "full_samples_ge_500": sample_ok,
        }
        correctness = all(gates[k] for k in (
            "argmax_direct_tie",
            "graph_safe_equals_eager",
            "graph_safe_deterministic",
            "repeated_last_prompt_pathology_absent",
            "extra_vram_lt_64MiB",
        ))
        if args.mode == "full":
            passed = correctness and gates["full_speed_gain_ge_2_5ms"] and gates["full_samples_ge_500"]
        else:
            passed = correctness

        payload.update({
            "arms": {
                "EGR": {"timing_ms": eager_p},
                "GRAPH_SAFE": {"timing_ms": graph_p, "extra_vram_bytes": extra_vram},
            },
            "per_prompt": per_prompt,
            "determinism": det,
            "external_anchor_informative": anchor_info,
            "gates": gates,
            "summary": {
                "eager_p50_ms": eager_p["p50"],
                "graph_safe_p50_ms": graph_p["p50"],
                "gain_ms": gain,
                "eager_tok_s": None if not eager_p["p50"] else 1000.0 / float(eager_p["p50"]),
                "graph_safe_tok_s": None if not graph_p["p50"] else 1000.0 / float(graph_p["p50"]),
            },
            "status": "pass" if passed else "gate_failed",
            "completed_utc": utc_now(),
        })

        del rt
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
        "argmax_probe": payload.get("argmax_probe"),
    })
    return 0 if payload.get("status") in {"pass", "gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
