"""PRO-G2: exact K-token epoch graphs.

The existing full-token graph is causal: argmax writes the next token id back to
the device buffer that the following replay embeds. This experiment asks whether
K replays can themselves be captured as one parent graph. If supported, one host
launch advances K exact autoregressive tokens and can amortize launch/readback
cost without a drafter, speculation, or changed model semantics.

Unsupported nested capture is an honest technical result, not silently replaced
by a different method.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    REPO,
    environment_snapshot,
    first_divergence,
    load_json,
    percentiles,
    require_gpu_free,
    require_model_dir,
    result_path,
    utc_now,
    write_json_atomic,
)

OUT = result_path("PRO_G2_EPOCH_GRAPH.json")
TS200 = REPO / "reports" / "treesweep200"


def _new_runtime():
    sys.path.insert(0, str(REPO / "src"))
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(
        require_model_dir(), contexts_max=4096, embed_on_host=True,
        fp8_kv=True, verbose=False,
    )
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.device_cache = True
    rt.deterministic_accum = True
    rt.setup_graph()
    return rt


def _prefill(rt, prompt_ids: list[int]) -> int:
    rt.reset()
    start = int(rt._ring_i)
    for token in prompt_ids:
        rt.step_graph(int(token))
    slot = (start + len(prompt_ids) - 1) % int(rt._ring_size)
    return int(rt.ring_harvest(slot, 1)[0])


def _build_parent_epoch(rt, k: int):
    import cupy as cp

    stream = cp.cuda.Stream(non_blocking=True)
    ring = cp.zeros(k, dtype=cp.int32)
    runtime = cp.cuda.runtime
    free_before = int(cp.cuda.Device(0).mem_info[0])
    with stream:
        stream.begin_capture()
        for j in range(k):
            rt._graph.launch(stream)
            runtime.memcpyAsync(
                int(ring.data.ptr) + 4 * j,
                int(rt._tok_dev.data.ptr),
                4,
                runtime.memcpyDeviceToDevice,
                int(stream.ptr),
            )
        parent = stream.end_capture()
    stream.synchronize()
    free_after = int(cp.cuda.Device(0).mem_info[0])
    return parent, stream, ring, int(free_before - free_after)


def _child_batch(rt, k: int, ring, stream) -> list[int]:
    import cupy as cp

    runtime = cp.cuda.runtime
    for j in range(k):
        rt._graph.launch(stream)
        runtime.memcpyAsync(
            int(ring.data.ptr) + 4 * j,
            int(rt._tok_dev.data.ptr),
            4,
            runtime.memcpyDeviceToDevice,
            int(stream.ptr),
        )
    stream.synchronize()
    return [int(x) for x in cp.asnumpy(ring)]


def _parent_batch(parent, stream, ring) -> list[int]:
    import cupy as cp

    parent.launch(stream)
    stream.synchronize()
    return [int(x) for x in cp.asnumpy(ring)]


def _generate_in_batches(rt, prompt_ids: list[int], total: int, k: int, *, parent=None, stream=None, ring=None) -> list[int]:
    first = _prefill(rt, prompt_ids)
    ids = [first]
    remaining = total - 1
    while remaining > 0:
        take = min(k, remaining)
        # Fixed-size epoch graph may overshoot at the very end; only compare the
        # requested prefix. State after the overshoot is discarded with reset.
        if parent is None:
            batch = _child_batch(rt, k, ring, stream)
        else:
            batch = _parent_batch(parent, stream, ring)
        ids.extend(batch[:take])
        remaining -= take
    return ids


def _time_batches(rt, prompt_ids: list[int], k: int, *, parent=None, stream=None, ring=None, rounds: int = 12) -> list[float]:
    samples: list[float] = []
    for _ in range(rounds):
        _prefill(rt, prompt_ids)
        t0 = time.perf_counter_ns()
        if parent is None:
            _child_batch(rt, k, ring, stream)
        else:
            _parent_batch(parent, stream, ring)
        samples.append((time.perf_counter_ns() - t0) / 1e6 / k)
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_g2_epoch_graph",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "claim_boundary": (
            "Exact parent-graph amortization experiment at context <=4096. "
            "Results apply to offline/queued single-stream throughput; per-token "
            "interactive latency still includes the chosen harvest cadence."
        ),
    }
    try:
        require_gpu_free()
        runtime_file = REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py"
        payload["environment"] = environment_snapshot((runtime_file, Path(__file__)))
        anchor = load_json(TS200 / "V36_DETERMINISTIC_ANCHOR.json")
        prompt = anchor["prompts"][0]
        prompt_ids = [int(x) for x in prompt["prompt_ids"]]
        total = 16 if args.mode == "smoke" else 128
        ks = [2, 4] if args.mode == "smoke" else [2, 4, 8, 16, 32]
        rounds = 3 if args.mode == "smoke" else 12

        import cupy as cp

        rt = _new_runtime()
        payload["config"] = {
            "prompt": prompt["prompt"],
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": total,
            "epoch_sizes": ks,
            "timing_rounds": rounds,
        }
        payload["epochs"] = {}

        for k in ks:
            rec: dict[str, Any] = {"k": k, "status": "started"}
            try:
                parent, pstream, pring, extra = _build_parent_epoch(rt, k)
                cstream = cp.cuda.Stream(non_blocking=True)
                cring = cp.zeros(k, dtype=cp.int32)

                child_ids = _generate_in_batches(
                    rt, prompt_ids, total, k, parent=None, stream=cstream, ring=cring
                )
                parent_ids = _generate_in_batches(
                    rt, prompt_ids, total, k, parent=parent, stream=pstream, ring=pring
                )
                child_ms = _time_batches(
                    rt, prompt_ids, k, parent=None, stream=cstream, ring=cring, rounds=rounds
                )
                parent_ms = _time_batches(
                    rt, prompt_ids, k, parent=parent, stream=pstream, ring=pring, rounds=rounds
                )
                cstat, pstat = percentiles(child_ms), percentiles(parent_ms)
                speedup = float(cstat["p50"] / pstat["p50"])
                rec.update({
                    "status": "measured",
                    "child_ids": child_ids,
                    "parent_ids": parent_ids,
                    "identical": child_ids == parent_ids,
                    "first_divergence": first_divergence(child_ids, parent_ids),
                    "queued_child_per_token_ms": cstat,
                    "parent_graph_per_token_ms": pstat,
                    "raw_child_per_token_ms": child_ms,
                    "raw_parent_per_token_ms": parent_ms,
                    "speedup": speedup,
                    "parent_extra_vram_bytes": extra,
                    "gates": {
                        "exact_ids": child_ids == parent_ids,
                        "speedup_ge_1_10": speedup >= 1.10,
                        "extra_vram_lt_64_mib": extra < 64 * 1024 * 1024,
                    },
                })
            except Exception as exc:
                rec.update({
                    "status": "unsupported_or_failed",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                })
            payload["epochs"][str(k)] = rec

        measured = [x for x in payload["epochs"].values() if x["status"] == "measured"]
        passing = [x for x in measured if all(x["gates"].values())]
        payload["best"] = None
        if passing:
            best = max(passing, key=lambda x: x["speedup"])
            payload["best"] = {
                "k": best["k"],
                "speedup": best["speedup"],
                "per_token_ms": best["parent_graph_per_token_ms"]["p50"],
                "tok_s": 1000.0 / best["parent_graph_per_token_ms"]["p50"],
            }
            payload["status"] = "pass"
        elif measured:
            payload["status"] = "gate_failed"
        else:
            payload["status"] = "technical_blocked"
        payload["completed_utc"] = utc_now()

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
    print(json.dumps({"status": payload["status"], "output": str(OUT)}, indent=2))
    return 0 if payload["status"] in {"pass", "gate_failed", "technical_blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
