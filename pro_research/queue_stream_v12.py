"""PRO V12: decouple autoregressive device execution from per-token host sync.

The full-token CUDA graph is already causal: argmax writes the next token id to
_tok_dev, and the next replay consumes it. The existing V6 benchmark nevertheless
calls ring_harvest() after every decode replay; ring_harvest synchronizes the
graph stream. PV2-20 showed exact K=2/4 queued replays around 19 ms/token, below
the 20 ms E50 threshold, even though a parent child-graph gave no extra speed.

V12 isolates that scheduler effect with the V6 arithmetic unchanged.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, first_divergence, percentiles, require_gpu_free, utc_now
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _load_prompt_set, _new_runtime
from layer_capacity import apply_nonuniform_capacity
from moe_dev_batched import install_batched_moe_dev
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

RESULT_DIR = REPO / "pro_research" / "results" / "v12_async"
OUT = RESULT_DIR / "PRO_V12_ASYNC_HARVEST.json"
PREREG = REPO / "pro_research" / "V12_ASYNC_HARVEST_PREREGISTRATION.md"


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _build_v6(capacity: int):
    rt = _new_runtime(capacity)
    dense = DenseERVF()
    down = DownProjBatchKernels()
    up = UpProjBatchKernels()
    rt.enable_cache(capacity)
    apply_nonuniform_capacity(rt)
    rt.device_cache = True
    rt.deterministic_accum = True
    restore_sel, sel_counts = _install_selective(rt, dense)
    restore_moe = install_batched_moe_dev(rt, down, up)
    rt.setup_graph()
    return rt, dense, down, up, restore_sel, restore_moe, sel_counts


def _reset_exact_state(rt) -> None:
    """Reset model state and device LRU state without invalidating graph pointers."""
    import cupy as cp

    rt._graph_stream.synchronize()
    rt.reset()
    for dev in getattr(rt, "_dev_cache", {}).values():
        for name in ("ids", "w", "slots", "need", "state2", "stats2"):
            if name in dev:
                dev[name].fill(0)
        if "slot_of" in dev:
            dev["slot_of"].fill(-1)
        if "expert_of" in dev:
            dev["expert_of"].fill(-1)
        if "last_used" in dev:
            dev["last_used"].fill(-1)
    rt._ring_i = 0
    rt._ring_np[:] = np.int32(-1)
    cp.cuda.Device(0).synchronize()


def _prefill(rt, prompt_ids: list[int]) -> int:
    start = int(rt._ring_i)
    for tok in prompt_ids:
        rt.step_graph(int(tok))
        # V3 fix: the same pinned 4-byte staging word must not be overwritten
        # until CUDA has consumed it. This is prompt-only and outside decode timing.
        rt._graph_stream.synchronize()
    slot = (start + len(prompt_ids) - 1) % int(rt._ring_size)
    return int(rt.ring_harvest(slot, 1)[0])


def _run_sync(rt, prompt_ids: list[int], n: int) -> tuple[list[int], list[float]]:
    _reset_exact_state(rt)
    first = _prefill(rt, prompt_ids)
    ids = [first]
    ms: list[float] = []
    for _ in range(n - 1):
        slot = int(rt._ring_i)
        t0 = time.perf_counter_ns()
        rt.step_graph(None)
        tok = int(rt.ring_harvest(slot, 1)[0])
        ms.append((time.perf_counter_ns() - t0) / 1e6)
        ids.append(tok)
    return ids, ms


def _run_queued(rt, prompt_ids: list[int], n: int, k: int) -> dict[str, Any]:
    _reset_exact_state(rt)
    first = _prefill(rt, prompt_ids)
    ids = [first]
    epoch_per_token: list[float] = []
    issue_per_token: list[float] = []
    total_tokens = 0
    total_ns = 0
    remaining = n - 1
    while remaining > 0:
        take = min(k, remaining)
        start = int(rt._ring_i)
        t0 = time.perf_counter_ns()
        for _ in range(take):
            rt.step_graph(None)
        t_issue = time.perf_counter_ns()
        batch = rt.ring_harvest(start, take)
        t1 = time.perf_counter_ns()
        ids.extend(int(x) for x in batch)
        elapsed = t1 - t0
        total_ns += elapsed
        total_tokens += take
        epoch_per_token.append(elapsed / 1e6 / take)
        issue_per_token.append((t_issue - t0) / 1e6 / take)
        remaining -= take
    return {
        "ids": ids,
        "epoch_per_token_ms": percentiles(epoch_per_token),
        "issue_per_token_ms": percentiles(issue_per_token),
        "total_decode_tokens": total_tokens,
        "total_ms": total_ns / 1e6,
        "throughput_tok_s": 1e9 * total_tokens / total_ns if total_ns else None,
    }


def _run_event_stream(rt, prompt_ids: list[int], n: int, k: int) -> dict[str, Any]:
    """Queue K replays but expose each token after its own D2H-copy event.

    Events are non-timing events and are only queried from the CPU; no
    event.synchronize() or stream.synchronize() is used in the hot decode loop.
    """
    import cupy as cp

    _reset_exact_state(rt)
    first = _prefill(rt, prompt_ids)
    ids = [first]
    events = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(k)]
    delivery_gaps_ms: list[float] = []
    queue_issue_ms: list[float] = []
    epoch_per_token_ms: list[float] = []
    poll_iterations = 0
    total_tokens = 0
    total_ns = 0
    prev_delivery_ns: int | None = None
    remaining = n - 1

    while remaining > 0:
        take = min(k, remaining)
        slots: list[int] = []
        t0 = time.perf_counter_ns()
        for j in range(take):
            slots.append(int(rt._ring_i))
            rt.step_graph(None)
            events[j].record(rt._graph_stream)
        t_issue = time.perf_counter_ns()
        queue_issue_ms.append((t_issue - t0) / 1e6 / take)

        for j in range(take):
            ev = events[j]
            while not bool(ev.done):
                poll_iterations += 1
            now = time.perf_counter_ns()
            tok = int(rt._ring_np[slots[j]])
            ids.append(tok)
            if prev_delivery_ns is not None:
                delivery_gaps_ms.append((now - prev_delivery_ns) / 1e6)
            prev_delivery_ns = now
        t1 = time.perf_counter_ns()
        elapsed = t1 - t0
        total_ns += elapsed
        total_tokens += take
        epoch_per_token_ms.append(elapsed / 1e6 / take)
        remaining -= take

    return {
        "ids": ids,
        "delivery_gap_ms": percentiles(delivery_gaps_ms),
        "epoch_per_token_ms": percentiles(epoch_per_token_ms),
        "issue_per_token_ms": percentiles(queue_issue_ms),
        "poll_iterations": poll_iterations,
        "total_decode_tokens": total_tokens,
        "total_ms": total_ns / 1e6,
        "throughput_tok_s": 1e9 * total_tokens / total_ns if total_ns else None,
    }


def _preheat(rt, prompt_ids: list[int], tokens: int) -> None:
    """Reach a steadier clock/power state before A/B/A timing."""
    _reset_exact_state(rt)
    _prefill(rt, prompt_ids)
    left = tokens
    while left > 0:
        take = min(32, left)
        start = int(rt._ring_i)
        for _ in range(take):
            rt.step_graph(None)
        rt.ring_harvest(start, take)
        left -= take


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_v12_async_harvest",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": {
            "sync": "blocking per-token host round-trip latency",
            "queued": "single-sequence exact generation throughput with K-token host harvest cadence",
            "event_stream": "single-sequence exact host-delivered streaming throughput; next graph replays are already queued and each token is exposed after its own D2H completion event",
        },
    }

    try:
        require_gpu_free()
        prompts, _expected, n, capacity = _load_prompt_set(args.mode)
        if args.mode == "smoke":
            n = min(n, 32)
            ks = [2, 4, 8]
            preheat_n = 32
        else:
            n = max(n, 256)
            ks = [2, 4, 8, 16, 32]
            preheat_n = 128

        payload["config"] = {
            "tokens_per_prompt": n,
            "prompt_count": len(prompts),
            "capacity": capacity,
            "queue_sizes": ks,
            "preheat_tokens": preheat_n,
        }
        payload["environment_start"] = environment_snapshot((
            REPO / "pro_research" / "queue_stream_v12.py",
            REPO / "pro_research" / "graph_v6_full_stack.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
        ))

        rt, dense, down, up, restore_sel, restore_moe, sel_counts = _build_v6(capacity)
        payload["selective_capture_counters"] = dict(sel_counts)

        _preheat(rt, prompts[0]["prompt_ids"], preheat_n)

        per_prompt: dict[str, Any] = {}
        sync_a_all: list[float] = []
        sync_b_all: list[float] = []

        for p in prompts:
            ids, ms = _run_sync(rt, p["prompt_ids"], n)
            per_prompt[p["prompt"]] = {"kind": p["kind"], "sync_a_ids": ids, "queued": {}, "event_stream": {}}
            sync_a_all.extend(ms)

        for k in ks:
            for p in prompts:
                rec = _run_queued(rt, p["prompt_ids"], n, k)
                per_prompt[p["prompt"]]["queued"][str(k)] = rec

        event_ks = ks if args.mode == "smoke" else [4, 8, 16]
        event_errors: dict[str, str] = {}
        for k in event_ks:
            try:
                for p in prompts:
                    rec = _run_event_stream(rt, p["prompt_ids"], n, k)
                    per_prompt[p["prompt"]]["event_stream"][str(k)] = rec
            except Exception as exc:
                event_errors[str(k)] = f"{type(exc).__name__}: {exc}"

        for p in prompts:
            ids, ms = _run_sync(rt, p["prompt_ids"], n)
            per_prompt[p["prompt"]]["sync_b_ids"] = ids
            sync_b_all.extend(ms)

        sync_a = percentiles(sync_a_all)
        sync_b = percentiles(sync_b_all)
        drift = abs(float(sync_a["p50"]) - float(sync_b["p50"]))

        queued_summary: dict[str, Any] = {}
        for k in ks:
            toks = 0
            ms_total = 0.0
            exact = True
            first_divs: dict[str, int | None] = {}
            for p in prompts:
                name = p["prompt"]
                r = per_prompt[name]["queued"][str(k)]
                toks += int(r["total_decode_tokens"])
                ms_total += float(r["total_ms"])
                div = first_divergence(per_prompt[name]["sync_a_ids"], r["ids"])
                first_divs[name] = div
                exact = exact and div is None
            tps = 1000.0 * toks / ms_total if ms_total else None
            queued_summary[str(k)] = {
                "exact": exact,
                "first_divergence": first_divs,
                "tokens": toks,
                "total_ms": ms_total,
                "tok_s": tps,
                "E50": bool(exact and tps is not None and tps >= 50.0),
            }

        event_summary: dict[str, Any] = {}
        for k in event_ks:
            if str(k) in event_errors:
                event_summary[str(k)] = {"status": "technical_subarm_failure", "error": event_errors[str(k)]}
                continue
            toks = 0
            ms_total = 0.0
            gaps: list[float] = []
            exact = True
            first_divs: dict[str, int | None] = {}
            for p in prompts:
                name = p["prompt"]
                r = per_prompt[name]["event_stream"][str(k)]
                toks += int(r["total_decode_tokens"])
                ms_total += float(r["total_ms"])
                div = first_divergence(per_prompt[name]["sync_a_ids"], r["ids"])
                first_divs[name] = div
                exact = exact and div is None
                if r["delivery_gap_ms"]["p50"] is not None:
                    gaps.append(float(r["delivery_gap_ms"]["p50"]))
            tps = 1000.0 * toks / ms_total if ms_total else None
            gap_p50_across_prompts = float(np.median(np.asarray(gaps, dtype=np.float64))) if gaps else None
            event_summary[str(k)] = {
                "status": "measured",
                "exact": exact,
                "first_divergence": first_divs,
                "tokens": toks,
                "total_ms": ms_total,
                "tok_s": tps,
                "median_of_prompt_p50_delivery_gap_ms": gap_p50_across_prompts,
                "E50_streamed": bool(exact and tps is not None and tps >= 50.0 and gap_p50_across_prompts is not None and gap_p50_across_prompts <= 20.0),
            }

        sync_exact = all(per_prompt[p["prompt"]]["sync_a_ids"] == per_prompt[p["prompt"]]["sync_b_ids"] for p in prompts)
        best_queue = max((v for v in queued_summary.values() if v.get("exact") and v.get("tok_s") is not None), key=lambda x: x["tok_s"], default=None)
        best_event = max((v for v in event_summary.values() if v.get("status") == "measured" and v.get("exact") and v.get("tok_s") is not None), key=lambda x: x["tok_s"], default=None)

        gates = {
            "sync_a_b_token_parity": sync_exact,
            "baseline_drift_le_1ms": drift <= 1.0,
            "all_queued_exact": all(v["exact"] for v in queued_summary.values()),
            "queued_E50_any": any(v["E50"] for v in queued_summary.values()),
            "event_stream_E50_any": any(v.get("E50_streamed", False) for v in event_summary.values()),
            "full_tokens_ge_500": (sum(v["tokens"] for v in queued_summary.values()) >= 500) if args.mode == "full" else None,
        }

        if not gates["sync_a_b_token_parity"] or not gates["all_queued_exact"]:
            status = "correctness_failed"
        elif args.mode == "full" and not gates["baseline_drift_le_1ms"]:
            status = "measurement_unstable"
        elif gates["event_stream_E50_any"]:
            status = "streamed_E50_candidate"
        elif gates["queued_E50_any"]:
            status = "queued_E50_candidate"
        else:
            status = "gate_failed"

        payload.update({
            "sync": {
                "A": sync_a,
                "B": sync_b,
                "p50_drift_ms": drift,
                "A_tok_s": 1000.0 / float(sync_a["p50"]),
                "B_tok_s": 1000.0 / float(sync_b["p50"]),
            },
            "queued_summary": queued_summary,
            "event_stream_summary": event_summary,
            "per_prompt": per_prompt,
            "gates": gates,
            "best_queued": best_queue,
            "best_event_stream": best_event,
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })

        restore_sel()
        restore_moe()
        del rt, dense, down, up
        gc.collect()
        import cupy as cp
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
        "sync": payload.get("sync"),
        "best_queued": payload.get("best_queued"),
        "best_event_stream": payload.get("best_event_stream"),
        "gates": payload.get("gates"),
    }, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
