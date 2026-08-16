"""PRO V12B: rolling credit-window streaming over the unchanged V6 graph.

Unlike V12's fixed-K event arm, this runner never intentionally drains the GPU
queue between epochs. Once the oldest token's D2H-copy event completes, the host
consumes that ring slot and immediately appends one new graph replay, keeping a
fixed number of causal token replays outstanding.
"""
from __future__ import annotations

import argparse
from collections import deque
import gc
import json
import time
import traceback
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, first_divergence, percentiles, require_gpu_free, utc_now
from graph_e1f22 import _load_prompt_set
from queue_stream_v12 import _build_v6, _prefill, _preheat, _reset_exact_state, _run_sync

RESULT_DIR = REPO / "pro_research" / "results" / "v12_async"
OUT = RESULT_DIR / "PRO_V12B_CREDIT_STREAM.json"
PREREG = REPO / "pro_research" / "V12B_CREDIT_STREAM_PREREGISTRATION.md"


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _run_credit(rt, prompt_ids: list[int], n: int, window: int) -> dict[str, Any]:
    """Maintain up to ``window`` exact token graph replays in flight.

    Each event is recorded on the graph stream *after* step_graph() has queued
    both the token graph and that token's D2H ring copy. The event object is not
    reused until its prior record is complete and the corresponding token has
    been consumed by the CPU.
    """
    import cupy as cp

    if window < 1:
        raise ValueError("window must be >=1")
    if window >= int(rt._ring_size):
        raise ValueError(f"window={window} must be < ring_size={rt._ring_size}")

    _reset_exact_state(rt)
    first = _prefill(rt, prompt_ids)
    ids = [int(first)]
    target = n - 1
    if target <= 0:
        return {
            "ids": ids,
            "window": window,
            "decode_tokens": 0,
            "total_ms": 0.0,
            "throughput_tok_s": None,
            "time_to_first_credit_delivery_ms": None,
            "delivery_gap_ms": percentiles([]),
            "steady_delivery_gap_ms": percentiles([]),
            "raw_delivery_gap_ms": [],
            "raw_steady_delivery_gap_ms": [],
            "poll_iterations": 0,
            "enqueue_cpu_ms": 0.0,
            "enqueue_cpu_us_per_token": None,
            "max_outstanding": 0,
            "safety_ok": True,
        }

    events = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(window)]
    outstanding: deque[tuple[int, int, int]] = deque()  # (event index, ring slot, ordinal)
    issued = 0
    delivered = 0
    poll_iterations = 0
    enqueue_cpu_ns = 0
    max_outstanding = 0
    raw_gaps_ms: list[float] = []
    delivery_times_ns: list[int] = []
    safety_ok = True

    def enqueue(event_index: int) -> None:
        nonlocal issued, enqueue_cpu_ns, max_outstanding, safety_ok
        if len(outstanding) >= window:
            safety_ok = False
            raise RuntimeError("credit-window overflow before enqueue")
        slot = int(rt._ring_i)
        t_issue0 = time.perf_counter_ns()
        rt.step_graph(None)
        events[event_index].record(rt._graph_stream)
        t_issue1 = time.perf_counter_ns()
        enqueue_cpu_ns += t_issue1 - t_issue0
        outstanding.append((event_index, slot, issued))
        issued += 1
        max_outstanding = max(max_outstanding, len(outstanding))
        if len(outstanding) > window or len(outstanding) >= int(rt._ring_size):
            safety_ok = False
            raise RuntimeError("ring/event outstanding invariant violated")

    t0 = time.perf_counter_ns()
    initial = min(window, target)
    for event_index in range(initial):
        enqueue(event_index)

    first_delivery_ns: int | None = None
    previous_delivery_ns: int | None = None

    while delivered < target:
        if not outstanding:
            safety_ok = False
            raise RuntimeError("GPU credit queue drained unexpectedly")

        event_index, slot, ordinal = outstanding[0]
        ev = events[event_index]
        while not bool(ev.done):
            poll_iterations += 1

        now = time.perf_counter_ns()
        outstanding.popleft()
        if ordinal != delivered:
            safety_ok = False
            raise RuntimeError(f"delivery order mismatch: ordinal={ordinal}, delivered={delivered}")

        tok = int(rt._ring_np[slot])
        ids.append(tok)
        delivery_times_ns.append(now)
        if first_delivery_ns is None:
            first_delivery_ns = now
        if previous_delivery_ns is not None:
            raw_gaps_ms.append((now - previous_delivery_ns) / 1e6)
        previous_delivery_ns = now
        delivered += 1

        # Reuse this event only after its preceding record is known complete and
        # its ring slot has been consumed. Appending immediately keeps the stream
        # fed instead of waiting for an entire fixed-K epoch to drain.
        if issued < target:
            enqueue(event_index)

    t1 = delivery_times_ns[-1]
    total_ns = t1 - t0
    drop = min(window, len(raw_gaps_ms) // 4)
    steady_gaps = raw_gaps_ms[drop:]

    return {
        "ids": ids,
        "window": window,
        "decode_tokens": target,
        "total_ms": total_ns / 1e6,
        "throughput_tok_s": 1e9 * target / total_ns if total_ns else None,
        "time_to_first_credit_delivery_ms": None if first_delivery_ns is None else (first_delivery_ns - t0) / 1e6,
        "delivery_gap_ms": percentiles(raw_gaps_ms),
        "steady_delivery_gap_ms": percentiles(steady_gaps),
        "steady_drop_count": drop,
        "raw_delivery_gap_ms": raw_gaps_ms,
        "raw_steady_delivery_gap_ms": steady_gaps,
        "poll_iterations": poll_iterations,
        "enqueue_cpu_ms": enqueue_cpu_ns / 1e6,
        "enqueue_cpu_us_per_token": enqueue_cpu_ns / 1e3 / target,
        "max_outstanding": max_outstanding,
        "safety_ok": bool(safety_ok and max_outstanding <= window),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_v12b_credit_stream",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": (
            "Exact single-sequence greedy streaming. Credit-window arms may keep later "
            "causal graph replays queued while each completed token is individually "
            "delivered to the host; this is not arbitrary host-in-the-loop per-token latency."
        ),
    }

    try:
        require_gpu_free()
        prompts, _expected, n, capacity = _load_prompt_set(args.mode)
        if args.mode == "smoke":
            n = min(max(n, 32), 32)
            windows = [1, 2, 4, 8]
            preheat_n = 48
        else:
            n = max(n, 256)
            windows = [1, 2, 4, 8, 16, 32]
            preheat_n = 160

        payload["config"] = {
            "tokens_per_prompt": n,
            "prompt_count": len(prompts),
            "capacity": capacity,
            "windows": windows,
            "preheat_tokens": preheat_n,
            "steady_gap_drop_rule": "min(window, floor(number_of_gaps/4))",
        }
        payload["environment_start"] = environment_snapshot((
            REPO / "pro_research" / "credit_stream_v12b.py",
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
            per_prompt[p["prompt"]] = {
                "kind": p["kind"],
                "sync_a_ids": ids,
                "credit": {},
            }
            sync_a_all.extend(ms)

        for window in windows:
            for p in prompts:
                per_prompt[p["prompt"]]["credit"][str(window)] = _run_credit(
                    rt, p["prompt_ids"], n, window
                )

        for p in prompts:
            ids, ms = _run_sync(rt, p["prompt_ids"], n)
            per_prompt[p["prompt"]]["sync_b_ids"] = ids
            sync_b_all.extend(ms)

        sync_a = percentiles(sync_a_all)
        sync_b = percentiles(sync_b_all)
        drift = abs(float(sync_a["p50"]) - float(sync_b["p50"]))
        sync_mid = (float(sync_a["p50"]) + float(sync_b["p50"])) / 2.0
        sync_exact = all(
            per_prompt[p["prompt"]]["sync_a_ids"] == per_prompt[p["prompt"]]["sync_b_ids"]
            for p in prompts
        )

        summary: dict[str, Any] = {}
        for window in windows:
            total_tokens = 0
            total_ms = 0.0
            exact = True
            safe = True
            first_divs: dict[str, int | None] = {}
            prompt_steady_p50: dict[str, float | None] = {}
            prompt_steady_p95: dict[str, float | None] = {}
            first_delivery_ms: dict[str, float | None] = {}
            max_outstanding = 0
            polls = 0
            enqueue_ms = 0.0
            for p in prompts:
                name = p["prompt"]
                rec = per_prompt[name]["credit"][str(window)]
                div = first_divergence(per_prompt[name]["sync_a_ids"], rec["ids"])
                first_divs[name] = div
                exact = exact and div is None
                safe = safe and bool(rec["safety_ok"])
                total_tokens += int(rec["decode_tokens"])
                total_ms += float(rec["total_ms"])
                prompt_steady_p50[name] = rec["steady_delivery_gap_ms"]["p50"]
                prompt_steady_p95[name] = rec["steady_delivery_gap_ms"]["p95"]
                first_delivery_ms[name] = rec["time_to_first_credit_delivery_ms"]
                max_outstanding = max(max_outstanding, int(rec["max_outstanding"]))
                polls += int(rec["poll_iterations"])
                enqueue_ms += float(rec["enqueue_cpu_ms"])

            tok_s = 1000.0 * total_tokens / total_ms if total_ms else None
            finite_p50 = [float(v) for v in prompt_steady_p50.values() if v is not None]
            max_prompt_p50 = max(finite_p50) if finite_p50 else None
            signal = bool(
                exact and safe and tok_s is not None and tok_s >= 50.0
                and max_prompt_p50 is not None and max_prompt_p50 <= 20.0
            )
            enough = total_tokens >= 500
            full_verified = bool(signal and drift <= 1.0 and enough) if args.mode == "full" else None
            summary[str(window)] = {
                "exact": exact,
                "safety_ok": safe,
                "first_divergence": first_divs,
                "decode_tokens": total_tokens,
                "total_ms": total_ms,
                "tok_s": tok_s,
                "prompt_steady_p50_gap_ms": prompt_steady_p50,
                "prompt_steady_p95_gap_ms": prompt_steady_p95,
                "max_prompt_steady_p50_gap_ms": max_prompt_p50,
                "time_to_first_credit_delivery_ms": first_delivery_ms,
                "max_outstanding": max_outstanding,
                "poll_iterations": polls,
                "enqueue_cpu_ms": enqueue_ms,
                "E50_streamed_signal": signal,
                "full_tokens_ge_500": enough if args.mode == "full" else None,
                "E50_streamed_credit_verified": full_verified,
            }

        all_credit_exact = all(v["exact"] for v in summary.values())
        all_safety = all(v["safety_ok"] for v in summary.values())
        any_signal = any(v["E50_streamed_signal"] for v in summary.values())
        any_verified = any(bool(v.get("E50_streamed_credit_verified")) for v in summary.values())
        sync_e50 = bool(sync_mid <= 20.0 and sync_exact and drift <= 1.0)

        best = max(
            (dict(v, window=int(k)) for k, v in summary.items() if v["exact"] and v["safety_ok"] and v["tok_s"] is not None),
            key=lambda x: x["tok_s"],
            default=None,
        )

        gates = {
            "sync_a_b_token_parity": sync_exact,
            "all_credit_exact": all_credit_exact,
            "all_ring_event_safety": all_safety,
            "baseline_drift_le_1ms": drift <= 1.0,
            "E50_sync": sync_e50 if args.mode == "full" else None,
            "E50_streamed_signal_any": any_signal,
            "E50_streamed_credit_verified_any": any_verified if args.mode == "full" else None,
        }

        if not sync_exact or not all_credit_exact or not all_safety:
            status = "correctness_failed"
        elif args.mode == "full" and drift > 1.0:
            status = "measurement_unstable"
        elif args.mode == "full" and any_verified:
            status = "streamed_E50_verified"
        elif args.mode == "smoke":
            status = "smoke_pass_E50_signal" if any_signal else "smoke_pass"
        else:
            status = "gate_failed"

        payload.update({
            "sync": {
                "A": sync_a,
                "B": sync_b,
                "p50_midpoint_ms": sync_mid,
                "p50_drift_ms": drift,
                "midpoint_tok_s": 1000.0 / sync_mid,
            },
            "credit_summary": summary,
            "per_prompt": per_prompt,
            "best_credit": best,
            "gates": gates,
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })

        restore_sel()
        restore_moe()
        del rt, dense, down, up
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })

    _write(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "output": str(OUT),
        "sync": payload.get("sync"),
        "best_credit": payload.get("best_credit"),
        "gates": payload.get("gates"),
    }, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
