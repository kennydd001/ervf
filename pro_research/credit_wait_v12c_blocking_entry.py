"""V12C entrypoint that freezes cudaEventBlockingSync semantics explicitly.

CuPy's Event(block=True) maps the host wait to CUDA's blocking-sync event flag.
Only the token-delivery events use this setting; the V6 runtime's internal CUDA
events are created before this replacement function is invoked and remain
unchanged.
"""
from __future__ import annotations

from collections import deque
import time
from typing import Any

import cupy as cp

import credit_wait_v12c as experiment
from queue_stream_v12 import _prefill, _reset_exact_state
from common import percentiles


def run_event_wait_blocking(rt, prompt_ids: list[int], n: int, window: int) -> dict[str, Any]:
    if window < 1 or window >= int(rt._ring_size):
        raise ValueError(f"invalid window={window} for ring_size={rt._ring_size}")

    _reset_exact_state(rt)
    first = _prefill(rt, prompt_ids)
    ids = [int(first)]
    target = n - 1
    if target <= 0:
        return {"ids": ids, "decode_tokens": 0, "total_ms": 0.0,
                "throughput_tok_s": None, "safety_ok": True}

    # Critical V12C variable: only these delivery events are blocking-sync.
    events = [cp.cuda.Event(block=True, disable_timing=True) for _ in range(window)]
    q: deque[tuple[int, int, int]] = deque()
    issued = delivered = max_outstanding = 0
    enqueue_cpu_ns = event_wait_ns = 0
    raw_wait_ms: list[float] = []
    raw_gaps_ms: list[float] = []
    delivery_ns: list[int] = []
    safety_ok = True

    def enqueue(ei: int) -> None:
        nonlocal issued, max_outstanding, enqueue_cpu_ns, safety_ok
        if len(q) >= window:
            safety_ok = False
            raise RuntimeError("credit overflow")
        slot = int(rt._ring_i)
        s0 = time.perf_counter_ns()
        rt.step_graph(None)
        events[ei].record(rt._graph_stream)
        s1 = time.perf_counter_ns()
        enqueue_cpu_ns += s1 - s0
        q.append((ei, slot, issued))
        issued += 1
        max_outstanding = max(max_outstanding, len(q))
        if len(q) > window or len(q) >= int(rt._ring_size):
            safety_ok = False
            raise RuntimeError("outstanding/ring safety violation")

    t0 = time.perf_counter_ns()
    for ei in range(min(window, target)):
        enqueue(ei)

    prev: int | None = None
    first_delivery: int | None = None
    while delivered < target:
        if not q:
            safety_ok = False
            raise RuntimeError("credit queue drained")
        ei, slot, ordinal = q[0]
        w0 = time.perf_counter_ns()
        events[ei].synchronize()
        w1 = time.perf_counter_ns()
        event_wait_ns += w1 - w0
        raw_wait_ms.append((w1 - w0) / 1e6)
        q.popleft()
        if ordinal != delivered:
            safety_ok = False
            raise RuntimeError(f"delivery order mismatch {ordinal=} {delivered=}")

        now = time.perf_counter_ns()
        ids.append(int(rt._ring_np[slot]))
        delivery_ns.append(now)
        if first_delivery is None:
            first_delivery = now
        if prev is not None:
            raw_gaps_ms.append((now - prev) / 1e6)
        prev = now
        delivered += 1
        if issued < target:
            enqueue(ei)

    t1 = delivery_ns[-1]
    total_ns = t1 - t0
    drop = min(window, len(raw_gaps_ms) // 4)
    steady = raw_gaps_ms[drop:]
    return {
        "ids": ids,
        "window": window,
        "event_blocking_sync": True,
        "decode_tokens": target,
        "total_ms": total_ns / 1e6,
        "throughput_tok_s": 1e9 * target / total_ns if total_ns else None,
        "time_to_first_credit_delivery_ms": None if first_delivery is None else (first_delivery - t0) / 1e6,
        "delivery_gap_ms": percentiles(raw_gaps_ms),
        "steady_delivery_gap_ms": percentiles(steady),
        "steady_drop_count": drop,
        "event_wait_ms": percentiles(raw_wait_ms),
        "raw_delivery_gap_ms": raw_gaps_ms,
        "raw_steady_delivery_gap_ms": steady,
        "raw_event_wait_ms": raw_wait_ms,
        "event_wait_total_ms": event_wait_ns / 1e6,
        "enqueue_cpu_ms": enqueue_cpu_ns / 1e6,
        "enqueue_cpu_us_per_token": enqueue_cpu_ns / 1e3 / target,
        "max_outstanding": max_outstanding,
        "safety_ok": bool(safety_ok and max_outstanding <= window),
    }


experiment._run_event_wait = run_event_wait_blocking

if __name__ == "__main__":
    raise SystemExit(experiment.main())
