"""PRO V12C: rolling exact streaming using cudaEventSynchronize on oldest token.

The event is recorded after step_graph() queues the graph replay and its D2H
ring copy. Waiting on that event exposes one token without synchronizing later
work already queued on the stream.
"""
from __future__ import annotations

import argparse
from collections import deque
import gc
import json
import time
import traceback
from typing import Any

import cupy as cp
import numpy as np

from common import REPO, environment_snapshot, first_divergence, percentiles, require_gpu_free, utc_now
from graph_e1f22 import _load_prompt_set
from queue_stream_v12 import _build_v6, _prefill, _preheat, _reset_exact_state, _run_sync

RESULT_DIR = REPO / "pro_research" / "results" / "v12_async"
OUT = RESULT_DIR / "PRO_V12C_EVENT_WAIT.json"
PREREG = REPO / "pro_research" / "V12C_EVENT_WAIT_PREREGISTRATION.md"


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _run_event_wait(rt, prompt_ids: list[int], n: int, window: int) -> dict[str, Any]:
    if window < 1 or window >= int(rt._ring_size):
        raise ValueError(f"invalid window={window} for ring_size={rt._ring_size}")

    _reset_exact_state(rt)
    first = _prefill(rt, prompt_ids)
    ids = [int(first)]
    target = n - 1
    if target <= 0:
        return {"ids": ids, "decode_tokens": 0, "total_ms": 0.0,
                "throughput_tok_s": None, "safety_ok": True}

    events = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(window)]
    q: deque[tuple[int, int, int]] = deque()  # event_idx, ring_slot, ordinal
    issued = 0
    delivered = 0
    max_outstanding = 0
    enqueue_cpu_ns = 0
    event_wait_ns = 0
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
        events[ei].synchronize()  # waits only through this token's D2H event
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload: dict[str, Any] = {
        "kind": "pro_v12c_event_wait",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
    }

    try:
        require_gpu_free()
        prompts, _expected, n, capacity = _load_prompt_set(args.mode)
        if args.mode == "smoke":
            n, windows, preheat_n = 32, [1, 2, 4, 8], 48
        else:
            n, windows, preheat_n = max(n, 256), [1, 2, 4, 8, 16, 32], 160
        payload["config"] = {"tokens_per_prompt": n, "prompt_count": len(prompts),
                             "capacity": capacity, "windows": windows,
                             "preheat_tokens": preheat_n,
                             "steady_gap_drop_rule": "min(window, floor(number_of_gaps/4))"}
        payload["environment_start"] = environment_snapshot((
            REPO / "pro_research" / "credit_wait_v12c.py",
            REPO / "pro_research" / "queue_stream_v12.py",
            REPO / "pro_research" / "graph_v6_full_stack.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
        ))

        rt, dense, down, up, restore_sel, restore_moe, sel_counts = _build_v6(capacity)
        payload["selective_capture_counters"] = dict(sel_counts)
        _preheat(rt, prompts[0]["prompt_ids"], preheat_n)

        per_prompt: dict[str, Any] = {}
        sync_a_ms: list[float] = []
        sync_b_ms: list[float] = []
        for p in prompts:
            ids, ms = _run_sync(rt, p["prompt_ids"], n)
            per_prompt[p["prompt"]] = {"kind": p["kind"], "sync_a_ids": ids,
                                        "event_wait": {}}
            sync_a_ms.extend(ms)

        for w in windows:
            for p in prompts:
                per_prompt[p["prompt"]]["event_wait"][str(w)] = _run_event_wait(
                    rt, p["prompt_ids"], n, w)

        for p in prompts:
            ids, ms = _run_sync(rt, p["prompt_ids"], n)
            per_prompt[p["prompt"]]["sync_b_ids"] = ids
            sync_b_ms.extend(ms)

        sa, sb = percentiles(sync_a_ms), percentiles(sync_b_ms)
        drift = abs(float(sa["p50"]) - float(sb["p50"]))
        sync_mid = (float(sa["p50"]) + float(sb["p50"])) / 2.0
        sync_exact = all(per_prompt[p["prompt"]]["sync_a_ids"] ==
                         per_prompt[p["prompt"]]["sync_b_ids"] for p in prompts)

        summary: dict[str, Any] = {}
        for w in windows:
            exact = True
            safe = True
            tokens = 0
            total_ms = 0.0
            divs: dict[str, int | None] = {}
            p50s: dict[str, float | None] = {}
            p95s: dict[str, float | None] = {}
            first_deliveries: dict[str, float | None] = {}
            wait_p50s: dict[str, float | None] = {}
            max_out = 0
            enqueue_ms = 0.0
            for p in prompts:
                name = p["prompt"]
                rec = per_prompt[name]["event_wait"][str(w)]
                div = first_divergence(per_prompt[name]["sync_a_ids"], rec["ids"])
                divs[name] = div
                exact = exact and div is None
                safe = safe and bool(rec["safety_ok"])
                tokens += int(rec["decode_tokens"])
                total_ms += float(rec["total_ms"])
                p50s[name] = rec["steady_delivery_gap_ms"]["p50"]
                p95s[name] = rec["steady_delivery_gap_ms"]["p95"]
                first_deliveries[name] = rec["time_to_first_credit_delivery_ms"]
                wait_p50s[name] = rec["event_wait_ms"]["p50"]
                max_out = max(max_out, int(rec["max_outstanding"]))
                enqueue_ms += float(rec["enqueue_cpu_ms"])
            tok_s = 1000.0 * tokens / total_ms if total_ms else None
            finite = [float(x) for x in p50s.values() if x is not None]
            max_p50 = max(finite) if finite else None
            signal = bool(exact and safe and tok_s is not None and tok_s >= 50.0
                          and max_p50 is not None and max_p50 <= 20.0)
            enough = tokens >= 500
            verified = bool(signal and drift <= 1.0 and enough) if args.mode == "full" else None
            summary[str(w)] = {
                "exact": exact, "safety_ok": safe, "first_divergence": divs,
                "decode_tokens": tokens, "total_ms": total_ms, "tok_s": tok_s,
                "prompt_steady_p50_gap_ms": p50s,
                "prompt_steady_p95_gap_ms": p95s,
                "max_prompt_steady_p50_gap_ms": max_p50,
                "time_to_first_credit_delivery_ms": first_deliveries,
                "prompt_event_wait_p50_ms": wait_p50s,
                "max_outstanding": max_out, "enqueue_cpu_ms": enqueue_ms,
                "E50_event_wait_signal": signal,
                "full_tokens_ge_500": enough if args.mode == "full" else None,
                "E50_event_wait_verified": verified,
            }

        all_exact = all(x["exact"] for x in summary.values())
        all_safe = all(x["safety_ok"] for x in summary.values())
        any_signal = any(x["E50_event_wait_signal"] for x in summary.values())
        any_verified = any(bool(x.get("E50_event_wait_verified")) for x in summary.values())
        best = max((dict(v, window=int(k)) for k, v in summary.items()
                    if v["exact"] and v["safety_ok"] and v["tok_s"] is not None),
                   key=lambda x: x["tok_s"], default=None)
        gates = {
            "sync_a_b_token_parity": sync_exact,
            "all_event_wait_exact": all_exact,
            "all_ring_event_safety": all_safe,
            "baseline_drift_le_1ms": drift <= 1.0,
            "E50_sync": bool(sync_mid <= 20.0 and sync_exact and drift <= 1.0) if args.mode == "full" else None,
            "E50_event_wait_signal_any": any_signal,
            "E50_event_wait_verified_any": any_verified if args.mode == "full" else None,
        }
        if not sync_exact or not all_exact or not all_safe:
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
            "sync": {"A": sa, "B": sb, "p50_midpoint_ms": sync_mid,
                     "p50_drift_ms": drift, "midpoint_tok_s": 1000.0 / sync_mid},
            "event_wait_summary": summary,
            "per_prompt": per_prompt,
            "best_event_wait": best,
            "gates": gates,
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })
        restore_sel(); restore_moe()
        del rt, dense, down, up
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({"status": "technical_failure", "completed_utc": utc_now(),
                        "error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()}})

    _write(payload)
    print(json.dumps({"status": payload.get("status"), "output": str(OUT),
                      "sync": payload.get("sync"),
                      "best_event_wait": payload.get("best_event_wait"),
                      "gates": payload.get("gates")}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
