"""PRO V14-G — B3 overlap, but captured into a CUDA graph.

V14 eager came out **bit-exact and +3.65 ms/token slower**. `diag_event_op_cost`
then priced the machinery: a bare `Event.record` costs **0.285 us** (free), but
the full cross-stream `wait_event`/`record` round-trip pattern costs **~183 us
per iteration** in eager mode. V14 issues 23 layers x 6 slots = 138 such
fork/join hops per token, which is the entire regression.

So the eager result does not refute the overlap; it prices the *scheduler*, not
the *idea*. Inside a captured CUDA graph the fork/join edges are static graph
topology resolved once at capture time, and no runtime dependency-resolution API
is called during replay. This arm is therefore the decisive test of B3, and B3
is the gating condition for 100 tok/s:

    VRAM floor    2048 MB / 249 GB/s  = 8.22 ms
    PCIe gather     64 MB / 25.9 GB/s = 2.47 ms
    serial        10.69 ms =  93.6 tok/s   -> 100 unreachable
    overlapped     8.22 ms = 122   tok/s   -> 100 reachable at 82%

## Capture legality

Stream capture requires side streams to be forked from the capturing stream via
an event and joined back into it before capture ends. `moe_dev_overlap` already
has exactly that shape: `main.record(...)` -> `gather_stream.wait_event(...)`
forks, and the last `main.wait_event(g_done[top_k-1])` joins. During capture
`cp.cuda.get_current_stream()` is the capturing stream, so `main` binds
correctly. If capture rejects the topology that is itself a reportable result,
not a silent fallback.

## Re-capturing between arms

`setup_graph()` early-returns when `_graph` is not None, and re-allocates the
0.656 GiB pinned embedding table on every call -- calling it three times would
leak ~2 GB of pinned host memory. `_recapture()` therefore redoes only the
capture, reusing every buffer `setup_graph()` already allocated. That keeps the
V12 drift recipe intact (one runtime, no reallocation between arms), which is
what got drift down to 0.04-0.24 ms.

## Arms and gates

  BASE_A   V6 stack, captured
  CAND     V6 stack + B3 overlap, captured
  BASE_B   V6 stack, captured, after CAND

  G-V14G-C1  CAND ids == BASE_A ids, every prompt, every token
  G-V14G-C2  BASE_A ids == BASE_B ids
  G-V14G-D1  |BASE_A.p50 - BASE_B.p50| <= 1.0 ms
  G-V14G-P1  CAND.p50 <= midpoint - 0.8 ms

Graph timing uses SYNC semantics (one replay, one ring harvest per token),
matching how the 21.0923 ms V6 record was measured, so these numbers ARE
comparable to it -- unlike V13/V14's eager arms.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from typing import Any

from common import REPO, environment_snapshot, first_divergence, percentiles, require_gpu_free, utc_now
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _load_prompt_set, _new_runtime
from layer_capacity import apply_nonuniform_capacity
from moe_dev_batched import install_batched_moe_dev
from gather_small_grid import GatherSmallGrid
from moe_dev_overlap import install_overlap_moe_dev
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

RESULT_DIR = REPO / "pro_research" / "results" / "v14_overlap"
OUT = RESULT_DIR / "PRO_V14G_OVERLAP_GRAPH.json"


def _out_path(gb):
    return RESULT_DIR / (f"PRO_V14G_OVERLAP_GRAPH_gb{gb}.json" if gb else "PRO_V14G_OVERLAP_GRAPH.json")


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = _out_path(payload.get("gather_blocks") if isinstance(payload.get("gather_blocks"), int) else 0)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(out)


def _recapture(rt) -> None:
    """Redo only the capture, reusing everything setup_graph() allocated."""
    cp = rt.cp
    s = rt._graph_stream
    rt._graph = None
    rt._step_body_graph()                 # warm/compile outside capture
    cp.cuda.Device(0).synchronize()
    s.begin_capture()
    with s:
        rt._step_body_graph()
    rt._graph = s.end_capture()
    s.synchronize()
    rt.reset()


def _reset_exact_state(rt) -> None:
    import cupy as cp
    import numpy as np

    rt._graph_stream.synchronize()
    rt.reset()
    for dev in getattr(rt, "_dev_cache", {}).values():
        for name in ("ids", "w", "slots", "need", "state2", "stats2"):
            if name in dev:
                dev[name].fill(0)
        for name, val in (("slot_of", -1), ("expert_of", -1), ("last_used", -1)):
            if name in dev:
                dev[name].fill(val)
    rt._ring_i = 0
    rt._ring_np[:] = np.int32(-1)
    cp.cuda.Device(0).synchronize()


def _prefill(rt, prompt_ids: list[int]) -> int:
    start = int(rt._ring_i)
    for tok in prompt_ids:
        rt.step_graph(int(tok))
        rt._graph_stream.synchronize()
    slot = (start + len(prompt_ids) - 1) % int(rt._ring_size)
    return int(rt.ring_harvest(slot, 1)[0])


def _run(rt, prompt_ids: list[int], n: int) -> tuple[list[int], list[float]]:
    _reset_exact_state(rt)
    first = _prefill(rt, prompt_ids)
    ids, ms = [first], []
    for _ in range(n - 1):
        slot = int(rt._ring_i)
        t0 = time.perf_counter_ns()
        rt.step_graph(None)
        tok = int(rt.ring_harvest(slot, 1)[0])
        ms.append((time.perf_counter_ns() - t0) / 1e6)
        ids.append(tok)
    return ids, ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--gather-blocks", type=int, default=0,
                    help="0 = production gather (247 blocks, worst-case sized); "
                         "otherwise use the grid-stride gather at this many blocks, "
                         "leaving SM room for the down_masked it overlaps with")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_v14g_overlap_graph",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "opens_from": "V14 eager was bit-exact but +3.65 ms/token; diag_event_op_cost priced a bare Event.record at 0.285 us but the full cross-stream wait/record round-trip at ~183 us, and V14 issues 138 of those per token. Capture turns them into static graph edges.",
        "gather_blocks": None,
        "claim_boundary": "SYNC semantics (one replay + one ring harvest per token), the same regime the 21.0923 ms V6 record was measured in, so these numbers ARE comparable to it.",
    }

    try:
        require_gpu_free()
        import cupy as cp

        prompts, _expected, n, capacity = _load_prompt_set(args.mode)
        n = min(n, 32) if args.mode == "smoke" else max(n, 256)
        preheat_n = 32 if args.mode == "smoke" else 128
        payload["gather_blocks"] = args.gather_blocks or "production_247"
        payload["config"] = {"tokens_per_prompt": n, "prompt_count": len(prompts),
                             "capacity": capacity, "preheat_tokens": preheat_n}
        payload["environment_start"] = environment_snapshot((
            REPO / "pro_research" / "moe_dev_overlap.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
        ))

        rt = _new_runtime(capacity)
        dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
        rt.enable_cache(capacity)
        apply_nonuniform_capacity(rt)
        rt.device_cache = True
        rt.deterministic_accum = True
        restore_sel, sel_counts = _install_selective(rt, dense)
        restore_v6 = install_batched_moe_dev(rt, down, up)
        payload["selective_capture_counters"] = dict(sel_counts)
        rt.setup_graph()

        _reset_exact_state(rt)
        nxt = _prefill(rt, prompts[0]["prompt_ids"])
        for _ in range(preheat_n):
            rt.step_graph(None)
        rt._graph_stream.synchronize()

        base_a, base_a_ms = {}, []
        for p in prompts:
            ids, ms = _run(rt, p["prompt_ids"], n)
            base_a[p["prompt"]] = ids
            base_a_ms.extend(ms)

        # ---- CAND: overlap installed, then RE-captured -------------------
        restore_v6()
        gsmall = GatherSmallGrid() if args.gather_blocks else None
        restore_cand = install_overlap_moe_dev(rt, down, up, gsmall, args.gather_blocks)
        capture_error = None
        try:
            _recapture(rt)
        except Exception as exc:
            capture_error = f"{type(exc).__name__}: {exc}"
        payload["capture_error"] = capture_error
        if capture_error is not None:
            payload.update({"status": "capture_rejected", "completed_utc": utc_now()})
            _write(payload)
            print(json.dumps({"status": payload["status"],
                              "capture_error": capture_error}, indent=2))
            return 2

        cand, cand_ms = {}, []
        for p in prompts:
            ids, ms = _run(rt, p["prompt_ids"], n)
            cand[p["prompt"]] = ids
            cand_ms.extend(ms)

        restore_cand()
        install_batched_moe_dev(rt, down, up)
        _recapture(rt)
        base_b, base_b_ms = {}, []
        for p in prompts:
            ids, ms = _run(rt, p["prompt_ids"], n)
            base_b[p["prompt"]] = ids
            base_b_ms.extend(ms)

        a, b, c = percentiles(base_a_ms), percentiles(base_b_ms), percentiles(cand_ms)
        drift = abs(float(a["p50"]) - float(b["p50"]))
        midpoint = (float(a["p50"]) + float(b["p50"])) / 2.0
        delta = float(c["p50"]) - midpoint

        divs_cand = {p["prompt"]: first_divergence(base_a[p["prompt"]], cand[p["prompt"]])
                     for p in prompts}
        divs_bb = {p["prompt"]: first_divergence(base_a[p["prompt"]], base_b[p["prompt"]])
                   for p in prompts}

        gates = {
            "G_V14G_C1_cand_bitexact_vs_base_a": all(v is None for v in divs_cand.values()),
            "G_V14G_C2_base_a_eq_base_b": all(v is None for v in divs_bb.values()),
            "G_V14G_D1_drift_le_1ms": drift <= 1.0,
            "G_V14G_P1_gain_ge_0_8ms": delta <= -0.8,
        }
        if not gates["G_V14G_C1_cand_bitexact_vs_base_a"] or not gates["G_V14G_C2_base_a_eq_base_b"]:
            status = "correctness_failed"
        elif not gates["G_V14G_D1_drift_le_1ms"]:
            status = "measurement_unstable"
        elif gates["G_V14G_P1_gain_ge_0_8ms"]:
            status = "adoption_candidate"
        else:
            status = "gate_failed"

        payload.update({
            "timing_graph_sync": {
                "BASE_A": a, "CAND": c, "BASE_B": b,
                "baseline_midpoint_ms": midpoint,
                "drift_ms": drift,
                "cand_minus_midpoint_ms": delta,
                "BASE_A_tok_s": 1000.0 / float(a["p50"]),
                "CAND_tok_s": 1000.0 / float(c["p50"]),
                "BASE_B_tok_s": 1000.0 / float(b["p50"]),
                "v6_record_ms_for_reference": 21.0923,
                "pcie_gather_ms_in_play": 2.47,
                "fraction_of_pcie_time_hidden": (-delta) / 2.47 if delta < 0 else 0.0,
            },
            "first_divergence_cand_vs_base_a": divs_cand,
            "first_divergence_base_b_vs_base_a": divs_bb,
            "gates": gates,
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })

        restore_sel()
        del rt, dense, down, up
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })

    _write(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "timing_graph_sync": payload.get("timing_graph_sync"),
        "gates": payload.get("gates"),
        "capture_error": payload.get("capture_error"),
        "error": (payload.get("error") or {}).get("message"),
    }, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
