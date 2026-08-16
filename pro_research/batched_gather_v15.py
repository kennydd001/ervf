"""PRO V15 — batched gather + batched down_masked, in the captured graph.

`diag_moe_subkernel_marginals` (all gates green, drift 0.179 ms) put the two
worst MoE sub-kernels at:

    gather        3.479 ms/token   18.4 GB/s against a 25.9 GB/s PCIe ceiling
    down_masked   1.655 ms/token   against a 0.257 ms floor -- 15% efficiency,
                                   1.40 ms of headroom, the worst path in the model

Both are launched SIX times per layer with small grids -- down_masked runs
(hidden/128, nchunks) = (21, 8) = 168 blocks of 128 threads, i.e. ~6.5 blocks
per SM for ~12 us, 138 times per token. Batching the slot dimension turns that
into one launch of (21, 8, 6) = 1008 blocks.

The batched kernels already exist (down_gather_batch_kernels.py) and were
verified bit-exact against the reference on real captured activations. They
were never adopted because they need top_k independent mirrors and the first
implementation allocated them PER LAYER: 23 x 6 x 2.806 MB = 387 MB, which the
VRAM gate rightly rejected. But the mirror is transient scratch consumed inside
the layer that fills it, exactly like the runtime's own mstate["mirror"], so one
global copy suffices: 16.8 MB. That is fixed in moe_dev_batched.py -- the same
bug class the VRAM gate already caught once during the V6 build (a redundant
per-layer mirror costing 61.6 MB).

So this arm is not new arithmetic; it is an already-verified kernel path that a
sizing bug kept out of the stack.

## Arms and gates

  BASE_A   V6 stack, captured (gather_kernels=None)
  CAND     V6 stack + batched gather/down_masked, captured
  BASE_B   V6 stack, captured, after CAND

  G-V15-C1  CAND ids == BASE_A ids, every prompt, every token
  G-V15-C2  BASE_A ids == BASE_B ids
  G-V15-D1  |BASE_A.p50 - BASE_B.p50| <= 1.0 ms
  G-V15-P1  CAND.p50 <= midpoint - 0.5 ms

SYNC semantics (one replay + one ring harvest per token), the regime the
21.0923 ms V6 record was measured in, so these numbers are comparable to it.
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
from down_gather_batch_kernels import DownGatherBatchKernels
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

RESULT_DIR = REPO / "pro_research" / "results" / "v15_batched_gather"
OUT = RESULT_DIR / "PRO_V15_BATCHED_GATHER_GRAPH.json"


def _out_path(gb):
    return RESULT_DIR / "PRO_V15_BATCHED_GATHER_GRAPH.json"


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
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_v15_batched_gather_graph",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "opens_from": "diag_moe_subkernel_marginals put down_masked at 1.655 ms against a 0.257 ms floor (15% efficiency, 1.40 ms headroom) and the gather at 3.479 ms. Both are launched six times per layer with small grids (down_masked: 21x8 = 168 blocks). The batched versions exist and were verified bit-exact in isolation, but were rejected on VRAM because their mirror was allocated PER LAYER (23 x 6 x 2.806 MB = 387 MB). It is transient scratch; one global copy is 16.8 MB.",
        "claim_boundary": "SYNC semantics (one replay + one ring harvest per token), the same regime the 21.0923 ms V6 record was measured in, so these numbers ARE comparable to it.",
    }

    try:
        require_gpu_free()
        import cupy as cp

        prompts, _expected, n, capacity = _load_prompt_set(args.mode)
        n = min(n, 32) if args.mode == "smoke" else max(n, 256)
        preheat_n = 32 if args.mode == "smoke" else 128
        payload["config"] = {"tokens_per_prompt": n, "prompt_count": len(prompts),
                             "capacity": capacity, "preheat_tokens": preheat_n}
        payload["vram_note"] = "batched mirror is now ONE global buffer (16.8 MB), not one per layer (387 MB)"
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
        gk = DownGatherBatchKernels()
        restore_cand = install_batched_moe_dev(rt, down, up, gather_kernels=gk)
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
            "G_V15_C1_cand_bitexact_vs_base_a": all(v is None for v in divs_cand.values()),
            "G_V15_C2_base_a_eq_base_b": all(v is None for v in divs_bb.values()),
            "G_V15_D1_drift_le_1ms": drift <= 1.0,
            "G_V15_P1_gain_ge_0_5ms": delta <= -0.5,
        }
        if not gates["G_V15_C1_cand_bitexact_vs_base_a"] or not gates["G_V15_C2_base_a_eq_base_b"]:
            status = "correctness_failed"
        elif not gates["G_V15_D1_drift_le_1ms"]:
            status = "measurement_unstable"
        elif gates["G_V15_P1_gain_ge_0_5ms"]:
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
