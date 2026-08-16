"""PRO V14 — B3: overlap the down_proj PCIe gather with VRAM compute.

Preregistered before the first target-hardware run. Base is the verified V6
stack; the only change is *when* the gather runs (see moe_dev_overlap.py).

## Why this arm and not another

Every other lever measured today either failed, was refuted, or is too small,
and the arithmetic that says so uses only measured numbers:

    VRAM floor    2048 MB / 249 GB/s  = 8.22 ms   diag_gemv_width32
    PCIe gather     64 MB / 25.9 GB/s = 2.47 ms   diag_gather_pcie_ceiling
    serial        10.69 ms =  93.6 tok/s  -> 100 tok/s UNREACHABLE
    overlapped     8.22 ms = 122   tok/s  -> 100 tok/s reachable at 82%

Independently, the component marginals (diag_component_marginals_v6, all gates
green, drift 0.0397 ms) put MoE at 11.004 ms of a 23.141 ms token with a 5.19 ms
floor -- and 2.47 ms of that headroom is pure PCIe time that no faster kernel
can remove. Only overlap can.

Refuted today, so not retried here: queue starvation (V12, issue cost is 0.02-
0.05 ms/token), the FP8 decode LUT (25-27% slower without it), 32-lane cacheline
geometry (bit-exact, 0.95-1.07x), and gather concurrency variants (unroll x4 and
2/4/8/16 warps per column, all slower).

## Arms (one variable: whether gather and compute may run concurrently)

  BASE_A   V6 stack, unmodified
  CAND     V6 stack + double-buffered mirror + dedicated gather stream
  BASE_B   V6 stack, unmodified, after CAND

Eager, like V13: the change is a stream/dependency change whose benefit is an
absolute ms/token quantity, and an eager A/B avoids re-capturing a graph between
arms. A graph arm follows only if this one passes -- and note that a captured
graph preserves the fork/join structure, so the overlap survives capture.

## Gates, frozen before running

  G-V14-C1  CAND token ids == BASE_A ids, every prompt, every token. The
            ping-pong hazard (gathering slot s+1 into the buffer slot s-1 is
            still reading) would show up here first, so this gate is the whole
            safety argument and is not negotiable.
  G-V14-C2  BASE_A ids == BASE_B ids.
  G-V14-D1  |BASE_A.p50 - BASE_B.p50| <= 1.0 ms.
  G-V14-P1  CAND.p50 <= baseline midpoint - 0.8 ms.

G-V14-P1 is set at 0.8 ms, well under the 2.47 ms of PCIe time in play, so a
partial overlap still reads as a partial overlap rather than a success.
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
from moe_dev_overlap import install_overlap_moe_dev
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

RESULT_DIR = REPO / "pro_research" / "results" / "v14_overlap"
OUT = RESULT_DIR / "PRO_V14_OVERLAP.json"


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _reset_exact_state(rt) -> None:
    import cupy as cp

    rt.reset()
    for dev in getattr(rt, "_dev_cache", {}).values():
        for name in ("ids", "w", "slots", "need", "state2", "stats2"):
            if name in dev:
                dev[name].fill(0)
        for name, val in (("slot_of", -1), ("expert_of", -1), ("last_used", -1)):
            if name in dev:
                dev[name].fill(val)
    cp.cuda.Device(0).synchronize()


def _run(rt, prompt_ids: list[int], n: int) -> tuple[list[int], list[float]]:
    import cupy as cp

    _reset_exact_state(rt)
    nxt = None
    for tok in prompt_ids:
        nxt = int(rt.step(int(tok)))
    cp.cuda.Device(0).synchronize()
    ids, ms = [nxt], []
    for _ in range(n - 1):
        t0 = time.perf_counter_ns()
        nxt = int(rt.step(nxt))
        ms.append((time.perf_counter_ns() - t0) / 1e6)
        ids.append(nxt)
    return ids, ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_v14_overlap",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "opens_from": "measured floors: VRAM 2048MB/249GB/s = 8.22 ms, PCIe gather 64MB/25.9GB/s = 2.47 ms; serial 10.69 ms = 93.6 tok/s vs overlapped 8.22 ms = 122 tok/s",
        "claim_boundary": "eager (non-graph) single-sequence A/B; absolute ms/token only, not comparable to the 21.0923 ms V6 graph record",
    }

    try:
        require_gpu_free()
        import cupy as cp

        prompts, _expected, n, capacity = _load_prompt_set(args.mode)
        n = min(n, 32) if args.mode == "smoke" else max(n, 256)
        preheat_n = 32 if args.mode == "smoke" else 128
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

        _reset_exact_state(rt)
        nxt = None
        for tok in prompts[0]["prompt_ids"]:
            nxt = int(rt.step(int(tok)))
        for _ in range(preheat_n):
            nxt = int(rt.step(nxt))

        base_a, base_a_ms = {}, []
        for p in prompts:
            ids, ms = _run(rt, p["prompt_ids"], n)
            base_a[p["prompt"]] = ids
            base_a_ms.extend(ms)

        restore_v6()
        restore_cand = install_overlap_moe_dev(rt, down, up)
        cand, cand_ms = {}, []
        for p in prompts:
            ids, ms = _run(rt, p["prompt_ids"], n)
            cand[p["prompt"]] = ids
            cand_ms.extend(ms)
        restore_cand()

        install_batched_moe_dev(rt, down, up)
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
            "G_V14_C1_cand_bitexact_vs_base_a": all(v is None for v in divs_cand.values()),
            "G_V14_C2_base_a_eq_base_b": all(v is None for v in divs_bb.values()),
            "G_V14_D1_drift_le_1ms": drift <= 1.0,
            "G_V14_P1_gain_ge_0_8ms": delta <= -0.8,
        }
        if not gates["G_V14_C1_cand_bitexact_vs_base_a"] or not gates["G_V14_C2_base_a_eq_base_b"]:
            status = "correctness_failed"
        elif not gates["G_V14_D1_drift_le_1ms"]:
            status = "measurement_unstable"
        elif gates["G_V14_P1_gain_ge_0_8ms"]:
            status = "adoption_candidate"
        else:
            status = "gate_failed"

        payload.update({
            "timing_eager": {
                "BASE_A": a, "CAND": c, "BASE_B": b,
                "baseline_midpoint_ms": midpoint,
                "drift_ms": drift,
                "cand_minus_midpoint_ms": delta,
                "BASE_A_tok_s": 1000.0 / float(a["p50"]),
                "CAND_tok_s": 1000.0 / float(c["p50"]),
                "BASE_B_tok_s": 1000.0 / float(b["p50"]),
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
        "timing_eager": payload.get("timing_eager"),
        "gates": payload.get("gates"),
        "first_divergence_cand": payload.get("first_divergence_cand_vs_base_a"),
        "error": (payload.get("error") or {}).get("message"),
    }, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
