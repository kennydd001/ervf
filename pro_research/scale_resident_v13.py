"""PRO V13 — H-SCALE: down_proj block-scale planes resident in VRAM.

Preregistered here before the first target-hardware run; base is the verified
V6 stack (device routing + graph-safe residency + selective dense ERVF +
batched panel_scan/reduce_partials/weighted_accumulate + batched up-proj ERVF +
per-layer capacity).

## What opens V13

`diag_gather_pcie_ceiling.json`, measured today on this machine:

  * the gather already runs at **64.0%** of a byte-verified **25.908 GB/s**
    ceiling (PCIe Gen5 x8) -- N2's old "4.3 GB/s, 6x worse in the loop" no
    longer holds, so there is no large efficiency gap left to close;
  * every concurrency variant (unroll x4; 2/4/8/16 warps per column) is
    **slower**, so the remaining 36% is the scattered access pattern itself,
    not a latency-hiding failure;
  * **52.2%** of all down-path PCIe bytes are FP8 block scales, not weights
    (90.1 active panels x 2688 B against 164.7 columns x 1344 B), because a
    panel's scale bytes are indexed by output row and are all needed whenever
    the panel is active, while only ~1.8 of its 16 columns are nonzero;
  * hypothesis arm v3, which skips exactly those scale planes and was verified
    to copy a strict subset of the production byte set, costs
    **-1.380 ms/token**.

So the byte count is the only remaining lever and half of it is metadata that
does not have to be re-fetched at all.

## No arithmetic change

Same expert, same panel, same row, same scale byte, same
`e4m3_lut[byte] * global_scale`, same fmaf order, same routing, same masks.
Only the address the byte is read from changes. This is the same class of
change as `enable_cache(mode="up_only")` and the panel-major repack.

## Arms (one variable: where the scale byte lives)

  BASE_A   V6 stack, unmodified
  CAND     V6 stack + H-SCALE
  BASE_B   V6 stack, unmodified, after CAND

Eager (no graph) in this first runner: the saving is a PCIe byte-count effect
and is an absolute ms/token quantity, so it does not need graph residency to be
visible, and an eager A/B avoids re-capturing a graph between arms (which
`setup_graph`'s early return and its 0.656 GiB pinned embedding re-allocation
make a separate piece of work). A graph arm follows only if this one passes.

## Gates, frozen before running

  G-V13-C1  CAND token ids == BASE_A token ids, every prompt, every token.
            A single mismatch closes the arm regardless of speed.
  G-V13-C2  BASE_A ids == BASE_B ids (the swap back is clean).
  G-V13-V1  planned plane VRAM <= measured free VRAM, checked BEFORE
            allocating, and the post-allocation free VRAM stays > 64 MiB.
  G-V13-D1  |BASE_A.p50 - BASE_B.p50| <= 1.0 ms, the drift gate the V12 harness
            recipe reached at 0.108 ms. Above it, no performance conclusion.
  G-V13-P1  CAND.p50 <= baseline midpoint - 0.5 ms. Adoption candidate only.

The measured -1.380 ms is the gather-side gross figure; the net expectation
after the extra per-miss plane traffic (20.24 misses/token x 311,808 B at
25.9 GB/s = +0.244 ms) is ~-1.14 ms/token. G-V13-P1 is deliberately set below
that so a partial result still reads as a partial result.
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
from moe_dev_scale_resident import install_scale_resident_moe_dev, planned_plane_bytes
from scale_resident_kernels import PLANE_BYTES, ScaleResidentKernels
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

RESULT_DIR = REPO / "pro_research" / "results" / "v13_scale_resident"
OUT = RESULT_DIR / "PRO_V13_SCALE_RESIDENT.json"


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _reset_exact_state(rt) -> None:
    """Model state and device LRU cleared without touching allocations."""
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
    ids = [nxt]
    ms: list[float] = []
    for _ in range(n - 1):
        t0 = time.perf_counter_ns()
        nxt = int(rt.step(nxt))
        ms.append((time.perf_counter_ns() - t0) / 1e6)
        ids.append(nxt)
    return ids, ms


def _preheat(rt, prompt_ids: list[int], tokens: int) -> None:
    _reset_exact_state(rt)
    nxt = None
    for tok in prompt_ids:
        nxt = int(rt.step(int(tok)))
    for _ in range(tokens):
        nxt = int(rt.step(nxt))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_v13_scale_resident",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "opens_from": "pro_research/diag_gather_pcie_ceiling.json (52.2% of down PCIe bytes are block scales; v3 arm prices removing them at -1.380 ms/token)",
        "claim_boundary": "eager (non-graph) single-sequence A/B. Absolute ms/token savings only; not a graph-resident tok/s record and not comparable to the 21.0923 ms V6 graph number.",
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
            REPO / "pro_research" / "scale_resident_kernels.py",
            REPO / "pro_research" / "moe_dev_scale_resident.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
        ))

        rt = _new_runtime(capacity)
        dense = DenseERVF()
        down = DownProjBatchKernels()
        up = UpProjBatchKernels()
        rt.enable_cache(capacity)
        apply_nonuniform_capacity(rt)
        rt.device_cache = True
        rt.deterministic_accum = True
        restore_sel, sel_counts = _install_selective(rt, dense)
        restore_v6 = install_batched_moe_dev(rt, down, up)
        payload["selective_capture_counters"] = dict(sel_counts)

        _preheat(rt, prompts[0]["prompt_ids"], preheat_n)

        # ---------------- BASE_A (V6) --------------------------------------
        base_a: dict[str, list[int]] = {}
        base_a_ms: list[float] = []
        for p in prompts:
            ids, ms = _run(rt, p["prompt_ids"], n)
            base_a[p["prompt"]] = ids
            base_a_ms.extend(ms)

        # ---------------- G-V13-V1: VRAM gate, BEFORE allocating -----------
        # mem_info reports DRIVER-level free memory, and CuPy's pool keeps
        # every block it has ever grown into, so a raw read here returns 0
        # even with hundreds of MiB idle inside the pool. Return the unused
        # blocks first, then ask -- otherwise the gate measures the allocator,
        # not the device.
        pool = cp.get_default_memory_pool()
        pool_held_before = int(pool.total_bytes())
        pool_used_before = int(pool.used_bytes())
        pool.free_all_blocks()
        planned = planned_plane_bytes(rt)
        free_before = int(cp.cuda.Device(0).mem_info[0])
        vram_gate_prealloc = planned <= free_before
        payload["vram"] = {
            "plane_bytes_per_slot": PLANE_BYTES,
            "total_slots": sum(int(c["cap"]) for c in rt.cache.values()),
            "planned_plane_bytes": planned,
            "planned_plane_mib": planned / 1024 / 1024,
            "pool_held_before_mib": pool_held_before / 1024 / 1024,
            "pool_used_before_mib": pool_used_before / 1024 / 1024,
            "free_before_mib": free_before / 1024 / 1024,
            "device_total_mib": int(cp.cuda.Device(0).mem_info[1]) / 1024 / 1024,
            "gate_prealloc_fits": vram_gate_prealloc,
        }
        if not vram_gate_prealloc:
            payload.update({"status": "vram_gate_failed", "completed_utc": utc_now()})
            _write(payload)
            print(json.dumps({"status": payload["status"], "vram": payload["vram"]}, indent=2))
            return 2

        # ---------------- CAND (H-SCALE) -----------------------------------
        sres = ScaleResidentKernels()
        restore_v6()
        restore_cand = install_scale_resident_moe_dev(rt, down, up, sres)

        cand: dict[str, list[int]] = {}
        cand_ms: list[float] = []
        for p in prompts:
            ids, ms = _run(rt, p["prompt_ids"], n)
            cand[p["prompt"]] = ids
            cand_ms.extend(ms)

        # ---------------- CAND_PROBE: price the plane fetch itself ---------
        # S12's marginal method. fetch_planes is idempotent -- it writes the
        # same bytes of the same experts into the same slots -- so calling it
        # a second time per layer changes no output at all (the id gate below
        # checks that) and the delta against CAND is the fetch's own in-loop
        # cost. That is the term that has to explain the difference between
        # the isolated -1.380 ms and whatever CAND actually delivers.
        probe_ms: list[float] = []
        probe_ids: dict[str, list[int]] = {}
        _orig_moe_dev = rt._moe_dev

        import types as _types

        def _probe_moe_dev(self, i, out):
            r = _orig_moe_dev(i, out)
            sres.fetch_planes(self.bank[i]["down_base_ptr"], sres.planes[i],
                              self._dev_cache[i], self.top_k)
            return r

        rt._moe_dev = _types.MethodType(_probe_moe_dev, rt)
        for p in prompts:
            ids, ms = _run(rt, p["prompt_ids"], n)
            probe_ids[p["prompt"]] = ids
            probe_ms.extend(ms)
        rt._moe_dev = _orig_moe_dev

        free_after = int(cp.cuda.Device(0).mem_info[0])
        payload["vram"]["free_after_mib"] = free_after / 1024 / 1024
        payload["vram"]["gate_headroom_gt_64mib"] = free_after > 64 * 1024 * 1024

        # ---------------- BASE_B (V6 again) --------------------------------
        restore_cand()
        install_batched_moe_dev(rt, down, up)
        base_b: dict[str, list[int]] = {}
        base_b_ms: list[float] = []
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
            "G_V13_C1_cand_bitexact_vs_base_a": all(v is None for v in divs_cand.values()),
            "G_V13_C2_base_a_eq_base_b": all(v is None for v in divs_bb.values()),
            "G_V13_V1_vram": bool(vram_gate_prealloc and payload["vram"]["gate_headroom_gt_64mib"]),
            "G_V13_D1_drift_le_1ms": drift <= 1.0,
            "G_V13_P1_gain_ge_0_5ms": delta <= -0.5,
        }

        if not gates["G_V13_C1_cand_bitexact_vs_base_a"] or not gates["G_V13_C2_base_a_eq_base_b"]:
            status = "correctness_failed"
        elif not gates["G_V13_V1_vram"]:
            status = "vram_gate_failed"
        elif not gates["G_V13_D1_drift_le_1ms"]:
            status = "measurement_unstable"
        elif gates["G_V13_P1_gain_ge_0_5ms"]:
            status = "adoption_candidate"
        else:
            status = "gate_failed"

        pr = percentiles(probe_ms)
        plane_fetch_ms = float(pr["p50"]) - float(c["p50"])
        divs_probe = {p["prompt"]: first_divergence(base_a[p["prompt"]], probe_ids[p["prompt"]])
                      for p in prompts}
        gates["G_V13_C3_probe_bitexact"] = all(v is None for v in divs_probe.values())

        payload.update({
            "plane_fetch_marginal": {
                "CAND_PROBE": pr,
                "plane_fetch_ms_per_token": plane_fetch_ms,
                "probe_ids_match_base_a": gates["G_V13_C3_probe_bitexact"],
                "implied_gather_saving_ms_per_token": -delta + plane_fetch_ms,
                "isolated_gather_saving_ms_per_token": 1.380,
                "note": "CAND_PROBE = CAND with one extra idempotent fetch_planes per MoE layer. Its delta over CAND is the plane fetch's own in-loop cost; net gain + that cost = the gather-side saving actually realised in the loop, to compare against the 1.380 ms measured in isolation.",
            },
            "timing_eager": {
                "BASE_A": a, "CAND": c, "BASE_B": b,
                "baseline_midpoint_ms": midpoint,
                "drift_ms": drift,
                "cand_minus_midpoint_ms": delta,
                "BASE_A_tok_s": 1000.0 / float(a["p50"]),
                "CAND_tok_s": 1000.0 / float(c["p50"]),
                "BASE_B_tok_s": 1000.0 / float(b["p50"]),
            },
            "first_divergence_cand_vs_base_a": divs_cand,
            "first_divergence_base_b_vs_base_a": divs_bb,
            "gates": gates,
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })

        restore_sel()
        del rt, dense, down, up, sres
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
        "output": str(OUT),
        "vram": payload.get("vram"),
        "timing_eager": payload.get("timing_eager"),
        "gates": payload.get("gates"),
        "error": payload.get("error", {}).get("message") if payload.get("error") else None,
    }, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
