"""Causal A/B/A/CTL for batched gather_down_sparse_ind +
gemv_down_masked_partial_ind, isolating this ONE addition on top of the
already-verified V5 (down_proj panel_scan/reduce_partials/accumulate) +
up-proj batching. Single variable under test: gather_kernels on/off.

Step 1 (verify_down_gather_batch_real_full.py, verify_gather_batch_real_full.py
-- real captured model data, not synthetic) passed bit-exact, zero NaN,
before this file was run. The earlier synthetic-random-data unit test
(verify_down_gather_batch_kernels.py) showed NaN and was NOT the basis for
this integration -- superseded by the real-data verification.
"""

from __future__ import annotations

import argparse
import gc
import traceback
from typing import Any

from common import (
    REPO,
    environment_snapshot,
    first_divergence,
    percentiles,
    require_gpu_free,
    result_path,
    utc_now,
    write_json_atomic,
)
from down_gather_batch_kernels import DownGatherBatchKernels
from down_proj_batch_kernels import DownProjBatchKernels
from graph_e1f22 import _load_prompt_set, _new_runtime, _run_eager_timed
from moe_dev_batched import install_batched_moe_dev
from up_proj_batch_kernels import UpProjBatchKernels

OUT = result_path("PRO_GATHER_BATCHED_AB.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "gather_batched_ab",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "scope": "isolates batched gather_down_sparse_ind + gemv_down_masked_partial_ind on top of already-verified V5+up-proj batching; single variable = gather_kernels on/off",
    }

    try:
        require_gpu_free()
        prompts, expected, n, capacity = _load_prompt_set(args.mode)
        payload["config"] = {"tokens_per_prompt": n, "capacity": capacity, "prompt_count": len(prompts)}
        payload["environment"] = environment_snapshot((
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "fused_nvfp4.py",
            REPO / "pro_research" / "down_gather_batch_kernels.py",
            REPO / "pro_research" / "moe_dev_batched.py",
        ))

        rt = _new_runtime(capacity)
        batch_kernels = DownProjBatchKernels()
        up_kernels = UpProjBatchKernels()
        gather_kernels = DownGatherBatchKernels()

        def run_arm(with_gather: bool):
            restore = install_batched_moe_dev(rt, batch_kernels, up_kernels, gather_kernels if with_gather else None)
            ids_by_prompt = {}
            ms = []
            for p in prompts:
                pid, pms = _run_eager_timed(rt, p["prompt_ids"], n)
                ids_by_prompt[p["prompt"]] = pid
                ms.extend(pms)
            restore()
            return ids_by_prompt, ms

        base_a_ids, base_a_ms = run_arm(with_gather=False)
        batched_ids, batched_ms = run_arm(with_gather=True)
        base_b_ids, base_b_ms = run_arm(with_gather=False)

        restore_ctl = install_batched_moe_dev(rt, batch_kernels, up_kernels, gather_kernels)
        rt._bad_pick = 1
        ctl_n = min(n, 64)
        ctl = {}
        for p in prompts:
            ids, _ = _run_eager_timed(rt, p["prompt_ids"], ctl_n)
            ref = base_a_ids[p["prompt"]][:ctl_n]
            ctl[p["prompt"]] = {"identical": ids == ref, "first_divergence": first_divergence(ids, ref)}
        rt._bad_pick = 0
        restore_ctl()

        parity = {}
        for p in prompts:
            name = p["prompt"]
            a, b, c = base_a_ids[name], batched_ids[name], base_b_ids[name]
            parity[name] = {
                "all_identical": a == b == c,
                "batched_vs_base_a_first_divergence": first_divergence(b, a),
                "base_b_vs_base_a_first_divergence": first_divergence(c, a),
            }

        pa = percentiles(base_a_ms)
        pbatch = percentiles(batched_ms)
        pb = percentiles(base_b_ms)
        base_mid = (pa["p50"] + pb["p50"]) / 2.0 if pa["p50"] and pb["p50"] else None
        gain_ms = (base_mid - pbatch["p50"]) if base_mid and pbatch["p50"] else None
        gain_fraction = (gain_ms / base_mid) if gain_ms and base_mid else None
        base_drift = abs(pa["p50"] - pb["p50"]) if pa["p50"] and pb["p50"] else None
        control_diverges = any(not v["identical"] for v in ctl.values())

        gates = {
            "batched_equals_base_bitexact": all(v["all_identical"] for v in parity.values()),
            "base_drift_le_1ms": bool(base_drift is not None and base_drift <= 1.0),
            "ctl_diverges": control_diverges,
            "candidate_gain_ge_0_2ms_or_1pct": bool(gain_ms is not None and (gain_ms >= 0.2 or (gain_fraction or 0) >= 0.01)),
            "full_samples_ge_500": (len(batched_ms) >= 500) if args.mode == "full" else None,
        }
        correctness = gates["batched_equals_base_bitexact"] and gates["base_drift_le_1ms"] and gates["ctl_diverges"]
        passed = (correctness and gates["candidate_gain_ge_0_2ms_or_1pct"] and gates["full_samples_ge_500"]) if args.mode == "full" else correctness

        payload.update({
            "arms": {"BASE_A_no_gather": {"timing_ms": pa}, "BATCHED_with_gather": {"timing_ms": pbatch}, "BASE_B_no_gather": {"timing_ms": pb}},
            "parity": parity,
            "control": ctl,
            "gates": gates,
            "summary": {
                "base_a_p50_ms": pa["p50"], "batched_p50_ms": pbatch["p50"], "base_b_p50_ms": pb["p50"],
                "baseline_mid_p50_ms": base_mid, "gain_ms": gain_ms, "gain_fraction": gain_fraction,
                "base_drift_ms": base_drift, "batched_tok_s": (1000.0 / pbatch["p50"]) if pbatch["p50"] else None,
            },
            "status": "pass" if passed else "gate_failed",
            "completed_utc": utc_now(),
        })

        del rt, batch_kernels, up_kernels, gather_kernels
        gc.collect()
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload["status"] = "technical_failure"
        payload["completed_utc"] = utc_now()
        payload["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}

    write_json_atomic(OUT, payload)
    print({"status": payload.get("status"), "output": str(OUT), "summary": payload.get("summary"), "gates": payload.get("gates")})
    return 0 if payload.get("status") in {"pass", "gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
