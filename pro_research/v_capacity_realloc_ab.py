"""Causal A/B/A/CTL for the per-layer cache capacity reallocation
(layer_capacity.py), isolated on top of production kernels (no batching) so
the single variable under test is capacity allocation alone. Precedent:
A1's own adoption test already proved capacity changes (72 vs 56) preserve
bit-exact output with deterministic accumulation (D1) in place -- this is
the same class of change, just per-layer instead of uniform.

diag_per_layer_capacity.py already measured the hit-rate effect (-14.3%
misses, hit rate 85.61% -> 87.66%). This measures the actual ms/token.
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
from graph_e1f22 import _load_prompt_set, _new_runtime, _run_eager_timed
from layer_capacity import BASELINE_CAP, BOOST_DELTA, BOOST_LAYERS, REDUCE_DELTA, REDUCE_LAYERS, apply_nonuniform_capacity

OUT = result_path("PRO_CAPACITY_REALLOC_AB.json")


def _reset_uniform(rt, capacity: int):
    rt.enable_cache(capacity)
    rt.device_cache = True
    rt.deterministic_accum = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "capacity_realloc_ab",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "scope": "isolates per-layer cache capacity reallocation on the production (unbached) kernel path; single variable = uniform vs non-uniform capacity at constant total budget",
        "config": {
            "baseline_cap": BASELINE_CAP,
            "reduce_layers": REDUCE_LAYERS, "reduce_delta": REDUCE_DELTA,
            "boost_layers": BOOST_LAYERS, "boost_delta": BOOST_DELTA,
        },
    }

    try:
        require_gpu_free()
        prompts, expected, n, capacity = _load_prompt_set(args.mode)
        payload["config"]["tokens_per_prompt"] = n
        payload["config"]["prompt_count"] = len(prompts)
        payload["environment"] = environment_snapshot((
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
            REPO / "pro_research" / "layer_capacity.py",
        ))

        rt = _new_runtime(BASELINE_CAP)

        def run_arm(nonuniform: bool):
            _reset_uniform(rt, BASELINE_CAP)
            if nonuniform:
                apply_nonuniform_capacity(rt)
            ids_by_prompt = {}
            ms = []
            for p in prompts:
                pid, pms = _run_eager_timed(rt, p["prompt_ids"], n)
                ids_by_prompt[p["prompt"]] = pid
                ms.extend(pms)
            return ids_by_prompt, ms

        base_a_ids, base_a_ms = run_arm(nonuniform=False)
        nonuniform_ids, nonuniform_ms = run_arm(nonuniform=True)
        base_b_ids, base_b_ms = run_arm(nonuniform=False)

        # CTL: sabotage the router while non-uniform capacity is active --
        # must diverge, proving the harness has discriminating power.
        _reset_uniform(rt, BASELINE_CAP)
        apply_nonuniform_capacity(rt)
        rt._bad_pick = 1
        ctl_n = min(n, 64)
        ctl = {}
        for p in prompts:
            ids, _ = _run_eager_timed(rt, p["prompt_ids"], ctl_n)
            ref = base_a_ids[p["prompt"]][:ctl_n]
            ctl[p["prompt"]] = {"identical": ids == ref, "first_divergence": first_divergence(ids, ref)}
        rt._bad_pick = 0

        parity = {}
        for p in prompts:
            name = p["prompt"]
            a, b, c = base_a_ids[name], nonuniform_ids[name], base_b_ids[name]
            parity[name] = {
                "all_identical": a == b == c,
                "nonuniform_vs_base_a_first_divergence": first_divergence(b, a),
                "base_b_vs_base_a_first_divergence": first_divergence(c, a),
            }

        pa = percentiles(base_a_ms)
        pnu = percentiles(nonuniform_ms)
        pb = percentiles(base_b_ms)
        base_mid = (pa["p50"] + pb["p50"]) / 2.0 if pa["p50"] and pb["p50"] else None
        gain_ms = (base_mid - pnu["p50"]) if base_mid and pnu["p50"] else None
        gain_fraction = (gain_ms / base_mid) if gain_ms and base_mid else None
        base_drift = abs(pa["p50"] - pb["p50"]) if pa["p50"] and pb["p50"] else None
        control_diverges = any(not v["identical"] for v in ctl.values())

        gates = {
            "nonuniform_equals_base_bitexact": all(v["all_identical"] for v in parity.values()),
            "base_drift_le_1ms": bool(base_drift is not None and base_drift <= 1.0),
            "ctl_diverges": control_diverges,
            "candidate_gain_ge_0_1ms_or_0_5pct": bool(gain_ms is not None and (gain_ms >= 0.1 or (gain_fraction or 0) >= 0.005)),
            "full_samples_ge_500": (len(nonuniform_ms) >= 500) if args.mode == "full" else None,
        }
        correctness = gates["nonuniform_equals_base_bitexact"] and gates["base_drift_le_1ms"] and gates["ctl_diverges"]
        passed = (correctness and gates["candidate_gain_ge_0_1ms_or_0_5pct"] and gates["full_samples_ge_500"]) if args.mode == "full" else correctness

        payload.update({
            "arms": {"BASE_A_uniform": {"timing_ms": pa}, "NONUNIFORM": {"timing_ms": pnu}, "BASE_B_uniform": {"timing_ms": pb}},
            "parity": parity,
            "control": ctl,
            "gates": gates,
            "summary": {
                "base_a_p50_ms": pa["p50"], "nonuniform_p50_ms": pnu["p50"], "base_b_p50_ms": pb["p50"],
                "baseline_mid_p50_ms": base_mid, "gain_ms": gain_ms, "gain_fraction": gain_fraction,
                "base_drift_ms": base_drift, "nonuniform_tok_s": (1000.0 / pnu["p50"]) if pnu["p50"] else None,
            },
            "status": "pass" if passed else "gate_failed",
            "completed_utc": utc_now(),
        })

        del rt
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
