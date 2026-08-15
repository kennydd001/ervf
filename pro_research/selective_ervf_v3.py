"""PRO V3-G1B: causal A/B/A for the preregistered selective dense ERVF policy.

The original G1 experiment is left untouched.  V3-G1B uses ERVF only on the
four real shapes that were already bit-exact AND faster in G1, and dispatches
all small K/V/router projections to the production kernels.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from typing import Any

import numpy as np

from common import (
    REPO,
    environment_snapshot,
    first_divergence,
    load_json,
    percentiles,
    require_gpu_free,
    result_path,
    utc_now,
    write_json_atomic,
)
from ervf_dense import DenseERVF
from graph_e1f22 import _load_prompt_set, _new_runtime

OUT = result_path("PRO_V3_G1B_SELECTIVE_ERVF.json")
OLD_G1 = result_path("PRO_G1_DENSE_ERVF.json")

# Frozen from the already-observed G1 result, before this causal measurement.
BF16_ERVF_SHAPES = {(4096, 2688), (2688, 4096)}
FP8_ERVF_SHAPES = {(10304, 2688), (2688, 4096)}


def _install_selective(rt, dense: DenseERVF):
    orig_bf16 = rt.k.mv_bf16
    orig_fp8 = rt.k.mv_fp8_tensor
    orig_f32 = rt.k.mv_f32
    counters = {
        "bf16_ervf": 0,
        "bf16_prod": 0,
        "fp8_ervf": 0,
        "fp8_prod": 0,
        "f32_prod": 0,
    }

    def bf16(out, W, x, rows, cols):
        shape = (int(rows), int(cols))
        if shape in BF16_ERVF_SHAPES:
            counters["bf16_ervf"] += 1
            return dense.mv_bf16(out, W, x, int(rows), int(cols))
        counters["bf16_prod"] += 1
        return orig_bf16(out, W, x, rows, cols)

    def fp8(out, W, x, wscale, rows, cols):
        shape = (int(rows), int(cols))
        if shape in FP8_ERVF_SHAPES:
            counters["fp8_ervf"] += 1
            return dense.mv_fp8_tensor(out, W, x, float(wscale), int(rows), int(cols))
        counters["fp8_prod"] += 1
        return orig_fp8(out, W, x, wscale, rows, cols)

    def f32(out, W, x, rows, cols):
        counters["f32_prod"] += 1
        return orig_f32(out, W, x, rows, cols)

    rt.k.mv_bf16 = bf16
    rt.k.mv_fp8_tensor = fp8
    rt.k.mv_f32 = f32

    def restore():
        rt.k.mv_bf16 = orig_bf16
        rt.k.mv_fp8_tensor = orig_fp8
        rt.k.mv_f32 = orig_f32

    return restore, counters


def _arm(rt, prompts: list[dict[str, Any]], n: int, capacity: int) -> dict[str, Any]:
    import cupy as cp

    # Cache/LRU state is part of the execution state; rebuild it before each arm
    # so A/B/A begins from the same capacity/state definition.
    rt.enable_cache(capacity)
    rt.device_cache = True
    rt.deterministic_accum = True

    per_prompt = []
    all_ms: list[float] = []
    for p in prompts:
        rt.reset()
        nxt = None
        for token in p["prompt_ids"]:
            nxt = int(rt.step(int(token)))
        if nxt is None:
            raise ValueError("prompt must contain at least one token")
        cp.cuda.Device(0).synchronize()
        ids = [int(nxt)]
        cur = int(nxt)
        for _ in range(n - 1):
            t0 = time.perf_counter_ns()
            cur = int(rt.step(cur))
            cp.cuda.Device(0).synchronize()
            all_ms.append((time.perf_counter_ns() - t0) / 1e6)
            ids.append(cur)
        per_prompt.append({
            "prompt": p["prompt"],
            "kind": p["kind"],
            "ids": ids,
        })
    return {"prompts": per_prompt, "timing_ms": percentiles(all_ms), "raw_timing_ms": all_ms}


def _ids_by_prompt(arm: dict[str, Any]) -> dict[str, list[int]]:
    return {p["prompt"]: [int(x) for x in p["ids"]] for p in arm["prompts"]}


def _old_g1_evidence() -> dict[str, Any] | None:
    if not OLD_G1.exists():
        return None
    old = load_json(OLD_G1)
    cases = old.get("microbench", {}).get("cases", [])
    selected = {}
    for c in cases:
        shape = (int(c["rows"]), int(c["cols"]))
        if (c["kind"] == "bf16" and shape in BF16_ERVF_SHAPES) or (
            c["kind"] == "fp8" and shape in FP8_ERVF_SHAPES
        ):
            selected[c["name"]] = {
                "kind": c["kind"],
                "shape": list(shape),
                "bit_equal": bool(c.get("bit_equal")),
                "speedup": float(c.get("speedup")),
            }
    return {
        "source_status": old.get("status"),
        "source_file": str(OLD_G1),
        "selected_cases": selected,
        "all_selected_bit_exact": len(selected) == 4 and all(v["bit_equal"] for v in selected.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_v3_g1b_selective_ervf",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "preregistration": "pro_research/PRO_V3_PREREGISTRATION.md",
        "policy": {
            "bf16_ervf_shapes": [list(x) for x in sorted(BF16_ERVF_SHAPES)],
            "fp8_ervf_shapes": [list(x) for x in sorted(FP8_ERVF_SHAPES)],
            "f32": "production_only",
        },
    }

    try:
        require_gpu_free()
        prompts, _expected, n, capacity = _load_prompt_set(args.mode)
        payload["config"] = {"tokens_per_prompt": n, "capacity": capacity, "prompt_count": len(prompts)}
        payload["environment"] = environment_snapshot((
            REPO / "pro_research" / "selective_ervf_v3.py",
            REPO / "pro_research" / "ervf_dense.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
        ))
        payload["prior_g1_evidence"] = _old_g1_evidence()

        rt = _new_runtime(capacity)
        # _new_runtime from G0 already enables/loads the bank in its current
        # implementation; if a future version changes that, fail explicitly.
        if not getattr(rt, "bank", None):
            rt.load_routed_bank()
        rt.device_cache = True
        rt.deterministic_accum = True

        # Compile the candidate module before BASE_A so compilation cannot land
        # inside candidate timing and so both baselines see the same thermal era.
        dense = DenseERVF()

        base_a = _arm(rt, prompts, n, capacity)
        restore, counters = _install_selective(rt, dense)
        selective = _arm(rt, prompts, n, capacity)
        restore()
        base_b = _arm(rt, prompts, n, capacity)

        a_ids = _ids_by_prompt(base_a)
        s_ids = _ids_by_prompt(selective)
        b_ids = _ids_by_prompt(base_b)
        parity = {}
        for p in prompts:
            name = p["prompt"]
            parity[name] = {
                "all_identical": a_ids[name] == s_ids[name] == b_ids[name],
                "selective_vs_base_a_first_divergence": first_divergence(s_ids[name], a_ids[name]),
                "base_b_vs_base_a_first_divergence": first_divergence(b_ids[name], a_ids[name]),
            }

        pa = float(base_a["timing_ms"]["p50"])
        ps = float(selective["timing_ms"]["p50"])
        pb = float(base_b["timing_ms"]["p50"])
        base_mid = (pa + pb) / 2.0
        gain_ms = base_mid - ps
        gain_fraction = gain_ms / base_mid if base_mid else 0.0
        base_drift = abs(pa - pb)
        samples = int(selective["timing_ms"]["count"])

        prior_exact = bool(payload["prior_g1_evidence"] and payload["prior_g1_evidence"].get("all_selected_bit_exact"))
        gates = {
            "causal_token_parity": all(v["all_identical"] for v in parity.values()),
            "prior_selected_micro_bitexact": prior_exact,
            "base_a_b_p50_drift_le_1ms": base_drift <= 1.0,
            "candidate_gain_ge_1_5ms_or_5pct": gain_ms >= 1.5 or gain_fraction >= 0.05,
            "selective_samples_ge_500": samples >= 500 if args.mode == "full" else None,
            "selected_dispatch_was_exercised": counters["bf16_ervf"] > 0 and counters["fp8_ervf"] > 0,
            "f32_ervf_was_not_used": counters["f32_prod"] > 0,
        }
        required = [
            gates["causal_token_parity"],
            gates["prior_selected_micro_bitexact"],
            gates["base_a_b_p50_drift_le_1ms"],
            gates["candidate_gain_ge_1_5ms_or_5pct"],
            gates["selected_dispatch_was_exercised"],
            gates["f32_ervf_was_not_used"],
        ]
        if args.mode == "full":
            required.append(bool(gates["selective_samples_ge_500"]))

        payload.update({
            "arms": {"BASE_A": base_a, "SELECTIVE": selective, "BASE_B": base_b},
            "parity": parity,
            "dispatch_counters": counters,
            "gates": gates,
            "summary": {
                "base_a_p50_ms": pa,
                "selective_p50_ms": ps,
                "base_b_p50_ms": pb,
                "baseline_mid_p50_ms": base_mid,
                "gain_ms": gain_ms,
                "gain_fraction": gain_fraction,
                "base_drift_ms": base_drift,
                "selective_tok_s": 1000.0 / ps if ps else None,
            },
            "status": "pass" if all(required) else "gate_failed",
            "completed_utc": utc_now(),
        })

        del rt, dense
        gc.collect()
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload["status"] = "technical_failure"
        payload["completed_utc"] = utc_now()
        payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    write_json_atomic(OUT, payload)
    print({
        "status": payload.get("status"),
        "output": str(OUT),
        "summary": payload.get("summary"),
        "gates": payload.get("gates"),
        "dispatch_counters": payload.get("dispatch_counters"),
    })
    return 0 if payload.get("status") in {"pass", "gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
