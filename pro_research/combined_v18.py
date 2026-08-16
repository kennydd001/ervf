"""PRO V18 — H-SCALE and B3 overlap together, measured as one arm.

Two bit-exact mechanisms built earlier today each fell just under their own
preregistered gate:

    V13  H-SCALE   -0.374 ms/token   gate >=0.5   scale planes resident in VRAM,
                                                  so the gather moves 52% fewer
                                                  bytes
    V14G B3        -0.416 ms/token   gate >=0.8   double-buffered mirror plus a
                                                  gather stream, so slot s+1's
                                                  PCIe traffic runs while slot s
                                                  computes

They attack the same path from opposite sides. This project's working rule is
that component measurements are never summed into a claim, so the only way to
learn what they are worth together is to run them together -- and they may well
anti-compose, since a smaller gather leaves B3 less to hide. Somewhere between
-0.42 and -0.79 ms; a measurement decides.

Neither changes arithmetic: same expert, panel, row, scale byte,
`e4m3_lut[byte] * global_scale`, same fmaf order. Only where bytes live and when
they move.

## Arms and gates

  BASE_A   V6 stack, captured
  CAND     V6 stack + H-SCALE + B3, captured
  BASE_B   V6 stack, captured, after CAND

  G-V18-C1  CAND ids == BASE_A ids, every prompt, every token
  G-V18-C2  BASE_A ids == BASE_B ids
  G-V18-V1  planned plane VRAM fits, checked before allocating
  G-V18-D1  |BASE_A.p50 - BASE_B.p50| <= 1.0 ms
  G-V18-P1  CAND.p50 <= midpoint - 0.5 ms

P1 is set at 0.5 ms: above either mechanism alone, below their arithmetic sum,
so the gate answers "do they actually compose" rather than rewarding either one
being re-measured.

SYNC semantics, comparable to the 21.0923 ms V6 record.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
import types
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, first_divergence, percentiles, require_gpu_free, utc_now
from diag_component_marginals_graph import _prefill, _recapture, _reset_exact_state, _run
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _load_prompt_set, _new_runtime
from layer_capacity import apply_nonuniform_capacity
from moe_dev_batched import install_batched_moe_dev
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

RESULT_DIR = REPO / "pro_research" / "results" / "v18_combined"
OUT = RESULT_DIR / "PRO_V18_COMBINED.json"

from moe_dev_combined import install_combined_moe_dev
from moe_dev_scale_resident import planned_plane_bytes
from scale_resident_kernels import PLANE_BYTES, ScaleResidentKernels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_v18_combined",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "opens_from": "V13 H-SCALE (-0.374) and V14G B3 (-0.416) are each bit-exact and each just under their own gate; measured together here because component numbers are never summed. Superseded note: diag_glue_marginals measured 3.53 us per in-graph kernel launch and 0.370 ms for 105 norm/add launches, almost all fixed cost. diag_add_norm_fusion has the fusion bit-exact and 1.745x faster EAGER (-0.354 ms), whose implied 6.80 us/launch matches the eager 7.75 us rather than the in-graph 3.53 -- so the expected in-graph gain is about half, ~0.18 ms. Measured here rather than assumed.",
    }

    try:
        require_gpu_free()
        import cupy as cp

        prompts, _e, n, capacity = _load_prompt_set(args.mode)
        n = min(n, 32) if args.mode == "smoke" else max(n, 256)
        preheat = 32 if args.mode == "smoke" else 128
        payload["config"] = {"tokens_per_prompt": n, "prompt_count": len(prompts)}
        payload["environment"] = environment_snapshot()

        rt = _new_runtime(capacity)
        dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
        rt.enable_cache(capacity)
        apply_nonuniform_capacity(rt)
        rt.device_cache = True
        rt.deterministic_accum = True
        restore_sel, _ = _install_selective(rt, dense)
        install_batched_moe_dev(rt, down, up)
        rt.setup_graph()

        _reset_exact_state(rt)
        _prefill(rt, prompts[0]["prompt_ids"])
        for _ in range(preheat):
            rt.step_graph(None)
        rt._graph_stream.synchronize()

        def arm():
            ids_by, ms_all = {}, []
            for p in prompts:
                ids, ms = _run(rt, p["prompt_ids"], n)
                ids_by[p["prompt"]] = ids
                ms_all.extend(ms)
            return ids_by, percentiles(ms_all)

        base_a, a = arm()
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        planned = planned_plane_bytes(rt)
        free_before = int(cp.cuda.Device(0).mem_info[0])
        payload["vram"] = {"planned_plane_mib": planned / 1024 / 1024,
                           "free_before_mib": free_before / 1024 / 1024,
                           "fits": planned <= free_before}
        if planned > free_before:
            payload.update({"status": "vram_gate_failed", "completed_utc": utc_now()})
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps(payload["vram"], indent=2))
            return 2
        sres = ScaleResidentKernels()
        restore_fused = install_combined_moe_dev(rt, down, up, sres)
        _recapture(rt)
        cand, c = arm()
        restore_fused()
        _recapture(rt)
        base_b, b = arm()

        drift = abs(float(a["p50"]) - float(b["p50"]))
        mid = (float(a["p50"]) + float(b["p50"])) / 2.0
        delta = float(c["p50"]) - mid
        divs_c = {p["prompt"]: first_divergence(base_a[p["prompt"]], cand[p["prompt"]])
                  for p in prompts}
        divs_b = {p["prompt"]: first_divergence(base_a[p["prompt"]], base_b[p["prompt"]])
                  for p in prompts}

        gates = {
            "G_V18_C1_cand_bitexact": all(v is None for v in divs_c.values()),
            "G_V18_C2_base_a_eq_base_b": all(v is None for v in divs_b.values()),
            "G_V18_D1_drift_le_1ms": drift <= 1.0,
            "G_V18_P1_gain_ge_0_50ms": delta <= -0.50,
        }
        status = ("correctness_failed"
                  if not (gates["G_V18_C1_cand_bitexact"] and gates["G_V18_C2_base_a_eq_base_b"])
                  else "measurement_unstable" if not gates["G_V18_D1_drift_le_1ms"]
                  else "adoption_candidate" if gates["G_V18_P1_gain_ge_0_50ms"]
                  else "gate_failed")

        payload.update({
            "timing_graph_sync": {
                "BASE_A": a, "CAND": c, "BASE_B": b,
                "baseline_midpoint_ms": mid, "drift_ms": drift,
                "cand_minus_midpoint_ms": delta,
                "BASE_A_tok_s": 1000.0 / float(a["p50"]),
                "CAND_tok_s": 1000.0 / float(c["p50"]),
                "v13_alone_ms": -0.374, "v14g_alone_ms": -0.416,
                "arithmetic_sum_if_independent_ms": -0.790,
            },
            "first_divergence_cand": divs_c,
            "gates": gates, "status": status,
            "completed_utc": utc_now(),
        })

        restore_sel()
        del rt, dense, down, up
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({"status": "technical_failure",
                        "error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()},
                        "completed_utc": utc_now()})

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload.get("status"),
                      "timing_graph_sync": payload.get("timing_graph_sync"),
                      "gates": payload.get("gates"),
                      "error": (payload.get("error") or {}).get("message")}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
