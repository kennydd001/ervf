"""PRO V19 — V18 plus the block-per-(h,p) ssm_step.

V18 (H-SCALE + B3) set the record at 19.60-19.69 ms = 50.8-51.0 tok/s and
showed that two bit-exact mechanisms, each under its own gate, combined to
roughly twice their arithmetic sum. This tests whether that keeps going with a
third, deliberately chosen to be **disjoint**: the block-per-(h,p) ssm_step
lives in Mamba, not the down_proj path, so the two have no shared resource to
contend for.

The ssm variant is bit-exact on y and state and x1.031 in isolation
(-0.024 ms/token, 23 cold state buffers). Small alone -- which is exactly why
it is worth stacking rather than adopting on its own.

Expectation: additive at best, ~-0.02 ms, because disjointness cuts both ways
-- no contention, but also no shared bottleneck to relieve twice. If it comes
out super-additive again that is a finding about the machine, not about these
two kernels.

## Arms and gates

  BASE_A   V6, captured        CAND   V6 + H-SCALE + B3 + ssm-block, captured
  BASE_B   V6, captured, after CAND

  G-V19-C1  CAND ids == BASE_A ids, every prompt, every token
  G-V19-C2  BASE_A ids == BASE_B ids
  G-V19-D1  |BASE_A.p50 - BASE_B.p50| <= 1.0 ms
  G-V19-P1  CAND.p50 <= midpoint - 1.20 ms   (V18 already delivers 1.18-1.56;
                                              this asks the stack to hold that)

The ssm replacement is built with --use_fast_math to match gpu_kernels.py --
ssm_decode_step has two __expf calls, and omitting the flag reproduces
PV2-10's token-124 divergence exactly.
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

RESULT_DIR = REPO / "pro_research" / "results" / "v19_combined_ssm"
OUT = RESULT_DIR / "PRO_V19_COMBINED_SSM.json"

from moe_dev_combined import install_combined_moe_dev
from ssm_block_install import install_ssm_block
from moe_dev_scale_resident import planned_plane_bytes
from scale_resident_kernels import PLANE_BYTES, ScaleResidentKernels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_v19_combined_ssm",
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
        restore_moe = install_combined_moe_dev(rt, down, up, sres)
        restore_ssm = install_ssm_block(rt)
        restore_fused = lambda: (restore_ssm(), restore_moe())
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
            "G_V19_C1_cand_bitexact": all(v is None for v in divs_c.values()),
            "G_V19_C2_base_a_eq_base_b": all(v is None for v in divs_b.values()),
            "G_V19_D1_drift_le_1ms": drift <= 1.0,
            "G_V19_P1_gain_ge_1_20ms": delta <= -1.20,
        }
        status = ("correctness_failed"
                  if not (gates["G_V19_C1_cand_bitexact"] and gates["G_V19_C2_base_a_eq_base_b"])
                  else "measurement_unstable" if not gates["G_V19_D1_drift_le_1ms"]
                  else "adoption_candidate" if gates["G_V19_P1_gain_ge_1_20ms"]
                  else "gate_failed")

        payload.update({
            "timing_graph_sync": {
                "BASE_A": a, "CAND": c, "BASE_B": b,
                "baseline_midpoint_ms": mid, "drift_ms": drift,
                "cand_minus_midpoint_ms": delta,
                "BASE_A_tok_s": 1000.0 / float(a["p50"]),
                "CAND_tok_s": 1000.0 / float(c["p50"]),
                "v18_measured_ms": [-1.5594, -1.1823],
                "ssm_block_alone_ms": -0.024,
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
