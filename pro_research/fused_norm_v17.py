"""PRO V17 — add+norm fusion, measured in the captured graph.

The isolated benchmark (diag_add_norm_fusion.json) has the fusion bit-exact on
both `h` and `out` and **1.745x faster, -0.354 ms/token**. That number must not
be quoted for production: it is an EAGER harness, and its implied per-launch
saving of 6.80 us matches the eager launch cost of 7.75 us rather than the
in-graph 3.53 us measured by diag_glue_marginals. So the expected in-graph gain
is roughly half -- about 52 x 3.53 us = 0.18 ms -- plus a little from dropping
one of the five passes over `h`.

Halving an isolated result before believing it is the correction this session
has had to make four times. This measures it instead.

## What is fused

Per layer boundary `_step_body_graph` runs:

    k.add_(self.h, self.acc, self.hidden)            # h += acc
    k.norm(self.normed, self.h, d["norm"], ...)      # next layer's input norm

The add writes the buffer the norm immediately reads. `add_rmsnorm` does both in
one launch: 105 norm/add launches per token become 53.

Bit-exact because the add is elementwise (its thread mapping cannot change a
value) and the RMSNorm reduction is reproduced line for line -- same stride
loop, same `fmaf(v, v, acc)`, same warp shuffle, same in-order sum of warp sums
by thread 0, same `rsqrtf(s/n + eps)`, same scaling loop. Nothing is
re-associated.

## Arms and gates

  BASE_A   V6 stack, captured
  CAND     V6 stack + fused add/norm, captured
  BASE_B   V6 stack, captured, after CAND

  G-V17-C1  CAND ids == BASE_A ids, every prompt, every token
  G-V17-C2  BASE_A ids == BASE_B ids
  G-V17-D1  |BASE_A.p50 - BASE_B.p50| <= 1.0 ms
  G-V17-P1  CAND.p50 <= midpoint - 0.10 ms

P1 is set at 0.10 ms, well under the ~0.18 ms expectation, because the point is
to decide whether the in-graph gain is real -- not to demand the eager figure.
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

RESULT_DIR = REPO / "pro_research" / "results" / "v17_fused_norm"
OUT = RESULT_DIR / "PRO_V17_FUSED_NORM.json"

FUSED_SRC = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
extern "C" __global__ void add_rmsnorm(
    float* __restrict__ h, const float* __restrict__ addend,
    const unsigned short* __restrict__ w, float* __restrict__ out,
    const int n, const float eps)
{
    extern __shared__ float red[];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = h[i] + addend[i];
        h[i] = v;
        acc = fmaf(v, v, acc);
    }
    for (int o = warpSize >> 1; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        const int nw = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nw; ++i) s += red[i];
        red[31] = rsqrtf(s / (float)n + eps);
    }
    __syncthreads();
    const float scale = red[31];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = h[i] * scale * bf16_to_f32(w[i]);
}
"""


def install_fused_body(rt):
    """Rewrite _step_body_graph so each layer boundary is one launch, not two."""
    import cupy as cp

    # MUST match gpu_kernels.py:1525, which compiles with --use_fast_math.
    # Without it rsqrtf and the denormal mode differ, and the tiny drift
    # amplifies through routing into a token divergence around step 124 --
    # exactly the failure PV2-10 reported and left unexplained.
    mod = cp.RawModule(code=FUSED_SRC,
                       options=("-std=c++14", "--use_fast_math"))
    k_fused = mod.get_function("add_rmsnorm")
    orig = rt._step_body_graph
    block = rt.k.block
    smem = 32 * 4

    def body(self):
        k = self.k
        k.embed_gather(self.h, self._embed_tbl_ptr, self._tok_dev, self.hidden)
        n32, eps32 = np.int32(self.hidden), np.float32(self.eps)
        for i, ch in enumerate(self.pattern):
            d = self.layer[i]
            if i == 0:
                # no preceding add on the first layer
                k.norm(self.normed, self.h, d["norm"], self.hidden, self.eps)
            if ch == "M":
                self._mamba(i, self.acc)
            elif ch == "*":
                self._attention(i, self.acc)
            else:
                self._moe(i, self.acc)
            nxt = self.layer[i + 1]["norm"] if i + 1 < len(self.pattern) else self.norm_f
            k_fused((1,), (block,),
                    (self.h, self.acc, nxt, self.normed, n32, eps32),
                    shared_mem=smem)
        if self.lm_head_kind == "nvfp4":
            self.fused.gemv_into(self.logits, self.lm_head_codes,
                                 self.lm_head_scales, self.normed,
                                 self.lm_head_g, self.vocab, self.hidden)
        else:
            k.mv_bf16(self.logits, self.lm_head, self.normed, self.vocab,
                      self.hidden)
        k.argmax_logits(self._tok_dev, self.logits, self.vocab,
                        self._am_max, self._am_idx)
        k.pos_increment(self._pos_dev)

    rt._step_body_graph = types.MethodType(body, rt)
    return lambda: setattr(rt, "_step_body_graph", orig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "pro_v17_fused_norm",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "opens_from": "diag_glue_marginals measured 3.53 us per in-graph kernel launch and 0.370 ms for 105 norm/add launches, almost all fixed cost. diag_add_norm_fusion has the fusion bit-exact and 1.745x faster EAGER (-0.354 ms), whose implied 6.80 us/launch matches the eager 7.75 us rather than the in-graph 3.53 -- so the expected in-graph gain is about half, ~0.18 ms. Measured here rather than assumed.",
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
        restore_fused = install_fused_body(rt)
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
            "G_V17_C1_cand_bitexact": all(v is None for v in divs_c.values()),
            "G_V17_C2_base_a_eq_base_b": all(v is None for v in divs_b.values()),
            "G_V17_D1_drift_le_1ms": drift <= 1.0,
            "G_V17_P1_gain_ge_0_10ms": delta <= -0.10,
        }
        status = ("correctness_failed"
                  if not (gates["G_V17_C1_cand_bitexact"] and gates["G_V17_C2_base_a_eq_base_b"])
                  else "measurement_unstable" if not gates["G_V17_D1_drift_le_1ms"]
                  else "adoption_candidate" if gates["G_V17_P1_gain_ge_0_10ms"]
                  else "gate_failed")

        payload.update({
            "timing_graph_sync": {
                "BASE_A": a, "CAND": c, "BASE_B": b,
                "baseline_midpoint_ms": mid, "drift_ms": drift,
                "cand_minus_midpoint_ms": delta,
                "BASE_A_tok_s": 1000.0 / float(a["p50"]),
                "CAND_tok_s": 1000.0 / float(c["p50"]),
                "launches_removed_per_token": len(rt.pattern),
                "eager_isolated_saving_ms": 0.354,
                "expected_in_graph_saving_ms": 0.18,
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
