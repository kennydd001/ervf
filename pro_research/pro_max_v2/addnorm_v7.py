"""PV2-10: exact residual-add plus next RMSNorm fusion on top of V6."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
import types
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shared import (
    REPO, capture_v6, compare_arms, cuda_time, environment, graph_dot,
    new_v6_bundle, percentiles, prompt_set, result_path, run_arm, same_bits,
    status_from_gates, utc_now, write_json,
)

OUT = result_path("PV2_10_ADDNORM.json")

CUDA = r"""
__device__ __forceinline__ float pv2_bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

extern "C" __global__ void pv2_add_rmsnorm_bf16w(
    float* __restrict__ h,
    const float* __restrict__ residual,
    const unsigned short* __restrict__ w,
    float* __restrict__ out,
    const int n,
    const float eps)
{
    extern __shared__ float red[];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = h[i] + residual[i];
        h[i] = v;
        acc = fmaf(v, v, acc);
    }
    for (int o = warpSize >> 1; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
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
        out[i] = h[i] * scale * pv2_bf16_to_f32(w[i]);
}
"""


class AddNorm:
    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.mod = cp.RawModule(code=CUDA, options=("-std=c++14",))
        self.kernel = self.mod.get_function("pv2_add_rmsnorm_bf16w")

    def apply(self, h, residual, w, out, n: int, eps: float):
        self.kernel((1,), (256,),
                    (h, residual, w, out, np.int32(n), np.float32(eps)),
                    shared_mem=32 * 4)


def install(rt, candidate: AddNorm):
    original = rt._step_body_graph

    def body(self):
        k = self.k
        k.embed_gather(self.h, self._embed_tbl_ptr, self._tok_dev, self.hidden)
        # Preserve the production layer-0 norm exactly.
        k.norm(self.normed, self.h, self.layer[0]["norm"], self.hidden, self.eps)
        for i, ch in enumerate(self.pattern):
            if ch == "M":
                self._mamba(i, self.acc)
            elif ch == "*":
                self._attention(i, self.acc)
            else:
                self._moe(i, self.acc)
            next_w = self.layer[i + 1]["norm"] if i + 1 < len(self.pattern) else self.norm_f
            candidate.apply(self.h, self.acc, next_w, self.normed,
                            self.hidden, self.eps)
        if self.lm_head_kind == "nvfp4":
            self.fused.gemv_into(self.logits, self.lm_head_codes,
                                 self.lm_head_scales, self.normed,
                                 self.lm_head_g, self.vocab, self.hidden)
        else:
            k.mv_bf16(self.logits, self.lm_head, self.normed,
                      self.vocab, self.hidden)
        k.argmax_logits(self._tok_dev, self.logits, self.vocab,
                        self._am_max, self._am_idx)
        k.pos_increment(self._pos_dev)

    rt._step_body_graph = types.MethodType(body, rt)

    def restore():
        rt._step_body_graph = original
    return restore


def micro(rt, candidate: AddNorm, full: bool) -> dict[str, Any]:
    import cupy as cp
    rng = cp.random.RandomState(20260816)
    n = int(rt.hidden)
    h0 = rng.standard_normal(n, dtype=cp.float32)
    residual = rng.standard_normal(n, dtype=cp.float32)
    w = rt.layer[0]["norm"]
    hb, hc = h0.copy(), h0.copy()
    ob, oc = cp.empty(n, cp.float32), cp.empty(n, cp.float32)
    rt.k.add_(hb, residual, n)
    rt.k.norm(ob, hb, w, n, rt.eps)
    candidate.apply(hc, residual, w, oc, n, rt.eps)
    cp.cuda.Device(0).synchronize()
    exact_h = same_bits(hb, hc)
    exact_out = same_bits(ob, oc)

    def base_fn():
        cp.copyto(hb, h0)
        rt.k.add_(hb, residual, n)
        rt.k.norm(ob, hb, w, n, rt.eps)

    def cand_fn():
        cp.copyto(hc, h0)
        candidate.apply(hc, residual, w, oc, n, rt.eps)

    repeats = 300 if full else 60
    bt = cuda_time(base_fn, warmup=10, repeats=repeats)
    ct = cuda_time(cand_fn, warmup=10, repeats=repeats)
    bp, cpct = percentiles(bt), percentiles(ct)
    return {
        "exact_hidden": exact_h,
        "exact_normed": exact_out,
        "base_ms": bp,
        "candidate_ms": cpct,
        "speedup_p50": float(bp["p50"] / cpct["p50"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload: dict[str, Any] = {
        "kind": "pv2_addnorm", "status": "started", "mode": args.mode,
        "started_utc": utc_now(), "preregistration": "PREREGISTRATION.md",
    }
    bundle = None
    try:
        from shared import require_gpu_free
        require_gpu_free()
        prompts, _expected, n, _capacity = prompt_set(args.mode)
        payload["environment"] = environment((Path(__file__), HERE / "PREREGISTRATION.md"))
        bundle = new_v6_bundle()
        rt = bundle.rt
        candidate = AddNorm()
        payload["micro"] = micro(rt, candidate, args.mode == "full")

        capture_v6(bundle)
        base_a = run_arm(rt, prompts, n)

        restore = install(rt, candidate)
        extra = capture_v6(bundle)
        dot = graph_dot(rt).lower()
        cand = run_arm(rt, prompts, n)

        restore()
        capture_v6(bundle)
        base_b = run_arm(rt, prompts, n)

        par_a = compare_arms(base_a, cand)
        par_b = compare_arms(base_b, cand)
        pa = float(base_a["timing_ms"]["p50"])
        pc = float(cand["timing_ms"]["p50"])
        pb = float(base_b["timing_ms"]["p50"])
        mid = (pa + pb) / 2.0
        drift = abs(pa - pb)
        gain = mid - pc
        gates = {
            "micro_hidden_bitexact": bool(payload["micro"]["exact_hidden"]),
            "micro_normed_bitexact": bool(payload["micro"]["exact_normed"]),
            "micro_speedup_ge_1_02": payload["micro"]["speedup_p50"] >= 1.02,
            "graph_contains_candidate": "pv2_add_rmsnorm_bf16w" in dot,
            "causal_parity": all(x["identical"] for x in par_a.values()) and all(x["identical"] for x in par_b.values()),
            "base_drift_le_1ms": drift <= 1.0,
            "no_regression_gt_0_2pct": pc <= mid * 1.002,
            "samples_ge_500": int(cand["timing_ms"]["count"]) >= 500 if args.mode == "full" else None,
            "extra_vram_lt_64MiB": extra < 64 * 1024 * 1024,
        }
        required = ("micro_hidden_bitexact", "micro_normed_bitexact",
                    "micro_speedup_ge_1_02", "graph_contains_candidate",
                    "causal_parity", "base_drift_le_1ms",
                    "no_regression_gt_0_2pct", "extra_vram_lt_64MiB")
        if args.mode == "full":
            required += ("samples_ge_500",)
        payload.update({
            "graph": {"dot_length": len(dot), "candidate_name_present": "pv2_add_rmsnorm_bf16w" in dot, "extra_vram_bytes": extra},
            "arms": {"BASE_A": base_a, "ADDNORM": cand, "BASE_B": base_b},
            "parity": {"candidate_vs_base_a": par_a, "candidate_vs_base_b": par_b},
            "gates": gates,
            "adopt": all(bool(gates[k]) for k in required),
            "summary": {
                "base_a_p50_ms": pa, "candidate_p50_ms": pc,
                "base_b_p50_ms": pb, "baseline_mid_p50_ms": mid,
                "gain_ms": gain, "candidate_tok_s": 1000.0 / pc,
                "base_drift_ms": drift,
            },
            "status": status_from_gates(gates, required),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update(status="technical_failure", completed_utc=utc_now(),
                       error={"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()})
    finally:
        if bundle is not None:
            bundle.close()
    write_json(OUT, payload)
    print(json.dumps({"status": payload.get("status"), "adopt": payload.get("adopt"),
                      "summary": payload.get("summary"), "output": str(OUT)}, indent=2))
    return 0 if payload.get("status") in {"pass", "gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
