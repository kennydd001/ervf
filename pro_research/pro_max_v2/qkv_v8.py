"""PV2-11: exact mixed-shape one-launch Q/K/V projection on top of V6."""
from __future__ import annotations

import argparse
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
    capture_v6, compare_arms, cuda_time, environment, graph_dot,
    new_v6_bundle, percentiles, prompt_set, result_path, run_arm, same_bits,
    status_from_gates, utc_now, write_json,
)

OUT = result_path("PV2_11_QKV.json")

CUDA = r"""
__device__ __forceinline__ float pv2_bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
#define PV2_W 16
#define PV2_V 16
#define PV2_RPB 16

__device__ __forceinline__ float pv2_reduce_q(float acc[PV2_V]) {
    const int lane = threadIdx.x & 15;
    float s[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        float v = acc[g * 2] + acc[g * 2 + 1];
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o, 16);
        s[g] = v;
    }
    if (lane == 0) {
        const float a0 = s[0] + s[4];
        const float a1 = s[1] + s[5];
        const float a2 = s[2] + s[6];
        const float a3 = s[3] + s[7];
        return (a0 + a2) + (a1 + a3);
    }
    return 0.0f;
}

extern "C" __global__ void pv2_qkv_mixed_fused(
    const unsigned short* __restrict__ Wq,
    const unsigned short* __restrict__ Wk,
    const unsigned short* __restrict__ Wv,
    const float* __restrict__ x,
    float* __restrict__ qout,
    float* __restrict__ kout,
    float* __restrict__ vout,
    const int qrows, const int kvrows, const int cols)
{
    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __syncthreads();

    const int qblocks = (qrows + 15) >> 4;
    if ((int)blockIdx.x < qblocks) {
        const int sub = threadIdx.x >> 4;
        const int lane = threadIdx.x & 15;
        const int row = (int)blockIdx.x * 16 + sub;
        const bool valid = row < qrows;
        const unsigned short* w = Wq + (size_t)(valid ? row : 0) * cols;
        float part[16];
        #pragma unroll
        for (int vi = 0; vi < 16; ++vi) part[vi] = 0.0f;
        #pragma unroll
        for (int vi = 0; vi < 16; ++vi) {
            const int tid = lane + 16 * vi;
            if (valid)
                for (int k = tid; k < cols; k += 256)
                    part[vi] = fmaf(pv2_bf16_to_f32(w[k]), sx[k], part[vi]);
        }
        const float y = pv2_reduce_q(part);
        if (lane == 0 && valid) qout[row] = y;
        return;
    }

    int b = (int)blockIdx.x - qblocks;
    const bool is_v = b >= kvrows;
    if (is_v) b -= kvrows;
    if (b >= kvrows) return;
    const unsigned short* W = is_v ? Wv : Wk;
    float* out = is_v ? vout : kout;
    const unsigned short* w = W + (size_t)b * cols;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < cols; k += 256)
        acc = fmaf(pv2_bf16_to_f32(w[k]), sx[k], acc);
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float ws[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) ws[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float y = (lane < 8) ? ws[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            y += __shfl_down_sync(0xffffffffu, y, o);
        if (lane == 0) out[b] = y;
    }
}
"""


class QKV:
    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.mod = cp.RawModule(code=CUDA, options=("-std=c++14", "--use_fast_math"))
        self.kernel = self.mod.get_function("pv2_qkv_mixed_fused")

    def apply(self, q, k, v, Wq, Wk, Wv, x, qrows: int, kvrows: int, cols: int):
        blocks = (qrows + 15) // 16 + 2 * kvrows
        self.kernel((blocks,), (256,),
                    (Wq, Wk, Wv, x, q, k, v,
                     np.int32(qrows), np.int32(kvrows), np.int32(cols)),
                    shared_mem=cols * 4)


def install(rt, candidate: QKV):
    original = rt._attention

    def attention(self, i, out):
        if not (self.graph_mode and self.fp8_kv):
            return original(i, out)
        k, d = self.k, self.layer[i]
        qrows = self.n_heads * self.head_dim
        kvrows = self.n_kv * self.head_dim
        candidate.apply(self.qv, self.kv_, self.vv,
                        d["q_proj"], d["k_proj"], d["v_proj"], self.normed,
                        qrows, kvrows, self.hidden)
        scale = 1.0 / float(np.sqrt(self.head_dim))
        k.kv_write_fp8_dp(self.kc[i], self.kv_, self._pos_dev,
                          self.n_kv, self.head_dim, self.max_ctx)
        k.kv_write_fp8_dp(self.vc[i], self.vv, self._pos_dev,
                          self.n_kv, self.head_dim, self.max_ctx)
        k.attention_fp8_gqa4_dp(self.ctx, self.qv, self.kc[i], self.vc[i],
                                self._pos_dev, self.n_heads, self.head_dim,
                                self.groups, self.max_ctx, scale,
                                self.part_acc, self.part_ml)
        k.mv_bf16(out, d["o_proj"], self.ctx, self.hidden, qrows)

    rt._attention = types.MethodType(attention, rt)

    def restore():
        rt._attention = original
    return restore


def micro(rt, candidate: QKV, full: bool) -> dict[str, Any]:
    import cupy as cp
    i = rt.attn_layers[0]
    d = rt.layer[i]
    qrows = rt.n_heads * rt.head_dim
    kvrows = rt.n_kv * rt.head_dim
    cols = rt.hidden
    x = cp.random.RandomState(20260816).standard_normal(cols, dtype=cp.float32)
    qb, kb, vb = cp.empty(qrows, cp.float32), cp.empty(kvrows, cp.float32), cp.empty(kvrows, cp.float32)
    qc, kc, vc = cp.empty_like(qb), cp.empty_like(kb), cp.empty_like(vb)
    rt.k.mv_bf16(qb, d["q_proj"], x, qrows, cols)
    rt.k.mv_bf16(kb, d["k_proj"], x, kvrows, cols)
    rt.k.mv_bf16(vb, d["v_proj"], x, kvrows, cols)
    candidate.apply(qc, kc, vc, d["q_proj"], d["k_proj"], d["v_proj"],
                    x, qrows, kvrows, cols)
    cp.cuda.Device(0).synchronize()
    exact = {"q": same_bits(qb, qc), "k": same_bits(kb, kc), "v": same_bits(vb, vc)}

    def base_fn():
        rt.k.mv_bf16(qb, d["q_proj"], x, qrows, cols)
        rt.k.mv_bf16(kb, d["k_proj"], x, kvrows, cols)
        rt.k.mv_bf16(vb, d["v_proj"], x, kvrows, cols)

    def cand_fn():
        candidate.apply(qc, kc, vc, d["q_proj"], d["k_proj"], d["v_proj"],
                        x, qrows, kvrows, cols)

    repeats = 250 if full else 50
    bt, ct = cuda_time(base_fn, 10, repeats), cuda_time(cand_fn, 10, repeats)
    bp, cpct = percentiles(bt), percentiles(ct)
    return {"exact": exact, "base_ms": bp, "candidate_ms": cpct,
            "speedup_p50": float(bp["p50"] / cpct["p50"]),
            "shape": {"qrows": qrows, "kvrows": kvrows, "cols": cols}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload: dict[str, Any] = {"kind": "pv2_qkv", "status": "started",
                               "mode": args.mode, "started_utc": utc_now(),
                               "preregistration": "PREREGISTRATION.md"}
    bundle = None
    try:
        from shared import require_gpu_free
        require_gpu_free()
        prompts, _expected, n, _capacity = prompt_set(args.mode)
        payload["environment"] = environment((Path(__file__), HERE / "PREREGISTRATION.md"))
        bundle = new_v6_bundle()
        rt = bundle.rt
        candidate = QKV()
        payload["micro"] = micro(rt, candidate, args.mode == "full")

        capture_v6(bundle); base_a = run_arm(rt, prompts, n)
        restore = install(rt, candidate)
        extra = capture_v6(bundle); dot = graph_dot(rt).lower(); cand = run_arm(rt, prompts, n)
        restore()
        capture_v6(bundle); base_b = run_arm(rt, prompts, n)

        par_a, par_b = compare_arms(base_a, cand), compare_arms(base_b, cand)
        pa, pc, pb = (float(base_a["timing_ms"]["p50"]),
                      float(cand["timing_ms"]["p50"]),
                      float(base_b["timing_ms"]["p50"]))
        mid, drift = (pa + pb) / 2.0, abs(pa - pb)
        gates = {
            "micro_qkv_bitexact": all(payload["micro"]["exact"].values()),
            "micro_speedup_ge_1_02": payload["micro"]["speedup_p50"] >= 1.02,
            "graph_contains_candidate": "pv2_qkv_mixed_fused" in dot,
            "causal_parity": all(x["identical"] for x in par_a.values()) and all(x["identical"] for x in par_b.values()),
            "base_drift_le_1ms": drift <= 1.0,
            "no_regression_gt_0_2pct": pc <= mid * 1.002,
            "samples_ge_500": int(cand["timing_ms"]["count"]) >= 500 if args.mode == "full" else None,
            "extra_vram_lt_64MiB": extra < 64 * 1024 * 1024,
        }
        required = ("micro_qkv_bitexact", "micro_speedup_ge_1_02",
                    "graph_contains_candidate", "causal_parity",
                    "base_drift_le_1ms", "no_regression_gt_0_2pct",
                    "extra_vram_lt_64MiB")
        if args.mode == "full": required += ("samples_ge_500",)
        payload.update({
            "graph": {"dot_length": len(dot), "candidate_name_present": "pv2_qkv_mixed_fused" in dot, "extra_vram_bytes": extra},
            "arms": {"BASE_A": base_a, "QKV": cand, "BASE_B": base_b},
            "parity": {"candidate_vs_base_a": par_a, "candidate_vs_base_b": par_b},
            "gates": gates, "adopt": all(bool(gates[k]) for k in required),
            "summary": {"base_a_p50_ms": pa, "candidate_p50_ms": pc,
                        "base_b_p50_ms": pb, "baseline_mid_p50_ms": mid,
                        "gain_ms": mid - pc, "candidate_tok_s": 1000.0 / pc,
                        "base_drift_ms": drift},
            "status": status_from_gates(gates, required), "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update(status="technical_failure", completed_utc=utc_now(),
                       error={"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()})
    finally:
        if bundle is not None: bundle.close()
    write_json(OUT, payload)
    print(json.dumps({"status": payload.get("status"), "adopt": payload.get("adopt"),
                      "summary": payload.get("summary"), "output": str(OUT)}, indent=2))
    return 0 if payload.get("status") in {"pass", "gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
