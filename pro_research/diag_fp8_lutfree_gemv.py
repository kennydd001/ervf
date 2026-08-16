"""Is the shared-memory FP8 decode LUT the reason dense GEMVs run at 37.8% of
DRAM bandwidth?

## Why this is the biggest remaining lever

Exact per-token byte accounting (safetensors headers, validated by reproducing
the project's 165 tok/s roofline to three digits):

    Mamba        892.0 MB/token  43.6%   FP8 projections
    attention    280.8 MB/token  13.7%   BF16 projections
    shared+gate  290.0 MB/token  14.2%
    lm_head      198.2 MB/token   9.7%
    ------------------------------------ 1661 of 2048 MB go through dense GEMV

E5 measured that GEMV suite at a weighted **127.9 GB/s**. An independent check
today (diag_vram_bandwidth_check.json, 512 MiB buffers, byte-verified) puts this
device at **345.9 GB/s** read. So the dense GEMVs run at **37.0%** of what the
memory system delivers, on 81% of the model's per-token bytes. Nothing else in
the system is that large and that far from its ceiling.

## The specific suspect

`pro_gemv_fp8_tensor_ervf16` (pro_research/ervf_dense.py) stages a 256-entry
e4m3 decode table in shared memory and then does, per uchar4 of weights:

    acc = fmaf(lut[q.x], sx[k],   acc);   ... and three more

`q.x` is a weight byte, so the index is data-dependent and effectively random.
Sixteen lanes issuing four random 256-entry shared-memory lookups each is a
bank-conflict generator on the innermost line of the hottest loop in the model.
At 892 MB of FP8 for Mamba alone that is ~892M conflicted lookups per token.

E4M3 decoding does not need a table. For E != 0 the IEEE-754 bit layout falls
straight out: exponent field E+120, mantissa m<<20. Only the E == 0 subnormal
branch needs arithmetic, and it is one int-to-float and one multiply. Six ALU
ops, no memory traffic, no divergence if written branchlessly.

## Arms

  P0  exhaustive decode equivalence: all 256 byte values, LUT vs arithmetic.
      A single mismatch stops everything -- there is no point timing a kernel
      that computes different numbers.
  P1  the two real Mamba shapes (in_proj 10304x2688, out_proj 2688x4096) and
      the attention/lm_head-sized shapes, reference kernel vs LUT-free kernel:
      outputs compared bit-exact, then GB/s.

Read-only diagnostic on synthetic weights -- random bytes exercise the whole
decode domain, which real weights would not. It is a kernel-level bandwidth
question; integration only follows if P0 and P1 both pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_gpu_free, utc_now, write_json_atomic

ROUNDS = 200

SHAPES = [
    ("mamba_in_proj", 10304, 2688),
    ("mamba_out_proj", 2688, 4096),
    ("attn_qkv_like", 4608, 2688),
    ("attn_o_like", 2688, 4096),
]

SRC = r"""
// Verbatim from pro_research/ervf_dense.py -- the reference decode.
__device__ __forceinline__ float pro_e4m3_decode(unsigned char x) {
    const int s = x >> 7, E = (x >> 3) & 0xF, m = x & 7;
    const float v = (E == 0) ? ((float)m * 1.953125e-3f)
                             : ((float)(8 + m) * exp2f((float)(E - 10)));
    return s ? -v : v;
}

// LUT-free. For E != 0, (8+m)*2^(E-10) == (1 + m/8)*2^(E-7), which is exactly
// the IEEE-754 single with exponent field E-7+127 = E+120 and mantissa m<<20.
// The E == 0 subnormal case is m*2^-9, one int-to-float and one multiply.
// Branchless so no warp diverges on weight content.
__device__ __forceinline__ float e4m3_arith(unsigned char x) {
    const unsigned int s = ((unsigned int)(x & 0x80)) << 24;
    const int E = (x >> 3) & 0xF;
    const int m = x & 7;
    const float norm = __uint_as_float(((unsigned int)(E + 120) << 23)
                                       | ((unsigned int)m << 20));
    const float sub = (float)m * 1.953125e-3f;
    const float v = (E == 0) ? sub : norm;
    return __uint_as_float(__float_as_uint(v) | s);
}

extern "C" __global__ void decode_table(float* lut, float* arith) {
    const int i = threadIdx.x;
    if (i < 256) {
        lut[i] = pro_e4m3_decode((unsigned char)i);
        arith[i] = e4m3_arith((unsigned char)i);
    }
}

#define PRO_WIDTH 16
#define PRO_VIRTUAL (256 / PRO_WIDTH)
#define PRO_ROWS_PER_BLOCK (256 / PRO_WIDTH)

__device__ __forceinline__ float pro_reduce_exact(float acc[PRO_VIRTUAL]) {
    const int lane = threadIdx.x & (PRO_WIDTH - 1);
    float s[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        float v = acc[g * 2] + acc[g * 2 + 1];
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o, PRO_WIDTH);
        s[g] = v;
    }
    if (lane == 0) {
        const float a0 = s[0] + s[4];
        const float a1 = s[1] + s[5];
        const float a2 = s[2] + s[6];
        const float a3 = s[3] + s[7];
        const float b0 = a0 + a2;
        const float b1 = a1 + a3;
        return b0 + b1;
    }
    return 0.0f;
}

// REFERENCE: verbatim pro_gemv_fp8_tensor_ervf16.
extern "C" __global__ void gemv_fp8_ref(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ out, const float wscale, const int rows, const int cols)
{
    extern __shared__ float smem[];
    float* sx = smem;
    float* lut = smem + cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = pro_e4m3_decode((unsigned char)i);
    __syncthreads();

    const int sub = threadIdx.x / PRO_WIDTH;
    const int lane = threadIdx.x & (PRO_WIDTH - 1);
    const int row = blockIdx.x * PRO_ROWS_PER_BLOCK + sub;
    const bool valid = row < rows;
    const unsigned char* w = W + (size_t)(valid ? row : 0) * cols;

    float acc[PRO_VIRTUAL];
    #pragma unroll
    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) acc[vi] = 0.0f;
    const int nvec = cols >> 2;
    const uchar4* w4 = reinterpret_cast<const uchar4*>(w);
    #pragma unroll
    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) {
        const int tid = lane + PRO_WIDTH * vi;
        if (valid) {
            for (int qidx = tid; qidx < nvec; qidx += 256) {
                const uchar4 q = w4[qidx];
                const int k = qidx << 2;
                acc[vi] = fmaf(lut[q.x], sx[k],     acc[vi]);
                acc[vi] = fmaf(lut[q.y], sx[k + 1], acc[vi]);
                acc[vi] = fmaf(lut[q.z], sx[k + 2], acc[vi]);
                acc[vi] = fmaf(lut[q.w], sx[k + 3], acc[vi]);
            }
            for (int b = (nvec << 2) + tid; b < cols; b += 256)
                acc[vi] = fmaf(lut[w[b]], sx[b], acc[vi]);
        }
    }
    const float v = pro_reduce_exact(acc);
    if (lane == 0 && valid) out[row] = v * wscale;
}

// CANDIDATE: identical in every respect except that the decode is computed
// instead of looked up. Same loop structure, same fmaf order, same reduction,
// so the result must be bit-identical whenever P0 holds.
extern "C" __global__ void gemv_fp8_lutfree(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ out, const float wscale, const int rows, const int cols)
{
    extern __shared__ float smem[];
    float* sx = smem;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __syncthreads();

    const int sub = threadIdx.x / PRO_WIDTH;
    const int lane = threadIdx.x & (PRO_WIDTH - 1);
    const int row = blockIdx.x * PRO_ROWS_PER_BLOCK + sub;
    const bool valid = row < rows;
    const unsigned char* w = W + (size_t)(valid ? row : 0) * cols;

    float acc[PRO_VIRTUAL];
    #pragma unroll
    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) acc[vi] = 0.0f;
    const int nvec = cols >> 2;
    const uchar4* w4 = reinterpret_cast<const uchar4*>(w);
    #pragma unroll
    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) {
        const int tid = lane + PRO_WIDTH * vi;
        if (valid) {
            for (int qidx = tid; qidx < nvec; qidx += 256) {
                const uchar4 q = w4[qidx];
                const int k = qidx << 2;
                acc[vi] = fmaf(e4m3_arith(q.x), sx[k],     acc[vi]);
                acc[vi] = fmaf(e4m3_arith(q.y), sx[k + 1], acc[vi]);
                acc[vi] = fmaf(e4m3_arith(q.z), sx[k + 2], acc[vi]);
                acc[vi] = fmaf(e4m3_arith(q.w), sx[k + 3], acc[vi]);
            }
            for (int b = (nvec << 2) + tid; b < cols; b += 256)
                acc[vi] = fmaf(e4m3_arith(w[b]), sx[b], acc[vi]);
        }
    }
    const float v = pro_reduce_exact(acc);
    if (lane == 0 && valid) out[row] = v * wscale;
}
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    mod = cp.RawModule(code=SRC, options=("-std=c++14",))
    k_tab = mod.get_function("decode_table")
    k_ref = mod.get_function("gemv_fp8_ref")
    k_cand = mod.get_function("gemv_fp8_lutfree")

    # ---- P0: exhaustive decode equivalence over all 256 byte values -------
    lut = cp.zeros(256, dtype=cp.float32)
    ari = cp.zeros(256, dtype=cp.float32)
    k_tab((1,), (256,), (lut, ari))
    cp.cuda.Device(0).synchronize()
    lut_h, ari_h = cp.asnumpy(lut), cp.asnumpy(ari)
    bitwise = lut_h.view(np.uint32) == ari_h.view(np.uint32)
    p0_pass = bool(bitwise.all())
    mismatches = [{"byte": int(i), "lut": float(lut_h[i]), "arith": float(ari_h[i])}
                  for i in np.flatnonzero(~bitwise)[:16]]

    payload = {
        "kind": "diag_fp8_lutfree_gemv",
        "created_utc": utc_now(),
        "note": "Does the shared-memory e4m3 decode LUT cost bandwidth in the dense FP8 GEMV? Dense GEMV carries 1661 of the model's 2048 MB/token and E5 measured it at 127.9 GB/s against a device that delivers 345.9 GB/s (verified today).",
        "P0_decode_equivalence": {
            "all_256_bitwise_identical": p0_pass,
            "mismatches": mismatches,
        },
    }

    if not p0_pass:
        payload["status"] = "P0_failed"
        write_json_atomic(REPO / "pro_research" / "diag_fp8_lutfree_gemv.json",
                          payload, archive=False)
        print(json.dumps(payload, indent=2))
        return 2

    # ---- P1: the real shapes ---------------------------------------------
    rng = np.random.default_rng(20260816)
    arms = {}
    for name, rows, cols in SHAPES:
        W = cp.asarray(rng.integers(0, 256, size=rows * cols, dtype=np.uint8))
        x = cp.asarray(rng.standard_normal(cols).astype(np.float32))
        o_ref = cp.zeros(rows, dtype=cp.float32)
        o_cand = cp.zeros(rows, dtype=cp.float32)
        blocks = (rows + 15) // 16
        smem_ref = (cols + 256) * 4
        smem_cand = cols * 4
        args_ref = (W, x, o_ref, np.float32(1.0), np.int32(rows), np.int32(cols))
        args_cand = (W, x, o_cand, np.float32(1.0), np.int32(rows), np.int32(cols))

        k_ref((blocks,), (256,), args_ref, shared_mem=smem_ref)
        k_cand((blocks,), (256,), args_cand, shared_mem=smem_cand)
        cp.cuda.Device(0).synchronize()
        exact = bool(np.array_equal(cp.asnumpy(o_ref).view(np.uint32),
                                    cp.asnumpy(o_cand).view(np.uint32)))

        def timed(fn):
            fn()
            cp.cuda.Device(0).synchronize()
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record()
            for _ in range(ROUNDS):
                fn()
            e1.record()
            e1.synchronize()
            ms = cp.cuda.get_elapsed_time(e0, e1) / ROUNDS
            return ms, (rows * cols) / (ms * 1e-3) / 1e9

        ms_r, gb_r = timed(lambda: k_ref((blocks,), (256,), args_ref, shared_mem=smem_ref))
        ms_c, gb_c = timed(lambda: k_cand((blocks,), (256,), args_cand, shared_mem=smem_cand))
        arms[name] = {
            "rows": rows, "cols": cols, "weight_bytes": rows * cols,
            "bit_exact": exact,
            "ref_ms": ms_r, "ref_gb_s": gb_r,
            "lutfree_ms": ms_c, "lutfree_gb_s": gb_c,
            "speedup": ms_r / ms_c if ms_c else None,
            "smem_saved_bytes_per_block": 1024,
        }
        del W, x, o_ref, o_cand
        cp.get_default_memory_pool().free_all_blocks()

    all_exact = all(v["bit_exact"] for v in arms.values())
    speedups = [v["speedup"] for v in arms.values() if v["speedup"]]
    payload.update({
        "P1_shapes": arms,
        "summary": {
            "all_shapes_bit_exact": all_exact,
            "min_speedup": min(speedups) if speedups else None,
            "max_speedup": max(speedups) if speedups else None,
            "ref_gb_s_range": [min(v["ref_gb_s"] for v in arms.values()),
                               max(v["ref_gb_s"] for v in arms.values())],
            "lutfree_gb_s_range": [min(v["lutfree_gb_s"] for v in arms.values()),
                                   max(v["lutfree_gb_s"] for v in arms.values())],
            "device_read_gb_s_measured_today": 345.9,
        },
        "status": "measured" if all_exact else "correctness_failed",
    })
    write_json_atomic(REPO / "pro_research" / "diag_fp8_lutfree_gemv.json",
                      payload, archive=False)
    print(json.dumps(payload, indent=2))
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
