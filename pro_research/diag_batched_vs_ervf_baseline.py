"""Is the batch speedup measured against the baseline production actually runs?

`diag_batched_gemv_fp8` compared its batched kernel against the **production
one-block-per-row** `gemv_fp8_tensor` geometry. But production does not use that
for Mamba's big projection: `(10304, 2688)` is in `FP8_ERVF_SHAPES`, so
`_install_selective` routes it to **ERVF-16**, and E5 measured ERVF at 1.6-1.9x
the old geometry on the shapes it whitelisted.

If ERVF is meaningfully faster at N=1, then every speedup reported so far is
against a baseline the runtime does not run, and the real batch win is smaller
by that factor. That would change the projection materially, so it has to be
checked before B1 is built on it.

Corroborating signal that something is off: the isolated FP8 N=1 figure was
297.6 us/token for 27.7 MB = **93 GB/s**, while the in-loop Mamba marginal
implies ~170 GB/s. The in-loop path being *faster* than the isolated one is the
signature of exactly this -- the loop runs ERVF, the benchmark did not.

## Arms (one variable: the N=1 reference)

  prod_rowblock   one block per row (what the earlier benchmark used)
  prod_ervf16     ERVF-16, which is what production actually runs for this shape
  batched_nN      the batched kernel, N in {1,2,4,8}

Reported both ways: speedup against the row-block baseline (comparable to the
earlier result) and against the ERVF baseline (the honest one). Gate: every arm
bit-exact against `prod_rowblock` at N=1, and finite.

Shapes: the two FP8 Mamba projections. `(10304, 2688)` is ERVF-whitelisted;
`(2688, 4096)` is also in FP8_ERVF_SHAPES, so both are affected.
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

SHAPES = [("mamba_in_proj", 10304, 2688, 637.4), ("mamba_out_proj", 2688, 4096, 253.2)]
NS = [1, 2, 4, 8]
CYCLE = 8
ROUNDS = 100

BASE = r"""
__device__ __forceinline__ float e4m3_decode(unsigned char x) {
    const int s = x >> 7, E = (x >> 3) & 0xF, m = x & 7;
    const float v = (E == 0) ? ((float)m * 1.953125e-3f)
                             : ((float)(8 + m) * exp2f((float)(E - 10)));
    return s ? -v : v;
}

// (a) one block per row -- the geometry the earlier benchmark used as baseline
extern "C" __global__ void prod_rowblock(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ out, const float wscale, const int rows, const int cols)
{
    extern __shared__ float smem[];
    float* sx = smem; float* lut = smem + cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = e4m3_decode((unsigned char)i);
    __syncthreads();
    const int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned char* __restrict__ w = W + (size_t)row * cols;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < cols; k += blockDim.x)
        acc = fmaf(lut[w[k]], sx[k], acc);
    for (int o = warpSize >> 1; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float ws1[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) ws1[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        float v = (lane < nw) ? ws1[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) out[row] = v * wscale;
    }
}

// (b) ERVF-16 -- verbatim pro_gemv_fp8_tensor_ervf16, what production runs here
#define W16 16
#define V16 (256 / W16)
__device__ __forceinline__ float reduce16(float acc[V16]) {
    const int lane = threadIdx.x & (W16 - 1);
    float s[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        float v = acc[g * 2] + acc[g * 2 + 1];
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o, W16);
        s[g] = v;
    }
    if (lane == 0) {
        const float a0 = s[0] + s[4], a1 = s[1] + s[5];
        const float a2 = s[2] + s[6], a3 = s[3] + s[7];
        return (a0 + a2) + (a1 + a3);
    }
    return 0.0f;
}
extern "C" __global__ void prod_ervf16(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ out, const float wscale, const int rows, const int cols)
{
    extern __shared__ float smem[];
    float* sx = smem; float* lut = smem + cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = e4m3_decode((unsigned char)i);
    __syncthreads();
    const int sub = threadIdx.x / W16;
    const int lane = threadIdx.x & (W16 - 1);
    const int row = blockIdx.x * (256 / W16) + sub;
    const bool valid = row < rows;
    const unsigned char* __restrict__ w = W + (size_t)(valid ? row : 0) * cols;
    float acc[V16];
    #pragma unroll
    for (int vi = 0; vi < V16; ++vi) acc[vi] = 0.0f;
    const int nvec = cols >> 2;
    const uchar4* w4 = reinterpret_cast<const uchar4*>(w);
    #pragma unroll
    for (int vi = 0; vi < V16; ++vi) {
        const int tid = lane + W16 * vi;
        if (valid) {
            for (int q = tid; q < nvec; q += 256) {
                const uchar4 c = w4[q];
                const int k = q << 2;
                acc[vi] = fmaf(lut[c.x], sx[k],     acc[vi]);
                acc[vi] = fmaf(lut[c.y], sx[k + 1], acc[vi]);
                acc[vi] = fmaf(lut[c.z], sx[k + 2], acc[vi]);
                acc[vi] = fmaf(lut[c.w], sx[k + 3], acc[vi]);
            }
            for (int b = (nvec << 2) + tid; b < cols; b += 256)
                acc[vi] = fmaf(lut[w[b]], sx[b], acc[vi]);
        }
    }
    const float v = reduce16(acc);
    if (lane == 0 && valid) out[row] = v * wscale;
}
"""

BATCH = r"""
extern "C" __global__ void batched_n%(N)d(
    const unsigned char* __restrict__ W, const float* __restrict__ X,
    float* __restrict__ Y, const float wscale, const int rows, const int cols)
{
    const int N = %(N)d;
    extern __shared__ float lut[];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = e4m3_decode((unsigned char)i);
    __syncthreads();
    const int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned char* __restrict__ w = W + (size_t)row * cols;
    float acc[%(N)d];
    #pragma unroll
    for (int n = 0; n < N; ++n) acc[n] = 0.0f;
    for (int k = threadIdx.x; k < cols; k += blockDim.x) {
        const float wv = lut[w[k]];
        #pragma unroll
        for (int n = 0; n < N; ++n)
            acc[n] = fmaf(wv, X[(size_t)n * cols + k], acc[n]);
    }
    __shared__ float ws[%(N)d][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nw = (blockDim.x + 31) >> 5;
    #pragma unroll
    for (int n = 0; n < N; ++n) {
        float a = acc[n];
        for (int o = warpSize >> 1; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) ws[n][warp] = a;
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int n = 0; n < N; ++n) {
            float v = (lane < nw) ? ws[n][lane] : 0.0f;
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0) Y[(size_t)n * rows + row] = v * wscale;
        }
    }
}
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    mod = cp.RawModule(code=BASE + "".join(BATCH % {"N": n} for n in NS),
                       options=("-std=c++14",))
    k_row = mod.get_function("prod_rowblock")
    k_ervf = mod.get_function("prod_ervf16")
    k_b = {n: mod.get_function(f"batched_n{n}") for n in NS}

    rng = np.random.default_rng(20260816)
    results = {}

    for label, rows, cols, mb in SHAPES:
        mats = [cp.asarray(rng.integers(0, 256, size=rows * cols, dtype=np.uint8))
                for _ in range(CYCLE)]
        X = cp.asarray(rng.standard_normal((max(NS), cols)).astype(np.float32).ravel())
        o_row = cp.zeros(rows, dtype=cp.float32)
        o_erv = cp.zeros(rows, dtype=cp.float32)
        Y = cp.zeros(max(NS) * rows, dtype=cp.float32)
        ws = np.float32(0.0123)
        sm_full, sm_lut = (cols + 256) * 4, 256 * 4
        wb = rows * cols

        k_row((rows,), (256,), (mats[0], X, o_row, ws, np.int32(rows), np.int32(cols)),
              shared_mem=sm_full)
        k_ervf(((rows + 15) // 16,), (256,),
               (mats[0], X, o_erv, ws, np.int32(rows), np.int32(cols)), shared_mem=sm_full)
        k_b[1]((rows,), (256,), (mats[0], X, Y, ws, np.int32(rows), np.int32(cols)),
               shared_mem=sm_lut)
        cp.cuda.Device(0).synchronize()
        ref = cp.asnumpy(o_row)
        ervf_exact = bool(np.array_equal(ref.view(np.uint32),
                                         cp.asnumpy(o_erv).view(np.uint32)))
        batch_exact = bool(np.array_equal(ref.view(np.uint32),
                                          cp.asnumpy(Y[:rows]).view(np.uint32)))

        def timed(fn):
            fn(0)
            cp.cuda.Device(0).synchronize()
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record()
            for i in range(ROUNDS):
                fn(i)
            e1.record()
            e1.synchronize()
            return cp.cuda.get_elapsed_time(e0, e1) / ROUNDS

        ms_row = timed(lambda i: k_row((rows,), (256,),
                                       (mats[i % CYCLE], X, o_row, ws,
                                        np.int32(rows), np.int32(cols)), shared_mem=sm_full))
        ms_erv = timed(lambda i: k_ervf(((rows + 15) // 16,), (256,),
                                        (mats[i % CYCLE], X, o_erv, ws,
                                         np.int32(rows), np.int32(cols)), shared_mem=sm_full))
        per_n = {}
        for N in NS:
            ms = timed(lambda i, N=N: k_b[N]((rows,), (256,),
                                             (mats[i % CYCLE], X, Y, ws,
                                              np.int32(rows), np.int32(cols)),
                                             shared_mem=sm_lut))
            per_n[str(N)] = {
                "ms_per_batch_step": ms,
                "ms_per_token": ms / N,
                "speedup_vs_rowblock": ms_row / (ms / N),
                "speedup_vs_ervf_PRODUCTION": ms_erv / (ms / N),
            }

        results[label] = {
            "rows": rows, "cols": cols, "mb_per_token_in_model": mb,
            "ervf_bitexact_vs_rowblock": ervf_exact,
            "batched_n1_bitexact_vs_rowblock": batch_exact,
            "finite": bool(np.isfinite(ref).all()),
            "rowblock_ms": ms_row, "ervf_ms": ms_erv,
            "ervf_speedup_over_rowblock": ms_row / ms_erv,
            "rowblock_gb_s": wb / (ms_row * 1e-3) / 1e9,
            "ervf_gb_s": wb / (ms_erv * 1e-3) / 1e9,
            "by_N": per_n,
        }
        del mats, X, o_row, o_erv, Y
        cp.get_default_memory_pool().free_all_blocks()

    tot = sum(v["mb_per_token_in_model"] for v in results.values())
    w_row = {str(N): sum(v["mb_per_token_in_model"] * v["by_N"][str(N)]["speedup_vs_rowblock"]
                         for v in results.values()) / tot for N in NS}
    w_erv = {str(N): sum(v["mb_per_token_in_model"] * v["by_N"][str(N)]["speedup_vs_ervf_PRODUCTION"]
                         for v in results.values()) / tot for N in NS}
    all_exact = all(v["ervf_bitexact_vs_rowblock"] and v["batched_n1_bitexact_vs_rowblock"]
                    and v["finite"] for v in results.values())

    payload = {
        "kind": "diag_batched_vs_ervf_baseline",
        "created_utc": utc_now(),
        "note": "The earlier batch measurements used the one-block-per-row FP8 kernel as the N=1 baseline, but production routes these shapes to ERVF-16 via FP8_ERVF_SHAPES. Speedup against a baseline the runtime does not run overstates the batch win. Reported both ways; speedup_vs_ervf_PRODUCTION is the honest one.",
        "rounds": ROUNDS, "matrices_in_rotation": CYCLE,
        "shapes": results,
        "all_bitexact_and_finite": all_exact,
        "mb_weighted_speedup_vs_rowblock": w_row,
        "mb_weighted_speedup_vs_ervf_PRODUCTION": w_erv,
        "status": "measured" if all_exact else "correctness_failed",
    }
    write_json_atomic(REPO / "pro_research" / "diag_batched_vs_ervf_baseline.json",
                      payload, archive=False)
    print(json.dumps({
        "all_bitexact_and_finite": all_exact,
        "ervf_speedup_over_rowblock": {k: round(v["ervf_speedup_over_rowblock"], 3)
                                       for k, v in results.items()},
        "gb_s_N1": {k: {"rowblock": round(v["rowblock_gb_s"], 1),
                        "ervf": round(v["ervf_gb_s"], 1)} for k, v in results.items()},
        "mb_weighted_vs_rowblock": {k: round(v, 3) for k, v in w_row.items()},
        "mb_weighted_vs_ervf_PRODUCTION": {k: round(v, 3) for k, v in w_erv.items()},
    }, indent=2))
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
