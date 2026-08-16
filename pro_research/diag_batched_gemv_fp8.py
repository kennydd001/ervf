"""Correction + extension: batch scaling for the FP8 shapes, in the right dtype.

`diag_batched_gemv_scaling.json` measured all six shapes as **BF16**. That is
wrong for the two largest ones. The Mamba layer file is 38 782 608 B for
38.7M parameters -- **one byte per parameter, i.e. FP8 tensor-scaled**, which the
config confirms (`quantization_config` targets `mixer.in_proj` / `mixer.out_proj`
at 8 bits). Benchmarking them as BF16 doubles their byte count and therefore
misweights the headline MB-weighted average: those two shapes carry 890 of the
1686 MB/token that result covered.

That matters beyond bookkeeping. FP8 halves the bytes per parameter, so the
arithmetic intensity per byte is already twice BF16's, and batching could
saturate *earlier* -- exactly the kind of thing that would make the projection
optimistic. So it has to be measured rather than assumed to carry over.

Same method as the BF16 run, same gates:

  G1  at N=1 the batched kernel is bit-exact against the production
      `gemv_fp8_tensor` kernel, and all outputs are finite. (Random uint8 is
      safe here -- unlike random uint16 read as bf16, every e4m3 byte except
      0x7F/0xFF is a finite value, and the finiteness gate catches those.)
  G2  per-token time is ms_per_batch_step / N, never anything else.

Shapes: Mamba in_proj (10304, 2688) and out_proj (2688, 4096), FP8, which are
637.4 and 253.2 MB/token respectively in the real model -- 42% of all per-token
VRAM traffic between them.
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

SHAPES = [
    ("mamba_in_proj_fp8", 10304, 2688, 637.4),
    ("mamba_out_proj_fp8", 2688, 4096, 253.2),
]
NS = [1, 2, 4, 8]
CYCLE = 8
ROUNDS = 100

PROD = r"""
__device__ __forceinline__ float e4m3_decode(unsigned char x) {
    const int s = x >> 7, E = (x >> 3) & 0xF, m = x & 7;
    const float v = (E == 0) ? ((float)m * 1.953125e-3f)
                             : ((float)(8 + m) * exp2f((float)(E - 10)));
    return s ? -v : v;
}

// PRODUCTION reference geometry: one block per row, x staged in shared memory,
// FP8 decoded through a shared-memory table, one tensor-wide weight scale.
extern "C" __global__ void gemv_fp8_prod(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ out, const float wscale, const int rows, const int cols)
{
    extern __shared__ float smem[];
    float* sx = smem;
    float* lut = smem + cols;
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
"""

BATCH = r"""
extern "C" __global__ void gemv_fp8_batched_n%(N)d(
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
        const float wv = lut[w[k]];                  // one load + decode, N uses
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
        for (int o = warpSize >> 1; o > 0; o >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, o);
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

    src = PROD + "".join(BATCH % {"N": n} for n in NS)
    mod = cp.RawModule(code=src, options=("-std=c++14",))
    k_prod = mod.get_function("gemv_fp8_prod")
    k_batch = {n: mod.get_function(f"gemv_fp8_batched_n{n}") for n in NS}

    rng = np.random.default_rng(20260816)
    results = {}

    for label, rows, cols, mb_token in SHAPES:
        mats = [cp.asarray(rng.integers(0, 256, size=rows * cols, dtype=np.uint8))
                for _ in range(CYCLE)]
        X = cp.asarray(rng.standard_normal((max(NS), cols)).astype(np.float32).ravel())
        out_p = cp.zeros(rows, dtype=cp.float32)
        Y = cp.zeros(max(NS) * rows, dtype=cp.float32)
        wscale = np.float32(0.0123)
        smem_prod = (cols + 256) * 4
        smem_batch = 256 * 4
        wbytes = rows * cols

        out_p.fill(0)
        Y.fill(0)
        k_prod((rows,), (256,), (mats[0], X, out_p, wscale, np.int32(rows), np.int32(cols)),
               shared_mem=smem_prod)
        k_batch[1]((rows,), (256,), (mats[0], X, Y, wscale, np.int32(rows), np.int32(cols)),
                   shared_mem=smem_batch)
        cp.cuda.Device(0).synchronize()
        ref = cp.asnumpy(out_p)
        exact = bool(np.array_equal(ref.view(np.uint32),
                                    cp.asnumpy(Y[:rows]).view(np.uint32)))
        finite = bool(np.isfinite(ref).all())

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

        per_n = {}
        for N in NS:
            ms = timed(lambda i, N=N: k_batch[N]((rows,), (256,),
                                                 (mats[i % CYCLE], X, Y, wscale,
                                                  np.int32(rows), np.int32(cols)),
                                                 shared_mem=smem_batch))
            per_n[str(N)] = {"ms_per_batch_step": ms, "ms_per_token": ms / N,
                             "weight_gb_s": wbytes / (ms * 1e-3) / 1e9}
        base = per_n["1"]["ms_per_token"]
        for N in NS:
            per_n[str(N)]["speedup_per_token_vs_N1"] = base / per_n[str(N)]["ms_per_token"]

        results[label] = {
            "rows": rows, "cols": cols, "dtype": "fp8_e4m3_tensor_scaled",
            "mb_per_token_in_model": mb_token, "weight_bytes": wbytes,
            "n1_bitexact_vs_production": exact, "finite_outputs": finite,
            "by_N": per_n,
            "speedup_at_N4": per_n["4"]["speedup_per_token_vs_N1"],
            "speedup_at_N8": per_n["8"]["speedup_per_token_vs_N1"],
        }
        del mats, X, out_p, Y
        cp.get_default_memory_pool().free_all_blocks()

    all_exact = all(v["n1_bitexact_vs_production"] and v["finite_outputs"]
                    for v in results.values())
    tot = sum(v["mb_per_token_in_model"] for v in results.values())
    weighted = {str(N): sum(v["mb_per_token_in_model"] * v["by_N"][str(N)]["speedup_per_token_vs_N1"]
                            for v in results.values()) / tot for N in NS}

    payload = {
        "kind": "diag_batched_gemv_fp8",
        "created_utc": utc_now(),
        "note": "Correction to diag_batched_gemv_scaling, which measured the two Mamba shapes as BF16. They are FP8 tensor-scaled (38 782 608 B for 38.7M params = 1 byte/param; config quantization_config targets mixer.in_proj/out_proj at 8 bits). Those two carry 890.6 of the 1685.7 MB/token that result weighted, so the headline average needs this.",
        "supersedes_for_shapes": ["mamba_in_proj", "mamba_out_proj"],
        "rounds": ROUNDS, "matrices_in_rotation": CYCLE,
        "shapes": results,
        "all_n1_bitexact_and_finite": all_exact,
        "mb_weighted_speedup_per_token": weighted,
        "covered_mb_per_token": tot,
        "status": "measured" if all_exact else "correctness_failed",
    }
    write_json_atomic(REPO / "pro_research" / "diag_batched_gemv_fp8.json", payload,
                      archive=False)
    print(json.dumps({"all_n1_bitexact_and_finite": all_exact,
                      "mb_weighted_speedup_per_token": {k: round(v, 3) for k, v in weighted.items()},
                      "per_shape": {k: {"exact": v["n1_bitexact_vs_production"],
                                        "finite": v["finite_outputs"],
                                        "N4": round(v["speedup_at_N4"], 3),
                                        "N8": round(v["speedup_at_N8"], 3),
                                        "us_tok_N1": round(v["by_N"]["1"]["ms_per_token"] * 1000, 1),
                                        "us_tok_N8": round(v["by_N"]["8"]["ms_per_token"] * 1000, 1)}
                                    for k, v in results.items()}}, indent=2))
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
