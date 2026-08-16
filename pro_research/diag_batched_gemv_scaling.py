"""De-risk the whole batch program in one isolated measurement.

`agents/DECISION_SINGLE_STREAM_VS_BATCH.md` recommends abandoning single-stream
for PRO-E100-BATCH, and it rests on one physical claim:

    at N>1 a GEMV becomes a GEMM with small N, so the weight matrix is read once
    for N tokens instead of N times, and the kernels currently pinned at
    157-172 GB/s -- Mamba, up_proj, shared_expert, 1569 MB/token between them --
    stop being bandwidth-bound.

That claim is arithmetic, not a measurement. Before anyone spends weeks on a
runtime rewrite (every buffer in it is 1D and single-sequence), it should be
checked on this hardware, with these shapes. If the per-token cost does not fall
roughly N-fold, the batch program's entire justification is wrong and it is far
cheaper to learn that now.

This is the same discipline the single-stream work just paid for: nine
interventions were built off plausible arithmetic today and eight returned
nothing.

## What is measured

Y[N, rows] = W[rows, cols] @ X[N, cols], for N in {1, 2, 4, 8}, on the real
shapes, with a cold rotation of distinct weight matrices so no L2 artifact can
inflate the result (an earlier measurement today reported 336 GB/s that way and
it was wrong -- the honest cold figure is 230-261).

The kernel is the production `gemv_bf16` geometry with N accumulators per
thread: `w[k]` is loaded ONCE and used for all N inputs. That is exactly the
sharing the batch program is built on, so it is the right thing to price.

## Gates

  G1  at N=1 the batched kernel is bit-exact against the production kernel.
      (It reads x from global rather than shared memory, which changes nothing
      about the values or their order -- and the gate checks that rather than
      asserting it.)
  G2  reported per-token time is ms_per_batch_step / N, never anything else.

## What the numbers mean

`speedup_per_token` at N is the honest aggregate win from batching that shape.
If it tracks N, the decision note holds. If it saturates early, the shape has
hit a second limit (x traffic, occupancy, or compute) and the batch ceiling is
lower than the note claims -- which is exactly what this is for.
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

# (label, rows, cols, MB/token in the real model, which component)
SHAPES = [
    ("mamba_in_proj",  10304, 2688, 637.4, "Mamba"),
    ("mamba_out_proj",  2688, 4096, 253.2, "Mamba"),
    ("q_proj",          4096, 2688, 132.1, "attention"),
    ("o_proj",          2688, 4096, 132.1, "attention"),
    ("shared_up",       3712, 2688, 143.6, "shared expert"),
    ("routed_up",       1856, 2688, 387.3, "routed experts"),
]
NS = [1, 2, 4, 8]
CYCLE = 6
ROUNDS = 100

SRC_TMPL = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

// PRODUCTION reference: one block per row, x staged in shared memory.
extern "C" __global__ void gemv_prod(
    const unsigned short* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ out, const int rows, const int cols)
{
    extern __shared__ float sx[];
    const int row = blockIdx.x;
    if (row >= rows) return;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    const unsigned short* __restrict__ w = W + (size_t)row * cols;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < cols; k += blockDim.x)
        acc = fmaf(bf16_to_f32(w[k]), sx[k], acc);
    for (int o = warpSize >> 1; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float ws1[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) ws1[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        float v = (lane < nw) ? ws1[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) out[row] = v;
    }
}
"""

# One kernel per N, with N a COMPILE-TIME constant so the accumulator array is
# fully unrolled into registers. The first version took N as a runtime argument,
# which forced dynamic indexing of a local array and faulted at N>=4 -- a bug in
# the benchmark, not in the idea being tested.
BATCH_TMPL = r"""
extern "C" __global__ void gemv_batched_n%(N)d(
    const unsigned short* __restrict__ W, const float* __restrict__ X,
    float* __restrict__ Y, const int rows, const int cols)
{
    const int N = %(N)d;
    const int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short* __restrict__ w = W + (size_t)row * cols;
    float acc[%(N)d];
    #pragma unroll
    for (int n = 0; n < N; ++n) acc[n] = 0.0f;
    for (int k = threadIdx.x; k < cols; k += blockDim.x) {
        const float wv = bf16_to_f32(w[k]);          // one load, N uses
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
            if (lane == 0) Y[(size_t)n * rows + row] = v;
        }
    }
}
"""



def main() -> int:
    require_gpu_free()
    import cupy as cp

    src = SRC_TMPL + "".join(BATCH_TMPL % {"N": n} for n in NS)
    mod = cp.RawModule(code=src, options=("-std=c++14",))
    k_prod = mod.get_function("gemv_prod")
    k_batch = {n: mod.get_function(f"gemv_batched_n{n}") for n in NS}

    rng = np.random.default_rng(20260816)
    results = {}

    for label, rows, cols, mb_token, comp in SHAPES:
        # Realistic bf16: truncate real float32 values. Random uint16 would
        # land on NaN/Inf exponent patterns, and a bitwise output comparison
        # would then pass on garbage (NaN bits equal NaN bits).
        mats = [cp.asarray(((rng.standard_normal(rows * cols) * 0.05)
                            .astype(np.float32).view(np.uint32) >> 16).astype(np.uint16))
                for _ in range(CYCLE)]
        X = cp.asarray(rng.standard_normal((max(NS), cols)).astype(np.float32).ravel())
        out_p = cp.zeros(rows, dtype=cp.float32)
        Y = cp.zeros(max(NS) * rows, dtype=cp.float32)
        smem = cols * 4
        wbytes = rows * cols * 2

        # ---- G1: N=1 bit-exactness against production -------------------
        out_p.fill(0)
        Y.fill(0)
        k_prod((rows,), (256,), (mats[0], X, out_p, np.int32(rows), np.int32(cols)),
               shared_mem=smem)
        k_batch[1]((rows,), (256,), (mats[0], X, Y, np.int32(rows), np.int32(cols)))
        cp.cuda.Device(0).synchronize()
        exact = bool(np.array_equal(cp.asnumpy(out_p).view(np.uint32),
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

        ms_prod = timed(lambda i: k_prod((rows,), (256,),
                                         (mats[i % CYCLE], X, out_p,
                                          np.int32(rows), np.int32(cols)), shared_mem=smem))
        per_n = {}
        for N in NS:
            ms = timed(lambda i, N=N: k_batch[N]((rows,), (256,),
                                                 (mats[i % CYCLE], X, Y,
                                                  np.int32(rows), np.int32(cols))))
            per_n[str(N)] = {
                "ms_per_batch_step": ms,
                "ms_per_token": ms / N,
                "weight_gb_s": wbytes / (ms * 1e-3) / 1e9,
                "speedup_per_token_vs_N1": None,
            }
        base_tok = per_n["1"]["ms_per_token"]
        for N in NS:
            per_n[str(N)]["speedup_per_token_vs_N1"] = base_tok / per_n[str(N)]["ms_per_token"]

        results[label] = {
            "rows": rows, "cols": cols, "component": comp,
            "mb_per_token_in_model": mb_token,
            "weight_bytes": wbytes,
            "n1_bitexact_vs_production": exact,
            "finite_outputs": bool(np.isfinite(cp.asnumpy(out_p)).all()),
            "production_ms": ms_prod,
            "by_N": per_n,
            "speedup_at_N4": per_n["4"]["speedup_per_token_vs_N1"],
            "speedup_at_N8": per_n["8"]["speedup_per_token_vs_N1"],
        }
        del mats, X, out_p, Y
        cp.get_default_memory_pool().free_all_blocks()

    all_exact = all(v["n1_bitexact_vs_production"] for v in results.values())

    # weight the shapes by the MB/token they actually carry in the model
    tot_mb = sum(v["mb_per_token_in_model"] for v in results.values())
    weighted = {}
    for N in NS:
        w = sum(v["mb_per_token_in_model"] * v["by_N"][str(N)]["speedup_per_token_vs_N1"]
                for v in results.values()) / tot_mb
        weighted[str(N)] = w

    payload = {
        "kind": "diag_batched_gemv_scaling",
        "created_utc": utc_now(),
        "note": "Prices the physical claim the batch decision rests on: that at N>1 the weight matrix is read once for N tokens, so bandwidth-bound GEMVs stop being bandwidth-bound. Isolated, cold rotation of 6 distinct matrices per shape so no L2 artifact can inflate it. Per-token time is always ms_per_batch_step / N.",
        "decision_doc": "agents/DECISION_SINGLE_STREAM_VS_BATCH.md",
        "rounds": ROUNDS, "matrices_in_rotation": CYCLE,
        "shapes": results,
        "all_n1_bitexact": all_exact,
        "mb_weighted_speedup_per_token": weighted,
        "covered_mb_per_token": tot_mb,
        "status": "measured" if all_exact else "correctness_failed",
    }
    write_json_atomic(REPO / "pro_research" / "diag_batched_gemv_scaling.json",
                      payload, archive=False)
    print(json.dumps({"all_n1_bitexact": all_exact,
                      "mb_weighted_speedup_per_token": weighted,
                      "per_shape": {k: {"speedup_N4": round(v["speedup_at_N4"], 3),
                                        "speedup_N8": round(v["speedup_at_N8"], 3),
                                        "gb_s_N1": round(v["by_N"]["1"]["weight_gb_s"], 1),
                                        "gb_s_N8": round(v["by_N"]["8"]["weight_gb_s"], 1)}
                                    for k, v in results.items()}}, indent=2))
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
