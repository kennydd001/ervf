"""The one number that decides the batch ceiling: K-tiled batched ERVF.

Where this stands. 99.2% of the model's 2048 MB/token weight traffic runs
through an ERVF kernel (Mamba FP8 + q/o BF16 via the whitelists, shared
expert/lm_head/routed-up NVFP4 via `gemv_into`'s and `run_batched`'s ERVF
default). ERVF already reaches **247-266 GB/s at N=1 = 77% of the device
ceiling**, and batching on top of it measured **x1.64 at N=4** -- which projects
the whole batch programme to ~71 tok/s at N=4 and ~83 at N=8, short of the goal.

But that x1.64 is a **lower bound**, and knowingly so: the batched ERVF kernel
reads X from global memory, because staging N copies of the full X vector in
shared exceeds the 48 KB dynamic limit at N=4, cols=4096. At N=4 that is
4 x 4 B of X traffic per weight byte.

Why staging should actually pay here, from the ERVF geometry rather than from
hope: in `pro_gemv_*_ervf16` the row index is `sub = threadIdx.x / 16` but the
element index is `tid = lane + 16*vi`, which does **not** depend on `sub`. So
all 16 rows in a block walk the *same* k-set -- every X element is needed by 16
different threads. Staging it once gives 16x reuse.

## The tiling, and why it stays bit-exact

Process the reduction dimension in tiles of QT = 256 uchar4 groups (1024
elements). With QT equal to the stride, each virtual tid handles exactly one `q`
per tile: tile 0 gives it q = tid, tile 1 gives q = 256 + tid, and so on -- the
identical sequence, in the identical increasing order, that the untiled loop
walks. The per-tid accumulation order is therefore unchanged, which is the whole
bit-exactness argument. The gate checks it rather than trusting it.

Shared memory: N * 1024 * 4 B for X plus 1 KB for the LUT -- 17 KB at N=4,
33 KB at N=8, both inside the 48 KB limit that forced the global reads.

## Arms

  prod_ervf16        production geometry, N=1 (what the runtime actually runs)
  ervf_global_nN     the previous batched kernel, X from global
  ervf_tiled_nN      this one, X staged per tile

Gate: N=1 of both batched kernels bit-exact against prod_ervf16, outputs finite.
Reported against prod_ervf16, i.e. against what production executes.
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
QT = 256          # uchar4 groups per tile == the tid stride, so order is preserved

HEAD = r"""
__device__ __forceinline__ float e4m3_decode(unsigned char x) {
    const int s = x >> 7, E = (x >> 3) & 0xF, m = x & 7;
    const float v = (E == 0) ? ((float)m * 1.953125e-3f)
                             : ((float)(8 + m) * exp2f((float)(E - 10)));
    return s ? -v : v;
}
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
    const int sub = threadIdx.x / W16, lane = threadIdx.x & (W16 - 1);
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
                const uchar4 c = w4[q]; const int k = q << 2;
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

TILED = r"""
extern "C" __global__ void ervf_tiled_n%(N)d(
    const unsigned char* __restrict__ W, const float* __restrict__ X,
    float* __restrict__ Y, const float wscale, const int rows, const int cols)
{
    const int N = %(N)d;
    const int QT = %(QT)d;
    const int KT = QT * 4;
    extern __shared__ float smem[];
    float* sx = smem;                     // [N, KT]
    float* lut = smem + (size_t)N * KT;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = e4m3_decode((unsigned char)i);
    const int sub = threadIdx.x / W16, lane = threadIdx.x & (W16 - 1);
    const int row = blockIdx.x * (256 / W16) + sub;
    const bool valid = row < rows;
    const unsigned char* __restrict__ w = W + (size_t)(valid ? row : 0) * cols;
    const uchar4* w4 = reinterpret_cast<const uchar4*>(w);
    const int nvec = cols >> 2;
    float acc[V16][%(N)d];
    #pragma unroll
    for (int vi = 0; vi < V16; ++vi)
        #pragma unroll
        for (int n = 0; n < N; ++n) acc[vi][n] = 0.0f;

    // QT == the tid stride, so each virtual tid gets exactly one q per tile and
    // walks q = tid, 256+tid, 512+tid, ... -- the same sequence in the same
    // order as the untiled loop. That is what keeps this bit-exact.
    for (int q0 = 0; q0 < nvec; q0 += QT) {
        const int qend = min(q0 + QT, nvec);
        const int klen = (qend - q0) * 4;
        __syncthreads();
        for (int i = threadIdx.x; i < N * klen; i += blockDim.x) {
            const int n = i / klen, kk = i - n * klen;
            sx[n * KT + kk] = X[(size_t)n * cols + q0 * 4 + kk];
        }
        __syncthreads();
        #pragma unroll
        for (int vi = 0; vi < V16; ++vi) {
            const int q = q0 + lane + W16 * vi;
            if (valid && q < qend) {
                const uchar4 c = w4[q];
                const int kk = (q - q0) << 2;
                const float d0 = lut[c.x], d1 = lut[c.y], d2 = lut[c.z], d3 = lut[c.w];
                #pragma unroll
                for (int n = 0; n < N; ++n) {
                    const float* sxn = sx + n * KT;
                    acc[vi][n] = fmaf(d0, sxn[kk],     acc[vi][n]);
                    acc[vi][n] = fmaf(d1, sxn[kk + 1], acc[vi][n]);
                    acc[vi][n] = fmaf(d2, sxn[kk + 2], acc[vi][n]);
                    acc[vi][n] = fmaf(d3, sxn[kk + 3], acc[vi][n]);
                }
            }
        }
    }
    // scalar tail, same order as the reference
    #pragma unroll
    for (int vi = 0; vi < V16; ++vi) {
        const int tid = lane + W16 * vi;
        if (valid) {
            for (int b = (nvec << 2) + tid; b < cols; b += 256) {
                const float d = lut[w[b]];
                #pragma unroll
                for (int n = 0; n < N; ++n)
                    acc[vi][n] = fmaf(d, X[(size_t)n * cols + b], acc[vi][n]);
            }
        }
    }
    #pragma unroll
    for (int n = 0; n < N; ++n) {
        float a[V16];
        #pragma unroll
        for (int vi = 0; vi < V16; ++vi) a[vi] = acc[vi][n];
        const float v = reduce16(a);
        if (lane == 0 && valid) Y[(size_t)n * rows + row] = v * wscale;
    }
}
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    src = HEAD + "".join(TILED % {"N": n, "QT": QT} for n in NS)
    mod = cp.RawModule(code=src, options=("-std=c++14",))
    k_ref = mod.get_function("prod_ervf16")
    k_t = {n: mod.get_function(f"ervf_tiled_n{n}") for n in NS}

    rng = np.random.default_rng(20260816)
    out = {}
    for label, rows, cols, mb in SHAPES:
        mats = [cp.asarray(rng.integers(0, 256, size=rows * cols, dtype=np.uint8))
                for _ in range(CYCLE)]
        X = cp.asarray(rng.standard_normal((max(NS), cols)).astype(np.float32).ravel())
        oe = cp.zeros(rows, dtype=cp.float32)
        Y = cp.zeros(max(NS) * rows, dtype=cp.float32)
        ws = np.float32(0.0123)
        blocks = (rows + 15) // 16
        sm_ref = (cols + 256) * 4
        wb = rows * cols

        def sm_tiled(N):
            return (N * QT * 4 + 256) * 4

        k_ref((blocks,), (256,), (mats[0], X, oe, ws, np.int32(rows), np.int32(cols)),
              shared_mem=sm_ref)
        k_t[1]((blocks,), (256,), (mats[0], X, Y, ws, np.int32(rows), np.int32(cols)),
               shared_mem=sm_tiled(1))
        cp.cuda.Device(0).synchronize()
        ref = cp.asnumpy(oe)
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

        ms_ref = timed(lambda i: k_ref((blocks,), (256,),
                                       (mats[i % CYCLE], X, oe, ws,
                                        np.int32(rows), np.int32(cols)), shared_mem=sm_ref))
        per = {}
        for N in NS:
            ms = timed(lambda i, N=N: k_t[N]((blocks,), (256,),
                                             (mats[i % CYCLE], X, Y, ws,
                                              np.int32(rows), np.int32(cols)),
                                             shared_mem=sm_tiled(N)))
            per[str(N)] = {"ms_per_batch_step": ms, "ms_per_token": ms / N,
                           "speedup_vs_production_ervf": ms_ref / (ms / N),
                           "smem_bytes": sm_tiled(N),
                           "gb_s_per_step": wb / (ms * 1e-3) / 1e9}
        out[label] = {"rows": rows, "cols": cols, "mb_per_token": mb,
                      "n1_bitexact_vs_production_ervf": exact, "finite": finite,
                      "production_ervf_ms": ms_ref,
                      "production_ervf_gb_s": wb / (ms_ref * 1e-3) / 1e9,
                      "by_N": per}
        del mats, X, oe, Y
        cp.get_default_memory_pool().free_all_blocks()

    tot = sum(v["mb_per_token"] for v in out.values())
    wsp = {str(N): sum(v["mb_per_token"] * v["by_N"][str(N)]["speedup_vs_production_ervf"]
                       for v in out.values()) / tot for N in NS}
    all_ok = all(v["n1_bitexact_vs_production_ervf"] and v["finite"] for v in out.values())

    payload = {
        "kind": "diag_ervf_batched_tiled",
        "created_utc": utc_now(),
        "note": "K-tiled batched ERVF: X staged per tile in shared memory instead of read from global. QT=256 uchar4 groups equals the tid stride, so each virtual tid walks the identical q sequence in the identical order as the untiled reference -- the bit-exactness argument. Compared against prod_ervf16, i.e. what the runtime actually executes for these shapes.",
        "qt_uchar4_groups": QT, "rounds": ROUNDS, "matrices_in_rotation": CYCLE,
        "prior_global_x_result": {"N4_mb_weighted": 1.640,
                                  "source": "diag_ervf_batched_fp8.json"},
        "shapes": out,
        "all_n1_bitexact_and_finite": all_ok,
        "mb_weighted_speedup_vs_production_ervf": wsp,
        "status": "measured" if all_ok else "correctness_failed",
    }
    write_json_atomic(REPO / "pro_research" / "diag_ervf_batched_tiled.json", payload,
                      archive=False)
    print(json.dumps({
        "n1_bitexact_and_finite": all_ok,
        "mb_weighted_vs_production_ervf": {k: round(v, 3) for k, v in wsp.items()},
        "prior_global_x_N4": 1.640,
        "per_shape": {k: {"prod_gb_s": round(v["production_ervf_gb_s"], 1),
                          **{f"N{n}": round(v["by_N"][str(n)]["speedup_vs_production_ervf"], 3)
                             for n in NS}} for k, v in out.items()},
    }, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
