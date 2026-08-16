"""Should (256, 2688) -- the K/V projections -- be on the ERVF whitelist?

The in-graph stage attribution measured `kv_proj` at **34.6 GB/s** against
`q_proj`'s 112.9 -- 3.3x worse per byte, well outside the ~0.3 ms noise floor.
My first reading was an occupancy problem, and that was **wrong**: K/V are
(256, 2688), which is not in `BF16_ERVF_SHAPES = {(4096,2688), (2688,4096)}`, so
they run on the PRODUCTION `gemv_bf16`, which launches **one block per row** --
256 blocks, not 16.

Reading that kernel gives the real reason:

    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];

**every block stages the entire x vector**. Per block that is 2688 floats =
10 752 B of x for only 2688 BF16 = 5 376 B of useful weights. Across 256 blocks:
2.75 MB of x traffic against 1.376 MB of W. The projection moves ~3x its own
weight bytes, which is exactly the kind of ratio that turns into 34.6 GB/s when
you score it against weight bytes alone.

ERVF-16 gives 16 rows per block, so for 256 rows it stages x **16 times instead
of 256** -- a 16x cut in the redundant traffic -- at the cost of 16 blocks on a
26-SM device. Which of those wins is an empirical question, hence this.

## Why it can be bit-exact

Traced both reduction trees by hand, then checked:

  production: 8 warps reduce with offsets 16,8,4,2,1; then warp 0 reduces the 8
  warp sums, where offsets 16 and 8 are no-ops (lanes 8..31 hold zero), leaving
  `((ws0+ws4)+(ws2+ws6)) + ((ws1+ws5)+(ws3+ws7))`.

  ERVF-16: `acc[2g] + acc[2g+1]` IS the offset-16 step (virtual tid t = lane+16*vi
  puts vi=2g at lanes 0-15 and vi=2g+1 at lanes 16-31 of warp g), then offsets
  8,4,2,1 as 16-wide shuffles, then
  `((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))`.

Identical, and the element partition matches too: production thread `tid` takes
k = tid, tid+256, ...; ERVF virtual tid t = lane+16*vi takes k = t, t+256, ...
Same k for the same virtual index. So this should hold for any (rows, cols) with
rows divisible by 16 and 256 threads per block -- 256/16 = 16 exactly.

That is an argument, not a result. The gate below decides it.

Arms: production `gemv_bf16` vs `pro_gemv_bf16_ervf16` on the real K/V shape,
plus Q's shape as a control (already whitelisted, so it should show the known
win) and a cold rotation so no L2 artifact can inflate either.
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
    ("kv_proj", 256, 2688),      # K and V -- NOT whitelisted today
    ("q_proj", 4096, 2688),      # control: already whitelisted
    ("o_proj", 2688, 4096),      # control: already whitelisted
]
CYCLE = 8
ROUNDS = 200
CALLS_PER_TOKEN = {"kv_proj": 12, "q_proj": 6, "o_proj": 6}   # 6 attention layers

SRC = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

// PRODUCTION: verbatim gemv_bf16 from gpu_kernels.py -- one block per row.
extern "C" __global__ void gemv_bf16_prod(
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
    __shared__ float ws[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) ws[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        float v = (lane < nw) ? ws[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) out[row] = v;
    }
}

// ERVF-16: verbatim pro_gemv_bf16_ervf16 from ervf_dense.py -- 16 rows per
// block, so x is staged rows/16 times instead of rows times.
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
        const float a0 = s[0] + s[4], a1 = s[1] + s[5];
        const float a2 = s[2] + s[6], a3 = s[3] + s[7];
        return (a0 + a2) + (a1 + a3);
    }
    return 0.0f;
}

extern "C" __global__ void gemv_bf16_ervf16(
    const unsigned short* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ out, const int rows, const int cols)
{
    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    const int sub = threadIdx.x / PRO_WIDTH;
    const int lane = threadIdx.x & (PRO_WIDTH - 1);
    const int row = blockIdx.x * PRO_ROWS_PER_BLOCK + sub;
    const bool valid = row < rows;
    const unsigned short* __restrict__ w = W + (size_t)(valid ? row : 0) * cols;
    float acc[PRO_VIRTUAL];
    #pragma unroll
    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) acc[vi] = 0.0f;
    #pragma unroll
    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) {
        const int tid = lane + PRO_WIDTH * vi;
        if (valid)
            for (int k = tid; k < cols; k += 256)
                acc[vi] = fmaf(bf16_to_f32(w[k]), sx[k], acc[vi]);
    }
    const float v = pro_reduce_exact(acc);
    if (lane == 0 && valid) out[row] = v;
}
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    mod = cp.RawModule(code=SRC, options=("-std=c++14",))
    k_prod = mod.get_function("gemv_bf16_prod")
    k_ervf = mod.get_function("gemv_bf16_ervf16")

    rng = np.random.default_rng(20260816)
    arms = {}

    for name, rows, cols in SHAPES:
        mats = [cp.asarray(rng.integers(0, 65536, size=rows * cols, dtype=np.uint16))
                for _ in range(CYCLE)]
        x = cp.asarray(rng.standard_normal(cols).astype(np.float32))
        o_p = cp.zeros(rows, dtype=cp.float32)
        o_e = cp.zeros(rows, dtype=cp.float32)
        smem = cols * 4
        bp, be = rows, (rows + 15) // 16

        exact = True
        for m in mats:
            o_p.fill(0)
            o_e.fill(0)
            k_prod((bp,), (256,), (m, x, o_p, np.int32(rows), np.int32(cols)), shared_mem=smem)
            k_ervf((be,), (256,), (m, x, o_e, np.int32(rows), np.int32(cols)), shared_mem=smem)
            cp.cuda.Device(0).synchronize()
            if not np.array_equal(cp.asnumpy(o_p).view(np.uint32),
                                  cp.asnumpy(o_e).view(np.uint32)):
                exact = False
                break

        def timed(k, blocks, out):
            def run(i):
                k((blocks,), (256,), (mats[i % CYCLE], x, out,
                                      np.int32(rows), np.int32(cols)), shared_mem=smem)
            run(0)
            cp.cuda.Device(0).synchronize()
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record()
            for i in range(ROUNDS):
                run(i)
            e1.record()
            e1.synchronize()
            return cp.cuda.get_elapsed_time(e0, e1) / ROUNDS

        ms_p, ms_e = timed(k_prod, bp, o_p), timed(k_ervf, be, o_e)
        calls = CALLS_PER_TOKEN[name]
        wbytes = rows * cols * 2
        arms[name] = {
            "rows": rows, "cols": cols,
            "whitelisted_today": (rows, cols) in {(4096, 2688), (2688, 4096)},
            "prod_blocks": bp, "ervf_blocks": be,
            "x_staged_bytes_prod": bp * cols * 4,
            "x_staged_bytes_ervf": be * cols * 4,
            "weight_bytes": wbytes,
            "bit_exact": exact,
            "prod_ms": ms_p, "ervf_ms": ms_e,
            "speedup": ms_p / ms_e if ms_e else None,
            "calls_per_token": calls,
            "saving_ms_per_token": (ms_p - ms_e) * calls,
        }
        del mats, x, o_p, o_e
        cp.get_default_memory_pool().free_all_blocks()

    all_exact = all(v["bit_exact"] for v in arms.values())
    kv = arms["kv_proj"]

    payload = {
        "kind": "diag_kv_proj_ervf",
        "created_utc": utc_now(),
        "note": "Is (256,2688) missing from BF16_ERVF_SHAPES? The production gemv_bf16 launches one block per row and each block stages the whole x vector, so at 256 rows it moves 2.75 MB of x for 1.376 MB of weights. ERVF-16 stages x rows/16 times instead of rows times. Cold rotation of 8 matrices so no L2 artifact can inflate either arm.",
        "current_whitelist": {"bf16": [[4096, 2688], [2688, 4096]]},
        "rounds": ROUNDS, "matrices_in_rotation": CYCLE,
        "arms": arms,
        "all_bit_exact": all_exact,
        "kv_proj_speedup": kv["speedup"],
        "kv_proj_saving_ms_per_token": kv["saving_ms_per_token"],
        "status": "measured" if all_exact else "correctness_failed",
    }
    write_json_atomic(REPO / "pro_research" / "diag_kv_proj_ervf.json", payload,
                      archive=False)
    print(json.dumps(payload, indent=2))
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
