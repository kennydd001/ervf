"""ERVF at 32 lanes per row instead of 16 — the gating condition for 100 tok/s.

`diag_gemv_l2_vs_dram.json` established the honest numbers: this device streams
**345.9 GB/s**, but the dense GEMV kernel gets only **209-229 GB/s on a cold
working set** (60-66%). At 229 GB/s the per-token VRAM floor is 2048/229 =
8.94 ms, and with the down_proj's 2.47 ms of PCIe beside it the machine floor is
11.4 ms = 88 tok/s. **100 tok/s is unreachable until this kernel moves.**

## The suspect, and why 32 lanes should address it

`PRO_WIDTH 16` gives `PRO_ROWS_PER_BLOCK = 256/16 = 16`. Two consequences:

  * a warp's 32 threads span two different rows, so each load instruction emits
    two 64-byte requests -- **half a cacheline each** -- at addresses `cols`
    apart;
  * a block keeps 16 row streams open at once; with ~130 blocks resident that is
    ~2080 concurrent DRAM streams, each at its own row offset.

At `PRO_WIDTH 32` a warp covers exactly one row: 32 lanes x 4 B = **128 bytes,
one full cacheline per instruction**, and 8 row streams per block instead of 16.

## Why it can still be bit-exact

ERVF's whole point is reproducing the reference reduction tree under a different
physical geometry, and at width 32 the mapping is even more direct than at 16.

The production kernel used 256 threads per row: five shuffle steps (16,8,4,2,1)
inside each of 8 warps, then lane 0 combining the 8 warp sums as
`((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))`.

At width 16, virtual tid `t = lane + 16*vi` lands in reference warp `vi>>1` at
position `lane + 16*(vi&1)`, so the code has to fold the offset-16 step by hand
(`acc[2g] + acc[2g+1]`) before doing 16-wide shuffles.

At width 32, virtual tid `t = lane + 32*vj` lands in reference warp **exactly
`vj`**, at position exactly `lane`. So each accumulator `acc[vj]` IS reference
warp `vj`, its full 16/8/4/2/1 shuffle reduction is a plain 32-wide shuffle
reduction, and the same final tree closes it. Same elements per virtual tid
(`qidx == t (mod 256)`), same order, same tree -- bit-identical by construction,
and checked rather than argued.

## Arms

  P0  bit-exactness of width-32 against the production width-16 kernel on every
      real shape. A single differing bit stops the experiment.
  P1  cold-DRAM bandwidth, using diag_gemv_l2_vs_dram.py's rotation harness
      (enough distinct matrices that the working set is ~10x L2), so the L2
      artifact that inflated the earlier isolated figure cannot recur.

Read-only diagnostic on synthetic weights; random bytes exercise the whole
decode domain. Integration only follows if P0 and P1 both pass.
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
CYCLE = 12          # ~10x L2 for the Mamba shapes -- cold DRAM, no L2 reuse
SHAPES = [
    ("mamba_in_proj", 10304, 2688),
    ("mamba_out_proj", 2688, 4096),
    ("attn_qkv_like", 4608, 2688),
    ("attn_o_like", 2688, 4096),
]

SRC = r"""
__device__ __forceinline__ float pro_e4m3_decode(unsigned char x) {
    const int s = x >> 7, E = (x >> 3) & 0xF, m = x & 7;
    const float v = (E == 0) ? ((float)m * 1.953125e-3f)
                             : ((float)(8 + m) * exp2f((float)(E - 10)));
    return s ? -v : v;
}

// ---------------- production geometry: 16 lanes per row -------------------
#define W16 16
#define V16 (256 / W16)

__device__ __forceinline__ float reduce16(float acc[V16]) {
    const int lane = threadIdx.x & (W16 - 1);
    float s[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        float v = acc[g * 2] + acc[g * 2 + 1];
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o, W16);
        s[g] = v;
    }
    if (lane == 0) {
        const float a0 = s[0] + s[4], a1 = s[1] + s[5];
        const float a2 = s[2] + s[6], a3 = s[3] + s[7];
        return (a0 + a2) + (a1 + a3);
    }
    return 0.0f;
}

extern "C" __global__ void gemv_fp8_w16(
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

    const int sub = threadIdx.x / W16;
    const int lane = threadIdx.x & (W16 - 1);
    const int row = blockIdx.x * (256 / W16) + sub;
    const bool valid = row < rows;
    const unsigned char* w = W + (size_t)(valid ? row : 0) * cols;

    float acc[V16];
    #pragma unroll
    for (int vi = 0; vi < V16; ++vi) acc[vi] = 0.0f;
    const int nvec = cols >> 2;
    const uchar4* w4 = reinterpret_cast<const uchar4*>(w);
    #pragma unroll
    for (int vi = 0; vi < V16; ++vi) {
        const int tid = lane + W16 * vi;
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
    const float v = reduce16(acc);
    if (lane == 0 && valid) out[row] = v * wscale;
}

// ---------------- candidate geometry: 32 lanes per row --------------------
// Virtual tid t = lane + 32*vj lands in reference warp vj at position lane, so
// acc[vj] IS reference warp vj and a plain 32-wide shuffle reduction reproduces
// its 16/8/4/2/1 tree exactly. The final 8-way tree is byte-for-byte the same.
#define W32 32
#define V32 (256 / W32)

__device__ __forceinline__ float reduce32(float acc[V32]) {
    const int lane = threadIdx.x & (W32 - 1);
    float s[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        float v = acc[g];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o, W32);
        s[g] = v;
    }
    if (lane == 0) {
        const float a0 = s[0] + s[4], a1 = s[1] + s[5];
        const float a2 = s[2] + s[6], a3 = s[3] + s[7];
        return (a0 + a2) + (a1 + a3);
    }
    return 0.0f;
}

extern "C" __global__ void gemv_fp8_w32(
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

    const int sub = threadIdx.x / W32;
    const int lane = threadIdx.x & (W32 - 1);
    const int row = blockIdx.x * (256 / W32) + sub;
    const bool valid = row < rows;
    const unsigned char* w = W + (size_t)(valid ? row : 0) * cols;

    float acc[V32];
    #pragma unroll
    for (int vj = 0; vj < V32; ++vj) acc[vj] = 0.0f;
    const int nvec = cols >> 2;
    const uchar4* w4 = reinterpret_cast<const uchar4*>(w);
    #pragma unroll
    for (int vj = 0; vj < V32; ++vj) {
        const int tid = lane + W32 * vj;
        if (valid) {
            for (int qidx = tid; qidx < nvec; qidx += 256) {
                const uchar4 q = w4[qidx];
                const int k = qidx << 2;
                acc[vj] = fmaf(lut[q.x], sx[k],     acc[vj]);
                acc[vj] = fmaf(lut[q.y], sx[k + 1], acc[vj]);
                acc[vj] = fmaf(lut[q.z], sx[k + 2], acc[vj]);
                acc[vj] = fmaf(lut[q.w], sx[k + 3], acc[vj]);
            }
            for (int b = (nvec << 2) + tid; b < cols; b += 256)
                acc[vj] = fmaf(lut[w[b]], sx[b], acc[vj]);
        }
    }
    const float v = reduce32(acc);
    if (lane == 0 && valid) out[row] = v * wscale;
}
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    l2 = int(props.get("l2CacheSize", 0))
    mod = cp.RawModule(code=SRC, options=("-std=c++14",))
    k16 = mod.get_function("gemv_fp8_w16")
    k32 = mod.get_function("gemv_fp8_w32")

    rng = np.random.default_rng(20260816)
    arms = {}

    for name, rows, cols in SHAPES:
        wbytes = rows * cols
        mats = [cp.asarray(rng.integers(0, 256, size=wbytes, dtype=np.uint8))
                for _ in range(CYCLE)]
        x = cp.asarray(rng.standard_normal(cols).astype(np.float32))
        o16 = cp.zeros(rows, dtype=cp.float32)
        o32 = cp.zeros(rows, dtype=cp.float32)
        smem = (cols + 256) * 4
        b16 = (rows + 15) // 16      # 16 rows per block
        b32 = (rows + 7) // 8        # 8 rows per block

        # ---- P0: bit-exactness on every rotation matrix --------------------
        exact = True
        for m in mats:
            o16.fill(0)
            o32.fill(0)
            k16((b16,), (256,), (m, x, o16, np.float32(1.0), np.int32(rows), np.int32(cols)),
                shared_mem=smem)
            k32((b32,), (256,), (m, x, o32, np.float32(1.0), np.int32(rows), np.int32(cols)),
                shared_mem=smem)
            cp.cuda.Device(0).synchronize()
            if not np.array_equal(cp.asnumpy(o16).view(np.uint32),
                                  cp.asnumpy(o32).view(np.uint32)):
                exact = False
                break

        def timed(k, blocks, out):
            def run(i):
                k((blocks,), (256,),
                  (mats[i % CYCLE], x, out, np.float32(1.0),
                   np.int32(rows), np.int32(cols)), shared_mem=smem)
            run(0)
            cp.cuda.Device(0).synchronize()
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record()
            for i in range(ROUNDS):
                run(i)
            e1.record()
            e1.synchronize()
            ms = cp.cuda.get_elapsed_time(e0, e1) / ROUNDS
            return ms, wbytes / (ms * 1e-3) / 1e9

        ms16, gb16 = timed(k16, b16, o16)
        ms32, gb32 = timed(k32, b32, o32)
        arms[name] = {
            "rows": rows, "cols": cols, "weight_bytes": wbytes,
            "working_set_bytes": wbytes * CYCLE,
            "working_set_over_l2": (wbytes * CYCLE) / l2 if l2 else None,
            "bit_exact_w32_vs_w16": exact,
            "w16_ms": ms16, "w16_gb_s": gb16,
            "w32_ms": ms32, "w32_gb_s": gb32,
            "speedup": ms16 / ms32 if ms32 else None,
            "w16_frac_of_device": gb16 / 345.9,
            "w32_frac_of_device": gb32 / 345.9,
        }
        del mats, x, o16, o32
        cp.get_default_memory_pool().free_all_blocks()

    all_exact = all(v["bit_exact_w32_vs_w16"] for v in arms.values())
    sp = [v["speedup"] for v in arms.values() if v["speedup"]]
    best32 = max(v["w32_gb_s"] for v in arms.values())

    # what the new kernel rate would imply for the machine floor
    vram_mb, pcie_ms = 2048.0, 2.47
    floor_ms = vram_mb * 1e6 / (best32 * 1e9) * 1e3 + pcie_ms

    payload = {
        "kind": "diag_gemv_width32",
        "created_utc": utc_now(),
        "note": "32 lanes per row (one full cacheline per instruction, 8 row streams per block) against the production 16 (half a cacheline, 16 streams). Cold working set ~10x L2 so the L2 artifact that inflated the earlier isolated figure cannot recur.",
        "device": {"l2_cache_mib": l2 / 1024 / 1024 if l2 else None,
                   "read_gb_s_measured": 345.9},
        "rounds": ROUNDS, "matrices_in_rotation": CYCLE,
        "arms": arms,
        "summary": {
            "all_shapes_bit_exact": all_exact,
            "min_speedup": min(sp) if sp else None,
            "max_speedup": max(sp) if sp else None,
            "best_w32_gb_s": best32,
            "best_w32_frac_of_device": best32 / 345.9,
            "implied_machine_floor_ms_serial": floor_ms,
            "implied_machine_floor_tok_s_serial": 1000.0 / floor_ms,
            "note": "the implied floor assumes ALL 2048 MB/token move at the best measured GEMV rate and the 2.47 ms of down_proj PCIe stays serial; it is a floor, not a prediction of the runner",
        },
        "status": "measured" if all_exact else "correctness_failed",
    }
    write_json_atomic(REPO / "pro_research" / "diag_gemv_width32.json",
                      payload, archive=False)
    print(json.dumps(payload, indent=2))
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
