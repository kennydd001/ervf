"""One thread per weight BYTE instead of per output row.

`diag_down_masked_chain` cornered the problem. Sweeping `nchunks` (which is
`gridDim.y`, and the panel loop strides by it, so it sets the dependent-chain
length per thread without changing total work):

    nchunks   blocks   loads/thread   ms/token
       2         42        124.5        4.004
       4         84         62.2        2.535
       8        168         31.1        1.549   <- production, the knee
      16        336         15.6        1.772
      32        672          7.8        1.603
      64       1344          3.9        1.613

Below 8 it is chain-limited (each halving of the chain halves the time); at and
above 8 it is **flat**. So the kernel has a hard floor near 1.55 ms that more
parallelism cannot move -- and it is not bandwidth (64 MB / 249 GB/s = 0.26 ms),
not instructions (~0.08 ms), not launch overhead or occupancy (V15 gave 6x the
blocks in one launch, neutral).

## What the access pattern actually does

    hb = row >> 1;  hi = row & 1;
    byte = pcodes[c * rowhalf + hb];
    w = e2m1[hi ? (byte >> 4) : (byte & 15)] * s;

Rows 2k and 2k+1 compute `hb = k`, so **two threads load the same byte** and
each keeps one nibble. Within a warp, 32 threads therefore touch only 16
distinct byte addresses: every warp-load requests 16 useful bytes, and half the
load instructions in the hottest loop of the model are literally redundant.

## The candidate

Let one thread own the byte, i.e. rows 2t and 2t+1, and carry two accumulators.
That halves the global load instructions and doubles the useful bytes per
request. Each row's accumulator still visits the same panels in the same order
with the same `fmaf` sequence, so both rows come out **bit-identical** -- this
reorganises which thread does the work, not what is computed. The candidate is
compared byte-for-byte against the reference before any timing is read.

Row count is even (2688), so there is no tail case; the kernel asserts it by
construction (`rowhalf = rows >> 1` threads cover all rows exactly).

Fewer threads means fewer blocks for the same nchunks, and the sweep above shows
the knee sits at ~168 blocks -- so nchunks is swept here too rather than fixed,
because halving the thread count could otherwise push the candidate below the
knee and hide a real win.

Timing-only for the reference/candidate comparison of `partials`; the reduce
step is not run.
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

HIDDEN = 2688
INTER = 1856
NPANEL = INTER // 16
NZ_FRACTION = 0.09
TOP_K = 6
MOE_LAYERS = 23
CHUNKS = [8, 16, 32]
ROUNDS = 30

SRC = r"""
// REFERENCE: verbatim gemv_down_masked_partial_ind.
extern "C" __global__ void down_masked_ref(
    const unsigned char* __restrict__ bank, const int* __restrict__ id_ptr,
    const float* __restrict__ globals, const float* __restrict__ act,
    const int* __restrict__ panel_list, const unsigned int* __restrict__ panel_masks,
    const int* __restrict__ panel_count, const float* __restrict__ e2m1_lut,
    const float* __restrict__ e4m3_lut, float* __restrict__ partials,
    const int rows, const int inter)
{
    const float global_scale = globals[(*id_ptr) * 2 + 0];
    const int nchunks = gridDim.y;
    const int chunk = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    __shared__ float s_e2m1[16];
    __shared__ float s_e4m3[256];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    if (threadIdx.x < 256) s_e4m3[threadIdx.x] = e4m3_lut[threadIdx.x];
    __syncthreads();
    const int hb = row >> 1, hi = row & 1, rowhalf = rows >> 1;
    const int pcount = *panel_count;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    float acc = 0.0f;
    for (int pi = chunk; pi < pcount; pi += nchunks) {
        const int p = panel_list[pi];
        const unsigned char* __restrict__ pbase = bank + (size_t)p * panel_stride;
        const float s = s_e4m3[pbase[row]] * global_scale;
        const unsigned char* __restrict__ pcodes = pbase + rows;
        unsigned int m = panel_masks[p];
        while (m) {
            const int c = __ffs(m) - 1; m &= m - 1;
            const unsigned char byte = pcodes[(size_t)c * rowhalf + hb];
            const float w = s_e2m1[hi ? (byte >> 4) : (byte & 15)] * s;
            acc = fmaf(w, act[(p << 4) + c], acc);
        }
    }
    partials[(size_t)chunk * rows + row] = acc;
}

// CANDIDATE: one thread per weight BYTE, i.e. rows 2t and 2t+1, two
// accumulators. Same panels in the same order, same fmaf sequence per row --
// only the thread-to-row mapping changes.
extern "C" __global__ void down_masked_byterow(
    const unsigned char* __restrict__ bank, const int* __restrict__ id_ptr,
    const float* __restrict__ globals, const float* __restrict__ act,
    const int* __restrict__ panel_list, const unsigned int* __restrict__ panel_masks,
    const int* __restrict__ panel_count, const float* __restrict__ e2m1_lut,
    const float* __restrict__ e4m3_lut, float* __restrict__ partials,
    const int rows, const int inter)
{
    const float global_scale = globals[(*id_ptr) * 2 + 0];
    const int nchunks = gridDim.y;
    const int chunk = blockIdx.y;
    const int hb = blockIdx.x * blockDim.x + threadIdx.x;   // byte index
    const int rowhalf = rows >> 1;
    if (hb >= rowhalf) return;
    __shared__ float s_e2m1[16];
    __shared__ float s_e4m3[256];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    if (threadIdx.x < 256) s_e4m3[threadIdx.x] = e4m3_lut[threadIdx.x];
    __syncthreads();
    const int row0 = hb << 1, row1 = row0 + 1;
    const int pcount = *panel_count;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int pi = chunk; pi < pcount; pi += nchunks) {
        const int p = panel_list[pi];
        const unsigned char* __restrict__ pbase = bank + (size_t)p * panel_stride;
        const float s0 = s_e4m3[pbase[row0]] * global_scale;
        const float s1 = s_e4m3[pbase[row1]] * global_scale;
        const unsigned char* __restrict__ pcodes = pbase + rows;
        unsigned int m = panel_masks[p];
        while (m) {
            const int c = __ffs(m) - 1; m &= m - 1;
            const unsigned char byte = pcodes[(size_t)c * rowhalf + hb];
            const float a = act[(p << 4) + c];
            acc0 = fmaf(s_e2m1[byte & 15] * s0, a, acc0);
            acc1 = fmaf(s_e2m1[byte >> 4] * s1, a, acc1);
        }
    }
    partials[(size_t)chunk * rows + row0] = acc0;
    partials[(size_t)chunk * rows + row1] = acc1;
}
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    from moe_lab.lightningstream_nemotron import nvfp4

    mod = cp.RawModule(code=SRC, options=("-std=c++14",))
    k_ref = mod.get_function("down_masked_ref")
    k_cand = mod.get_function("down_masked_byterow")

    rng = np.random.default_rng(20260816)
    panel_stride = HIDDEN + 16 * (HIDDEN // 2)
    mirror = cp.asarray(rng.integers(0, 256, size=NPANEL * panel_stride, dtype=np.uint8))
    e2m1 = cp.asarray(nvfp4.E2M1_TABLE, dtype=cp.float32)
    e4m3 = cp.asarray(nvfp4.E4M3_TABLE, dtype=cp.float32)
    globals_dev = cp.asarray(np.array([[1.0, 1.0]], dtype=np.float32))
    id_dev = cp.zeros(1, dtype=cp.int32)
    act = cp.asarray(rng.standard_normal(INTER).astype(np.float32))

    nz_mask = rng.random(INTER) < NZ_FRACTION
    pb = nz_mask.reshape(NPANEL, 16)
    masks = np.zeros(NPANEL, dtype=np.uint32)
    for p in range(NPANEL):
        for c in range(16):
            if pb[p, c]:
                masks[p] |= np.uint32(1 << c)
    plist = np.flatnonzero(masks != 0).astype(np.int32)
    masks_d, plist_d = cp.asarray(masks), cp.asarray(plist)
    pcount_d = cp.asarray(np.int32([plist.size]))

    arms = {}
    for nc in CHUNKS:
        p_ref = cp.zeros(nc * HIDDEN, dtype=cp.float32)
        p_cand = cp.zeros(nc * HIDDEN, dtype=cp.float32)
        args_ref = (mirror, id_dev, globals_dev, act, plist_d, masks_d, pcount_d,
                    e2m1, e4m3, p_ref, np.int32(HIDDEN), np.int32(INTER))
        args_cand = (mirror, id_dev, globals_dev, act, plist_d, masks_d, pcount_d,
                     e2m1, e4m3, p_cand, np.int32(HIDDEN), np.int32(INTER))
        bx_ref = (HIDDEN + 127) // 128
        bx_cand = ((HIDDEN // 2) + 127) // 128

        k_ref((bx_ref, nc), (128,), args_ref)
        k_cand((bx_cand, nc), (128,), args_cand)
        cp.cuda.Device(0).synchronize()
        exact = bool(np.array_equal(cp.asnumpy(p_ref).view(np.uint32),
                                    cp.asnumpy(p_cand).view(np.uint32)))

        def timed(k, bx, args):
            def run():
                for _ in range(MOE_LAYERS * TOP_K):
                    k((bx, nc), (128,), args)
            run()
            cp.cuda.Device(0).synchronize()
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record()
            for _ in range(ROUNDS):
                run()
            e1.record()
            e1.synchronize()
            return cp.cuda.get_elapsed_time(e0, e1) / ROUNDS

        ms_r = timed(k_ref, bx_ref, args_ref)
        ms_c = timed(k_cand, bx_cand, args_cand)
        arms[str(nc)] = {
            "nchunks": nc,
            "ref_blocks": bx_ref * nc, "cand_blocks": bx_cand * nc,
            "bit_exact": exact,
            "ref_ms_per_token": ms_r, "cand_ms_per_token": ms_c,
            "speedup": ms_r / ms_c if ms_c else None,
        }
        del p_ref, p_cand
        cp.get_default_memory_pool().free_all_blocks()

    all_exact = all(v["bit_exact"] for v in arms.values())
    best = min((v for v in arms.values() if v["bit_exact"]),
               key=lambda v: v["cand_ms_per_token"], default=None)
    prod = arms["8"]["ref_ms_per_token"]

    payload = {
        "kind": "diag_down_masked_byterow",
        "created_utc": utc_now(),
        "note": "TIMING + bit-exactness of partials only; the reduce step is not run. Rows 2k and 2k+1 load the SAME weight byte in the reference, so 32 threads of a warp touch only 16 distinct addresses. The candidate gives one thread the byte and both its rows.",
        "geometry": {"hidden": HIDDEN, "inter": INTER,
                     "active_panels": int(plist.size),
                     "nz_columns": int(nz_mask.sum()),
                     "launches_per_token": MOE_LAYERS * TOP_K},
        "arms": arms,
        "all_bit_exact": all_exact,
        "production_ref_ms_per_token_nchunks8": prod,
        "best_candidate": best,
        "best_speedup_vs_production": (prod / best["cand_ms_per_token"]) if best else None,
        "in_loop_marginal_ms_reference": 1.655,
        "status": "measured" if all_exact else "correctness_failed",
    }
    write_json_atomic(REPO / "pro_research" / "diag_down_masked_byterow.json",
                      payload, archive=False)
    print(json.dumps(payload, indent=2))
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
