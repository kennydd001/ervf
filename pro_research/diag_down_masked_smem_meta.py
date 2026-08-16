"""Stage the panel metadata in shared memory once per block.

`diag_down_masked_ablate` located the cost. Ablating every memory access and all
the arithmetic inside the inner loop -- both weight loads, both LUT lookups, the
activation read -- still leaves **1.316 ms of the 1.530 ms**:

    full            1.530 ms
    no_code_load    1.578   (removing the load makes it SLOWER)
    no_scale_load   1.551
    no_luts         1.524
    no_act          1.507
    loop_only       1.316   <- 86% of the kernel, with all data access removed

So the weights, the scales, the LUTs and the activations together account for
about 14% of this kernel. The remaining 86% is the panel walk itself, and the
reason is visible in two lines:

    const int p = panel_list[pi];        // global load
    unsigned int m = panel_masks[p];     // global load INDEXED BY THE FIRST

That is a pointer-chase: the second load cannot issue until the first returns.
Each thread runs it `pcount/nchunks ~ 11.5` times, so ~23 serialised global
round-trips per thread -- and every thread in the block chases the *same*
pointers, redundantly, because `pi` does not depend on `row`.

It also explains why five earlier hypotheses came back empty: the bottleneck was
never in the data path at all, so widening it (more blocks, shorter chains,
fewer redundant weight loads, batching) could not help.

## The fix

Both arrays are tiny -- at most `npanel = 116` entries, 928 bytes together --
and identical for every thread in the block. Stage them into shared memory once,
cooperatively (128 threads cover 116 panels in a single parallel step, so the
whole chase costs two dependent loads per block instead of 23 per thread), then
read the loop's metadata from SMEM.

Bit-exact by construction: same `p`, same `m`, same panels in the same order,
same `fmaf` sequence. Only where the metadata is read from changes -- the same
class of change as H-SCALE. Checked against the reference rather than argued.

Timing + bit-exactness of `partials`; the reduce step is not run.
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

HIDDEN, INTER = 2688, 1856
NPANEL = INTER // 16
NZ_FRACTION = 0.09
CHUNKS = [4, 8, 16]
LAUNCHES = 23 * 6
ROUNDS = 30

SRC = r"""
#define MAXPANEL 116

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

// CANDIDATE: panel_list and panel_masks[p] staged into shared memory once per
// block, cooperatively. Identical p, identical m, identical order, identical
// fmaf sequence -- only where the metadata is read from changes.
extern "C" __global__ void down_masked_smem(
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
    __shared__ float s_e2m1[16];
    __shared__ float s_e4m3[256];
    __shared__ int   s_plist[MAXPANEL];
    __shared__ unsigned int s_masks[MAXPANEL];
    const int pcount = *panel_count;
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    if (threadIdx.x < 256) s_e4m3[threadIdx.x] = e4m3_lut[threadIdx.x];
    // The pointer-chase now happens once per block, in parallel, instead of
    // pcount/nchunks times per thread.
    for (int i = threadIdx.x; i < pcount; i += blockDim.x) {
        const int p = panel_list[i];
        s_plist[i] = p;
        s_masks[i] = panel_masks[p];
    }
    __syncthreads();
    if (row >= rows) return;
    const int hb = row >> 1, hi = row & 1, rowhalf = rows >> 1;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    float acc = 0.0f;
    for (int pi = chunk; pi < pcount; pi += nchunks) {
        const int p = s_plist[pi];
        const unsigned char* __restrict__ pbase = bank + (size_t)p * panel_stride;
        const float s = s_e4m3[pbase[row]] * global_scale;
        const unsigned char* __restrict__ pcodes = pbase + rows;
        unsigned int m = s_masks[pi];
        while (m) {
            const int c = __ffs(m) - 1; m &= m - 1;
            const unsigned char byte = pcodes[(size_t)c * rowhalf + hb];
            const float w = s_e2m1[hi ? (byte >> 4) : (byte & 15)] * s;
            acc = fmaf(w, act[(p << 4) + c], acc);
        }
    }
    partials[(size_t)chunk * rows + row] = acc;
}
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    from moe_lab.lightningstream_nemotron import nvfp4

    mod = cp.RawModule(code=SRC, options=("-std=c++14",))
    k_ref = mod.get_function("down_masked_ref")
    k_smem = mod.get_function("down_masked_smem")

    rng = np.random.default_rng(20260816)
    panel_stride = HIDDEN + 16 * (HIDDEN // 2)
    mirror = cp.asarray(rng.integers(0, 256, size=NPANEL * panel_stride, dtype=np.uint8))
    e2m1 = cp.asarray(nvfp4.E2M1_TABLE, dtype=cp.float32)
    e4m3 = cp.asarray(nvfp4.E4M3_TABLE, dtype=cp.float32)
    globals_dev = cp.asarray(np.array([[1.0, 1.0]], dtype=np.float32))
    id_dev = cp.zeros(1, dtype=cp.int32)
    act = cp.asarray(rng.standard_normal(INTER).astype(np.float32))

    nz = rng.random(INTER) < NZ_FRACTION
    pb = nz.reshape(NPANEL, 16)
    masks = np.zeros(NPANEL, dtype=np.uint32)
    for p in range(NPANEL):
        for c in range(16):
            if pb[p, c]:
                masks[p] |= np.uint32(1 << c)
    plist = np.flatnonzero(masks != 0).astype(np.int32)
    masks_d, plist_d = cp.asarray(masks), cp.asarray(plist)
    pcount_d = cp.asarray(np.int32([plist.size]))
    bx = (HIDDEN + 127) // 128

    arms = {}
    for nc in CHUNKS:
        p_ref = cp.zeros(nc * HIDDEN, dtype=cp.float32)
        p_cand = cp.zeros(nc * HIDDEN, dtype=cp.float32)
        a_ref = (mirror, id_dev, globals_dev, act, plist_d, masks_d, pcount_d,
                 e2m1, e4m3, p_ref, np.int32(HIDDEN), np.int32(INTER))
        a_cand = (mirror, id_dev, globals_dev, act, plist_d, masks_d, pcount_d,
                  e2m1, e4m3, p_cand, np.int32(HIDDEN), np.int32(INTER))
        k_ref((bx, nc), (128,), a_ref)
        k_smem((bx, nc), (128,), a_cand)
        cp.cuda.Device(0).synchronize()
        exact = bool(np.array_equal(cp.asnumpy(p_ref).view(np.uint32),
                                    cp.asnumpy(p_cand).view(np.uint32)))

        def timed(k, args):
            def run():
                for _ in range(LAUNCHES):
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

        ms_r, ms_c = timed(k_ref, a_ref), timed(k_smem, a_cand)
        arms[str(nc)] = {"nchunks": nc, "bit_exact": exact,
                         "ref_ms_per_token": ms_r, "smem_ms_per_token": ms_c,
                         "speedup": ms_r / ms_c if ms_c else None,
                         "saved_ms_per_token": ms_r - ms_c}
        del p_ref, p_cand
        cp.get_default_memory_pool().free_all_blocks()

    all_exact = all(v["bit_exact"] for v in arms.values())
    best = min((v for v in arms.values() if v["bit_exact"]),
               key=lambda v: v["smem_ms_per_token"], default=None)
    prod = arms["8"]["ref_ms_per_token"]

    payload = {
        "kind": "diag_down_masked_smem_meta",
        "created_utc": utc_now(),
        "note": "Stages panel_list and panel_masks[p] into shared memory once per block. diag_down_masked_ablate showed 1.316 of 1.530 ms remains with ALL data access removed, and the cause is the dependent pair `p = panel_list[pi]; m = panel_masks[p]` -- a pointer-chase run ~11.5 times per thread by every thread redundantly.",
        "geometry": {"hidden": HIDDEN, "inter": INTER,
                     "active_panels": int(plist.size), "nz_columns": int(nz.sum()),
                     "launches_per_token": LAUNCHES,
                     "smem_added_bytes": MAXPANEL_BYTES if (MAXPANEL_BYTES := 116 * 8) else 0},
        "arms": arms,
        "all_bit_exact": all_exact,
        "production_ref_ms_per_token_nchunks8": prod,
        "best": best,
        "best_saving_vs_production_ms": (prod - best["smem_ms_per_token"]) if best else None,
        "in_loop_marginal_ms_reference": 1.655,
        "status": "measured" if all_exact else "correctness_failed",
    }
    write_json_atomic(REPO / "pro_research" / "diag_down_masked_smem_meta.json",
                      payload, archive=False)
    print(json.dumps(payload, indent=2))
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
