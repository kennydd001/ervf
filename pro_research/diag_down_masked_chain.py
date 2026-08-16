"""Is `gemv_down_masked_partial_ind` limited by its dependent-load chain?

The arithmetic does not close on any usual axis. Measured in the real loop
(diag_moe_subkernel_marginals, all gates green): **1.655 ms/token** against

    bandwidth     64 MB / 249 GB/s              = 0.26 ms
    instructions  ~60M FMAs x ~8 supporting     = ~0.08 ms

both an order below. And V15 showed it is not launch overhead, grid size or
occupancy either: batching the slot dimension into one launch of
(21, 8, 6) = 1008 blocks instead of six launches of 168 was **neutral**
(+0.035 ms), despite 6x the parallelism.

What is left is the shape of the access. Each thread owns one output row and
walks its share of the panels:

    for (pi = chunk; pi < pcount; pi += nchunks)      // ~90/8 ~ 11 panels
        s = lut[pbase[row]]                           // 1 dependent load
        while (m) { byte = pcodes[c*rowhalf + hb] ... } // ~1.8 more

so ~31 dependent global loads per thread, each 1344 B from the last. Nothing in
the loop lets a thread have more than one of them in flight.

`nchunks` is exactly the knob that changes the chain length without changing
the total work: it is `gridDim.y`, and the panel loop strides by it. Doubling it
halves the panels per thread and doubles the blocks.

  * if the kernel is limited by the dependent chain, time falls with nchunks
  * if it is limited by request or bandwidth throughput, time stays flat

The partials buffer is sized for the swept nchunks so nothing is written out of
bounds, but the reduce step is NOT run and no output is claimed to be correct --
this is a timing-only diagnostic, and its numbers may never be reported as a
correctness result. Real panel/mask/nz metadata is generated to match the
measured census (9% nonzero, non-clustered).
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
ROWHALF = HIDDEN // 2
PANEL_STRIDE = HIDDEN + 16 * ROWHALF
DOWN_PANEL_BYTES = NPANEL * PANEL_STRIDE
TOP_K = 6
MOE_LAYERS = 23
NZ_FRACTION = 0.09
CHUNKS = [1, 2, 4, 8, 16, 32, 64]
ROUNDS = 30

SRC = r"""
// Verbatim gemv_down_masked_partial_ind from fused_nvfp4.py.
extern "C" __global__ void down_masked_ind(
    const unsigned char* __restrict__ bank,
    const int*           __restrict__ id_ptr,
    const float*         __restrict__ globals,
    const float*         __restrict__ act,
    const int*           __restrict__ panel_list,
    const unsigned int*  __restrict__ panel_masks,
    const int*           __restrict__ panel_count,
    const float*         __restrict__ e2m1_lut,
    const float*         __restrict__ e4m3_lut,
    float*               __restrict__ partials,
    const int rows,
    const int inter)
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

    const int hb = row >> 1;
    const int hi = row & 1;
    const int rowhalf = rows >> 1;
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
            const int c = __ffs(m) - 1;
            m &= m - 1;
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
    k = mod.get_function("down_masked_ind")

    rng = np.random.default_rng(20260816)
    mirror = cp.asarray(rng.integers(0, 256, size=DOWN_PANEL_BYTES, dtype=np.uint8))
    e2m1 = cp.asarray(nvfp4.E2M1_TABLE, dtype=cp.float32)
    e4m3 = cp.asarray(nvfp4.E4M3_TABLE, dtype=cp.float32)
    globals_dev = cp.asarray(np.array([[1.0, 1.0]], dtype=np.float32))
    id_dev = cp.zeros(1, dtype=cp.int32)
    act = cp.asarray(rng.standard_normal(INTER).astype(np.float32))

    nz_mask = rng.random(INTER) < NZ_FRACTION
    pmask_bool = nz_mask.reshape(NPANEL, 16)
    masks = np.zeros(NPANEL, dtype=np.uint32)
    for p in range(NPANEL):
        for c in range(16):
            if pmask_bool[p, c]:
                masks[p] |= np.uint32(1 << c)
    plist = np.flatnonzero(masks != 0).astype(np.int32)
    masks_d = cp.asarray(masks)
    plist_d = cp.asarray(plist)
    pcount_d = cp.asarray(np.int32([plist.size]))

    blocks_x = (HIDDEN + 127) // 128
    arms = {}
    for nc in CHUNKS:
        partials = cp.zeros(nc * HIDDEN, dtype=cp.float32)
        args = (mirror, id_dev, globals_dev, act, plist_d, masks_d, pcount_d,
                e2m1, e4m3, partials, np.int32(HIDDEN), np.int32(INTER))

        def run():
            # one token's worth: every layer, every expert slot
            for _ in range(MOE_LAYERS * TOP_K):
                k((blocks_x, nc), (128,), args)

        run()
        cp.cuda.Device(0).synchronize()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        for _ in range(ROUNDS):
            run()
        e1.record()
        e1.synchronize()
        ms = cp.cuda.get_elapsed_time(e0, e1) / ROUNDS

        arms[str(nc)] = {
            "nchunks": nc,
            "blocks_per_launch": blocks_x * nc,
            "panels_per_thread": float(plist.size) / nc,
            "dependent_loads_per_thread": float(plist.size) / nc * (1.0 + float(nz_mask.sum()) / plist.size),
            "ms_per_token": ms,
            "partials_bytes": int(partials.nbytes),
        }
        del partials
        cp.get_default_memory_pool().free_all_blocks()

    base = arms["8"]["ms_per_token"]
    best = min(arms.values(), key=lambda v: v["ms_per_token"])
    verdict = ("dependent_chain_limited" if best["ms_per_token"] < 0.75 * base
               else "flat_not_chain_limited")

    payload = {
        "kind": "diag_down_masked_chain",
        "created_utc": utc_now(),
        "note": "TIMING ONLY. The reduce step is not run and no output is correct by construction; these numbers may never be reported as a correctness result. nchunks is gridDim.y and the panel loop strides by it, so it changes the dependent-load chain length per thread without changing total work.",
        "geometry": {"hidden": HIDDEN, "inter": INTER, "npanel": NPANEL,
                     "active_panels": int(plist.size),
                     "nz_columns": int(nz_mask.sum()),
                     "launches_per_token": MOE_LAYERS * TOP_K},
        "production_nchunks": 8,
        "production_ms_per_token": base,
        "in_loop_marginal_ms_reference": 1.655,
        "bandwidth_floor_ms_reference": 0.26,
        "arms": arms,
        "best": best,
        "speedup_best_over_production": base / best["ms_per_token"],
        "verdict": verdict,
    }
    write_json_atomic(REPO / "pro_research" / "diag_down_masked_chain.json",
                      payload, archive=False)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
