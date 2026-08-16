"""Why is the sparse down_proj gather 6x slower in the loop than in isolation?

The project's own numbers make this the largest unattacked inefficiency in the
system:

  * N2 (docs/LIGHTNINGSTREAM_NEMOTRON_RESEARCH_LOG.md) measured
    `gather_down_sparse` pulling ~35 MB/token from mapped host at **~4.3 GB/s
    effective**, against the **25.05 GB/s** that S5 measured for the same
    uchar4 coalesced mapped-read pattern in isolation -- "6x slechter in de lus".
  * The down pipeline is 6.5058 ms of a 22.53 ms token in-graph
    (diag_down_ablation_timing.json) = 28.9%; the STUB arm without down_proj
    runs at 16.02 ms = 62.4 tok/s.
  * The link is PCIe **Gen5 x8** (nvidia-smi: gen 5, width 8), i.e. ~31.5 GB/s
    theoretical, so 25 GB/s is a plausible practical ceiling and 4.3 GB/s is
    ~14% of it.

S8 and S11 both concluded "the MoE term is not transfer-bound", but both
measured the *DMA* path (bulk cudaMemcpyAsync of up_proj misses). This kernel
is a different mechanism: SM-side zero-copy reads over PCIe. Those two can
disagree, and nothing in the log separates them. This does.

## What is measured

One token's worth of gather work: 23 MoE layers x 6 experts = 138 calls, on a
real-sized pinned panel-major bank (128 experts x 2,806,272 B = 359 MiB, so the
scattered-over-a-large-region property is preserved).

Sparsity is synthetic but calibrated to the measured census: ReLU^2 leaves ~9%
of the 1856 intermediates nonzero and they do not cluster (S2: 30.6% of the
16-column blocks are all-zero). The arms differ only in the copy kernel, never
in which bytes are copied, so a bandwidth comparison is unaffected by the
synthetic draw -- and every arm is byte-compared against arm v0's output.

## Arms (single variable: the copy kernel; identical byte set)

  v0_production   verbatim body of gather_down_sparse_ind, production launch
                  geometry (grid sized for worst case inter+npanel warps)
  v1_unroll4      same warp->column mapping, but 4 independent loads issued
                  into registers before any store (tests: is each warp limited
                  to one outstanding PCIe read?)
  v2_splitW       W warps per column/panel instead of 1 (tests: is the limit
                  the number of concurrent requests, not per-warp latency?)
  contig_sm       SM-side uchar4 copy of the same total bytes, fully
                  contiguous -- replicates S5's 25.05 GB/s isolated number on
                  today's machine
  dma_contig      cudaMemcpyAsync of the same total bytes -- the copy engine's
                  practical H2D ceiling on this link today

contig_sm and dma_contig are ceilings, not candidates: they move the same
number of bytes but not the same bytes, so they are labelled reference arms
and can never be reported as a gather result.

Read-only diagnostic. No model load, no runtime import, no adoption claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, require_gpu_free, utc_now, write_json_atomic

HIDDEN = 2688          # down_proj output rows
INTER = 1856           # down_proj reduction dim / intermediate size
NPANEL = INTER // 16   # 116
ROWHALF = HIDDEN // 2  # 1344 B per column (nibbles)
PANEL_STRIDE = HIDDEN + 16 * ROWHALF          # 24192 B
DOWN_PANEL_BYTES = NPANEL * PANEL_STRIDE      # 2806272 B
N_EXPERTS = 128
MOE_LAYERS = 23
TOP_K = 6
CALLS_PER_TOKEN = MOE_LAYERS * TOP_K          # 138

NZ_FRACTION = 0.09     # S2 census: ReLU^2 leaves ~9% nonzero
ROUNDS = 20            # tokens' worth of gather per timed arm
SEED = 20260816

SRC = r"""
extern "C" __global__ void gather_v0(
    const unsigned char* __restrict__ down_base,
    const int*           __restrict__ id_ptr,
    const size_t         panel_bytes,
    unsigned char*       __restrict__ dst_base,
    const int*           __restrict__ panel_list,
    const int*           __restrict__ panel_count,
    const int*           __restrict__ nz_list,
    const int*           __restrict__ nz_count,
    const int rows)
{
    const unsigned char* __restrict__ src_base =
        down_base + (size_t)(*id_ptr) * panel_bytes;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int ncol = *nz_count;
    const int rowhalf = rows >> 1;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    if (warp < ncol) {
        const int j = nz_list[warp];
        const size_t off = (size_t)(j >> 4) * panel_stride + rows
                         + (size_t)(j & 15) * rowhalf;
        const uchar4* s = reinterpret_cast<const uchar4*>(src_base + off);
        uchar4* d = reinterpret_cast<uchar4*>(dst_base + off);
        for (int k = lane; k < rowhalf / 4; k += 32) d[k] = s[k];
    } else if (warp < ncol + *panel_count) {
        const int p = panel_list[warp - ncol];
        const size_t off = (size_t)p * panel_stride;
        const uchar4* s = reinterpret_cast<const uchar4*>(src_base + off);
        uchar4* d = reinterpret_cast<uchar4*>(dst_base + off);
        for (int k = lane; k < rows / 4; k += 32) d[k] = s[k];
    }
}

// Same warp->column mapping, same bytes, but four loads are issued into
// registers before the first store, so a warp can have four PCIe reads in
// flight instead of one.
__device__ __forceinline__ void copy_unroll4(const uchar4* __restrict__ s,
                                             uchar4* __restrict__ d,
                                             int n, int lane)
{
    int k = lane;
    for (; k + 96 < n; k += 128) {
        uchar4 r0 = s[k];
        uchar4 r1 = s[k + 32];
        uchar4 r2 = s[k + 64];
        uchar4 r3 = s[k + 96];
        d[k]      = r0;
        d[k + 32] = r1;
        d[k + 64] = r2;
        d[k + 96] = r3;
    }
    for (; k < n; k += 32) d[k] = s[k];
}

extern "C" __global__ void gather_v1_unroll4(
    const unsigned char* __restrict__ down_base,
    const int*           __restrict__ id_ptr,
    const size_t         panel_bytes,
    unsigned char*       __restrict__ dst_base,
    const int*           __restrict__ panel_list,
    const int*           __restrict__ panel_count,
    const int*           __restrict__ nz_list,
    const int*           __restrict__ nz_count,
    const int rows)
{
    const unsigned char* __restrict__ src_base =
        down_base + (size_t)(*id_ptr) * panel_bytes;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int ncol = *nz_count;
    const int rowhalf = rows >> 1;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    if (warp < ncol) {
        const int j = nz_list[warp];
        const size_t off = (size_t)(j >> 4) * panel_stride + rows
                         + (size_t)(j & 15) * rowhalf;
        copy_unroll4(reinterpret_cast<const uchar4*>(src_base + off),
                     reinterpret_cast<uchar4*>(dst_base + off),
                     rowhalf / 4, lane);
    } else if (warp < ncol + *panel_count) {
        const int p = panel_list[warp - ncol];
        const size_t off = (size_t)p * panel_stride;
        copy_unroll4(reinterpret_cast<const uchar4*>(src_base + off),
                     reinterpret_cast<uchar4*>(dst_base + off),
                     rows / 4, lane);
    }
}

// SPLIT: W warps cooperate on one column (or one panel scale block), so the
// number of concurrent outstanding reads grows by W without changing the byte
// set or the access granularity per request.
extern "C" __global__ void gather_v2_split(
    const unsigned char* __restrict__ down_base,
    const int*           __restrict__ id_ptr,
    const size_t         panel_bytes,
    unsigned char*       __restrict__ dst_base,
    const int*           __restrict__ panel_list,
    const int*           __restrict__ panel_count,
    const int*           __restrict__ nz_list,
    const int*           __restrict__ nz_count,
    const int rows,
    const int wpc)                       // warps per column
{
    const unsigned char* __restrict__ src_base =
        down_base + (size_t)(*id_ptr) * panel_bytes;
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int ncol = *nz_count;
    const int rowhalf = rows >> 1;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    const int unit = gwarp / wpc;        // which column / panel
    const int sub  = gwarp - unit * wpc; // which slice of it
    if (unit < ncol) {
        const int j = nz_list[unit];
        const size_t off = (size_t)(j >> 4) * panel_stride + rows
                         + (size_t)(j & 15) * rowhalf;
        const uchar4* s = reinterpret_cast<const uchar4*>(src_base + off);
        uchar4* d = reinterpret_cast<uchar4*>(dst_base + off);
        for (int k = lane + sub * 32; k < rowhalf / 4; k += 32 * wpc) d[k] = s[k];
    } else if (unit < ncol + *panel_count) {
        const int p = panel_list[unit - ncol];
        const size_t off = (size_t)p * panel_stride;
        const uchar4* s = reinterpret_cast<const uchar4*>(src_base + off);
        uchar4* d = reinterpret_cast<uchar4*>(dst_base + off);
        for (int k = lane + sub * 32; k < rows / 4; k += 32 * wpc) d[k] = s[k];
    }
}

// HYPOTHESIS ARM: identical to v0 except the per-panel FP8 block-scale planes
// are NOT fetched. It copies a strict SUBSET of v0's bytes, so it is not a
// drop-in candidate -- it prices the question "what would the gather cost if
// the scale planes were already resident in VRAM?".
extern "C" __global__ void gather_v3_no_scales(
    const unsigned char* __restrict__ down_base,
    const int*           __restrict__ id_ptr,
    const size_t         panel_bytes,
    unsigned char*       __restrict__ dst_base,
    const int*           __restrict__ nz_list,
    const int*           __restrict__ nz_count,
    const int rows)
{
    const unsigned char* __restrict__ src_base =
        down_base + (size_t)(*id_ptr) * panel_bytes;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int rowhalf = rows >> 1;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    if (warp < *nz_count) {
        const int j = nz_list[warp];
        const size_t off = (size_t)(j >> 4) * panel_stride + rows
                         + (size_t)(j & 15) * rowhalf;
        const uchar4* s = reinterpret_cast<const uchar4*>(src_base + off);
        uchar4* d = reinterpret_cast<uchar4*>(dst_base + off);
        for (int k = lane; k < rowhalf / 4; k += 32) d[k] = s[k];
    }
}

// REFERENCE ARM ONLY -- same byte COUNT, fully contiguous, different bytes.
extern "C" __global__ void contig_sm(
    const uchar4* __restrict__ s, uchar4* __restrict__ d, const long long n4)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) d[i] = s[i];
}
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    mod = cp.RawModule(code=SRC, options=("-std=c++14",))
    k_v0 = mod.get_function("gather_v0")
    k_v1 = mod.get_function("gather_v1_unroll4")
    k_v2 = mod.get_function("gather_v2_split")
    k_contig = mod.get_function("contig_sm")

    rng = np.random.default_rng(SEED)

    # ---- pinned panel-major bank, real size (359 MiB), never uploaded -------
    pin = cp.cuda.alloc_pinned_memory(N_EXPERTS * DOWN_PANEL_BYTES)
    bank = np.frombuffer(pin, dtype=np.uint8, count=N_EXPERTS * DOWN_PANEL_BYTES)
    # Fill with a cheap deterministic pattern; content is irrelevant to a copy
    # benchmark but must not be all-zero (some paths could special-case it).
    bank[:] = np.arange(bank.size % 251, bank.size % 251 + bank.size,
                        dtype=np.int64).astype(np.uint8)
    bank_ptr = bank.ctypes.data

    mirror = cp.zeros(DOWN_PANEL_BYTES, dtype=cp.uint8)
    ref_mirror = cp.zeros(DOWN_PANEL_BYTES, dtype=cp.uint8)

    # ---- one realistic (nz_list, panel_list) per call ----------------------
    calls = []
    total_bytes = 0
    for _ in range(CALLS_PER_TOKEN):
        nz_mask = rng.random(INTER) < NZ_FRACTION
        nz = np.flatnonzero(nz_mask).astype(np.int32)
        pmask = nz_mask.reshape(NPANEL, 16).any(axis=1)
        plist = np.flatnonzero(pmask).astype(np.int32)
        eid = int(rng.integers(0, N_EXPERTS))
        calls.append({
            "nz": cp.asarray(nz),
            "nzc": cp.asarray(np.int32([nz.size])),
            "plist": cp.asarray(plist),
            "pcount": cp.asarray(np.int32([plist.size])),
            "id": cp.asarray(np.int32([eid])),
            "bytes": int(nz.size) * ROWHALF + int(plist.size) * HIDDEN,
        })
        total_bytes += calls[-1]["bytes"]

    nz_counts = [int(c["nzc"][0]) for c in calls]
    p_counts = [int(c["pcount"][0]) for c in calls]

    # production launch geometry: grid sized for the worst case so the shape is
    # capture-stable, exactly as down_masked_into_indirect does it
    max_warps = INTER + NPANEL
    blocks_v0 = (max_warps * 32 + 255) // 256

    def run_v0(dst):
        for c in calls:
            k_v0((blocks_v0,), (256,),
                 (np.uint64(bank_ptr), c["id"], np.uint64(DOWN_PANEL_BYTES),
                  dst, c["plist"], c["pcount"], c["nz"], c["nzc"],
                  np.int32(HIDDEN)))

    def run_v1(dst):
        for c in calls:
            k_v1((blocks_v0,), (256,),
                 (np.uint64(bank_ptr), c["id"], np.uint64(DOWN_PANEL_BYTES),
                  dst, c["plist"], c["pcount"], c["nz"], c["nzc"],
                  np.int32(HIDDEN)))

    def make_run_v2(wpc):
        blocks = (max_warps * wpc * 32 + 255) // 256

        def run(dst):
            for c in calls:
                k_v2((blocks,), (256,),
                     (np.uint64(bank_ptr), c["id"], np.uint64(DOWN_PANEL_BYTES),
                      dst, c["plist"], c["pcount"], c["nz"], c["nzc"],
                      np.int32(HIDDEN), np.int32(wpc)))
        return run, blocks

    def timed(fn, dst, rounds=ROUNDS):
        fn(dst)                       # warm
        cp.cuda.Device(0).synchronize()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        for _ in range(rounds):
            fn(dst)
        e1.record()
        e1.synchronize()
        ms = cp.cuda.get_elapsed_time(e0, e1) / rounds
        return ms, total_bytes / (ms * 1e-3) / 1e9

    results = {}

    # ---- v0: production body, and the byte-exactness reference -------------
    run_v0(ref_mirror)
    cp.cuda.Device(0).synchronize()
    ms, gbs = timed(run_v0, mirror)
    exact_v0 = bool(cp.array_equal(mirror, ref_mirror))
    results["v0_production"] = {"ms_per_token": ms, "gb_s": gbs,
                                "blocks": blocks_v0, "bytes_match_v0": exact_v0}

    # ---- v1: four independent loads in flight per warp ---------------------
    mirror.fill(0)
    ms, gbs = timed(run_v1, mirror)
    results["v1_unroll4"] = {"ms_per_token": ms, "gb_s": gbs,
                             "blocks": blocks_v0,
                             "bytes_match_v0": bool(cp.array_equal(mirror, ref_mirror))}

    # ---- v2: W warps per column -------------------------------------------
    for wpc in (2, 4, 8, 16):
        run, blocks = make_run_v2(wpc)
        mirror.fill(0)
        ms, gbs = timed(run, mirror)
        results[f"v2_split{wpc}"] = {"ms_per_token": ms, "gb_s": gbs,
                                     "blocks": blocks, "warps_per_column": wpc,
                                     "bytes_match_v0": bool(cp.array_equal(mirror, ref_mirror))}

    # ---- v3: what the gather would cost with scale planes already resident -
    k_v3 = mod.get_function("gather_v3_no_scales")
    nz_bytes = sum(int(c["nzc"][0]) for c in calls) * ROWHALF

    def run_v3(dst):
        for c in calls:
            k_v3((blocks_v0,), (256,),
                 (np.uint64(bank_ptr), c["id"], np.uint64(DOWN_PANEL_BYTES),
                  dst, c["nz"], c["nzc"], np.int32(HIDDEN)))

    mirror.fill(0)
    run_v3(mirror)
    cp.cuda.Device(0).synchronize()
    v3_subset_ok = bool(cp.all((mirror == ref_mirror) | (mirror == 0)))
    ms, gbs_v3 = timed(run_v3, mirror)
    results["v3_no_scales_HYPOTHESIS"] = {
        "ms_per_token": ms,
        "gb_s": nz_bytes / (ms * 1e-3) / 1e9,
        "gathered_bytes_per_token": nz_bytes,
        "is_strict_subset_of_v0_bytes": v3_subset_ok,
        "saving_vs_v0_ms_per_token": results["v0_production"]["ms_per_token"] - ms,
        "note": "HYPOTHESIS ARM, not a candidate: copies only the nonzero weight columns and skips the per-panel FP8 scale planes. Prices H-SCALE (scale planes resident in VRAM). The masked GEMV still needs those scales -- they would have to come from device memory, which this arm does not build.",
    }

    # ---- reference ceilings (same byte COUNT, not the same bytes) ----------
    # uchar4 is FOUR bytes, not sixteen. Getting this wrong once already
    # produced a 97.9 GB/s "ceiling" over a ~31.5 GB/s link; the element count
    # is therefore asserted against the copied bytes below.
    ref_n = total_bytes // 4 * 4
    ref_dst = cp.zeros(ref_n, dtype=cp.uint8)
    n4 = ref_n // 4

    def run_contig(_dst=None):
        k_contig((1024,), (256,), (np.uint64(bank_ptr), ref_dst, np.int64(n4)))

    ref_dst.fill(0)
    run_contig()
    cp.cuda.Device(0).synchronize()
    contig_exact = bool(np.array_equal(cp.asnumpy(ref_dst), bank[:ref_n]))
    ms, gbs = timed(run_contig, None)
    results["ref_contig_sm"] = {"ms_per_token": ms, "gb_s": gbs,
                                "bytes_verified": contig_exact,
                                "note": "REFERENCE: contiguous SM-side uchar4 read of the same byte count; replicates S5's isolated 25.05 GB/s measurement on today's machine"}

    rt = cp.cuda.runtime
    stream = cp.cuda.Stream(non_blocking=True)

    def run_dma(_dst=None):
        rt.memcpyAsync(int(ref_dst.data.ptr), int(bank_ptr), ref_n,
                       rt.memcpyHostToDevice, stream.ptr)
        stream.synchronize()

    ref_dst.fill(0)
    run_dma()
    dma_exact = bool(np.array_equal(cp.asnumpy(ref_dst), bank[:ref_n]))
    ms, gbs = timed(run_dma, None, rounds=10)
    results["ref_dma_contig"] = {"ms_per_token": ms, "gb_s": gbs,
                                 "bytes_verified": dma_exact,
                                 "note": "REFERENCE: cudaMemcpyAsync of the same byte count; the copy engine's practical H2D ceiling on this link today"}

    best_gather = max(
        (v["gb_s"] for k, v in results.items()
         if not k.startswith("ref_") and v.get("bytes_match_v0")),
        default=None)
    ceiling = max(v["gb_s"] for k, v in results.items()
                  if k.startswith("ref_") and v.get("bytes_verified"))

    payload = {
        "kind": "diag_gather_pcie_ceiling",
        "created_utc": utc_now(),
        "note": "read-only bandwidth diagnostic; every non-reference arm copies the IDENTICAL byte set and is byte-compared against the production body (v0). Reference arms move the same byte COUNT but different bytes and are ceilings, never gather results. Sparsity is synthetic, calibrated to the S2 census (9% nonzero, non-clustered); the byte set is held fixed across arms so the comparison is unaffected.",
        "environment": environment_snapshot((
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "fused_nvfp4.py",
        )),
        "geometry": {
            "hidden_rows": HIDDEN, "intermediate": INTER, "npanel": NPANEL,
            "panel_stride_bytes": PANEL_STRIDE,
            "down_panel_bytes_per_expert": DOWN_PANEL_BYTES,
            "bank_experts": N_EXPERTS,
            "bank_bytes": N_EXPERTS * DOWN_PANEL_BYTES,
            "calls_per_token": CALLS_PER_TOKEN,
            "rounds_timed": ROUNDS,
        },
        "workload": {
            "nz_fraction_target": NZ_FRACTION,
            "nz_per_call_mean": float(np.mean(nz_counts)),
            "active_panels_per_call_mean": float(np.mean(p_counts)),
            "all_zero_panel_fraction": 1.0 - float(np.mean(p_counts)) / NPANEL,
            "gathered_bytes_per_token": total_bytes,
            "scale_byte_fraction": float(np.mean(p_counts)) * HIDDEN /
                                   (float(np.mean(nz_counts)) * ROWHALF +
                                    float(np.mean(p_counts)) * HIDDEN),
        },
        "arms": results,
        "summary": {
            "production_gb_s": results["v0_production"]["gb_s"],
            "best_exact_variant_gb_s": best_gather,
            "reference_ceiling_gb_s": ceiling,
            "production_fraction_of_ceiling": results["v0_production"]["gb_s"] / ceiling if ceiling else None,
            "best_speedup_vs_production": best_gather / results["v0_production"]["gb_s"] if best_gather else None,
        },
    }
    write_json_atomic(REPO / "pro_research" / "diag_gather_pcie_ceiling.json",
                      payload, archive=False)
    print(json.dumps({"workload": payload["workload"], "arms": results,
                      "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
