"""A profiler substitute: ablate the pieces INSIDE down_masked and time each.

`ncu`, `nsys` and `compute-sanitizer` are not on PATH on this machine (PV2-21),
and installing software into the user's environment is not something to do
unasked. But the question that needs answering is narrow enough to answer with
ablation, which is a technique this project already uses (the STUB arms in
diag_v6_component_breakdown / diag_down_ablation_timing).

Five hypotheses have already been measured and refuted for this kernel --
bandwidth, instruction throughput, launch/grid/occupancy, dependent-chain
length, and redundant loads plus sector waste -- leaving a hard ~1.5 ms floor
with no explanation. What has never been separated is *which statement in the
inner loop* the time sits in.

## Arms (TIMING ONLY -- every arm except `full` computes WRONG numbers by
## design and none of them may ever be quoted as a correctness result)

  full            the reference kernel
  no_code_load    the per-column weight byte comes from arithmetic on (p, c)
                  instead of from `pcodes[c*rowhalf + hb]`
  no_scale_load   the per-panel scale byte comes from `p` instead of from
                  `pbase[row]`
  no_luts         both loads happen, but the decoded value is the raw byte
                  rather than `s_e2m1[...] * s_e4m3[...]` -- isolates the two
                  shared-memory lookups
  no_act          both loads and both LUTs happen, but `act[(p<<4)+c]` is
                  replaced by a constant -- isolates the shared-memory
                  broadcast read
  loop_only       neither load, no LUT, no act: the panel/mask walk alone

Every arm still accumulates whatever it loaded into `acc` and writes `acc` out,
so no load can be eliminated as dead code -- that is the trap that makes naive
ablation lie, and it is why each variant consumes its value.

The differences are marginal costs of each statement, not a partition: removing
a load also removes its latency from the chain, so the arms will not sum to
`full`. They are read as "which statement dominates", which is exactly what a
profiler line-level report would give.
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
NCHUNKS = 8
LAUNCHES = 23 * 6
ROUNDS = 30

BODY = r"""
extern "C" __global__ void down_masked_%(name)s(
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
        const unsigned char sb = %(scale_byte)s;
        const float s = %(scale_val)s;
        const unsigned char* __restrict__ pcodes = pbase + rows;
        unsigned int m = panel_masks[p];
        while (m) {
            const int c = __ffs(m) - 1; m &= m - 1;
            const unsigned char byte = %(code_byte)s;
            const float w = %(weight)s;
            const float a = %(act_val)s;
            acc = fmaf(w, a, acc);
        }
    }
    partials[(size_t)chunk * rows + row] = acc;
}
"""

REAL_SCALE_BYTE = "pbase[row]"
FAKE_SCALE_BYTE = "(unsigned char)(p & 0xFF)"
REAL_CODE_BYTE = "pcodes[(size_t)c * rowhalf + hb]"
FAKE_CODE_BYTE = "(unsigned char)((p * 31 + c) & 0xFF)"
REAL_SCALE_VAL = "s_e4m3[sb] * global_scale"
FAKE_SCALE_VAL = "(float)sb * global_scale"
REAL_WEIGHT = "s_e2m1[hi ? (byte >> 4) : (byte & 15)] * s"
FAKE_WEIGHT = "(float)byte * s"
REAL_ACT = "act[(p << 4) + c]"
FAKE_ACT = "1.0009765625f"

ARMS = {
    "full":          dict(scale_byte=REAL_SCALE_BYTE, code_byte=REAL_CODE_BYTE,
                          scale_val=REAL_SCALE_VAL, weight=REAL_WEIGHT, act_val=REAL_ACT),
    "no_code_load":  dict(scale_byte=REAL_SCALE_BYTE, code_byte=FAKE_CODE_BYTE,
                          scale_val=REAL_SCALE_VAL, weight=REAL_WEIGHT, act_val=REAL_ACT),
    "no_scale_load": dict(scale_byte=FAKE_SCALE_BYTE, code_byte=REAL_CODE_BYTE,
                          scale_val=REAL_SCALE_VAL, weight=REAL_WEIGHT, act_val=REAL_ACT),
    "no_luts":       dict(scale_byte=REAL_SCALE_BYTE, code_byte=REAL_CODE_BYTE,
                          scale_val=FAKE_SCALE_VAL, weight=FAKE_WEIGHT, act_val=REAL_ACT),
    "no_act":        dict(scale_byte=REAL_SCALE_BYTE, code_byte=REAL_CODE_BYTE,
                          scale_val=REAL_SCALE_VAL, weight=REAL_WEIGHT, act_val=FAKE_ACT),
    "loop_only":     dict(scale_byte=FAKE_SCALE_BYTE, code_byte=FAKE_CODE_BYTE,
                          scale_val=FAKE_SCALE_VAL, weight=FAKE_WEIGHT, act_val=FAKE_ACT),
}


def main() -> int:
    require_gpu_free()
    import cupy as cp

    from moe_lab.lightningstream_nemotron import nvfp4

    src = "\n".join((BODY % {"name": n, **cfg}) for n, cfg in ARMS.items())
    mod = cp.RawModule(code=src, options=("-std=c++14",))

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
    partials = cp.zeros(NCHUNKS * HIDDEN, dtype=cp.float32)
    args = (mirror, id_dev, globals_dev, act, plist_d, masks_d, pcount_d,
            e2m1, e4m3, partials, np.int32(HIDDEN), np.int32(INTER))
    bx = (HIDDEN + 127) // 128

    results = {}
    for name in ARMS:
        k = mod.get_function(f"down_masked_{name}")

        def run():
            for _ in range(LAUNCHES):
                k((bx, NCHUNKS), (128,), args)

        run()
        cp.cuda.Device(0).synchronize()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        for _ in range(ROUNDS):
            run()
        e1.record()
        e1.synchronize()
        results[name] = cp.cuda.get_elapsed_time(e0, e1) / ROUNDS

    full = results["full"]
    attribution = {n: {"ms_per_token": v,
                       "removed_ms": full - v,
                       "removed_fraction_of_full": (full - v) / full}
                   for n, v in results.items()}

    ranked = sorted(((n, v["removed_ms"]) for n, v in attribution.items() if n != "full"),
                    key=lambda x: -x[1])

    payload = {
        "kind": "diag_down_masked_ablate",
        "created_utc": utc_now(),
        "note": "TIMING ONLY. Every arm except `full` computes wrong numbers by design and may never be quoted as a correctness result. Each variant still consumes whatever it loaded so no load is eliminated as dead code. Arms do not sum to `full` -- removing a load also removes its latency from the chain -- and are read as 'which statement dominates'.",
        "why": "ncu/nsys/compute-sanitizer are not on PATH (PV2-21) and five end-to-end hypotheses have already been refuted for this kernel, leaving a ~1.5 ms floor unexplained. This separates the inner-loop statements instead.",
        "geometry": {"hidden": HIDDEN, "inter": INTER, "nchunks": NCHUNKS,
                     "active_panels": int(plist.size), "nz_columns": int(nz.sum()),
                     "launches_per_token": LAUNCHES},
        "arms_ms_per_token": results,
        "attribution": attribution,
        "dominant_statement": ranked[0][0] if ranked else None,
        "ranking_by_removed_ms": ranked,
    }
    write_json_atomic(REPO / "pro_research" / "diag_down_masked_ablate.json",
                      payload, archive=False)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
