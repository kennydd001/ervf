"""Guard on diag_gather_pcie_ceiling.py's reference arm before its number is used.

That run reported the contiguous SM-side arm at 97.9 GB/s, which is physically
impossible over the measured link (nvidia-smi: PCIe Gen5 x8, ~31.5 GB/s
theoretical) and 3.8x the copy engine's 25.76 GB/s on the same buffer in the
same run. One of the two must be an artifact, and the working rule here is to
resolve disagreeing measurements rather than keep the convenient one.

Two candidate artifacts, both testable:
  1. the loads were never issued / never landed -> check the copied bytes;
  2. the 64 MiB source region was re-read 20 times and served from device L2
     -> re-measure while striding a fresh region of the 359 MiB pinned bank on
     every round, so nothing can be reused.

Also measures a device->device copy of the same size as an upper sanity bound
(VRAM bandwidth), which no host-sourced arm may exceed.

Read-only. No model load.
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

BANK_BYTES = 128 * 2806272        # 359 MiB, same as the ceiling script
CHUNK = 64 * 1024 * 1024          # 64 MiB per timed copy
ROUNDS = 5

SRC = r"""
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
    k = mod.get_function("contig_sm")

    pin = cp.cuda.alloc_pinned_memory(BANK_BYTES)
    bank = np.frombuffer(pin, dtype=np.uint8, count=BANK_BYTES)
    rng = np.random.default_rng(7)
    bank[:] = rng.integers(0, 256, size=BANK_BYTES, dtype=np.uint8)
    base = bank.ctypes.data

    dst = cp.zeros(CHUNK, dtype=cp.uint8)
    n4 = CHUNK // 16
    n_offsets = BANK_BYTES // CHUNK          # 5 disjoint 64 MiB windows

    rt = cp.cuda.runtime
    stream = cp.cuda.Stream(non_blocking=True)
    out = {}

    # ---- 1. correctness: did the SM arm actually copy the bytes? -----------
    dst.fill(0)
    k((1024,), (256,), (np.uint64(base), dst, np.int64(n4)))
    cp.cuda.Device(0).synchronize()
    got = cp.asnumpy(dst)
    out["sm_copy_bytes_correct"] = bool(np.array_equal(got, bank[:CHUNK]))

    def time_it(fn, rounds=ROUNDS):
        fn(0)
        cp.cuda.Device(0).synchronize()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        for r in range(rounds):
            fn(r)
        e1.record()
        e1.synchronize()
        ms = cp.cuda.get_elapsed_time(e0, e1) / rounds
        return ms, CHUNK / (ms * 1e-3) / 1e9

    # ---- 2a. SM arm, SAME window every round (what the first script did) ---
    ms, gbs = time_it(lambda r: k((1024,), (256,), (np.uint64(base), dst, np.int64(n4))))
    out["sm_same_window"] = {"ms": ms, "gb_s": gbs}

    # ---- 2b. SM arm, a FRESH 64 MiB window every round ---------------------
    ms, gbs = time_it(lambda r: k((1024,), (256,),
                                  (np.uint64(base + (r % n_offsets) * CHUNK),
                                   dst, np.int64(n4))))
    out["sm_fresh_window"] = {"ms": ms, "gb_s": gbs, "windows": n_offsets}

    # ---- 2c. DMA, same and fresh windows ----------------------------------
    def dma(r, off=True):
        rt.memcpyAsync(int(dst.data.ptr),
                       int(base + ((r % n_offsets) * CHUNK if off else 0)),
                       CHUNK, rt.memcpyHostToDevice, stream.ptr)
        stream.synchronize()

    ms, gbs = time_it(lambda r: dma(r, False))
    out["dma_same_window"] = {"ms": ms, "gb_s": gbs}
    ms, gbs = time_it(lambda r: dma(r, True))
    out["dma_fresh_window"] = {"ms": ms, "gb_s": gbs}

    # ---- 3. device->device sanity bound (VRAM bandwidth) -------------------
    dsrc = cp.zeros(CHUNK, dtype=cp.uint8)
    ms, gbs = time_it(lambda r: k((1024,), (256,),
                                  (np.uint64(int(dsrc.data.ptr)), dst, np.int64(n4))))
    out["d2d_sm"] = {"ms": ms, "gb_s": gbs,
                     "note": "device->device: reads+writes VRAM, so ~2x this in total traffic; no host arm may exceed it"}

    verdict = ("sm_arm_was_reuse_artifact"
               if out["sm_fresh_window"]["gb_s"] < 0.6 * out["sm_same_window"]["gb_s"]
               else "sm_arm_reproduces_on_fresh_data")

    payload = {
        "kind": "diag_gather_ceiling_check",
        "created_utc": utc_now(),
        "note": "guard on the 97.9 GB/s reference arm in diag_gather_pcie_ceiling.json; PCIe Gen5 x8 is ~31.5 GB/s theoretical so that number cannot stand as reported",
        "chunk_bytes": CHUNK, "bank_bytes": BANK_BYTES, "rounds": ROUNDS,
        "arms": out,
        "verdict": verdict,
    }
    write_json_atomic(REPO / "pro_research" / "diag_gather_ceiling_check.json",
                      payload, archive=False)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
