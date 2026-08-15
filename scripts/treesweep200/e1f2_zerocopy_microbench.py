"""E1 fase 2, microbench E1-2.0: bulk zero-copy kernel-reads over PCIe.

Preregistration: reports/treesweep200/E1F2_ZEROCOPY_PREREGISTRATION_2026-08-15.md
(gates frozen before any measurement; kill-gate G-E1F2-K1 decides whether the
graph-resident design with in-kernel host reads is worth building).

M0: UVA correctness -- kernel checksum over a pinned host buffer.
M1: streaming bulk read, pinned pool vs device copy.
M2: the production ERVF gemv_into on a synthetic NVFP4 expert record, weights
    mapped on host vs resident on device, cycling 24 distinct records.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import cupy as cp  # noqa: E402

from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4  # noqa: E402
from moe_lab.lightningstream_nemotron.runtime import UP_CODE, UP_SCALE  # noqa: E402

OUT = REPO / "reports" / "treesweep200" / "E1F2_ZEROCOPY_MICROBENCH.json"

_SRC = r"""
extern "C" __global__ void checksum_u4(const uint4* __restrict__ p,
                                       const size_t n4,
                                       unsigned long long* __restrict__ sink) {
    unsigned long long acc = 0ull;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n4; i += stride) {
        const uint4 v = p[i];
        acc ^= (unsigned long long)v.x ^ ((unsigned long long)v.y << 1)
             ^ ((unsigned long long)v.z << 2) ^ ((unsigned long long)v.w << 3);
    }
    for (int o = 16; o > 0; o >>= 1)
        acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
    __shared__ unsigned long long red[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long r = 0ull;
        for (int w = 0; w < (blockDim.x >> 5); w++) r ^= red[w];
        if (r != 0ull || n4 == 0) sink[blockIdx.x] = r;  // keep alive
        if (n4 == 0) sink[blockIdx.x] = 1ull;
    }
}
"""


def gpu_free() -> bool:
    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return not o.stdout.strip()


def timed(fn, warmup=3, reps=50) -> float:
    for _ in range(warmup):
        fn()
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        ts.append(cp.cuda.get_elapsed_time(e0, e1))
    return float(np.median(ts))


def main() -> int:
    if not gpu_free():
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    mod = cp.RawModule(code=_SRC, options=("-std=c++14",))
    checksum = mod.get_function("checksum_u4")
    results: dict = {}

    # ---------------- M0: UVA correctness ----------------
    n0 = 4 * 1024 * 1024
    pin0 = cp.cuda.alloc_pinned_memory(n0)
    host0 = np.frombuffer(pin0, dtype=np.uint8, count=n0)
    rs = np.random.RandomState(20260815)
    host0[:] = rs.randint(0, 256, size=n0, dtype=np.uint8)
    sink = cp.zeros(4096, dtype=cp.uint64)
    ptr0 = host0.ctypes.data
    blocks = 512
    checksum((blocks,), (256,),
             (np.uint64(ptr0), np.uint64(n0 // 16), sink.data.ptr))
    cp.cuda.Device().synchronize()
    expected = np.zeros((), dtype=np.uint64)
    hv = host0.view(np.uint64).reshape(-1, 4)
    acc = np.bitwise_xor.reduce(hv[:, 0]) ^ np.bitwise_xor.reduce(
        (hv[:, 1].astype(np.uint64) << 1) & np.uint64(0xFFFFFFFFFFFFFFFF))
    # exact reference via the same xor folding as the kernel is awkward in
    # numpy; instead recompute the kernel's value with a second, simpler
    # kernel-independent rule: compare against device copy of the same bytes.
    dev0 = cp.asarray(host0)
    sink_d = cp.zeros(4096, dtype=cp.uint64)
    checksum((blocks,), (256,),
             (dev0.data.ptr, np.uint64(n0 // 16), sink_d.data.ptr))
    cp.cuda.Device().synchronize()
    m0_ok = bool(cp.array_equal(cp.sort(sink), cp.sort(sink_d)))
    results["M0_uva_correct"] = m0_ok
    del dev0

    # ---------------- M1: streaming bulk read ----------------
    pool_bytes = 256 * 1024 * 1024
    pin1 = cp.cuda.alloc_pinned_memory(pool_bytes)
    host1 = np.frombuffer(pin1, dtype=np.uint8, count=pool_bytes)
    host1[:] = rs.randint(0, 256, size=pool_bytes, dtype=np.uint8)
    dev1 = cp.asarray(host1)
    sink1 = cp.zeros(4096, dtype=cp.uint64)
    n4 = pool_bytes // 16

    ms_host = timed(lambda: checksum((1024,), (256,),
                    (np.uint64(host1.ctypes.data), np.uint64(n4),
                     sink1.data.ptr)), warmup=2, reps=10)
    ms_dev = timed(lambda: checksum((1024,), (256,),
                   (dev1.data.ptr, np.uint64(n4), sink1.data.ptr)),
                   warmup=2, reps=10)
    results["M1_streaming"] = {
        "bytes": pool_bytes,
        "host_ms": ms_host, "device_ms": ms_dev,
        "host_gbps": pool_bytes / ms_host / 1e6,
        "device_gbps": pool_bytes / ms_dev / 1e6,
    }
    del dev1
    cp.get_default_memory_pool().free_all_blocks()

    # ---------------- M2: production ERVF up-GEMV, host vs device ----------------
    fused = FusedNVFP4()
    rows, cols = 1856, 2688
    n_rec = 24
    rec_bytes = UP_CODE + UP_SCALE
    pin2 = cp.cuda.alloc_pinned_memory(n_rec * UP_CODE)
    pin3 = cp.cuda.alloc_pinned_memory(n_rec * UP_SCALE)
    h_codes = np.frombuffer(pin2, dtype=np.uint8, count=n_rec * UP_CODE)
    h_scales = np.frombuffer(pin3, dtype=np.uint8, count=n_rec * UP_SCALE)
    h_codes[:] = rs.randint(0, 256, size=h_codes.size, dtype=np.uint8)
    h_scales[:] = rs.randint(0, 256, size=h_scales.size, dtype=np.uint8)
    # remap e4m3 NaN patterns in scales
    nan_pat = (h_scales & 0x7F) == 0x7F
    h_scales[nan_pat] &= 0xFE

    d_codes = cp.asarray(h_codes)
    d_scales = cp.asarray(h_scales)
    x = cp.random.RandomState(7).standard_normal(cols).astype(cp.float32)
    out_h = cp.zeros(rows, dtype=cp.float32)
    out_d = cp.zeros(rows, dtype=cp.float32)

    cbase, sbase = h_codes.ctypes.data, h_scales.ctypes.data

    def arm_host(i):
        fused.gemv_into(out_h, np.uint64(cbase + i * UP_CODE),
                        np.uint64(sbase + i * UP_SCALE), x, 0.01,
                        rows, cols, apply_relu2=True)

    def arm_device(i):
        fused.gemv_into(out_d, d_codes[i * UP_CODE:(i + 1) * UP_CODE],
                        d_scales[i * UP_SCALE:(i + 1) * UP_SCALE], x, 0.01,
                        rows, cols, apply_relu2=True)

    # correctness: bitexact same output on every record
    m2_bitexact = True
    for i in range(n_rec):
        arm_host(i)
        arm_device(i)
        cp.cuda.Device().synchronize()
        if not bool(cp.array_equal(out_h, out_d)):
            m2_bitexact = False
            results["M2_first_mismatch_record"] = i
            break
    results["M2_bitexact"] = m2_bitexact

    ms_m2_host = timed(lambda: arm_host(int(rs.randint(n_rec))), warmup=8, reps=50)
    ms_m2_dev = timed(lambda: arm_device(int(rs.randint(n_rec))), warmup=8, reps=50)
    results["M2_ervf_up_gemv"] = {
        "rows": rows, "cols": cols, "records": n_rec,
        "record_bytes": rec_bytes,
        "host_ms": ms_m2_host, "device_ms": ms_m2_dev,
        "host_gbps": rec_bytes / ms_m2_host / 1e6,
        "device_gbps": rec_bytes / ms_m2_dev / 1e6,
    }

    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
