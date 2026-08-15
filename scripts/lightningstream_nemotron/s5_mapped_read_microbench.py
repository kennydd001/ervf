"""S5 microbench: what can mapped pinned host memory actually deliver?

Three read patterns over the SAME pinned buffer, 512 MiB working set:
  1. byte-per-thread masked-style reads (the current S5 kernel pattern)
  2. uchar4 wide coalesced streaming reads (SM-as-DMA gather pattern)
  3. uint4 wide coalesced streaming reads
Plus the copy-engine reference: pinned H2D memcpy bandwidth.

Component measurement only. No tok/s claims.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

SRC = r"""
extern "C" __global__ void read_bytes(
    const unsigned char* __restrict__ src, float* out, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i & 1023] += src[i];
}

extern "C" __global__ void read_uchar4(
    const uchar4* __restrict__ src, float* out, size_t nvec)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nvec) {
        const uchar4 q = src[i];
        out[i & 1023] += q.x + q.y + q.z + q.w;
    }
}

extern "C" __global__ void read_uint4(
    const uint4* __restrict__ src, float* out, size_t nvec)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nvec) {
        const uint4 q = src[i];
        out[i & 1023] += (float)(q.x ^ q.y ^ q.z ^ q.w);
    }
}
"""


def bench(fn, args, blocks, threads, reps=20):
    fn((blocks,), (threads,), args)
    import cupy as cp
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        fn((blocks,), (threads,), args)
    cp.cuda.Device(0).synchronize()
    return (time.perf_counter_ns() - t0) / reps / 1e6


def main() -> int:
    import cupy as cp

    mib = 512
    n = mib * 1024 * 1024
    pm = cp.cuda.alloc_pinned_memory(n)
    host = np.frombuffer(pm, dtype=np.uint8, count=n)
    host[:] = np.arange(n, dtype=np.uint64) % 251  # touch every page
    out = cp.zeros(1024, dtype=cp.float32)

    mod = cp.RawModule(code=SRC, options=("-std=c++14",))
    ptr = np.uint64(host.ctypes.data)

    r = {}
    fn = mod.get_function("read_bytes")
    th, bl = 256, (n + 255) // 256
    ms = bench(fn, (ptr, out, np.uint64(n)), bl, th)
    r["byte_per_thread_gbs"] = n / (ms * 1e6)

    fn = mod.get_function("read_uchar4")
    n4 = n // 4
    th, bl = 256, (n4 + 255) // 256
    ms = bench(fn, (ptr, out, np.uint64(n4)), bl, th)
    r["uchar4_gbs"] = n / (ms * 1e6)

    fn = mod.get_function("read_uint4")
    n16 = n // 16
    th, bl = 256, (n16 + 255) // 256
    ms = bench(fn, (ptr, out, np.uint64(n16)), bl, th)
    r["uint4_gbs"] = n / (ms * 1e6)

    # copy engine reference
    dev = cp.empty(n, dtype=cp.uint8)
    s = cp.cuda.Stream(non_blocking=True)
    with s:
        dev.set(host, stream=s)
    s.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(10):
        with s:
            dev.set(host, stream=s)
    s.synchronize()
    r["memcpy_h2d_gbs"] = n / ((time.perf_counter_ns() - t0) / 10 / 1e6)

    # scattered-column copy-engine pattern: 1344 B copies, 168 per "expert",
    # 49 experts = one token's miss columns; measures small-copy overhead.
    cols, cbytes = 168 * 49, 1344
    dst = cp.empty(cols * cbytes, dtype=cp.uint8)
    t0 = time.perf_counter_ns()
    with s:
        for k in range(cols):
            off = (k * 7919) % (n - cbytes)
            dst[k * cbytes:(k + 1) * cbytes].set(host[off:off + cbytes], stream=s)
    s.synchronize()
    dt = (time.perf_counter_ns() - t0) / 1e6
    r["scattered_1344B_copies"] = {"n": cols, "ms": dt,
                                   "effective_gbs": cols * cbytes / (dt * 1e6)}

    for k, v in r.items():
        print(f"{k}: {v if not isinstance(v, dict) else v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
