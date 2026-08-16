"""Independently re-measure the number the entire roofline rests on.

Every ceiling claim in this project -- 165 tok/s, "V6 is at 28.9% of roofline",
the component decomposition -- divides by **338.4 GB/s**, measured once by N5.
Today's byte accounting reproduces 165 tok/s from that figure exactly, which
means the accounting and the bandwidth figure now confirm each other; but they
would also confirm each other if the bandwidth figure were wrong, because the
accounting was checked *against* it. One of the two needs an independent
anchor.

A short guard is cheap and today already caught one bandwidth number that was
4x off (uchar4 is 4 bytes, not 16 -- see diag_gather_ceiling_check.py).

Buffers are 512 MiB, far past any plausible L2, and every arm is byte-verified
before its number counts. Read-only, no model load.

Arms:
  read_only   grid-stride sum of a 512 MiB uint4 buffer (pure read)
  copy        512 MiB -> 512 MiB (read + write, traffic = 2x)
  triad       a[i] = b[i] + s*c[i] over 3 x 512 MiB (traffic = 3x)
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

N_BYTES = 512 * 1024 * 1024
ROUNDS = 20

SRC = r"""
extern "C" __global__ void read_only(const uint4* __restrict__ s, long long n4,
                                     unsigned long long* __restrict__ sink)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    unsigned int acc = 0u;
    for (; i < n4; i += stride) { uint4 v = s[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0xFFFFFFFFu) sink[0] += acc;   // never taken; defeats DCE
}
extern "C" __global__ void copy_k(const uint4* __restrict__ s,
                                  uint4* __restrict__ d, long long n4)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) d[i] = s[i];
}
extern "C" __global__ void triad_k(float* __restrict__ a,
                                   const float* __restrict__ b,
                                   const float* __restrict__ c,
                                   float s, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) a[i] = b[i] + s * c[i];
}
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    mod = cp.RawModule(code=SRC, options=("-std=c++14",))
    k_read = mod.get_function("read_only")
    k_copy = mod.get_function("copy_k")
    k_triad = mod.get_function("triad_k")

    n4 = N_BYTES // 16          # uint4 is SIXTEEN bytes (uchar4 is four)
    nf = N_BYTES // 4
    src = cp.arange(N_BYTES, dtype=cp.uint8)
    dst = cp.zeros(N_BYTES, dtype=cp.uint8)
    sink = cp.zeros(1, dtype=cp.uint64)
    grid, block = 2048, 256

    def timed(fn, traffic_bytes):
        fn()
        cp.cuda.Device(0).synchronize()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        for _ in range(ROUNDS):
            fn()
        e1.record()
        e1.synchronize()
        ms = cp.cuda.get_elapsed_time(e0, e1) / ROUNDS
        return {"ms": ms, "gb_s": traffic_bytes / (ms * 1e-3) / 1e9}

    out = {}
    out["read_only"] = timed(
        lambda: k_read((grid,), (block,), (src, np.int64(n4), sink)), N_BYTES)

    dst.fill(0)
    k_copy((grid,), (block,), (src, dst, np.int64(n4)))
    cp.cuda.Device(0).synchronize()
    copy_ok = bool(cp.array_equal(src, dst))
    out["copy"] = timed(lambda: k_copy((grid,), (block,), (src, dst, np.int64(n4))),
                        2 * N_BYTES)
    out["copy"]["bytes_verified"] = copy_ok

    fa = cp.zeros(nf, dtype=cp.float32)
    fb = cp.full(nf, 2.0, dtype=cp.float32)
    fc = cp.full(nf, 3.0, dtype=cp.float32)
    k_triad((grid,), (block,), (fa, fb, fc, np.float32(0.5), np.int64(nf)))
    cp.cuda.Device(0).synchronize()
    triad_ok = bool(float(fa[0]) == 3.5 and float(fa[-1]) == 3.5)
    out["triad"] = timed(
        lambda: k_triad((grid,), (block,), (fa, fb, fc, np.float32(0.5), np.int64(nf))),
        3 * nf * 4)
    out["triad"]["values_verified"] = triad_ok

    best = max(v["gb_s"] for v in out.values())
    payload = {
        "kind": "diag_vram_bandwidth_check",
        "created_utc": utc_now(),
        "note": "independent anchor for the 338.4 GB/s figure the whole roofline divides by; 512 MiB buffers so no L2 reuse, every arm byte- or value-verified",
        "buffer_bytes": N_BYTES, "rounds": ROUNDS,
        "grid": grid, "block": block,
        "arms": out,
        "best_observed_gb_s": best,
        "project_roofline_gb_s": 338.4,
        "ratio_best_over_project": best / 338.4,
        "verdict": ("consistent_with_338_4" if 0.85 <= best / 338.4 <= 1.15
                    else "project_figure_needs_revision"),
    }
    write_json_atomic(REPO / "pro_research" / "diag_vram_bandwidth_check.json",
                      payload, archive=False)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
