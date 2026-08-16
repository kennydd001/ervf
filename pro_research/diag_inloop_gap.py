"""Why does the same kernel reach only 64% of its isolated speed inside the loop?

    device, pure streaming                       345.9 GB/s
    ERVF isolated, cold rotation, N=1            248-267   (72-77%)
    Mamba in-loop (892 MB / 5.168 ms, in-graph)  172.6     (50%)

Same kernel, same shape, same dtype. If every GEMV ran at its isolated rate a
token would cost 7.67 ms of VRAM plus 2.47 ms of PCIe = 10.1 ms ~ 99 tok/s;
we measure 21.24 ms. Nearly half the token is in this gap, it is larger than
anything attempted today, and it is orthogonal to the single-stream/batch
choice -- closing it helps both.

Three hypotheses were filed. Two of them can be tested in isolation, without a
model load, by reproducing the loop's *environment* around the isolated kernel:

  H1  L2 eviction between layers. The per-token working set is 2 GB, so nothing
      the kernel leaves in L2 (32 MiB) survives until the same kernel runs
      again 52 layers later. The isolated benchmark re-runs back to back.
      Test: stream a 64 MiB buffer between calls to evict L2.

  H2  Bandwidth contention with `copy_stream`. In the real loop the MoE cache
      fetch pulls expert misses over PCIe on a second stream while the dense
      GEMVs run -- ~20 misses/token x 2.81 MB.
      Test: run a continuous H2D copy on a second stream underneath.

  H3  Thermal throttling. Not tested here (it needs a sustained run, and clocks
      are recorded per arm below so the effect would be visible if it appeared).

If an arm reproduces ~172 GB/s, that hypothesis explains the gap. If none does,
the cause is something the loop does that this harness does not, and that is
worth knowing too -- today has already shown twice that an isolated number can
mislead, so the arms are read as evidence about the gap, not as the gap itself.

Arms all run the identical ERVF kernel on the identical cold rotation; only the
surrounding environment changes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_gpu_free, utc_now, write_json_atomic
from diag_ervf_batched_tiled import HEAD

ROWS, COLS = 10304, 2688          # Mamba in_proj
CYCLE = 8
ROUNDS = 100
L2_FLUSH_MB = 64
PCIE_CHUNK_MB = 8

EXTRA = r"""
extern "C" __global__ void stream_touch(const uint4* __restrict__ s,
                                        unsigned long long* __restrict__ sink,
                                        long long n4)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    unsigned int acc = 0u;
    for (; i < n4; i += stride) { uint4 v = s[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0xFFFFFFFFu) sink[0] += acc;
}
"""


def _clocks() -> dict:
    try:
        o = subprocess.run(["nvidia-smi",
                            "--query-gpu=clocks.sm,clocks.mem,temperature.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        sm, mem, t = [x.strip() for x in o.split(",")]
        return {"sm_mhz": int(sm), "mem_mhz": int(mem), "temp_c": int(t)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    require_gpu_free()
    import cupy as cp

    mod = cp.RawModule(code=HEAD + EXTRA, options=("-std=c++14",))
    k_ervf = mod.get_function("prod_ervf16")
    k_touch = mod.get_function("stream_touch")

    rng = np.random.default_rng(20260816)
    mats = [cp.asarray(rng.integers(0, 256, size=ROWS * COLS, dtype=np.uint8))
            for _ in range(CYCLE)]
    X = cp.asarray(rng.standard_normal(COLS).astype(np.float32))
    out = cp.zeros(ROWS, dtype=cp.float32)
    ws = np.float32(0.0123)
    blocks = (ROWS + 15) // 16
    smem = (COLS + 256) * 4
    wbytes = ROWS * COLS

    flush = cp.zeros(L2_FLUSH_MB * 1024 * 1024, dtype=cp.uint8)
    sink = cp.zeros(1, dtype=cp.uint64)
    n4 = flush.size // 16

    pin = cp.cuda.alloc_pinned_memory(PCIE_CHUNK_MB * 1024 * 1024)
    hbuf = np.frombuffer(pin, dtype=np.uint8, count=PCIE_CHUNK_MB * 1024 * 1024)
    hbuf[:] = 7
    dbuf = cp.zeros(hbuf.size, dtype=cp.uint8)
    side = cp.cuda.Stream(non_blocking=True)
    rt = cp.cuda.runtime

    def gemv(i):
        k_ervf((blocks,), (256,), (mats[i % CYCLE], X, out, ws,
                                   np.int32(ROWS), np.int32(COLS)), shared_mem=smem)

    def l2_flush():
        k_touch((1024,), (256,), (flush, sink, np.int64(n4)))

    def pcie_burst():
        rt.memcpyAsync(int(dbuf.data.ptr), int(hbuf.ctypes.data), hbuf.size,
                       rt.memcpyHostToDevice, side.ptr)

    def measure(label, per_call=None, background=False):
        for i in range(3):
            gemv(i)
        cp.cuda.Device(0).synchronize()
        before = _clocks()
        if background:
            for _ in range(4):
                pcie_burst()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        for i in range(ROUNDS):
            if per_call is not None:
                per_call()
            if background and i % 4 == 0:
                pcie_burst()
            gemv(i)
        e1.record()
        e1.synchronize()
        side.synchronize()
        total = cp.cuda.get_elapsed_time(e0, e1) / ROUNDS
        after = _clocks()
        return {"ms_per_call_including_environment": total,
                "clocks_before": before, "clocks_after": after}

    # baseline first, then each environment, then baseline again as a drift check
    base_a = measure("baseline_a")
    ms_base = base_a["ms_per_call_including_environment"]
    gb_base = wbytes / (ms_base * 1e-3) / 1e9

    # H1: the flush kernel's own time must be subtracted -- measure it alone
    cp.cuda.Device(0).synchronize()
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    for _ in range(ROUNDS):
        l2_flush()
    e1.record()
    e1.synchronize()
    ms_flush = cp.cuda.get_elapsed_time(e0, e1) / ROUNDS

    h1 = measure("l2_evicted", per_call=l2_flush)
    h2 = measure("pcie_contention", background=True)
    h3 = measure("both", per_call=l2_flush, background=True)
    base_b = measure("baseline_b")

    def net(rec, subtract=0.0):
        ms = rec["ms_per_call_including_environment"] - subtract
        return {"gemv_ms": ms, "gemv_gb_s": wbytes / (ms * 1e-3) / 1e9,
                "clocks_after": rec["clocks_after"]}

    arms = {
        "baseline_a": net(base_a),
        "l2_evicted": net(h1, ms_flush),
        "pcie_contention": net(h2),
        "both": net(h3, ms_flush),
        "baseline_b": net(base_b),
    }
    drift = abs(arms["baseline_a"]["gemv_gb_s"] - arms["baseline_b"]["gemv_gb_s"])
    target = 172.6

    def closeness(v):
        return abs(v["gemv_gb_s"] - target)

    best = min((k for k in ("l2_evicted", "pcie_contention", "both")),
               key=lambda k: closeness(arms[k]))
    verdict = (f"{best}_reproduces_in_loop_rate"
               if closeness(arms[best]) < 0.15 * target
               else "no_arm_reproduces_in_loop_rate")

    payload = {
        "kind": "diag_inloop_gap",
        "created_utc": utc_now(),
        "note": "Reproduces the loop's ENVIRONMENT around the isolated ERVF kernel to test why in-loop throughput is 64% of isolated. The L2-flush kernel's own measured time is subtracted from the arms that use it. Arms are evidence about the gap, not the gap itself.",
        "shape": [ROWS, COLS], "rounds": ROUNDS,
        "l2_flush_mb": L2_FLUSH_MB, "l2_flush_ms_alone": ms_flush,
        "pcie_chunk_mb": PCIE_CHUNK_MB,
        "reference": {"device_stream_gb_s": 345.9,
                      "isolated_ervf_gb_s": gb_base,
                      "in_loop_mamba_gb_s": target},
        "arms": arms,
        "baseline_drift_gb_s": drift,
        "verdict": verdict,
    }
    write_json_atomic(REPO / "pro_research" / "diag_inloop_gap.json", payload,
                      archive=False)
    print(json.dumps({
        "reference": payload["reference"],
        "arms_gb_s": {k: round(v["gemv_gb_s"], 1) for k, v in arms.items()},
        "baseline_drift_gb_s": round(drift, 1),
        "l2_flush_ms_alone": round(ms_flush, 4),
        "verdict": verdict,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
