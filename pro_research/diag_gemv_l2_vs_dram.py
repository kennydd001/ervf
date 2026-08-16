"""Why is the same FP8 GEMV kernel ~2.2x slower inside the decode loop than on
its own?

Two measurements of the identical kernel disagree:

  * isolated (diag_fp8_lutfree_gemv.json): 295-357 GB/s on the real Mamba and
    attention shapes -- 85-103% of the 345.9 GB/s this device delivers
    (diag_vram_bandwidth_check.json, 512 MiB buffers, verified);
  * in the real decode loop (diag_component_marginals_v6.json, S12 marginal
    method): Mamba 154.4 GB/s, attention 151.7 GB/s.

Both cannot describe the same hardware doing the same work, and the project's
working rule is to resolve a disagreement rather than keep the convenient half.
Today already produced two cases where the isolated number was the misleading
one (the gather's 1.380 ms became 0.701 ms in the loop; a 97.9 GB/s "ceiling"
turned out to be a four-byte type read as sixteen).

## The leading suspect

The isolated benchmark re-reads ONE matrix 200 times in a tight loop. Mamba's
in_proj is 27.7 MB and out_proj 11.0 MB. If those sit in L2 between rounds, the
"bandwidth" measured is partly L2 bandwidth, and no in-loop kernel can ever
reproduce it -- in the real loop each weight matrix is touched once per token,
with 2 GB of other traffic in between, so every read is a cold DRAM read.

## Arms (one variable: whether the working set can stay in L2)

  hot_single     the isolated benchmark's own pattern: one matrix, re-read
  cold_cycle_N   N distinct matrices of the same shape, visited round-robin,
                 sized so the total working set exceeds L2 several times over

Same kernel, same shape, same launch geometry, same round count. If cold_cycle
lands near the in-loop 152 GB/s, the isolated figure was an L2 artifact and the
GEMV is already at its DRAM roofline in the loop -- which would close the
"dense GEMV is only at 45% efficiency" lever entirely. If cold_cycle stays near
300 GB/s, the loop is losing the bandwidth to something else and the lever is
real.

Also records SM/memory clocks and temperature before and after each arm, since
a 36% sustained-load SM clock drop is already on record for this machine
(diag_lmhead_throttle_check.json), and reports the device's actual L2 size
rather than assuming one.

Read-only, no model load.
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
from diag_fp8_lutfree_gemv import SRC

ROUNDS = 200
SHAPES = [("mamba_in_proj", 10304, 2688), ("mamba_out_proj", 2688, 4096)]
CYCLE_COUNTS = [1, 4, 12]


def _clocks() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,pstate",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        sm, mem, temp, ps = [v.strip() for v in out.split(",")]
        return {"sm_mhz": int(sm), "mem_mhz": int(mem), "temp_c": int(temp), "pstate": ps}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    require_gpu_free()
    import cupy as cp

    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    l2_bytes = int(props.get("l2CacheSize", 0))

    mod = cp.RawModule(code=SRC, options=("-std=c++14",))
    k_ref = mod.get_function("gemv_fp8_ref")

    rng = np.random.default_rng(20260816)
    arms = {}

    for name, rows, cols in SHAPES:
        wbytes = rows * cols
        x = cp.asarray(rng.standard_normal(cols).astype(np.float32))
        out = cp.zeros(rows, dtype=cp.float32)
        blocks = (rows + 15) // 16
        smem = (cols + 256) * 4

        for ncyc in CYCLE_COUNTS:
            working_set = wbytes * ncyc
            if working_set > 3_500_000_000:
                continue
            mats = [cp.asarray(rng.integers(0, 256, size=wbytes, dtype=np.uint8))
                    for _ in range(ncyc)]

            def run(i):
                k_ref((blocks,), (256,),
                      (mats[i % ncyc], x, out, np.float32(1.0),
                       np.int32(rows), np.int32(cols)), shared_mem=smem)

            run(0)
            dev.synchronize()
            before = _clocks()
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record()
            for i in range(ROUNDS):
                run(i)
            e1.record()
            e1.synchronize()
            after = _clocks()
            ms = cp.cuda.get_elapsed_time(e0, e1) / ROUNDS

            label = f"{name}__cycle{ncyc}"
            arms[label] = {
                "shape": [rows, cols],
                "matrices_in_rotation": ncyc,
                "weight_bytes_per_call": wbytes,
                "working_set_bytes": working_set,
                "working_set_over_l2": (working_set / l2_bytes) if l2_bytes else None,
                "ms_per_call": ms,
                "gb_s": wbytes / (ms * 1e-3) / 1e9,
                "clocks_before": before,
                "clocks_after": after,
            }
            del mats
            cp.get_default_memory_pool().free_all_blocks()

        del x, out
        cp.get_default_memory_pool().free_all_blocks()

    def gb(label):
        return arms[label]["gb_s"] if label in arms else None

    hot = gb("mamba_in_proj__cycle1")
    cold = gb(f"mamba_in_proj__cycle{CYCLE_COUNTS[-1]}")
    verdict = "undetermined"
    if hot and cold:
        if cold < 0.7 * hot:
            verdict = "isolated_number_was_an_L2_artifact"
        elif cold > 0.9 * hot:
            verdict = "isolated_number_survives_cold_working_set"
        else:
            verdict = "partial_L2_effect"

    payload = {
        "kind": "diag_gemv_l2_vs_dram",
        "created_utc": utc_now(),
        "note": "resolves the disagreement between the isolated FP8 GEMV (295-357 GB/s) and the same kernel's in-loop marginal (152-154 GB/s) by varying only whether the working set can stay resident in L2",
        "device": {
            "name": props["name"].decode() if isinstance(props.get("name"), bytes) else str(props.get("name")),
            "l2_cache_bytes": l2_bytes,
            "l2_cache_mib": l2_bytes / 1024 / 1024 if l2_bytes else None,
            "multiprocessor_count": int(props.get("multiProcessorCount", 0)),
        },
        "reference_points": {
            "device_read_gb_s_512MiB_verified": 345.9,
            "in_loop_mamba_gb_s_marginal": 154.4,
            "in_loop_attn_gb_s_marginal": 151.7,
        },
        "rounds": ROUNDS,
        "arms": arms,
        "verdict": verdict,
    }
    write_json_atomic(REPO / "pro_research" / "diag_gemv_l2_vs_dram.json",
                      payload, archive=False)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
