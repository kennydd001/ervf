"""Head-to-head input: what does OUR NVFP4 ERVF GEMV cost on the shapes that
are already NVFP4 in the checkpoint?

C2b proved native Blackwell FP4 executes and sustains 292-303 GB/s, with M=2
free. But that only becomes a *format-preserving* win where the checkpoint is
already NVFP4 -- there the tensor core changes the accumulation order and
nothing else. Where the source is FP8 (Mamba, 892 MB/token) or BF16
(attention), moving to FP4 is a real quantisation change with an unmeasured
quality cost, and must not be counted in the same column.

So the decisive comparison is only on the genuinely-NVFP4 shapes:

    lm_head       131072 x 2688     already NVFP4   (C2b measured this exact shape)
    shared_up       3712 x 2688     already NVFP4
    shared_down     2688 x 3712     already NVFP4
    routed_up       1856 x 2688     already NVFP4, x6 experts x 23 layers

This measures our production path (`fused.gemv_into`, which takes the ERVF
branch by default) on those shapes under the SAME protocol C2b used: CUDA event
timing, p50 over rounds, and a cold rotation over enough distinct matrices that
the working set is several times the 32 MiB L2. That last part is not optional
-- re-reading one matrix measured 336 GB/s earlier today where the cold rate was
230, and a 1.46x L2 artifact is larger than the effect being tested.

NVFP4 weight bytes are counted as rows*cols*0.5625 (4-bit code + one F8_E4M3
block scale per 16 codes), matching the checkpoint's own footprint.

Read-only, synthetic weights, no model load. Nothing here is a tok/s claim.
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

ROUNDS = 200
L2_TARGET_MULTIPLE = 4.0     # cold: working set at least this many times L2

# (name, rows, cols, calls_per_token) -- all already NVFP4 in the checkpoint
SHAPES = [
    ("lm_head",     131072, 2688, 1),
    ("shared_up",     3712, 2688, 23),
    ("shared_down",   2688, 3712, 23),
    ("routed_up",     1856, 2688, 138),   # 6 experts x 23 layers
]

# C2b native-FP4 p50 (ms) for the shapes it covered, for the direct column.
C2B_NATIVE = {"lm_head": {"M1": 0.5820159912109375, "M2": 0.5841703891754151}}


def main() -> int:
    require_gpu_free()
    import cupy as cp
    from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

    fused = FusedNVFP4()
    l2 = int(cp.cuda.runtime.getDeviceProperties(0).get("l2CacheSize", 0))
    rng = np.random.default_rng(20260816)
    arms = {}

    for name, rows, cols in [(n, r, c) for n, r, c, _ in SHAPES]:
        code_bytes = rows * cols // 2
        scale_bytes = rows * (cols // 16)
        nvfp4_bytes = code_bytes + scale_bytes          # == rows*cols*0.5625

        cycle = max(2, int(np.ceil(L2_TARGET_MULTIPLE * l2 / nvfp4_bytes)))
        cycle = min(cycle, 24)                          # keep VRAM sane
        mats = []
        for _ in range(cycle):
            mats.append((
                cp.asarray(rng.integers(0, 256, size=code_bytes, dtype=np.uint8)),
                cp.asarray(rng.integers(0, 256, size=scale_bytes, dtype=np.uint8)),
            ))
        x = cp.asarray(rng.standard_normal(cols).astype(np.float32))
        out = cp.zeros(rows, dtype=cp.float32)

        def run(i):
            c, s = mats[i % cycle]
            fused.gemv_into(out, c, s, x, 1.0, rows, cols)

        run(0)
        cp.cuda.Device(0).synchronize()
        samples = []
        for _ in range(7):
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record()
            for i in range(ROUNDS):
                run(i)
            e1.record()
            e1.synchronize()
            samples.append(cp.cuda.get_elapsed_time(e0, e1) / ROUNDS)
        samples.sort()
        p50 = samples[len(samples) // 2]

        rec = {
            "rows": rows, "cols": cols,
            "nvfp4_weight_bytes": nvfp4_bytes,
            "matrices_in_rotation": cycle,
            "working_set_bytes": nvfp4_bytes * cycle,
            "working_set_over_l2": (nvfp4_bytes * cycle) / l2 if l2 else None,
            "ervf_p50_ms": p50,
            "ervf_gb_s": nvfp4_bytes / (p50 * 1e-3) / 1e9,
            "event_samples_ms": samples,
        }
        if name in C2B_NATIVE:
            n1 = C2B_NATIVE[name]["M1"]
            n2 = C2B_NATIVE[name]["M2"]
            rec["c2b_native_fp4_M1_ms"] = n1
            rec["c2b_native_fp4_M2_ms"] = n2
            rec["native_speedup_M1"] = p50 / n1
            rec["native_speedup_per_token_at_M2"] = p50 / (n2 / 2.0)
            rec["comparison_note"] = "format-preserving: this shape is NVFP4 in the checkpoint, so native FP4 changes the accumulation order only, not the quantisation"
        arms[name] = rec
        del mats, x, out
        cp.get_default_memory_pool().free_all_blocks()

    per_token = {n: arms[n]["ervf_p50_ms"] * calls for n, _, _, calls in SHAPES}

    payload = {
        "kind": "diag_nvfp4_ervf_reference_rates",
        "created_utc": utc_now(),
        "note": "our production NVFP4 ERVF GEMV on the already-NVFP4 shapes, measured under C2b's protocol with a cold rotation (>=4x the 32 MiB L2) so the earlier 1.46x L2 artifact cannot recur. Only these shapes permit a format-preserving comparison against native FP4; Mamba (FP8) and attention (BF16) would require a real quantisation change and belong in a different column.",
        "device_l2_mib": l2 / 1024 / 1024 if l2 else None,
        "rounds": ROUNDS,
        "arms": arms,
        "ervf_ms_per_token_by_shape": per_token,
        "ervf_ms_per_token_total_nvfp4_shapes": sum(per_token.values()),
        "claim_boundary": "kernel-level rates only; no tok/s claim, and native FP4 is not bit-exact against the ERVF reduction tree",
    }
    write_json_atomic(REPO / "pro_research" / "diag_nvfp4_ervf_reference_rates.json",
                      payload, archive=False)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
