"""X1 smoke: exercise the batched NVFP4 GEMM alone, on synthetic buffers.

Isolates the kernel from the oracle so a launch failure points at the kernel and
not at the harness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4  # noqa: E402
from moe_lab.lightningstream_nemotron.gpu_kernels import GPUKernels  # noqa: E402
from moe_lab.lightningstream_nemotron.sweepspec import SweepMoE  # noqa: E402


def main() -> int:
    import cupy as cp

    fused, k = FusedNVFP4(), GPUKernels()
    sweep = SweepMoE(fused, k)
    print(f"opt-in dynamic shared accepted: {sweep.max_shared} B")

    rows, cols = 1856, 2688
    rng = np.random.default_rng(0)
    codes = cp.asarray(rng.integers(0, 256, size=rows * cols // 2, dtype=np.uint8))
    scales = cp.asarray(rng.integers(0, 128, size=rows * cols // 16, dtype=np.uint8))
    gscale = 0.031
    xs = cp.asarray(rng.standard_normal(8 * cols).astype(np.float32))

    ref = cp.zeros(rows, dtype=cp.float32)
    out = cp.zeros(8 * rows, dtype=cp.float32)

    for B in range(1, 9):
        need = B * cols * 4
        nodes = cp.asarray(np.arange(B, dtype=np.int32))
        try:
            out.fill(0)
            sweep.gemm_into(out, codes, scales, xs, nodes, gscale, rows, cols, B,
                            apply_relu2=True)
            cp.cuda.Device(0).synchronize()
            ok_run = True
        except Exception as e:
            print(f"  B={B} needs {need} B: LAUNCH FAILED {type(e).__name__}: {e}")
            return 1
        # every b must equal the unbatched kernel on that same vector
        worst = 0
        for b in range(B):
            fused.gemv_into(ref, codes, scales, xs[b * cols:(b + 1) * cols],
                            gscale, rows, cols, apply_relu2=True)
            cp.cuda.Device(0).synchronize()
            same = bool(cp.array_equal(ref, out[b * rows:(b + 1) * rows]))
            worst += 0 if same else 1
        print(f"  B={B} needs {need:>6} B: run={ok_run} bit-identical rows: "
              f"{B - worst}/{B}")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
