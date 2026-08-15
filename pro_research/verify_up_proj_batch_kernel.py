"""Isolated bit-exact verification of gemv_nvfp4_ervf_ind_batched against
the per-slot reference kernel, on synthetic data at real model dimensions
(moe_intermediate_size 1856 rows, hidden_size 2688 cols, top_k 6). No
runtime/model load required -- structural equivalence test: the batched
kernel must match the sequential reference regardless of what the random
bytes decode to, since only slot/id/output addressing changes between them,
not the per-thread arithmetic.

Not a gated PRO experiment. Step 1 before any integration into _moe_dev.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_gpu_free, utc_now, write_json_atomic

ROWS = 1856     # moe_intermediate_size (up-proj output dim)
COLS = 2688     # hidden_size (up-proj input dim)
TOP_K = 6
CAP = 12        # synthetic cache capacity, > top_k so slot != id in general
N_EXPERTS = 128


def main() -> int:
    require_gpu_free()
    import cupy as cp
    from up_proj_batch_kernels import UpProjBatchKernels

    cp.random.seed(20260816)
    k = UpProjBatchKernels()

    code_stride = ROWS * (COLS // 2)
    scale_stride = ROWS * (COLS // 16)

    codes_base = cp.random.randint(0, 256, size=CAP * code_stride, dtype=cp.uint8)
    scales_base = cp.random.randint(0, 256, size=CAP * scale_stride, dtype=cp.uint8)
    globals_dev = cp.random.standard_normal(N_EXPERTS * 2, dtype=cp.float32)
    e2m1_lut = cp.random.standard_normal(16, dtype=cp.float32)
    e4m3_lut = cp.random.standard_normal(256, dtype=cp.float32)
    x = cp.random.standard_normal(COLS, dtype=cp.float32)

    checks = []
    for trial, apply_relu2 in enumerate((False, True, False)):
        cp.random.seed(1000 + trial)
        slots = cp.random.permutation(CAP)[:TOP_K].astype(cp.int32)
        ids = cp.random.permutation(N_EXPERTS)[:TOP_K].astype(cp.int32)

        ref_outs = []
        for s in range(TOP_K):
            out_s = cp.zeros(ROWS, dtype=cp.float32)
            k.run_ref(out_s, codes_base, scales_base, slots[s:s + 1], ids[s:s + 1],
                     globals_dev, 1, e2m1_lut, e4m3_lut, x, ROWS, COLS,
                     apply_relu2, code_stride, scale_stride)
            ref_outs.append(cp.asnumpy(out_s))
        import numpy as np
        ref_all = np.concatenate(ref_outs)

        out_batched = cp.zeros(TOP_K * ROWS, dtype=cp.float32)
        k.run_batched(out_batched, codes_base, scales_base, slots, ids,
                      globals_dev, 1, e2m1_lut, e4m3_lut, x, ROWS, COLS,
                      apply_relu2, code_stride, scale_stride, TOP_K)
        batched_np = cp.asnumpy(out_batched)

        ok = bool((ref_all == batched_np).all())
        max_abs_diff = float(abs(ref_all - batched_np).max())
        checks.append({"trial": trial, "apply_relu2": apply_relu2, "bit_exact": ok, "max_abs_diff": max_abs_diff})

    overall = all(c["bit_exact"] for c in checks)
    payload = {
        "kind": "verify_up_proj_batch_kernel",
        "created_utc": utc_now(),
        "note": "isolated bit-exact structural test, no runtime/model load",
        "dims": {"rows": ROWS, "cols": COLS, "top_k": TOP_K, "cap": CAP, "n_experts": N_EXPERTS},
        "checks": checks,
        "overall_pass": overall,
    }
    out = REPO / "pro_research" / "verify_up_proj_batch_kernel.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
