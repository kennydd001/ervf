"""V5 step 1: isolated bit-exact verification of panel_scan_batched and
reduce_partials_batched against their reference (per-slot) kernels, on
synthetic data matching the real model's dimensions (moe_intermediate_size
1856, hidden_size 2688, top_k 6). No runtime, no model weights, no GPU model
load required -- a pure kernel-level unit test, the correct first step
before touching _moe_dev's orchestration at all.

Not a gated PRO experiment. This only tests the two kernels in isolation;
it does NOT establish that integrating them into _moe_dev is correct (that
requires the full BASE_A/BATCHED/BASE_B/CTL causal A/B in
PRO_V5_PREREGISTRATION.md, not attempted until this passes).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_gpu_free, utc_now, write_json_atomic

INTER = 1856   # moe_intermediate_size
ROWS = 2688    # hidden_size
TOP_K = 6
NCHUNKS = 4    # matches nchunks used elsewhere for down_masked partials


def main() -> int:
    require_gpu_free()
    import cupy as cp
    from down_proj_batch_kernels import DownProjBatchKernels

    cp.random.seed(20260816)
    k = DownProjBatchKernels()

    results = {}

    # ---- panel_scan: sparsity levels matter (real data is ReLU2-sparse,
    # roughly 30-70% zero per the fused_nvfp4.py comment on down_proj
    # design) -- test several sparsity levels, not just one.
    panel_scan_checks = []
    for sparsity in (0.0, 0.3, 0.5, 0.7, 0.95, 1.0):
        act_batched = cp.random.standard_normal(TOP_K * INTER, dtype=cp.float32)
        zero_mask = cp.random.random(TOP_K * INTER) < sparsity
        act_batched = cp.where(zero_mask, cp.float32(0.0), act_batched)

        ref_masks, ref_list, ref_count, ref_nz, ref_nzc = [], [], [], [], []
        for s in range(TOP_K):
            act_s = act_batched[s * INTER:(s + 1) * INTER]
            m, l, c, nz, nzc = k.run_panel_scan_ref(act_s, INTER)
            ref_masks.append(cp.asnumpy(m))
            ref_list.append(cp.asnumpy(l))
            ref_count.append(int(cp.asnumpy(c)[0]))
            ref_nz.append(cp.asnumpy(nz))
            ref_nzc.append(int(cp.asnumpy(nzc)[0]))

        b_masks, b_list, b_count, b_nz, b_nzc = k.run_panel_scan_batched(act_batched, INTER, TOP_K)
        b_masks_np = cp.asnumpy(b_masks).reshape(TOP_K, -1)
        b_list_np = cp.asnumpy(b_list).reshape(TOP_K, -1)
        b_count_np = cp.asnumpy(b_count)
        b_nz_np = cp.asnumpy(b_nz).reshape(TOP_K, -1)
        b_nzc_np = cp.asnumpy(b_nzc)

        ok = True
        for s in range(TOP_K):
            n = ref_count[s]
            nzn = ref_nzc[s]
            if int(b_count_np[s]) != n or int(b_nzc_np[s]) != nzn:
                ok = False
                break
            if not (ref_masks[s] == b_masks_np[s]).all():
                ok = False
                break
            if not (ref_list[s][:n] == b_list_np[s][:n]).all():
                ok = False
                break
            if not (ref_nz[s][:nzn] == b_nz_np[s][:nzn]).all():
                ok = False
                break
        panel_scan_checks.append({"sparsity": sparsity, "bit_exact": bool(ok)})

    results["panel_scan"] = panel_scan_checks

    # ---- reduce_partials
    reduce_checks = []
    for trial in range(3):
        partials_batched = cp.random.standard_normal(TOP_K * NCHUNKS * ROWS, dtype=cp.float32)
        ref_outs = []
        for s in range(TOP_K):
            p_s = partials_batched[s * NCHUNKS * ROWS:(s + 1) * NCHUNKS * ROWS]
            out = k.run_reduce_partials_ref(p_s, ROWS, NCHUNKS)
            ref_outs.append(cp.asnumpy(out))
        ref_all = __import__("numpy").concatenate(ref_outs)

        b_out = k.run_reduce_partials_batched(partials_batched, ROWS, NCHUNKS, TOP_K)
        b_out_np = cp.asnumpy(b_out)

        ok = bool((ref_all == b_out_np).all())
        max_abs_diff = float(abs(ref_all - b_out_np).max())
        reduce_checks.append({"trial": trial, "bit_exact": ok, "max_abs_diff": max_abs_diff})

    results["reduce_partials"] = reduce_checks

    all_panel_ok = all(c["bit_exact"] for c in panel_scan_checks)
    all_reduce_ok = all(c["bit_exact"] for c in reduce_checks)

    payload = {
        "kind": "verify_down_proj_batch_kernels",
        "created_utc": utc_now(),
        "note": "isolated bit-exact kernel unit test, no runtime/model load; step 1 of PRO_V5_PREREGISTRATION.md",
        "dims": {"inter": INTER, "rows": ROWS, "top_k": TOP_K, "nchunks": NCHUNKS},
        "panel_scan_batched_bit_exact_all_sparsity_levels": all_panel_ok,
        "reduce_partials_batched_bit_exact_all_trials": all_reduce_ok,
        "results": results,
        "overall_pass": all_panel_ok and all_reduce_ok,
    }
    out = REPO / "pro_research" / "verify_down_proj_batch_kernels.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
