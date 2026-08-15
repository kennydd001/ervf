"""Isolated bit-exact verification of gather_down_sparse_ind_batched and
gemv_down_masked_partial_ind_batched against their per-slot reference
kernels, on synthetic data at real dimensions. No runtime/model load.

Uses panel_scan_batched (already verified in
verify_down_proj_batch_kernels.py) to generate realistic sparse panel
metadata from random ReLU2-like activations, then feeds that into both the
reference (sequential, one shared mirror reused per slot -- exactly how
_moe_dev works today) and batched (parallel, top_k independent mirrors)
gather+down_masked paths, comparing outputs.

Not a gated PRO experiment. Step 1 before any integration into _moe_dev.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_gpu_free, utc_now, write_json_atomic

ROWS = 2688      # hidden_size
INTER = 1856     # moe_intermediate_size
TOP_K = 6
NPANEL = INTER // 16   # 116
NCHUNKS = 8             # matches fused.nchunks
CAP = 12
N_EXPERTS = 128
ROWHALF = ROWS // 2
PANEL_STRIDE = ROWS + 16 * ROWHALF
MIRROR_BYTES = NPANEL * PANEL_STRIDE  # == panel_bytes for one expert record


def main() -> int:
    require_gpu_free()
    import cupy as cp
    import numpy as np
    from down_gather_batch_kernels import DownGatherBatchKernels
    from down_proj_batch_kernels import DownProjBatchKernels

    cp.random.seed(20260816)
    scan_k = DownProjBatchKernels()
    gk = DownGatherBatchKernels()

    # Synthetic "host" bank: sized for the full expert id range, since ids
    # (used to index into down_base, same as the real bank's expert-id
    # addressing) are drawn from 0..N_EXPERTS-1, not 0..CAP-1 (CAP is the
    # up-proj device-cache capacity, an unrelated concept -- down_base
    # indexes by raw expert id, not cache slot).
    down_base = cp.random.randint(0, 256, size=N_EXPERTS * MIRROR_BYTES, dtype=cp.uint8)
    globals_dev = cp.random.standard_normal(N_EXPERTS * 2, dtype=cp.float32)
    e2m1_lut = cp.random.standard_normal(16, dtype=cp.float32)
    e4m3_lut = cp.random.standard_normal(256, dtype=cp.float32)

    checks = []
    for trial, sparsity in enumerate((0.3, 0.5, 0.7)):
        cp.random.seed(2000 + trial)
        ids = cp.random.permutation(N_EXPERTS)[:TOP_K].astype(cp.int32)

        act_batched = cp.random.standard_normal(TOP_K * INTER, dtype=cp.float32)
        zero_mask = cp.random.random(TOP_K * INTER) < sparsity
        act_batched = cp.where(zero_mask, cp.float32(0.0), act_batched)

        # Generate realistic panel metadata via the already-verified batched scan.
        masks_b, plist_b, pcount_b, nz_b, nzc_b = scan_k.run_panel_scan_batched(act_batched, INTER, TOP_K)

        # ---- reference: sequential, ONE shared mirror reused per slot
        # (exactly _moe_dev's current pattern), one shared `partials`.
        mirror_ref = cp.zeros(MIRROR_BYTES, dtype=cp.uint8)
        ref_partials_all = []
        ref_mirrors = []
        for s in range(TOP_K):
            mirror_ref.fill(0)
            plist_s = plist_b[s * NPANEL:(s + 1) * NPANEL]
            masks_s = masks_b[s * NPANEL:(s + 1) * NPANEL]
            pcount_s = pcount_b[s:s + 1]
            nz_s = nz_b[s * INTER:(s + 1) * INTER]
            nzc_s = nzc_b[s:s + 1]
            act_s = act_batched[s * INTER:(s + 1) * INTER]
            id_s = ids[s:s + 1]

            blocks = ((INTER + NPANEL) * 32 + 255) // 256
            gk.run_gather_ref(down_base, id_s, MIRROR_BYTES, mirror_ref, plist_s, pcount_s, nz_s, nzc_s, ROWS, blocks)
            ref_mirrors.append(cp.asnumpy(mirror_ref).copy())

            partials_s = cp.zeros(NCHUNKS * ROWS, dtype=cp.float32)
            gk.run_down_masked_ref(mirror_ref, id_s, globals_dev, act_s, plist_s, masks_s,
                                   pcount_s, e2m1_lut, e4m3_lut, partials_s, ROWS, INTER, NCHUNKS)
            ref_partials_all.append(cp.asnumpy(partials_s))
        ref_all = np.concatenate(ref_partials_all)

        # ---- batched: top_k independent mirrors, one launch each.
        mirror_batched = cp.zeros(TOP_K * MIRROR_BYTES, dtype=cp.uint8)
        blocks = ((INTER + NPANEL) * 32 + 255) // 256
        gk.run_gather_batched(down_base, ids, MIRROR_BYTES, mirror_batched, plist_b, pcount_b,
                              nz_b, nzc_b, ROWS, NPANEL, INTER, MIRROR_BYTES, TOP_K, blocks)
        mirror_batched_np = cp.asnumpy(mirror_batched)
        mirror_mismatch = None
        for s in range(TOP_K):
            slot_mirror = mirror_batched_np[s * MIRROR_BYTES:(s + 1) * MIRROR_BYTES]
            if not (ref_mirrors[s] == slot_mirror).all():
                diff_idx = int(np.argmax(ref_mirrors[s] != slot_mirror))
                mirror_mismatch = {
                    "slot": s, "first_diff_index": diff_idx,
                    "ref_byte": int(ref_mirrors[s][diff_idx]), "batched_byte": int(slot_mirror[diff_idx]),
                    "mismatch_count": int((ref_mirrors[s] != slot_mirror).sum()),
                }
                break

        partials_batched = cp.zeros(TOP_K * NCHUNKS * ROWS, dtype=cp.float32)
        gk.run_down_masked_batched(mirror_batched, ids, globals_dev, act_batched, plist_b, masks_b,
                                   pcount_b, e2m1_lut, e4m3_lut, partials_batched, ROWS, INTER,
                                   NPANEL, MIRROR_BYTES, TOP_K, NCHUNKS)
        batched_np = cp.asnumpy(partials_batched)

        print(f"trial {trial} sparsity={sparsity} mirror_mismatch={mirror_mismatch}", flush=True)
        ref_nan = int(np.isnan(ref_all).sum())
        batched_nan = int(np.isnan(batched_np).sum())
        ok = bool(ref_nan == 0 and batched_nan == 0 and (ref_all == batched_np).all())
        finite = np.isfinite(ref_all) & np.isfinite(batched_np)
        with np.errstate(over="ignore", invalid="ignore"):
            diffs = np.where(finite, abs(ref_all.astype(np.float64) - batched_np.astype(np.float64)), 0.0)
        max_abs_diff = float(diffs.max()) if finite.any() else None
        checks.append({
            "trial": trial, "sparsity": sparsity, "bit_exact": ok,
            "max_abs_diff": max_abs_diff, "ref_nan_count": ref_nan, "batched_nan_count": batched_nan,
            "ref_first_nan_index": int(np.argmax(np.isnan(ref_all))) if ref_nan else None,
            "batched_first_nan_index": int(np.argmax(np.isnan(batched_np))) if batched_nan else None,
            "mirror_mismatch": mirror_mismatch,
        })

    overall = all(c["bit_exact"] for c in checks)
    payload = {
        "kind": "verify_down_gather_batch_kernels",
        "created_utc": utc_now(),
        "note": "isolated bit-exact structural test, no runtime/model load",
        "dims": {"rows": ROWS, "inter": INTER, "top_k": TOP_K, "npanel": NPANEL, "nchunks": NCHUNKS},
        "checks": checks,
        "overall_pass": overall,
    }
    out = REPO / "pro_research" / "verify_down_gather_batch_kernels.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
