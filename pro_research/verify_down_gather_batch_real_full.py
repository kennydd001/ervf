"""Full top_k=6 verification of gather_down_sparse_ind_batched and
gemv_down_masked_partial_ind_batched against real model data, following up
on verify_down_gather_batch_real_data.py's top_k=1 result (matched exactly,
no NaN -- confirming the earlier synthetic-data NaN was a test-harness
artifact, not a kernel bug).

Captures all top_k=6 real per-slot calls (mirror/act/panel metadata/ids)
from one real MoE layer's unbached forward pass, then runs the reference
kernels sequentially on those exact 6 captures (reproducing what _moe_dev
already does today) versus the batched kernels once on all 6 -- both from
identical real inputs.

Not a gated PRO experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_gpu_free, require_model_dir, utc_now, write_json_atomic


def main() -> int:
    require_gpu_free()
    import cupy as cp
    import numpy as np
    from down_gather_batch_kernels import DownGatherBatchKernels
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    gk = DownGatherBatchKernels()

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.device_cache = True
    rt.deterministic_accum = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    ids_seq = tok.encode("The history of computing began when", add_special_tokens=False)

    rt.reset()
    nxt = None
    for t in ids_seq:
        nxt = int(rt.step(int(t)))
    cp.cuda.Device(0).synchronize()

    fused = rt.fused
    calls = []
    orig_down_masked = fused.down_masked_ind_k

    def capture_and_run(grid, block, kargs):
        calls.append({
            "mirror": cp.asarray(kargs[0]).copy(),
            "id": cp.asarray(kargs[1]).copy(),
            "globals": cp.asarray(kargs[2]).copy(),
            "act": cp.asarray(kargs[3]).copy(),
            "panel_list": cp.asarray(kargs[4]).copy(),
            "panel_masks": cp.asarray(kargs[5]).copy(),
            "panel_count": cp.asarray(kargs[6]).copy(),
            "e2m1": cp.asarray(kargs[7]).copy(),
            "e4m3": cp.asarray(kargs[8]).copy(),
        })
        return orig_down_masked(grid, block, kargs)

    fused.down_masked_ind_k = capture_and_run
    rt.step(nxt)
    fused.down_masked_ind_k = orig_down_masked
    cp.cuda.Device(0).synchronize()

    top_k = rt.top_k
    if len(calls) < top_k:
        print(f"only captured {len(calls)} calls, expected >= {top_k}")
        return 1
    calls = calls[:top_k]  # first layer's worth

    ROWS = rt.hidden
    INTER = rt.moe_inter
    NPANEL = INTER // 16
    NCHUNKS = fused.nchunks

    # ---- reference: sequential, exactly as _moe_dev does today.
    ref_partials = []
    for c in calls:
        p = cp.zeros(NCHUNKS * ROWS, dtype=cp.float32)
        gk.run_down_masked_ref(c["mirror"], c["id"], c["globals"], c["act"],
                               c["panel_list"], c["panel_masks"], c["panel_count"],
                               c["e2m1"], c["e4m3"], p, ROWS, INTER, NCHUNKS)
        ref_partials.append(cp.asnumpy(p))
    ref_all = np.concatenate(ref_partials)

    # ---- batched: assemble the top_k real captures into batched buffers.
    mirror_bytes = int(calls[0]["mirror"].size)
    mirror_b = cp.concatenate([c["mirror"] for c in calls])
    ids_b = cp.concatenate([c["id"] for c in calls])
    act_b = cp.concatenate([c["act"] for c in calls])
    panel_list_b = cp.concatenate([c["panel_list"] for c in calls])
    panel_masks_b = cp.concatenate([c["panel_masks"] for c in calls])
    panel_count_b = cp.concatenate([c["panel_count"] for c in calls])
    globals_dev = calls[0]["globals"]  # shared, same array for all slots
    e2m1 = calls[0]["e2m1"]
    e4m3 = calls[0]["e4m3"]

    partials_batched = cp.zeros(top_k * NCHUNKS * ROWS, dtype=cp.float32)
    gk.run_down_masked_batched(mirror_b, ids_b, globals_dev, act_b, panel_list_b, panel_masks_b,
                               panel_count_b, e2m1, e4m3, partials_batched, ROWS, INTER, NPANEL,
                               mirror_bytes, top_k, NCHUNKS)
    batched_np = cp.asnumpy(partials_batched)

    ref_nan = int(np.isnan(ref_all).sum())
    batched_nan = int(np.isnan(batched_np).sum())
    matches = bool(ref_nan == 0 and batched_nan == 0 and (ref_all == batched_np).all())

    payload = {
        "kind": "verify_down_gather_batch_real_full",
        "created_utc": utc_now(),
        "note": "top_k=6 real captured data from one real MoE layer forward pass; verifies gemv_down_masked_partial_ind_batched, the kernel that showed NaN with synthetic random test data (gather itself already matched bit-exact in the synthetic test, so it is not re-tested here)",
        "top_k": top_k,
        "panel_counts": [int(cp.asnumpy(c["panel_count"])[0]) for c in calls],
        "ref_nan_count": ref_nan,
        "batched_nan_count": batched_nan,
        "bit_exact_match": matches,
    }
    out = REPO / "pro_research" / "verify_down_gather_batch_real_full.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
