"""Re-test gather_down_sparse_ind_batched / gemv_down_masked_partial_ind_batched
using REAL model activations and weights instead of synthetic random data.

verify_down_gather_batch_kernels.py (synthetic random data) found NaN in
BOTH the reference and batched arms -- but the reference kernel is a
verbatim copy of the exact kernel already proven bit-exact correct in
production use (V5/V6's causal A/B on the real model, 256 real tokens x 3
prompts). That strongly suggests the synthetic test's random e4m3/e2m1 LUTs
or activation pattern hit an edge case specific to the synthetic setup, not
a real bug -- this script checks that directly by using the real runtime's
actual bank/globals/LUTs/activations from one real forward pass.

Not a gated PRO experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_gpu_free, utc_now, write_json_atomic


def main() -> int:
    require_gpu_free()
    import cupy as cp
    import numpy as np
    from down_gather_batch_kernels import DownGatherBatchKernels
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    from common import require_model_dir

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

    # Find the first MoE layer and capture its REAL routing + activations by
    # monkeypatching gemv_ervf_indirect / route_topk transiently -- simplest
    # is to just step once more and snapshot state right after routing for
    # layer `target_layer`, using rt's own _dev_cache/mstate/act buffers.
    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]
    target_layer = moe_layers[5]  # an interior layer, not the first (edge effects)

    fused = rt.fused
    captured = {}
    orig_down_masked = fused.down_masked_ind_k

    def capture_and_run(grid, block, kargs):
        # kargs: (mirror, id_ptr, globals, act, panel_list, panel_masks,
        # panel_count, e2m1, e4m3, partials, hidden, inter) -- the RawKernel
        # is called as kernel(grid, block, args_tuple), not kernel(*args).
        if "act" not in captured:
            captured["mirror"] = cp.asarray(kargs[0]).copy()
            captured["id"] = cp.asarray(kargs[1]).copy()
            captured["globals"] = cp.asarray(kargs[2]).copy()
            captured["act"] = cp.asarray(kargs[3]).copy()
            captured["panel_list"] = cp.asarray(kargs[4]).copy()
            captured["panel_masks"] = cp.asarray(kargs[5]).copy()
            captured["panel_count"] = cp.asarray(kargs[6]).copy()
            captured["e2m1"] = cp.asarray(kargs[7]).copy()
            captured["e4m3"] = cp.asarray(kargs[8]).copy()
        return orig_down_masked(grid, block, kargs)

    fused.down_masked_ind_k = capture_and_run
    rt.step(nxt)
    fused.down_masked_ind_k = orig_down_masked
    cp.cuda.Device(0).synchronize()

    if "act" not in captured:
        print("capture failed -- no down_masked_ind_k call observed")
        return 1

    ROWS = rt.hidden
    INTER = rt.moe_inter
    NPANEL = INTER // 16
    NCHUNKS = fused.nchunks

    # Run the reference kernel again on the EXACT captured real inputs and
    # compare against a fresh copy -- sanity check the kernel is
    # deterministic on real data (it must be, called this way already
    # thousands of times correctly in V5/V6).
    partials_a = cp.zeros(NCHUNKS * ROWS, dtype=cp.float32)
    gk.run_down_masked_ref(captured["mirror"], captured["id"], captured["globals"], captured["act"],
                           captured["panel_list"], captured["panel_masks"], captured["panel_count"],
                           captured["e2m1"], captured["e4m3"], partials_a, ROWS, INTER, NCHUNKS)
    partials_b = cp.zeros(NCHUNKS * ROWS, dtype=cp.float32)
    gk.run_down_masked_ref(captured["mirror"], captured["id"], captured["globals"], captured["act"],
                           captured["panel_list"], captured["panel_masks"], captured["panel_count"],
                           captured["e2m1"], captured["e4m3"], partials_b, ROWS, INTER, NCHUNKS)
    a_np, b_np = cp.asnumpy(partials_a), cp.asnumpy(partials_b)
    ref_self_consistent = bool((a_np == b_np).all())
    ref_has_nan = bool(np.isnan(a_np).any())

    # Now the batched kernel with top_k=1 (trivial batching -- same slot
    # data replicated, output should equal the reference exactly).
    mirror_1 = captured["mirror"].copy()
    ids_1 = captured["id"].copy()
    panel_list_1 = captured["panel_list"].copy()
    panel_masks_1 = captured["panel_masks"].copy()
    panel_count_1 = captured["panel_count"].copy()
    act_1 = captured["act"].copy()
    partials_batched = cp.zeros(1 * NCHUNKS * ROWS, dtype=cp.float32)
    gk.run_down_masked_batched(mirror_1, ids_1, captured["globals"], act_1, panel_list_1, panel_masks_1,
                               panel_count_1, captured["e2m1"], captured["e4m3"], partials_batched,
                               ROWS, INTER, NPANEL, int(mirror_1.size), 1, NCHUNKS)
    batched_np = cp.asnumpy(partials_batched)
    batched_has_nan = bool(np.isnan(batched_np).any())
    top_k1_matches_ref = bool((a_np == batched_np).all())

    payload = {
        "kind": "verify_down_gather_batch_real_data",
        "created_utc": utc_now(),
        "note": "uses real captured model state, not synthetic random data",
        "target_layer": target_layer,
        "panel_count": int(cp.asnumpy(captured["panel_count"])[0]),
        "ref_self_consistent_same_inputs_twice": ref_self_consistent,
        "ref_has_nan_on_real_data": ref_has_nan,
        "batched_top_k1_has_nan": batched_has_nan,
        "batched_top_k1_matches_ref": top_k1_matches_ref,
    }
    out = REPO / "pro_research" / "verify_down_gather_batch_real_data.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
