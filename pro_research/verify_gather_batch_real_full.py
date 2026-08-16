"""Full top_k=6 real-data verification of gather_down_sparse_ind_batched,
mirroring verify_down_gather_batch_real_full.py's approach for
down_masked (which confirmed bit-exact, zero NaN on real data -- the
earlier synthetic-data NaN was a test-harness artifact). The synthetic
test already showed gather's mirror output matching bit-exact, but this
confirms it on real captured inputs too, for completeness before
integration.

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
    orig_gather = fused.gather_ind_k

    def capture_and_run(grid, block, kargs):
        # kargs: (down_base_ptr_uint64, id_ptr, panel_bytes, mirror_dst,
        # panel_list, panel_count, nz_list, nz_count, rows)
        calls.append({
            "down_base_ptr": int(kargs[0]),
            "id": cp.asarray(kargs[1]).copy(),
            "panel_bytes": int(kargs[2]),
            "dst_before": cp.asarray(kargs[3]).copy(),
            "panel_list": cp.asarray(kargs[4]).copy(),
            "panel_count": cp.asarray(kargs[5]).copy(),
            "nz_list": cp.asarray(kargs[6]).copy(),
            "nz_count": cp.asarray(kargs[7]).copy(),
            "rows": int(kargs[8]),
        })
        r = orig_gather(grid, block, kargs)
        calls[-1]["dst_after"] = cp.asarray(kargs[3]).copy()
        return r

    fused.gather_ind_k = capture_and_run
    rt.step(nxt)
    fused.gather_ind_k = orig_gather
    cp.cuda.Device(0).synchronize()

    top_k = rt.top_k
    if len(calls) < top_k:
        print(f"only captured {len(calls)} calls, expected >= {top_k}")
        return 1
    calls = calls[:top_k]

    ROWS = rt.hidden
    INTER = rt.moe_inter
    NPANEL = INTER // 16
    panel_bytes = calls[0]["panel_bytes"]

    # ---- reference: rerun gather fresh on each slot's real inputs (zeroed
    # destination, matching the real single-mirror-reused-per-slot pattern).
    blocks = ((INTER + NPANEL) * 32 + 255) // 256
    ref_mirrors = []
    for c in calls:
        dst = cp.zeros_like(c["dst_before"])
        gk.run_gather_ref(
            _host_ptr_array(cp, c["down_base_ptr"], panel_bytes), c["id"], panel_bytes, dst,
            c["panel_list"], c["panel_count"], c["nz_list"], c["nz_count"], ROWS, blocks)
        ref_mirrors.append(cp.asnumpy(dst))

    # ---- batched: same real down_base pointer, all top_k slots at once.
    ids_b = cp.concatenate([c["id"] for c in calls])
    panel_list_b = cp.concatenate([c["panel_list"] for c in calls])
    panel_count_b = cp.concatenate([c["panel_count"] for c in calls])
    nz_list_b = cp.concatenate([c["nz_list"] for c in calls])
    nz_count_b = cp.concatenate([c["nz_count"] for c in calls])
    dst_batched = cp.zeros(top_k * panel_bytes, dtype=cp.uint8)
    gk.run_gather_batched(
        _host_ptr_array(cp, calls[0]["down_base_ptr"], panel_bytes), ids_b, panel_bytes, dst_batched,
        panel_list_b, panel_count_b, nz_list_b, nz_count_b, ROWS, NPANEL, INTER, panel_bytes, top_k, blocks)
    dst_batched_np = cp.asnumpy(dst_batched)

    all_match = True
    mismatch = None
    for s in range(top_k):
        ref = ref_mirrors[s]
        batched_slot = dst_batched_np[s * panel_bytes:(s + 1) * panel_bytes]
        if not (ref == batched_slot).all():
            all_match = False
            diff_idx = int(np_argmax_neq(ref, batched_slot))
            mismatch = {"slot": s, "first_diff_index": diff_idx}
            break

    payload = {
        "kind": "verify_gather_batch_real_full",
        "created_utc": utc_now(),
        "note": "top_k=6 real captured data from one real MoE layer forward pass",
        "top_k": top_k,
        "bit_exact_match": all_match,
        "mismatch": mismatch,
    }
    out = REPO / "pro_research" / "verify_gather_batch_real_full.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0 if all_match else 1


def _host_ptr_array(cp, ptr, nbytes):
    """down_base_ptr captured from a live kernel call is a raw pointer
    (passed as np.uint64 in the real call) into the runtime's own
    pinned/mapped host bank -- re-wrap as np.uint64 for the same ABI type
    the kernel expects (a plain Python int would not match)."""
    import numpy as np
    return np.uint64(ptr)


def np_argmax_neq(a, b):
    import numpy as np
    return np.argmax(a != b)


if __name__ == "__main__":
    raise SystemExit(main())
