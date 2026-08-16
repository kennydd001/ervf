"""V5 step 2 (+ later extension): install a batched `_moe_dev` on a live
LightningRuntime instance via types.MethodType (the same non-invasive
pattern V3/V4 used for _install_selective) -- no edit to runtime.py or
fused_nvfp4.py.

Batches panel_scan, reduce_partials, AND weighted_accumulate_ind across the
top_k expert slots per layer (each verified bit-exact in isolation by
verify_down_proj_batch_kernels.py). gather_down_sparse_ind and
gemv_down_masked_partial_ind stay per-slot, unchanged, called with the same
arguments as the original _moe_dev -- only their panel_list/panel_masks/
panel_count/nz/nz_count inputs now come from the batched panel_scan output
(sliced per slot) instead of a reused single-slot scratch struct, and their
partials output goes into a batched buffer that a single reduce_partials
call reduces directly into self.contrib.

weighted_accumulate_ind is NOT batched the same mechanical way panel_scan/
reduce_partials were: the original calls sequentially accumulate into the
SAME `out` buffer (dst[i] = fmaf(src[i], w, dst[i]), 6x in a row), so a
naive per-slot-parallel batch would race and, worse, could silently change
the FP summation order (the exact class of bug D1 already found once in
this project). The batched kernel instead runs the same s=0..top_k-1 fmaf
sequence inside a single kernel launch, bit-identical to the original
launches -- see weighted_accumulate_ind_batched in down_proj_batch_kernels.py.

Optionally also batches the up-proj ERVF GEMV (gemv_nvfp4_ervf_ind, via
up_proj_batch_kernels.UpProjBatchKernels passed as `up_kernels`) -- this one
IS a mechanical, independent-per-slot batch like panel_scan/reduce_partials
(each slot writes its own output region, x is identical across slots, no
race), verified bit-exact in isolation against the reference kernel. Pass
up_kernels=None to keep the up-proj GEMV on the unbached production path
(useful for isolating V5's own down_proj-only contribution).

Optionally also batches gather_down_sparse_ind and
gemv_down_masked_partial_ind (via down_gather_batch_kernels.DownGatherBatchKernels
passed as `gather_kernels`). A first attempt at this failed its own isolated
unit test with synthetic random data (NaN in both reference and batched
arms, non-reproducible between runs) and was NOT integrated -- but a
follow-up using REAL captured model activations/weights (not synthetic
random data) showed both kernels bit-exact, zero NaN, on real 6-slot data
(verify_down_gather_batch_real_full.py, verify_gather_batch_real_full.py).
The earlier failure was a synthetic-test-data artifact, not a kernel bug.
Batching these requires top_k independent mirror buffers (bs["mirror_batched"])
instead of the single shared mstate["mirror"] the per-slot path reuses
sequentially -- budget ~top_k x 2.68 MB/layer when enabled. Pass
gather_kernels=None to keep this pair on the unbached, mstate-mirror-reusing
production path.

UP_CODE/UP_SCALE/DOWN_PANEL_BYTES constants and gather_ind_k's fixed-size
grid formula are copied from runtime.py/fused_nvfp4.py verbatim (not
imported, to keep this file's dependency on internals explicit and easy to
diff against the source it mirrors).
"""

from __future__ import annotations

import types

import numpy as np

CODE_BYTES = 4_988_928
SCALE_BYTES = 623_616
HALF_CODE = CODE_BYTES // 2
HALF_SCALE = SCALE_BYTES // 2
UP_CODE = HALF_CODE
UP_SCALE = HALF_SCALE
DOWN_PANEL_BYTES = HALF_CODE + HALF_SCALE


def install_batched_moe_dev(rt, batch_kernels, up_kernels=None, gather_kernels=None) -> callable:
    """Returns a `restore()` callable that puts the original _moe_dev back."""
    cp = rt.cp
    fused = rt.fused
    top_k = rt.top_k
    inter = rt.moe_inter
    hidden = rt.hidden
    npanel = inter // 16
    nchunks = fused.nchunks

    orig_moe_dev = rt._moe_dev
    batched_state: dict[str, dict] = {}

    def _alloc_batched(i: int) -> dict:
        d = {
            "act": cp.zeros(top_k * inter, dtype=cp.float32),
            "masks": cp.zeros(top_k * npanel, dtype=cp.uint32),
            "plist": cp.zeros(top_k * npanel, dtype=cp.int32),
            "pcount": cp.zeros(top_k, dtype=cp.int32),
            "nz": cp.zeros(top_k * inter, dtype=cp.int32),
            "nzc": cp.zeros(top_k, dtype=cp.int32),
            "partials": cp.zeros(top_k * nchunks * hidden, dtype=cp.float32),
            # gather/down_masked stay per-slot, sequential when
            # gather_kernels is None -- reuse the runtime's own single-slot
            # mstate["mirror"] (rt.mstate is allocated once at runtime init
            # regardless of this patch) rather than allocating a duplicate
            # ~2.68 MB/layer buffer. A fresh per-layer mirror here was the
            # cause of the first VRAM-gate failure (23 layers x 2.68 MB ~=
            # 61.6 MB, blowing the 64 MiB budget almost entirely on a
            # redundant allocation).
        }
        if gather_kernels is not None:
            d["mirror_batched"] = cp.zeros(top_k * DOWN_PANEL_BYTES, dtype=cp.uint8)
        return d

    def batched_moe_dev(self, i, out):
        cp2, k, d, fused2 = self.cp, self.k, self.layer[i], self.fused
        bank, c = self.bank[i], self.cache[i]
        if not hasattr(self, "_dev_cache"):
            self._dev_cache = {}
        if i not in self._dev_cache:
            self._dev_cache[i] = fused2.alloc_device_cache(
                self.n_experts, c["cap"], self.top_k, bank["globals"])
        dev = self._dev_cache[i]
        if i not in batched_state:
            batched_state[i] = _alloc_batched(i)
        bs = batched_state[i]

        k.mv_f32(self.rlog, d["gate_w"], self.normed, self.n_experts, self.hidden)
        fused2.route_topk(self.rlog, d["gate_b"], dev["ids"], dev["w"],
                          self.n_experts, self.top_k, self.scaling,
                          bad_pick=self._bad_pick)
        fused2.cache_assign(dev, dev["ids"], c["cap"], self.top_k)
        self.evt[0].record()
        with self.copy_stream:
            self.copy_stream.wait_event(self.evt[0])
            fused2.cache_fetch(bank["up_codes"].ctypes.data,
                               bank["up_scales"].ctypes.data,
                               c["codes"], c["scales"], dev,
                               UP_CODE, UP_SCALE, self.top_k)
            self.evt[1].record(self.copy_stream)

        out.fill(0)
        fused2.gemv_into(self._act_shared, d["sh_up_c"], d["sh_up_s"],
                         self.normed, d["sh_up_g"], self.shared_inter,
                         self.hidden, apply_relu2=True)
        fused2.gemv_into(out, d["sh_dn_c"], d["sh_dn_s"],
                         self._act_shared, d["sh_dn_g"],
                         self.hidden, self.shared_inter)

        cp2.cuda.get_current_stream().wait_event(self.evt[1])

        # ---- pass 1: up-proj GEMVs into per-slot slices of one batched
        # activation buffer (was: reused single self._act_moe). Batched into
        # ONE launch (up_kernels) when installed, else top_k sequential
        # calls to the production indirect ERVF GEMV (V5-only mode).
        if up_kernels is not None:
            up_kernels.run_batched(
                bs["act"], c["codes"], c["scales"], dev["slots"], dev["ids"],
                dev["globals"], 1, fused2.e2m1, fused2.e4m3, self.normed,
                self.moe_inter, self.hidden, True, UP_CODE, UP_SCALE, self.top_k)
        else:
            for s in range(self.top_k):
                fused2.gemv_ervf_indirect(
                    bs["act"][s * inter:(s + 1) * inter], c["codes"], c["scales"],
                    dev, s, dev["globals"], 1, self.normed,
                    self.moe_inter, self.hidden, True, UP_CODE, UP_SCALE)

        # ---- ONE batched panel_scan for all top_k slots (was: top_k calls).
        batch_kernels.panel_scan_batched(
            (top_k,), (256,),
            (bs["act"], np.int32(inter), bs["masks"], bs["plist"],
             bs["pcount"], bs["nz"], bs["nzc"]))

        # ---- pass 2: gather + masked-GEMV, sourcing panel metadata from
        # the batched panel_scan output. Batched into ONE gather + ONE
        # down_masked launch (gather_kernels) when installed, else the
        # original top_k sequential per-slot calls sharing one reused mirror.
        if gather_kernels is not None:
            max_warps = inter + npanel
            blocks = (max_warps * 32 + 255) // 256
            gather_kernels.run_gather_batched(
                np.uint64(bank["down_base_ptr"]), dev["ids"], DOWN_PANEL_BYTES,
                bs["mirror_batched"], bs["plist"], bs["pcount"], bs["nz"], bs["nzc"],
                hidden, npanel, inter, DOWN_PANEL_BYTES, self.top_k, blocks)
            gather_kernels.run_down_masked_batched(
                bs["mirror_batched"], dev["ids"], dev["globals"], bs["act"],
                bs["plist"], bs["masks"], bs["pcount"], fused2.e2m1, fused2.e4m3,
                bs["partials"], hidden, inter, npanel, DOWN_PANEL_BYTES, self.top_k, nchunks)
        else:
            max_warps = inter + npanel
            blocks = (max_warps * 32 + 255) // 256
            grid_dm = ((hidden + 127) // 128, nchunks)
            for s in range(self.top_k):
                plist_s = bs["plist"][s * npanel:(s + 1) * npanel]
                masks_s = bs["masks"][s * npanel:(s + 1) * npanel]
                pcount_s = bs["pcount"][s:s + 1]
                nz_s = bs["nz"][s * inter:(s + 1) * inter]
                nzc_s = bs["nzc"][s:s + 1]
                act_s = bs["act"][s * inter:(s + 1) * inter]
                partials_s = bs["partials"][s * nchunks * hidden:(s + 1) * nchunks * hidden]

                fused2.gather_ind_k((blocks,), (256,),
                                    (np.uint64(bank["down_base_ptr"]), dev["ids"][s:],
                                     np.uint64(DOWN_PANEL_BYTES), self.mstate["mirror"],
                                     plist_s, pcount_s, nz_s, nzc_s, np.int32(hidden)))
                fused2.down_masked_ind_k(grid_dm, (128,),
                                         (self.mstate["mirror"], dev["ids"][s:], dev["globals"],
                                          act_s, plist_s, masks_s, pcount_s,
                                          fused2.e2m1, fused2.e4m3, partials_s,
                                          np.int32(hidden), np.int32(inter)))

        # ---- ONE batched reduce_partials for all top_k slots, writing
        # directly into self.contrib (was: top_k calls into state["partials"]
        # then top_k separate reduce_partials calls).
        blocks_x = (hidden + 255) // 256
        batch_kernels.reduce_partials_batched(
            (blocks_x, top_k), (256,),
            (bs["partials"], self.contrib, np.int32(hidden), np.int32(nchunks)))

        # ---- ONE batched weighted-accumulate, replacing top_k sequential
        # accumulate_indirect calls. Preserves the exact s=0..top_k-1 fmaf
        # order into `out` (which already holds the shared-expert term) --
        # NOT a parallel/atomic reduction, which would change the FP
        # summation order (verified bit-exact in isolation against the
        # sequential reference in verify_down_proj_batch_kernels.py).
        batch_kernels.run_accumulate_batched(out, self.contrib, dev["w"], self.hidden, self.top_k)
        return None, None

    rt._moe_dev = types.MethodType(batched_moe_dev, rt)

    def restore():
        rt._moe_dev = orig_moe_dev

    return restore
