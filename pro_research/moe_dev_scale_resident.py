"""Install the H-SCALE `_moe_dev` on a live runtime: identical to the V6
batched `_moe_dev` (moe_dev_batched.install_batched_moe_dev) except that the
down_proj FP8 block-scale planes live in VRAM instead of being re-gathered over
PCIe on every expert call.

Same non-invasive `types.MethodType` pattern V3/V4/V5 used; runtime.py and
fused_nvfp4.py are untouched.

Exactly three differences from the V6 body, all of them data placement:

  1. per layer a `planes` buffer of `cap x 311,808 B` is allocated once;
  2. the copy stream stages a missed expert's scale plane next to its up_proj
     codes/scales, under the same `need[]/slots[]/ids[]` contract and the same
     wide-uint4 SM staging pattern, so the ordering and the graph-capturability
     of the existing fetch are preserved;
  3. the per-slot gather drops its panel-scale branch and the masked GEMV reads
     the scale byte from the resident plane.

Everything else -- routing, `cache_assign`, the shared expert, the batched
panel_scan / reduce_partials / weighted_accumulate, the up-proj ERVF batching,
the fmaf order, the `contrib` staging -- is byte-for-byte the V6 path.

The output must therefore be bit-identical to V6, and the runner gates on that
over 3 x 256 real tokens before any timing number is looked at.
"""

from __future__ import annotations

import types

import numpy as np

from moe_dev_batched import DOWN_PANEL_BYTES, UP_CODE, UP_SCALE
from scale_resident_kernels import PLANE_BYTES


def install_scale_resident_moe_dev(rt, batch_kernels, up_kernels, sres) -> callable:
    """Returns a `restore()` callable that puts the original _moe_dev back."""
    cp = rt.cp
    top_k = rt.top_k
    inter = rt.moe_inter
    hidden = rt.hidden
    npanel = inter // 16
    nchunks = rt.fused.nchunks

    orig_moe_dev = rt._moe_dev
    batched_state: dict[str, dict] = {}

    # worst case is every column nonzero; the panel-scale warps of the
    # production launch are gone, so the grid is inter warps instead of
    # inter + npanel
    gather_blocks = (inter * 32 + 255) // 256

    def _alloc_batched(i: int) -> dict:
        return {
            "act": cp.zeros(top_k * inter, dtype=cp.float32),
            "masks": cp.zeros(top_k * npanel, dtype=cp.uint32),
            "plist": cp.zeros(top_k * npanel, dtype=cp.int32),
            "pcount": cp.zeros(top_k, dtype=cp.int32),
            "nz": cp.zeros(top_k * inter, dtype=cp.int32),
            "nzc": cp.zeros(top_k, dtype=cp.int32),
            "partials": cp.zeros(top_k * nchunks * hidden, dtype=cp.float32),
        }

    def scale_resident_moe_dev(self, i, out):
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
        if i not in sres.planes:
            sres.alloc_planes(i, int(c["cap"]))
        planes = sres.planes[i]

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
            # H-SCALE (2): the missed expert's scale plane, same contract.
            sres.fetch_planes(bank["down_base_ptr"], planes, dev, self.top_k)
            self.evt[1].record(self.copy_stream)

        out.fill(0)
        fused2.gemv_into(self._act_shared, d["sh_up_c"], d["sh_up_s"],
                         self.normed, d["sh_up_g"], self.shared_inter,
                         self.hidden, apply_relu2=True)
        fused2.gemv_into(out, d["sh_dn_c"], d["sh_dn_s"],
                         self._act_shared, d["sh_dn_g"],
                         self.hidden, self.shared_inter)

        cp2.cuda.get_current_stream().wait_event(self.evt[1])

        up_kernels.run_batched(
            bs["act"], c["codes"], c["scales"], dev["slots"], dev["ids"],
            dev["globals"], 1, fused2.e2m1, fused2.e4m3, self.normed,
            self.moe_inter, self.hidden, True, UP_CODE, UP_SCALE, self.top_k)

        batch_kernels.panel_scan_batched(
            (top_k,), (256,),
            (bs["act"], np.int32(inter), bs["masks"], bs["plist"],
             bs["pcount"], bs["nz"], bs["nzc"]))

        grid_dm = ((hidden + 127) // 128, nchunks)
        for s in range(self.top_k):
            plist_s = bs["plist"][s * npanel:(s + 1) * npanel]
            masks_s = bs["masks"][s * npanel:(s + 1) * npanel]
            pcount_s = bs["pcount"][s:s + 1]
            nz_s = bs["nz"][s * inter:(s + 1) * inter]
            nzc_s = bs["nzc"][s:s + 1]
            act_s = bs["act"][s * inter:(s + 1) * inter]
            partials_s = bs["partials"][s * nchunks * hidden:(s + 1) * nchunks * hidden]

            # H-SCALE (3a): columns only, no panel-scale branch.
            sres.gather_cols(gather_blocks, bank["down_base_ptr"], dev["ids"][s:],
                             self.mstate["mirror"], nz_s, nzc_s, hidden)
            # H-SCALE (3b): scale byte from the resident plane.
            sres.down_masked_sres(grid_dm, self.mstate["mirror"], planes,
                                  dev["slots"][s:], dev["ids"][s:], dev["globals"],
                                  act_s, plist_s, masks_s, pcount_s,
                                  fused2.e2m1, fused2.e4m3, partials_s,
                                  hidden, inter)

        blocks_x = (hidden + 255) // 256
        batch_kernels.reduce_partials_batched(
            (blocks_x, top_k), (256,),
            (bs["partials"], self.contrib, np.int32(hidden), np.int32(nchunks)))
        batch_kernels.run_accumulate_batched(out, self.contrib, dev["w"],
                                             self.hidden, self.top_k)
        return None, None

    rt._moe_dev = types.MethodType(scale_resident_moe_dev, rt)

    def restore():
        rt._moe_dev = orig_moe_dev
        sres.planes.clear()

    return restore


def planned_plane_bytes(rt) -> int:
    """VRAM the resident planes will take, before allocating any of them."""
    return sum(int(c["cap"]) for c in rt.cache.values()) * PLANE_BYTES
