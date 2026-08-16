"""H-SCALE and B3 overlap in one `_moe_dev`, measured as one arm.

Two mechanisms built earlier today are each bit-exact and each fell just under
its own preregistered gate:

    V13  H-SCALE   -0.374 ms/token   gate was >=0.5   scale planes resident in
                                                      VRAM, so the gather moves
                                                      52% fewer bytes
    V14G B3        -0.416 ms/token   gate was >=0.8   double-buffered mirror and
                                                      a gather stream, so slot
                                                      s+1's PCIe traffic runs
                                                      while slot s computes

They attack the same path from opposite sides: H-SCALE shrinks the PCIe
transfer, B3 hides what is left behind compute. The project's own working rule
is that component measurements are never summed into a claim, so the only way to
know what they are worth together is to run them together.

They may well anti-compose -- a smaller gather leaves B3 less to hide -- and that
is precisely the question. The expectation is somewhere between -0.42 and
-0.79 ms, and only a measurement decides where.

## What this installs

Everything from `moe_dev_scale_resident` (per-layer `planes` buffer, the scale
plane staged on the copy stream under the same need[]/slots[]/ids[] contract,
the column-only gather, the masked GEMV reading scales from the resident plane)
PLUS everything from `moe_dev_overlap` (two mirrors ping-ponged, a dedicated
gather stream, and the hazard edge that stops the gather for slot s+1
overwriting the buffer slot s-1 is still reading).

The combination changes no arithmetic: same expert, panel, row, scale byte,
`e4m3_lut[byte] * global_scale`, same fmaf order. Only where bytes live and when
they move.

VRAM: the resident planes (492.4 MiB at cap 72 over 23 layers) plus one extra
global mirror (2.81 MB). The runner gates on it before allocating.
"""

from __future__ import annotations

import types

import numpy as np

from moe_dev_batched import DOWN_PANEL_BYTES, UP_CODE, UP_SCALE


def install_combined_moe_dev(rt, batch_kernels, up_kernels, sres) -> callable:
    """H-SCALE + B3 overlap. Returns a `restore()` callable."""
    cp = rt.cp
    top_k = rt.top_k
    inter = rt.moe_inter
    hidden = rt.hidden
    npanel = inter // 16
    nchunks = rt.fused.nchunks

    orig_moe_dev = rt._moe_dev
    batched_state: dict[int, dict] = {}

    # B3: two mirrors, ping-ponged. mstate["mirror"] is global scratch, so this
    # is 2 x 2.81 MB total rather than per layer.
    mirrors = [rt.mstate["mirror"], cp.zeros(DOWN_PANEL_BYTES, dtype=cp.uint8)]
    gather_stream = cp.cuda.Stream(non_blocking=True)
    g_done = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(top_k + 1)]
    m_done = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(top_k)]

    # H-SCALE: the column-only gather no longer needs the panel-scale warps.
    gather_blocks = (inter * 32 + 255) // 256

    def _alloc_batched() -> dict:
        return {
            "act": cp.zeros(top_k * inter, dtype=cp.float32),
            "masks": cp.zeros(top_k * npanel, dtype=cp.uint32),
            "plist": cp.zeros(top_k * npanel, dtype=cp.int32),
            "pcount": cp.zeros(top_k, dtype=cp.int32),
            "nz": cp.zeros(top_k * inter, dtype=cp.int32),
            "nzc": cp.zeros(top_k, dtype=cp.int32),
            "partials": cp.zeros(top_k * nchunks * hidden, dtype=cp.float32),
        }

    def combined_moe_dev(self, i, out):
        cp2, k, d, fused2 = self.cp, self.k, self.layer[i], self.fused
        bank, c = self.bank[i], self.cache[i]
        if not hasattr(self, "_dev_cache"):
            self._dev_cache = {}
        if i not in self._dev_cache:
            self._dev_cache[i] = fused2.alloc_device_cache(
                self.n_experts, c["cap"], self.top_k, bank["globals"])
        dev = self._dev_cache[i]
        if i not in batched_state:
            batched_state[i] = _alloc_batched()
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
            sres.fetch_planes(bank["down_base_ptr"], planes, dev, self.top_k)
            self.evt[1].record(self.copy_stream)

        out.fill(0)
        fused2.gemv_into(self._act_shared, d["sh_up_c"], d["sh_up_s"],
                         self.normed, d["sh_up_g"], self.shared_inter,
                         self.hidden, apply_relu2=True)
        fused2.gemv_into(out, d["sh_dn_c"], d["sh_dn_s"],
                         self._act_shared, d["sh_dn_g"],
                         self.hidden, self.shared_inter)

        main = cp2.cuda.get_current_stream()
        main.wait_event(self.evt[1])

        up_kernels.run_batched(
            bs["act"], c["codes"], c["scales"], dev["slots"], dev["ids"],
            dev["globals"], 1, fused2.e2m1, fused2.e4m3, self.normed,
            self.moe_inter, self.hidden, True, UP_CODE, UP_SCALE, self.top_k)

        batch_kernels.panel_scan_batched(
            (top_k,), (256,),
            (bs["act"], np.int32(inter), bs["masks"], bs["plist"],
             bs["pcount"], bs["nz"], bs["nzc"]))

        grid_dm = ((hidden + 127) // 128, nchunks)

        def issue_gather(s: int) -> None:
            sres.gather_cols(gather_blocks, bank["down_base_ptr"], dev["ids"][s:],
                             mirrors[s & 1],
                             bs["nz"][s * inter:(s + 1) * inter],
                             bs["nzc"][s:s + 1], hidden)

        main.record(g_done[top_k])
        gather_stream.wait_event(g_done[top_k])
        with gather_stream:
            issue_gather(0)
            g_done[0].record(gather_stream)

        for s in range(self.top_k):
            if s + 1 < self.top_k:
                with gather_stream:
                    if s >= 1:
                        gather_stream.wait_event(m_done[s - 1])
                    issue_gather(s + 1)
                    g_done[s + 1].record(gather_stream)

            main.wait_event(g_done[s])
            sres.down_masked_sres(
                grid_dm, mirrors[s & 1], planes, dev["slots"][s:], dev["ids"][s:],
                dev["globals"], bs["act"][s * inter:(s + 1) * inter],
                bs["plist"][s * npanel:(s + 1) * npanel],
                bs["masks"][s * npanel:(s + 1) * npanel],
                bs["pcount"][s:s + 1], fused2.e2m1, fused2.e4m3,
                bs["partials"][s * nchunks * hidden:(s + 1) * nchunks * hidden],
                hidden, inter)
            m_done[s].record(main)

        blocks_x = (hidden + 255) // 256
        batch_kernels.reduce_partials_batched(
            (blocks_x, top_k), (256,),
            (bs["partials"], self.contrib, np.int32(hidden), np.int32(nchunks)))
        batch_kernels.run_accumulate_batched(out, self.contrib, dev["w"],
                                             self.hidden, self.top_k)
        return None, None

    rt._moe_dev = types.MethodType(combined_moe_dev, rt)

    def restore():
        rt._moe_dev = orig_moe_dev
        sres.planes.clear()

    return restore
