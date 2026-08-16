"""B3 — overlap the down_proj PCIe gather with VRAM compute.

This is the step today's measurements identified as the gating condition for
100 tok/s, with no estimate anywhere in the chain:

    VRAM floor    2048 MB / 249 GB/s  = 8.22 ms   (diag_gemv_width32)
    PCIe gather     64 MB / 25.9 GB/s = 2.47 ms   (diag_gather_pcie_ceiling)
    serial        10.69 ms =  93.6 tok/s   -> 100 unreachable
    overlapped     8.22 ms = 122   tok/s   -> 100 reachable at 82% efficiency

And the component marginals put a floor under the same conclusion from the other
side: of MoE's 5.81 ms of headroom, 2.47 ms is pure PCIe time that no faster
kernel can ever remove -- only overlap can.

## What is serial today

`moe_dev_batched.py` runs, per expert slot, on one stream:

    gather_ind_k        PCIe zero-copy reads  ->  mstate["mirror"]
    down_masked_ind_k   VRAM compute          <-  mstate["mirror"]

Same stream, so they cannot overlap; and there is exactly ONE mirror shared by
all top_k slots, so they could not overlap even on two streams.

## What this changes

Two mirrors, ping-ponged, and a dedicated gather stream:

    prime:  G: gather(slot 0) -> mirror[0];  record g[0]
    s-loop: G: wait m[s-1]                   (mirror[(s+1)%2] free again)
            G: gather(slot s+1) -> mirror[(s+1)%2];  record g[s+1]
            M: wait g[s]
            M: down_masked(slot s) <- mirror[s%2];    record m[s]

So slot s+1's PCIe traffic is in flight while slot s computes. The `wait m[s-1]`
edge is the part that is easy to forget and would silently corrupt: without it
the gather for s+1 could overwrite the buffer slot s-1 is still reading.

Cost: one extra mirror. `mstate["mirror"]` is a single global scratch reused
across layers, not per-layer, so this is 2 x 2,806,272 B = 5.6 MB total, not
5.6 MB x 23.

## Why it stays bit-exact

Nothing about what is computed changes: same routing, same slots, same nz sets,
same panels, same masks, same kernels, same arguments, same fmaf order, same
reduce and accumulate. Only *when* the gather runs and *which* of two identical
buffers it writes. The gather writes exactly the columns and panel-scale blocks
that slot s's masked GEMV then reads, so the ping-pong is safe by the same
argument that makes the single-buffer version safe.

Installed non-invasively with types.MethodType, like V3-V6 and V13.
"""

from __future__ import annotations

import types

import numpy as np

from moe_dev_batched import DOWN_PANEL_BYTES, UP_CODE, UP_SCALE


def install_overlap_moe_dev(rt, batch_kernels, up_kernels,
                            gather_small=None, gather_blocks: int = 0) -> callable:
    """Returns a `restore()` callable that puts the original _moe_dev back.

    `gather_small` (a GatherSmallGrid) plus `gather_blocks` swaps the gather for
    the grid-stride variant at a chosen, much smaller static grid. The
    production launch is sized for the worst case (247 blocks) while only ~31
    have work at the measured sparsity, and those idle blocks claim SMs that
    `down_masked` needs -- which is the measured reason V14-G hid only 16.8% of
    the PCIe time. Leave it None to keep the production gather.
    """
    cp = rt.cp
    top_k = rt.top_k
    inter = rt.moe_inter
    hidden = rt.hidden
    npanel = inter // 16
    nchunks = rt.fused.nchunks

    orig_moe_dev = rt._moe_dev
    batched_state: dict[int, dict] = {}

    # One extra global mirror; mstate["mirror"] is global scratch, not per-layer.
    mirrors = [rt.mstate["mirror"], cp.zeros(DOWN_PANEL_BYTES, dtype=cp.uint8)]
    gather_stream = cp.cuda.Stream(non_blocking=True)
    g_done = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(top_k + 1)]
    m_done = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(top_k)]

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

    def overlap_moe_dev(self, i, out):
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

        max_warps = inter + npanel
        gblocks = (max_warps * 32 + 255) // 256
        grid_dm = ((hidden + 127) // 128, nchunks)

        def issue_gather(s: int) -> None:
            plist_s = bs["plist"][s * npanel:(s + 1) * npanel]
            pcount_s = bs["pcount"][s:s + 1]
            nz_s = bs["nz"][s * inter:(s + 1) * inter]
            nzc_s = bs["nzc"][s:s + 1]
            if gather_small is not None:
                gather_small.launch(gather_blocks, bank["down_base_ptr"],
                                    dev["ids"][s:], DOWN_PANEL_BYTES,
                                    mirrors[s & 1], plist_s, pcount_s,
                                    nz_s, nzc_s, hidden)
            else:
                fused2.gather_ind_k(
                    (gblocks,), (256,),
                    (np.uint64(bank["down_base_ptr"]), dev["ids"][s:],
                     np.uint64(DOWN_PANEL_BYTES), mirrors[s & 1],
                     plist_s, pcount_s, nz_s, nzc_s, np.int32(hidden)))

        # panel_scan ran on `main`; the gather stream must see its results.
        main.record(g_done[top_k])
        gather_stream.wait_event(g_done[top_k])
        with gather_stream:
            issue_gather(0)
            g_done[0].record(gather_stream)

        for s in range(self.top_k):
            if s + 1 < self.top_k:
                with gather_stream:
                    if s >= 1:
                        # mirror[(s+1)&1] == mirror[(s-1)&1] is still being read
                        # by slot s-1's masked GEMV until m_done[s-1].
                        gather_stream.wait_event(m_done[s - 1])
                    issue_gather(s + 1)
                    g_done[s + 1].record(gather_stream)

            main.wait_event(g_done[s])
            fused2.down_masked_ind_k(
                grid_dm, (128,),
                (mirrors[s & 1], dev["ids"][s:], dev["globals"],
                 bs["act"][s * inter:(s + 1) * inter],
                 bs["plist"][s * npanel:(s + 1) * npanel],
                 bs["masks"][s * npanel:(s + 1) * npanel],
                 bs["pcount"][s:s + 1],
                 fused2.e2m1, fused2.e4m3,
                 bs["partials"][s * nchunks * hidden:(s + 1) * nchunks * hidden],
                 np.int32(hidden), np.int32(inter)))
            m_done[s].record(main)

        blocks_x = (hidden + 255) // 256
        batch_kernels.reduce_partials_batched(
            (blocks_x, top_k), (256,),
            (bs["partials"], self.contrib, np.int32(hidden), np.int32(nchunks)))
        batch_kernels.run_accumulate_batched(out, self.contrib, dev["w"],
                                             self.hidden, self.top_k)
        return None, None

    rt._moe_dev = types.MethodType(overlap_moe_dev, rt)

    def restore():
        rt._moe_dev = orig_moe_dev

    return restore
