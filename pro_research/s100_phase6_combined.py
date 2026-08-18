"""V18/QFAST MoE installer for phase-6 exact backend experiments."""
from __future__ import annotations
import types
import numpy as np
from moe_dev_batched import DOWN_PANEL_BYTES, UP_CODE, UP_SCALE
from scale_resident_kernels import PLANE_BYTES

BACKENDS = ("ballot_fused", "direct", "direct_opt")


def install_phase6_combined(rt, batch, up, sres, p5thr, p6k, config, backend):
    if backend not in BACKENDS:
        raise ValueError(backend)
    cp = rt.cp
    max_k = int(rt.top_k)
    if max_k != 6:
        raise RuntimeError(f"phase6 requires QFAST K6 shell, got {max_k}")
    use_direct = backend in {"direct", "direct_opt"}
    use_ballot = backend in {"ballot_fused", "direct_opt"}
    use_fused = backend in {"ballot_fused", "direct_opt"}
    inter, hidden = int(rt.moe_inter), int(rt.hidden)
    npanel, nchunks = inter // 16, int(rt.fused.nchunks)
    original = rt._moe_dev
    state = {}

    mirrors = None
    gather_stream = None
    g_done = m_done = None
    gather_blocks = (inter * 32 + 255) // 256
    if not use_direct:
        mirrors = [rt.mstate["mirror"], cp.zeros(DOWN_PANEL_BYTES, dtype=cp.uint8)]
        gather_stream = cp.cuda.Stream(non_blocking=True)
        g_done = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(max_k + 1)]
        m_done = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(max_k)]

    def alloc():
        return {
            "act": cp.zeros(max_k * inter, dtype=cp.float32),
            "masks": cp.zeros(max_k * npanel, dtype=cp.uint32),
            "plist": cp.zeros(max_k * npanel, dtype=cp.int32),
            "pcount": cp.zeros(max_k, dtype=cp.int32),
            "nz": cp.zeros(max_k * inter, dtype=cp.int32),
            "nzc": cp.zeros(max_k, dtype=cp.int32),
            "partials": cp.zeros(max_k * nchunks * hidden, dtype=cp.float32),
            "max_act": cp.zeros(max_k, dtype=cp.float32),
        }

    def moe(self, i, out):
        cp2, k, d, fused = self.cp, self.k, self.layer[i], self.fused
        bank, cache = self.bank[i], self.cache[i]
        ki = int(config["layer_k"].get(int(i), max_k))
        alpha = float(config.get("alpha", 0.0))
        if ki not in (4, 5, 6):
            raise RuntimeError(f"layer {i}: invalid K={ki}")
        if not hasattr(self, "_dev_cache"):
            self._dev_cache = {}
        if i not in self._dev_cache:
            self._dev_cache[i] = fused.alloc_device_cache(
                self.n_experts, cache["cap"], max_k, bank["globals"]
            )
        dev = self._dev_cache[i]
        if i not in state:
            state[i] = alloc()
        bs = state[i]
        if i not in sres.planes:
            sres.alloc_planes(i, int(cache["cap"]))
        planes = sres.planes[i]

        k.mv_f32(self.rlog, d["gate_w"], self.normed, self.n_experts, self.hidden)
        fused.route_topk(
            self.rlog, d["gate_b"], dev["ids"], dev["w"], self.n_experts,
            ki, self.scaling, bad_pick=self._bad_pick,
        )
        fused.cache_assign(dev, dev["ids"], cache["cap"], ki)
        self.evt[0].record()
        with self.copy_stream:
            self.copy_stream.wait_event(self.evt[0])
            fused.cache_fetch(
                bank["up_codes"].ctypes.data, bank["up_scales"].ctypes.data,
                cache["codes"], cache["scales"], dev, UP_CODE, UP_SCALE, ki,
            )
            sres.fetch_planes(bank["down_base_ptr"], planes, dev, ki)
            self.evt[1].record(self.copy_stream)

        out.fill(0)
        fused.gemv_into(
            self._act_shared, d["sh_up_c"], d["sh_up_s"], self.normed,
            d["sh_up_g"], self.shared_inter, self.hidden, apply_relu2=True,
        )
        fused.gemv_into(
            out, d["sh_dn_c"], d["sh_dn_s"], self._act_shared,
            d["sh_dn_g"], self.hidden, self.shared_inter,
        )
        main = cp2.cuda.get_current_stream()
        main.wait_event(self.evt[1])

        up.run_batched(
            bs["act"], cache["codes"], cache["scales"], dev["slots"],
            dev["ids"], dev["globals"], 1, fused.e2m1, fused.e4m3,
            self.normed, self.moe_inter, self.hidden, True,
            UP_CODE, UP_SCALE, ki,
        )

        if use_ballot:
            if alpha <= 0.0:
                p6k.scan_exact(
                    (ki,), (256,),
                    (bs["act"], np.int32(inter), bs["masks"], bs["plist"],
                     bs["pcount"], bs["nz"], bs["nzc"]),
                )
            else:
                p6k.scan_threshold(
                    (ki,), (256,),
                    (bs["act"], np.int32(inter), np.float32(alpha),
                     bs["masks"], bs["plist"], bs["pcount"], bs["nz"],
                     bs["nzc"], bs["max_act"]),
                )
        else:
            if alpha <= 0.0:
                batch.panel_scan_batched(
                    (ki,), (256,),
                    (bs["act"], np.int32(inter), bs["masks"], bs["plist"],
                     bs["pcount"], bs["nz"], bs["nzc"]),
                )
            else:
                p5thr.panel_scan_threshold_batched(
                    (ki,), (256,),
                    (bs["act"], np.int32(inter), np.float32(alpha),
                     bs["masks"], bs["plist"], bs["pcount"], bs["nz"],
                     bs["nzc"], bs["max_act"]),
                )

        grid = ((hidden + 127) // 128, nchunks)
        if use_direct:
            for s in range(ki):
                p6k.down_direct(
                    grid, (128,),
                    (np.uint64(bank["down_base_ptr"]), dev["ids"][s:], planes,
                     dev["slots"][s:], dev["globals"],
                     bs["act"][s * inter:(s + 1) * inter],
                     bs["plist"][s * npanel:(s + 1) * npanel],
                     bs["masks"][s * npanel:(s + 1) * npanel],
                     bs["pcount"][s:s + 1], fused.e2m1, fused.e4m3,
                     bs["partials"][s * nchunks * hidden:(s + 1) * nchunks * hidden],
                     np.uint64(DOWN_PANEL_BYTES), np.uint64(PLANE_BYTES),
                     np.int32(hidden), np.int32(inter)),
                )
        else:
            def issue(s):
                sres.gather_cols(
                    gather_blocks, bank["down_base_ptr"], dev["ids"][s:],
                    mirrors[s & 1], bs["nz"][s * inter:(s + 1) * inter],
                    bs["nzc"][s:s + 1], hidden,
                )

            main.record(g_done[max_k])
            gather_stream.wait_event(g_done[max_k])
            with gather_stream:
                issue(0)
                g_done[0].record(gather_stream)
            for s in range(ki):
                if s + 1 < ki:
                    with gather_stream:
                        if s >= 1:
                            gather_stream.wait_event(m_done[s - 1])
                        issue(s + 1)
                        g_done[s + 1].record(gather_stream)
                main.wait_event(g_done[s])
                sres.down_masked_sres(
                    grid, mirrors[s & 1], planes, dev["slots"][s:],
                    dev["ids"][s:], dev["globals"],
                    bs["act"][s * inter:(s + 1) * inter],
                    bs["plist"][s * npanel:(s + 1) * npanel],
                    bs["masks"][s * npanel:(s + 1) * npanel],
                    bs["pcount"][s:s + 1], fused.e2m1, fused.e4m3,
                    bs["partials"][s * nchunks * hidden:(s + 1) * nchunks * hidden],
                    hidden, inter,
                )
                m_done[s].record(main)

        blocks = (hidden + 255) // 256
        if use_fused:
            p6k.reduce_accumulate(
                (blocks,), (256,),
                (bs["partials"], out, dev["w"], np.int32(hidden),
                 np.int32(nchunks), np.int32(ki)),
            )
        else:
            batch.reduce_partials_batched(
                (blocks, ki), (256,),
                (bs["partials"], self.contrib, np.int32(hidden),
                 np.int32(nchunks)),
            )
            batch.run_accumulate_batched(out, self.contrib, dev["w"], hidden, ki)
        return None, None

    rt._moe_dev = types.MethodType(moe, rt)
    rt._phase6_state = state

    def restore():
        rt._moe_dev = original
        sres.planes.clear()
        if hasattr(rt, "_phase6_state"):
            delattr(rt, "_phase6_state")

    return restore, state
