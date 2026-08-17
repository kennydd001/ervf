
"""Install exact packed routed-down execution into QFAST."""
from __future__ import annotations
import types
import numpy as np
from moe_dev_batched import DOWN_PANEL_BYTES, UP_CODE, UP_SCALE
from scale_resident_kernels import PLANE_BYTES


def install_phase7_packed(
    rt, batch, up, sres, threshold, packed_k, config
):
    cp = rt.cp
    max_k = int(rt.top_k)
    if max_k != 6:
        raise RuntimeError(f"packed backend requires K6 shell, got {max_k}")

    inter = int(rt.moe_inter)
    hidden = int(rt.hidden)
    npanel = inter // 16
    rowhalf = hidden // 2
    nchunks = int(rt.fused.nchunks)
    packed_bytes = inter * rowhalf

    original = rt._moe_dev
    state = {}
    packed_mirrors = [
        cp.zeros(packed_bytes, dtype=cp.uint8),
        cp.zeros(packed_bytes, dtype=cp.uint8),
    ]
    gather_stream = cp.cuda.Stream(non_blocking=True)
    gather_done = [
        cp.cuda.Event(block=False, disable_timing=True)
        for _ in range(max_k + 1)
    ]
    math_done = [
        cp.cuda.Event(block=False, disable_timing=True)
        for _ in range(max_k)
    ]
    gather_blocks = (inter * 32 + 255) // 256

    def allocate():
        return {
            "act": cp.zeros(max_k * inter, dtype=cp.float32),
            "masks": cp.zeros(max_k * npanel, dtype=cp.uint32),
            "plist": cp.zeros(max_k * npanel, dtype=cp.int32),
            "pcount": cp.zeros(max_k, dtype=cp.int32),
            "nz": cp.zeros(max_k * inter, dtype=cp.int32),
            "nzc": cp.zeros(max_k, dtype=cp.int32),
            "offsets": cp.zeros(max_k * npanel, dtype=cp.int32),
            "partials": cp.zeros(
                max_k * nchunks * hidden, dtype=cp.float32
            ),
            "max_act": cp.zeros(max_k, dtype=cp.float32),
        }

    def moe(self, layer, out):
        k = self.k
        d = self.layer[layer]
        fused = self.fused
        bank = self.bank[layer]
        cache = self.cache[layer]
        ki = int(config["layer_k"].get(int(layer), max_k))
        alpha = float(config.get("alpha", 0.0))
        if ki not in (4, 5, 6):
            raise RuntimeError(f"layer {layer}: invalid K={ki}")

        if not hasattr(self, "_dev_cache"):
            self._dev_cache = {}
        if layer not in self._dev_cache:
            self._dev_cache[layer] = fused.alloc_device_cache(
                self.n_experts, cache["cap"], max_k, bank["globals"]
            )
        dev = self._dev_cache[layer]

        if layer not in state:
            state[layer] = allocate()
        work = state[layer]
        if layer not in sres.planes:
            sres.alloc_planes(layer, int(cache["cap"]))
        planes = sres.planes[layer]

        k.mv_f32(
            self.rlog,
            d["gate_w"],
            self.normed,
            self.n_experts,
            self.hidden,
        )
        fused.route_topk(
            self.rlog,
            d["gate_b"],
            dev["ids"],
            dev["w"],
            self.n_experts,
            ki,
            self.scaling,
            bad_pick=self._bad_pick,
        )
        fused.cache_assign(dev, dev["ids"], cache["cap"], ki)

        self.evt[0].record()
        with self.copy_stream:
            self.copy_stream.wait_event(self.evt[0])
            fused.cache_fetch(
                bank["up_codes"].ctypes.data,
                bank["up_scales"].ctypes.data,
                cache["codes"],
                cache["scales"],
                dev,
                UP_CODE,
                UP_SCALE,
                ki,
            )
            sres.fetch_planes(
                bank["down_base_ptr"], planes, dev, ki
            )
            self.evt[1].record(self.copy_stream)

        out.fill(0)
        fused.gemv_into(
            self._act_shared,
            d["sh_up_c"],
            d["sh_up_s"],
            self.normed,
            d["sh_up_g"],
            self.shared_inter,
            self.hidden,
            apply_relu2=True,
        )
        fused.gemv_into(
            out,
            d["sh_dn_c"],
            d["sh_dn_s"],
            self._act_shared,
            d["sh_dn_g"],
            self.hidden,
            self.shared_inter,
        )

        main = cp.cuda.get_current_stream()
        main.wait_event(self.evt[1])
        up.run_batched(
            work["act"],
            cache["codes"],
            cache["scales"],
            dev["slots"],
            dev["ids"],
            dev["globals"],
            1,
            fused.e2m1,
            fused.e4m3,
            self.normed,
            inter,
            hidden,
            True,
            UP_CODE,
            UP_SCALE,
            ki,
        )

        if alpha <= 0.0:
            batch.panel_scan_batched(
                (ki,),
                (256,),
                (
                    work["act"],
                    np.int32(inter),
                    work["masks"],
                    work["plist"],
                    work["pcount"],
                    work["nz"],
                    work["nzc"],
                ),
            )
        else:
            threshold.panel_scan_threshold_batched(
                (ki,),
                (256,),
                (
                    work["act"],
                    np.int32(inter),
                    np.float32(alpha),
                    work["masks"],
                    work["plist"],
                    work["pcount"],
                    work["nz"],
                    work["nzc"],
                    work["max_act"],
                ),
            )

        packed_k.panel_offsets(
            (ki,),
            (32,),
            (
                work["masks"],
                np.int32(npanel),
                work["offsets"],
            ),
        )

        grid_down = ((hidden + 127) // 128, nchunks)

        def issue(slot):
            packed_k.gather_packed(
                (gather_blocks,),
                (256,),
                (
                    np.uint64(bank["down_base_ptr"]),
                    dev["ids"][slot:],
                    np.uint64(DOWN_PANEL_BYTES),
                    packed_mirrors[slot & 1],
                    work["nz"][slot * inter:(slot + 1) * inter],
                    work["nzc"][slot:slot + 1],
                    np.int32(hidden),
                ),
            )

        main.record(gather_done[max_k])
        gather_stream.wait_event(gather_done[max_k])
        with gather_stream:
            issue(0)
            gather_done[0].record(gather_stream)

        for slot in range(ki):
            if slot + 1 < ki:
                with gather_stream:
                    if slot >= 1:
                        gather_stream.wait_event(math_done[slot - 1])
                    issue(slot + 1)
                    gather_done[slot + 1].record(gather_stream)

            main.wait_event(gather_done[slot])
            packed_k.down_packed(
                grid_down,
                (128,),
                (
                    packed_mirrors[slot & 1],
                    planes,
                    dev["slots"][slot:],
                    dev["ids"][slot:],
                    dev["globals"],
                    work["act"][slot * inter:(slot + 1) * inter],
                    work["plist"][
                        slot * npanel:(slot + 1) * npanel
                    ],
                    work["masks"][
                        slot * npanel:(slot + 1) * npanel
                    ],
                    work["offsets"][
                        slot * npanel:(slot + 1) * npanel
                    ],
                    work["pcount"][slot:slot + 1],
                    fused.e2m1,
                    fused.e4m3,
                    work["partials"][
                        slot * nchunks * hidden:
                        (slot + 1) * nchunks * hidden
                    ],
                    np.uint64(PLANE_BYTES),
                    np.int32(hidden),
                    np.int32(inter),
                ),
            )
            math_done[slot].record(main)

        blocks = (hidden + 255) // 256
        batch.reduce_partials_batched(
            (blocks, ki),
            (256,),
            (
                work["partials"],
                self.contrib,
                np.int32(hidden),
                np.int32(nchunks),
            ),
        )
        batch.run_accumulate_batched(
            out, self.contrib, dev["w"], hidden, ki
        )
        return None, None

    rt._moe_dev = types.MethodType(moe, rt)
    rt._phase7_packed_state = state

    def restore():
        rt._moe_dev = original
        sres.planes.clear()
        if hasattr(rt, "_phase7_packed_state"):
            delattr(rt, "_phase7_packed_state")

    return restore, state
