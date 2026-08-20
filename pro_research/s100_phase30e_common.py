from __future__ import annotations

from common import REPO
from s100_phase24_common import make_synth
from s100_phase27_common import PipelinedMoEH4, Variant, phase27_gate
from s100_phase30e_group_kernels import GroupDispatchKernels
from s100_phase30e_shared_kernels import SharedOccupancyKernels

RESULTS = REPO / "pro_research" / "results" / "s100_phase30e"
FROZEN_PHASE27R = Variant(gather_y=4, batches=3, shared_overlap=True)


class Phase30EMoEH4(PipelinedMoEH4):
    """Phase27R plus exact shared-M4 and two-launch routed-UP dispatch."""

    def __init__(
        self,
        base,
        *,
        shared_direct: bool = True,
        group_dispatch: bool = True,
        diagnostic: bool = False,
    ):
        super().__init__(base, FROZEN_PHASE27R, diagnostic=diagnostic)
        self.shared_direct = bool(shared_direct)
        self.group_dispatch_enabled = bool(group_dispatch)
        self.shared_m4 = SharedOccupancyKernels() if self.shared_direct else None
        self.group_dispatch = (
            GroupDispatchKernels() if self.group_dispatch_enabled else None
        )
        self.parent_up = base.k.up
        if self.group_dispatch_enabled:
            # PipelinedMoEH4 invokes k.up for m=1..4. The m=1 call launches
            # exact M1-2 and M3-4 kernels; m=2..4 become no-ops.
            base.k.up = self._split_up

    def _shared_fork(self, i, d, normed, main):
        if not self.shared_direct:
            return super()._shared_fork(i, d, normed, main)
        b = self.base
        rt = self.rt
        self.shared_fork[i].record(main)
        with self.shared_stream:
            self.shared_stream.wait_event(self.shared_fork[i])
            self.shared_m4.nvfp4(
                d["sh_up_c"], d["sh_up_s"],
                rt.fused.e2m1, rt.fused.e4m3,
                normed, b.shared_act, d["sh_up_g"],
                b.shared, b.hidden, 4, False, True,
            )
            self.shared_m4.nvfp4(
                d["sh_dn_c"], d["sh_dn_s"],
                rt.fused.e2m1, rt.fused.e4m3,
                b.shared_act, self.shared_out, d["sh_dn_g"],
                b.hidden, b.shared, 4, False, False,
            )
            self.shared_done[i].record(self.shared_stream)

    def _split_up(
        self, m, cache_c, cache_s, slots, ids, globals_dev, e2, e4,
        normed, route_act, rows, cols, code_stride, scale_stride,
    ):
        if int(m) != 1:
            return
        b = self.base
        self.group_dispatch.split2(
            cache_c, cache_s, slots, ids, globals_dev,
            b.group_count, b.group_refs, e2, e4, normed, route_act,
            rows, cols, code_stride, scale_stride,
        )


def make_candidate(
    context: int,
    *,
    shared_direct: bool = True,
    group_dispatch: bool = True,
    diagnostic: bool = False,
):
    cfg, _, _ = phase27_gate()
    rt, graph, keep = make_synth(int(context), cfg)
    wrapped = Phase30EMoEH4(
        graph.gmoe,
        shared_direct=shared_direct,
        group_dispatch=group_dispatch,
        diagnostic=diagnostic,
    )
    graph.gmoe = wrapped
    graph.v.moeb = wrapped
    extra = [wrapped]
    if wrapped.shared_m4 is not None:
        extra.append(wrapped.shared_m4)
    if wrapped.group_dispatch is not None:
        extra.append(wrapped.group_dispatch)
    return rt, graph, list(keep) + extra


def compile_audit() -> dict:
    shared = SharedOccupancyKernels()
    grouped = GroupDispatchKernels()
    for fn in tuple(shared.f.values()) + tuple(grouped.f.values()):
        fn.compile()
    return {
        "shared": shared.resource_audit(),
        "group_dispatch": grouped.resource_audit(),
    }
