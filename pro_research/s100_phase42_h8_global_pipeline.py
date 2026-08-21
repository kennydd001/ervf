"""Phase42 explicit-global correction of the Phase41 range dispatch."""
from __future__ import annotations

from moe_dev_batched import UP_CODE, UP_SCALE
from s100_phase32_common import make_candidate as make_phase32
from s100_phase41_h8_full_pipeline import Phase41FullPipelineMoEH8
from s100_phase42_range_dispatch_kernels import Phase42RangeDispatchKernels


class Phase42GlobalPipelineMoEH8(Phase41FullPipelineMoEH8):
    def __init__(self, phase32_moe, *, serial_control: bool, batches: int = 3):
        super().__init__(
            phase32_moe,
            batches=int(batches),
            gather_y=4,
            serial_control=bool(serial_control),
        )
        self.dispatch = Phase42RangeDispatchKernels()

    def _up_range(self, multiplicity, normed, cache, dev, begin, end):
        b, rt = self.base, self.rt
        self.dispatch.up_range(
            multiplicity,
            cache["codes"], cache["scales"], b.slots, b.ids, dev["globals"],
            b.group_count, b.group_refs, rt.fused.e2m1, rt.fused.e4m3,
            normed, b.route_act, b.inter, b.hidden, UP_CODE, UP_SCALE,
            begin, end,
        )

    def _scan_range(self, begin, end):
        b = self.base
        self.dispatch.scan_range(
            b.route_act, b.group_count, b.group_refs,
            b.route_masks, b.route_plist, b.route_pcount,
            b.union_masks, b.union_plist, b.union_pcount,
            b.union_nz, b.union_nzc, b.inter, begin, end,
        )


def _make(context: int, *, serial_control: bool, batches: int = 3):
    rt, graph, keep = make_phase32(int(context), "dense_m8")
    wrapped = Phase42GlobalPipelineMoEH8(
        graph.gmoe, serial_control=bool(serial_control), batches=int(batches)
    )
    graph.gmoe = wrapped
    return rt, graph, list(keep) + [wrapped, wrapped.pipeline, wrapped.dispatch]


def make_serial(context: int):
    return _make(int(context), serial_control=True)


def make_overlap(context: int):
    return _make(int(context), serial_control=False)


def make_overlap_geometry(context: int, batches: int):
    return _make(int(context), serial_control=False, batches=int(batches))


def make_parent(context: int):
    return make_phase32(int(context), "dense_m8")
