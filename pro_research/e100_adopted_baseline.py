"""Exact single-RHS baseline matching the adopted V6 selective dense policy.

Correctness reference remains the original production kernel; this dispatcher
selects the fastest already-adopted exact kernel for performance comparisons.
"""
from __future__ import annotations

from ervf_dense import DenseERVF
from selective_ervf_v3 import BF16_ERVF_SHAPES, FP8_ERVF_SHAPES


class AdoptedSingleRHS:
    def __init__(self, rt):
        self.rt = rt
        self.dense = DenseERVF()
        self.counters = {
            "bf16_ervf": 0,
            "bf16_prod": 0,
            "fp8_ervf": 0,
            "fp8_prod": 0,
            "f32_prod": 0,
            "nvfp4_adopted_ervf": 0,
        }

    @property
    def policy(self):
        return {
            "bf16_ervf_shapes": [list(x) for x in sorted(BF16_ERVF_SHAPES)],
            "fp8_ervf_shapes": [list(x) for x in sorted(FP8_ERVF_SHAPES)],
            "f32": "production_only",
            "nvfp4": "FusedNVFP4 adopted ERVF path",
        }

    def call(self, case, out, x) -> None:
        rt = self.rt
        shape = (int(case.rows), int(case.cols))
        if case.kind == "bf16":
            if shape in BF16_ERVF_SHAPES:
                self.counters["bf16_ervf"] += 1
                return self.dense.mv_bf16(out, case.W, x, case.rows, case.cols)
            self.counters["bf16_prod"] += 1
            return rt.k.mv_bf16(out, case.W, x, case.rows, case.cols)
        if case.kind == "f32":
            self.counters["f32_prod"] += 1
            return rt.k.mv_f32(out, case.W, x, case.rows, case.cols)
        if case.kind == "fp8":
            if shape in FP8_ERVF_SHAPES:
                self.counters["fp8_ervf"] += 1
                return self.dense.mv_fp8_tensor(out, case.W, x, case.scale, case.rows, case.cols)
            self.counters["fp8_prod"] += 1
            return rt.k.mv_fp8_tensor(out, case.W, x, case.scale, case.rows, case.cols)
        if case.kind == "nvfp4":
            self.counters["nvfp4_adopted_ervf"] += 1
            return rt.fused.gemv_into(
                out, case.codes, case.scales, x, case.scale,
                case.rows, case.cols, apply_relu2=case.apply_relu2,
                out_scale=case.out_scale,
            )
        raise ValueError(case.kind)


def production_call(rt, case, out, x) -> None:
    """Original exact kernel, including legacy dense paths, for correctness only."""
    if case.kind == "bf16":
        return rt.k.mv_bf16(out, case.W, x, case.rows, case.cols)
    if case.kind == "f32":
        return rt.k.mv_f32(out, case.W, x, case.rows, case.cols)
    if case.kind == "fp8":
        return rt.k.mv_fp8_tensor(out, case.W, x, case.scale, case.rows, case.cols)
    if case.kind == "nvfp4":
        # FusedNVFP4.use_ervf is adopted by default. For NVFP4 the relevant
        # reference is the adopted exact path because the legacy row kernel is
        # not part of V6 performance anymore; earlier NERVF evidence already
        # established its equivalence.
        return rt.fused.gemv_into(
            out, case.codes, case.scales, x, case.scale,
            case.rows, case.cols, apply_relu2=case.apply_relu2,
            out_scale=case.out_scale,
        )
    raise ValueError(case.kind)
