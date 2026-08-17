"""Shared exact-V18 construction helpers for phase 3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from diag_component_marginals_graph import _recapture
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _new_runtime
from layer_capacity import apply_nonuniform_capacity
from moe_dev_batched import install_batched_moe_dev
from moe_dev_combined import install_combined_moe_dev
from moe_dev_scale_resident import planned_plane_bytes
from scale_resident_kernels import ScaleResidentKernels
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels
from s100_phase3_profiles import apply_phase3_profile, public_profile_record


@dataclass
class V18Bundle:
    rt: Any
    capacity: int
    dense: Any
    down: Any
    up: Any
    scale_resident: Any
    restore_selective: Any
    restore_combined: Any
    selective_counters: dict[str, int]
    applied_profile: dict[str, Any] | None
    planned_plane_bytes: int


def rebuild_cache(rt, capacity: int) -> None:
    rt.enable_cache(int(capacity))
    apply_nonuniform_capacity(rt)
    rt.device_cache = True
    rt.deterministic_accum = True


def install_combined_current(rt, down, up):
    sres = ScaleResidentKernels()
    restore = install_combined_moe_dev(rt, down, up, sres)
    return sres, restore


def build_v18_runtime(capacity: int = 72, profile: str | None = None) -> V18Bundle:
    import cupy as cp

    rt = _new_runtime(int(capacity))
    applied = apply_phase3_profile(rt, profile) if profile else None
    dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
    rebuild_cache(rt, int(capacity))
    restore_sel, counters = _install_selective(rt, dense)
    install_batched_moe_dev(rt, down, up)
    rt.setup_graph()
    cp.get_default_memory_pool().free_all_blocks()
    planned = int(planned_plane_bytes(rt))
    free_before = int(cp.cuda.Device(0).mem_info[0])
    if planned > free_before:
        raise RuntimeError(f"V18 resident scale planes do not fit: {planned} > {free_before}")
    sres, restore_combined = install_combined_current(rt, down, up)
    _recapture(rt)
    return V18Bundle(
        rt=rt, capacity=int(capacity), dense=dense, down=down, up=up,
        scale_resident=sres, restore_selective=restore_sel,
        restore_combined=restore_combined, selective_counters=counters,
        applied_profile=applied, planned_plane_bytes=planned,
    )


def public_bundle_record(bundle: V18Bundle) -> dict[str, Any]:
    return {
        "capacity": bundle.capacity,
        "top_k": int(bundle.rt.top_k),
        "planned_plane_bytes": bundle.planned_plane_bytes,
        "selective_ervf_counters": dict(bundle.selective_counters),
        "profile": public_profile_record(bundle.applied_profile)
        if bundle.applied_profile is not None else None,
    }
