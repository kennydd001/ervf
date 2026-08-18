
"""Build phase-7 thr_0020 with legacy or exact static routed-down cache."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from diag_component_marginals_graph import _recapture
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _new_runtime
from layer_capacity import apply_nonuniform_capacity
from moe_dev_batched import install_batched_moe_dev
from moe_dev_scale_resident import planned_plane_bytes
from scale_resident_kernels import ScaleResidentKernels
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels
from s100_phase3_profiles import (
    apply_phase3_profile,
    public_profile_record,
)
from s100_phase5_combined import install_phase5_combined
from s100_phase5_threshold_kernels import Phase5ThresholdKernels
from s100_phase8_static_cache import StaticDownCache
from s100_phase8_static_combined import install_static_down_moe


@dataclass
class Phase8Bundle:
    rt: Any
    backend: str
    config: dict
    profile: dict
    static_cache: Any
    restore_selective: Any
    restore_combined: Any
    planned_plane_bytes: int


def build_phase8_runtime(
    capacity=72,
    selection=None,
    backend="legacy",
):
    import cupy as cp

    if backend not in {"legacy", "static"}:
        raise ValueError(backend)
    selection = selection or {}

    rt = _new_runtime(int(capacity))
    profile = apply_phase3_profile(rt, "qfast")
    config = {
        "layer_k": {},
        "alpha": 0.0020,
    }

    dense = DenseERVF()
    down = DownProjBatchKernels()
    up = UpProjBatchKernels()
    rt.enable_cache(int(capacity))
    apply_nonuniform_capacity(rt)
    rt.device_cache = True
    rt.deterministic_accum = True

    restore_selective, _counters = _install_selective(rt, dense)
    install_batched_moe_dev(rt, down, up)
    rt.setup_graph()
    cp.get_default_memory_pool().free_all_blocks()

    planned = int(planned_plane_bytes(rt))
    free = int(cp.cuda.Device(0).mem_info[0])
    if planned > free:
        raise RuntimeError(
            f"scale planes do not fit: {planned}>{free}"
        )

    sres = ScaleResidentKernels()
    threshold = Phase5ThresholdKernels()
    static_cache = None
    if backend == "legacy":
        restore_combined = install_phase5_combined(
            rt,
            down,
            up,
            sres,
            threshold,
            config,
        )
    else:
        static_cache = StaticDownCache(
            rt,
            selection,
            reserve_bytes=planned,
        )
        restore_combined, _state = install_static_down_moe(
            rt,
            down,
            up,
            sres,
            threshold,
            static_cache,
            config,
        )

    _recapture(rt)
    return Phase8Bundle(
        rt=rt,
        backend=backend,
        config=config,
        profile=profile,
        static_cache=static_cache,
        restore_selective=restore_selective,
        restore_combined=restore_combined,
        planned_plane_bytes=planned,
    )


def recapture(bundle):
    _recapture(bundle.rt)


def public_record(bundle):
    return {
        "backend": bundle.backend,
        "top_k": int(bundle.rt.top_k),
        "config": {
            "layer_k": {},
            "alpha": 0.0020,
        },
        "profile": public_profile_record(bundle.profile),
        "planned_plane_bytes": bundle.planned_plane_bytes,
        "static_cache": (
            bundle.static_cache.public_record()
            if bundle.static_cache is not None
            else None
        ),
    }
