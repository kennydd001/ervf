"""Build QFAST plus legacy or phase-6 MoE backends."""
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
from s100_phase3_profiles import apply_phase3_profile, public_profile_record
from s100_phase5_combined import install_phase5_combined
from s100_phase5_threshold_kernels import Phase5ThresholdKernels
from s100_phase6_combined import BACKENDS, install_phase6_combined
from s100_phase6_kernels import Phase6Kernels

ALL_BACKENDS = ("legacy",) + BACKENDS

@dataclass
class Phase6Bundle:
    rt: Any
    capacity: int
    config: dict[str, Any]
    backend: str
    profile: dict[str, Any]
    dense: Any
    down: Any
    up: Any
    sres: Any
    p5thr: Any
    p6k: Any
    restore_selective: Any
    restore_combined: Any
    counters: dict[str, int]
    planned: int
    state: dict[int, dict[str, Any]]


def build_phase6_runtime(capacity=72, layer_k=None, alpha=0.0, backend="legacy"):
    import cupy as cp
    if backend not in ALL_BACKENDS:
        raise ValueError(backend)
    rt = _new_runtime(int(capacity))
    profile = apply_phase3_profile(rt, "qfast")
    if int(rt.top_k) != 6:
        raise RuntimeError("QFAST shell must remain K6")
    config = {
        "layer_k": {int(k): int(v) for k, v in (layer_k or {}).items()},
        "alpha": float(alpha),
    }
    dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
    rt.enable_cache(int(capacity))
    apply_nonuniform_capacity(rt)
    rt.device_cache = True
    rt.deterministic_accum = True
    restore_sel, counters = _install_selective(rt, dense)
    install_batched_moe_dev(rt, down, up)
    rt.setup_graph()
    cp.get_default_memory_pool().free_all_blocks()
    planned = int(planned_plane_bytes(rt))
    free = int(cp.cuda.Device(0).mem_info[0])
    if planned > free:
        raise RuntimeError(f"scale planes do not fit: {planned}>{free}")
    sres = ScaleResidentKernels()
    p5thr = Phase5ThresholdKernels()
    p6k = None if backend == "legacy" else Phase6Kernels()
    state = {}
    if backend == "legacy":
        restore_combined = install_phase5_combined(
            rt, down, up, sres, p5thr, config
        )
    else:
        restore_combined, state = install_phase6_combined(
            rt, down, up, sres, p5thr, p6k, config, backend
        )
    _recapture(rt)
    return Phase6Bundle(
        rt, int(capacity), config, backend, profile, dense, down, up, sres,
        p5thr, p6k, restore_sel, restore_combined, counters, planned, state,
    )


def recapture(bundle):
    _recapture(bundle.rt)


def record(bundle):
    return {
        "capacity": bundle.capacity,
        "top_k": int(bundle.rt.top_k),
        "backend": bundle.backend,
        "config": {
            "layer_k": {str(k): int(v) for k, v in sorted(bundle.config["layer_k"].items())},
            "alpha": float(bundle.config["alpha"]),
        },
        "profile": public_profile_record(bundle.profile),
        "planned_plane_bytes": bundle.planned,
    }
