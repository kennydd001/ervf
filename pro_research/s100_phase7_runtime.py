
"""Build legacy or exact packed QFAST runtimes."""
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
from s100_phase7_packed_combined import install_phase7_packed
from s100_phase7_packed_kernels import Phase7PackedKernels


@dataclass
class Phase7Bundle:
    rt: Any
    backend: str
    config: dict
    profile: dict
    restore_selective: Any
    restore_combined: Any
    planned_plane_bytes: int
    state: dict


def build_phase7_runtime(
    capacity=72, layer_k=None, alpha=0.0, backend="legacy"
):
    import cupy as cp

    if backend not in {"legacy", "packed"}:
        raise ValueError(backend)

    rt = _new_runtime(int(capacity))
    profile = apply_phase3_profile(rt, "qfast")
    config = {
        "layer_k": {
            int(k): int(v) for k, v in (layer_k or {}).items()
        },
        "alpha": float(alpha),
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
        raise RuntimeError(f"scale planes do not fit: {planned}>{free}")

    sres = ScaleResidentKernels()
    threshold = Phase5ThresholdKernels()
    state = {}
    if backend == "legacy":
        restore_combined = install_phase5_combined(
            rt, down, up, sres, threshold, config
        )
    else:
        packed = Phase7PackedKernels()
        restore_combined, state = install_phase7_packed(
            rt, down, up, sres, threshold, packed, config
        )

    _recapture(rt)
    return Phase7Bundle(
        rt=rt,
        backend=backend,
        config=config,
        profile=profile,
        restore_selective=restore_selective,
        restore_combined=restore_combined,
        planned_plane_bytes=planned,
        state=state,
    )


def recapture(bundle):
    _recapture(bundle.rt)


def public_record(bundle):
    return {
        "backend": bundle.backend,
        "top_k": int(bundle.rt.top_k),
        "config": {
            "layer_k": {
                str(k): int(v)
                for k, v in sorted(bundle.config["layer_k"].items())
            },
            "alpha": float(bundle.config["alpha"]),
        },
        "profile": public_profile_record(bundle.profile),
        "planned_plane_bytes": bundle.planned_plane_bytes,
    }
