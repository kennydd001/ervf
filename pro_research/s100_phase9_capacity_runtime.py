from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from diag_component_marginals_graph import _recapture
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _new_runtime
from layer_capacity import reallocate_layer
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


@dataclass
class Bundle:
    rt: Any
    capmap: dict[int, int]
    profile: dict[str, Any]
    sres: Any
    threshold: Any
    restore_sel: Any
    restore_combined: Any
    planned: int


def build(capmap):
    import cupy as cp

    runtime = _new_runtime(72)
    profile = apply_phase3_profile(runtime, "qfast")

    expected_layers = {int(layer) for layer in runtime.moe_layers}
    normalized = {int(key): int(value) for key, value in capmap.items()}
    if set(normalized) != expected_layers:
        raise RuntimeError(
            "capacity map does not contain exactly the live MoE layers"
        )
    if any(value <= 0 for value in normalized.values()):
        raise RuntimeError("capacity map contains a non-positive capacity")

    # Initialize cache metadata without first allocating 72 slots per layer.
    # The old implementation allocated ~4.3 GiB and then replaced it, causing
    # transient/memory-pool pressure before the candidate was even measured.
    runtime.enable_cache(0)
    for layer in runtime.moe_layers:
        reallocate_layer(runtime, int(layer), normalized[int(layer)])

    runtime.device_cache = True
    runtime.deterministic_accum = True

    dense = DenseERVF()
    down = DownProjBatchKernels()
    up = UpProjBatchKernels()
    restore_sel, _ = _install_selective(runtime, dense)
    install_batched_moe_dev(runtime, down, up)
    runtime.setup_graph()
    cp.get_default_memory_pool().free_all_blocks()

    planned = int(planned_plane_bytes(runtime))
    free = int(cp.cuda.Device(0).mem_info[0])
    if planned > free:
        raise RuntimeError(
            f"phase9 scale planes do not fit: {planned}>{free}"
        )

    sres = ScaleResidentKernels()
    threshold = Phase5ThresholdKernels()
    config = {"layer_k": {}, "alpha": 0.0003}
    restore_combined = install_phase5_combined(
        runtime, down, up, sres, threshold, config
    )
    _recapture(runtime)

    return Bundle(
        runtime,
        normalized,
        profile,
        sres,
        threshold,
        restore_sel,
        restore_combined,
        planned,
    )


def record(bundle):
    return {
        "profile": "qfast+thr_0003",
        "capacity_map": {
            str(key): value for key, value in sorted(bundle.capmap.items())
        },
        "total_slots": sum(bundle.capmap.values()),
        "planned_plane_bytes": bundle.planned,
        "profile_record": public_profile_record(bundle.profile),
    }
