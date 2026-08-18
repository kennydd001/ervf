"""Probe: waar verbruikt build() met de 'current'-map ~800 MiB meer dan met
de 'budget_neutral'-map, bij identiek totaal aantal slots (1656)?

Draait de build-stappen van s100_phase9_capacity_runtime.build() stap voor
stap en print free/pool-statistieken na elke stap. Geen enkel bestaand
bestand wordt aangepast; dit is een wegwerpinstrument.
"""
from __future__ import annotations

import argparse
import gc
import json

from common import REPO


def mib(x):
    return round(x / 2**20, 1)


def snap(cp, label, rows):
    pool = cp.get_default_memory_pool()
    free, total = cp.cuda.Device(0).mem_info
    row = {
        "step": label,
        "free_mib": mib(free),
        "used_mib": mib(total - free),
        "pool_used_mib": mib(pool.used_bytes()),
        "pool_total_mib": mib(pool.total_bytes()),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    args = ap.parse_args()

    import cupy as cp

    from diag_component_marginals_graph import _recapture
    from down_proj_batch_kernels import DownProjBatchKernels
    from ervf_dense import DenseERVF
    from graph_e1f22 import _new_runtime
    from layer_capacity import reallocate_layer
    from moe_dev_batched import install_batched_moe_dev
    from moe_dev_scale_resident import planned_plane_bytes
    from s100_phase3_profiles import apply_phase3_profile
    from s100_phase5_combined import install_phase5_combined
    from s100_phase5_threshold_kernels import Phase5ThresholdKernels
    from scale_resident_kernels import ScaleResidentKernels
    from selective_ervf_v3 import _install_selective
    from up_proj_batch_kernels import UpProjBatchKernels

    rows = []
    snap(cp, "start", rows)

    profiles = json.loads(
        (REPO / "pro_research" / "results" / "s100_phase9"
         / "S100_PHASE9_CAPACITY_PROFILES.json").read_text()
    )["profiles"]
    capmap = {int(k): int(v) for k, v in profiles[args.profile].items()}
    print("total_slots", sum(capmap.values()), flush=True)

    rt = _new_runtime(72)
    snap(cp, "new_runtime", rows)
    apply_phase3_profile(rt, "qfast")
    snap(cp, "phase3_profile", rows)

    rt.enable_cache(0)
    snap(cp, "enable_cache_0", rows)
    for layer in rt.moe_layers:
        reallocate_layer(rt, int(layer), capmap[int(layer)])
    snap(cp, "reallocate_all", rows)

    rt.device_cache = True
    rt.deterministic_accum = True
    dense = DenseERVF()
    down = DownProjBatchKernels()
    up = UpProjBatchKernels()
    _install_selective(rt, dense)
    snap(cp, "install_selective", rows)
    install_batched_moe_dev(rt, down, up)
    snap(cp, "install_batched", rows)
    rt.setup_graph()
    snap(cp, "setup_graph", rows)
    cp.get_default_memory_pool().free_all_blocks()
    snap(cp, "free_all_blocks", rows)

    planned = int(planned_plane_bytes(rt))
    print("planned_planes_mib", mib(planned), flush=True)

    sres = ScaleResidentKernels()
    threshold = Phase5ThresholdKernels()
    install_phase5_combined(
        rt, down, up, sres, threshold, {"layer_k": {}, "alpha": 0.0003}
    )
    snap(cp, "install_combined_planes", rows)
    _recapture(rt)
    snap(cp, "recapture", rows)
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    snap(cp, "final", rows)

    out = REPO / "pro_research" / "results" / "s100_phase9" / (
        f"PROBE_BUILD_{args.profile.upper()}.json"
    )
    out.write_text(
        json.dumps(
            {"profile": args.profile, "total_slots": sum(capmap.values()),
             "planned_planes_mib": mib(planned), "steps": rows},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
