
"""Profile QFAST+thr_0020 route locality on frozen calibration/validation."""
from __future__ import annotations

import gc
import hashlib
import json
import traceback

import numpy as np

from common import (
    REPO,
    require_model_dir,
    utc_now,
    write_json_atomic,
)
from diag_component_marginals_graph import (
    _prefill,
    _reset_exact_state,
)
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from moe_dev_batched import DOWN_PANEL_BYTES
from s100_phase3_fidelity import _advance
from s100_phase5_quality import load_trace
from s100_phase7_runtime import build_phase7_runtime
from s100_phase8_common import BUDGETS, selection_hash

OUT = (
    REPO / "pro_research" / "results"
    / "S100_PHASE8_ROUTE_PROFILE.json"
)

SNAPSHOT_CUDA = r"""
extern "C" __global__ void snapshot_route_ids(
    const unsigned long long* __restrict__ pointers,
    int* __restrict__ output,
    const int top_k)
{
    const int layer_index = blockIdx.x;
    const int slot = threadIdx.x;
    if (slot >= top_k) return;
    const int* src = reinterpret_cast<const int*>(
        pointers[layer_index]
    );
    output[layer_index * top_k + slot] = src[slot];
}
"""


class RouteSnapshot:
    def __init__(self, rt):
        import cupy as cp

        self.cp = cp
        self.layers = [int(x) for x in rt.moe_layers]
        missing = [
            layer
            for layer in self.layers
            if layer not in rt._dev_cache
        ]
        if missing:
            raise RuntimeError(
                f"device route arrays missing for layers {missing}"
            )
        ptrs = np.asarray(
            [
                int(rt._dev_cache[layer]["ids"].data.ptr)
                for layer in self.layers
            ],
            dtype=np.uint64,
        )
        self.pointers = cp.asarray(ptrs)
        self.output = cp.empty(
            len(self.layers) * int(rt.top_k),
            dtype=cp.int32,
        )
        self.kernel = cp.RawKernel(
            SNAPSHOT_CUDA, "snapshot_route_ids"
        )
        self.top_k = int(rt.top_k)

    def read(self, rt) -> np.ndarray:
        rt._graph_stream.synchronize()
        self.kernel(
            (len(self.layers),),
            (32,),
            (
                self.pointers,
                self.output,
                np.int32(self.top_k),
            ),
        )
        return self.cp.asnumpy(self.output).reshape(
            len(self.layers), self.top_k
        )


def profile_split(bundle, kind):
    from transformers import AutoTokenizer

    prompts, _indices, n, data, _meta = load_trace(kind)
    tokenizer = AutoTokenizer.from_pretrained(
        str(require_model_dir()),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    rt = bundle.rt
    snapshot = RouteSnapshot(rt)
    counts = {
        layer: np.zeros(int(rt.n_experts), dtype=np.int64)
        for layer in snapshot.layers
    }

    for prompt_index, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(
            prompt["prompt"], add_special_tokens=False
        )
        targets = data["target_ids"][prompt_index]
        _reset_exact_state(rt)
        _prefill(rt, prompt_ids)

        for token_index in range(n):
            route_ids = snapshot.read(rt)
            for row, layer in enumerate(snapshot.layers):
                np.add.at(counts[layer], route_ids[row], 1)
            if token_index + 1 < n:
                _advance(rt, int(targets[token_index]))

        print(
            f"route-profile {kind} "
            f"{prompt_index + 1:02d}/{len(prompts)} "
            f"{prompt['id']} ({n} positions)",
            flush=True,
        )

    total = sum(int(x.sum()) for x in counts.values())
    return counts, total


def choose(counts, budget):
    ranked = []
    for layer in sorted(counts):
        for expert, count in enumerate(counts[layer].tolist()):
            ranked.append(
                (-int(count), int(layer), int(expert))
            )
    ranked.sort()
    picked = ranked[: int(budget)]

    by_layer: dict[int, list[int]] = {}
    selected_count = 0
    for neg_count, layer, expert in picked:
        by_layer.setdefault(layer, []).append(expert)
        selected_count += -neg_count
    for experts in by_layer.values():
        experts.sort()

    return by_layer, selected_count


def hits(counts, selection):
    hit = 0
    total = 0
    per_layer = {}
    for layer, row in counts.items():
        layer_total = int(row.sum())
        layer_hit = sum(
            int(row[expert])
            for expert in selection.get(layer, [])
        )
        total += layer_total
        hit += layer_hit
        per_layer[str(layer)] = {
            "hit": layer_hit,
            "total": layer_total,
            "hit_rate": (
                layer_hit / layer_total if layer_total else 0.0
            ),
            "records": len(selection.get(layer, [])),
        }
    return {
        "hit": hit,
        "total": total,
        "hit_rate": hit / total if total else 0.0,
        "per_layer": per_layer,
    }


def main() -> int:
    payload = {
        "kind": "s100_phase8_route_profile",
        "status": "started",
        "started_utc": utc_now(),
        "budgets": list(BUDGETS),
        "parent": {
            "profile": "qfast",
            "alpha": 0.0020,
        },
    }
    try:
        payload["gpu_idle_preflight"] = _require_gpu_idle_wddm()
        import cupy as cp

        bundle = build_phase7_runtime(
            capacity=72,
            layer_k={},
            alpha=0.0020,
            backend="legacy",
        )
        calibration, calibration_total = profile_split(
            bundle, "calibration"
        )
        validation, validation_total = profile_split(
            bundle, "validation"
        )

        selections = {}
        for budget in BUDGETS:
            by_layer, selected_count = choose(
                calibration, budget
            )
            record = {
                "budget": int(budget),
                "physical_bytes": (
                    int(budget) * int(DOWN_PANEL_BYTES)
                ),
                "physical_mib": (
                    int(budget)
                    * int(DOWN_PANEL_BYTES)
                    / (1024**2)
                ),
                "selection_sha256": selection_hash(by_layer),
                "by_layer": {
                    str(layer): experts
                    for layer, experts in sorted(by_layer.items())
                },
                "calibration": hits(calibration, by_layer),
                "validation": hits(validation, by_layer),
            }
            selections[str(budget)] = record
            print(
                f"budget {budget}: "
                f"cal={record['calibration']['hit_rate']:.4f} "
                f"val={record['validation']['hit_rate']:.4f} "
                f"VRAM={record['physical_mib']:.1f} MiB",
                flush=True,
            )

        payload.update(
            {
                "status": "profile_ready",
                "calibration_total_routes": calibration_total,
                "validation_total_routes": validation_total,
                "selections": selections,
                "completed_utc": utc_now(),
            }
        )
        bundle.restore_combined()
        bundle.restore_selective()
        del bundle
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    except Exception as exc:
        payload.update(
            {
                "status": "technical_failure",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "completed_utc": utc_now(),
            }
        )

    write_json_atomic(OUT, payload, archive=True)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "selections": {
                    key: {
                        "physical_mib": value["physical_mib"],
                        "calibration_hit_rate": value[
                            "calibration"
                        ]["hit_rate"],
                        "validation_hit_rate": value[
                            "validation"
                        ]["hit_rate"],
                    }
                    for key, value in payload.get(
                        "selections", {}
                    ).items()
                },
                "error": (payload.get("error") or {}).get(
                    "message"
                ),
                "output": str(OUT),
            },
            indent=2,
        )
    )
    return 2 if payload.get("status") == "technical_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
