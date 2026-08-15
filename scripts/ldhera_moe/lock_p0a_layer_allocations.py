from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from moe_lab.reporting import ROOT


DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, EXPERTS, TOP_K = 48, 128, 8
TOKENS, CONTEXT_TOKENS, TOTAL_SLOTS = 32_768, 1_024, 56
PREREG = ROOT / "reports/ldhera_moe/P0A_LAYER_CACHE_PREREGISTRATION.md"
DOMAIN_LOCK = ROOT / "reports/dchera_moe/p0a_domain_base_lock.json"
OUTPUT = ROOT / "reports/ldhera_moe/p0a_layer_allocation_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lru_miss_curve(routes: np.ndarray, base_experts: set[int]) -> list[int]:
    first_references = 0
    reuse_histogram = np.zeros(EXPERTS, dtype=np.int64)
    for start in range(0, TOKENS, CONTEXT_TOKENS):
        stack: list[int] = []  # LRU -> MRU, unbounded stack property
        for expert_value in routes[start : start + CONTEXT_TOKENS].reshape(-1):
            expert = int(expert_value)
            if expert in base_experts:
                continue
            try:
                position = stack.index(expert)
            except ValueError:
                first_references += 1
            else:
                distance = len(stack) - 1 - position
                reuse_histogram[distance] += 1
                stack.pop(position)
            stack.append(expert)
    curve = []
    for capacity in range(TOTAL_SLOTS + 1):
        if capacity == 0:
            curve.append(first_references + int(reuse_histogram.sum()))
        else:
            curve.append(first_references + int(reuse_histogram[capacity:].sum()))
    return curve


def exact_allocate(curves: list[list[int]]) -> tuple[list[int], int]:
    states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for curve in curves:
        next_states = {}
        for used, (misses, allocation) in states.items():
            for capacity in range(TOTAL_SLOTS - used + 1):
                candidate = (misses + curve[capacity], allocation + (capacity,))
                total = used + capacity
                incumbent = next_states.get(total)
                if incumbent is None or (candidate[0], tuple(-x for x in candidate[1])) < (
                    incumbent[0],
                    tuple(-x for x in incumbent[1]),
                ):
                    next_states[total] = candidate
        states = next_states
    misses, allocation = states[TOTAL_SLOTS]
    return list(allocation), misses


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    lock = json.loads(DOMAIN_LOCK.read_text(encoding="utf-8"))
    allocations = {}
    training = {}
    route_hashes = {}
    for domain in DOMAINS:
        curves = []
        for layer in range(LAYERS):
            path = ROOT / f"reports/runs/hera_moe/p0_routes/layer_{layer:02d}.safetensors"
            route_hashes[str(layer)] = sha256(path)
            routes = load_file(path)[f"{domain}_router_ids"].numpy()
            base_experts = {
                row["expert"]
                for row in lock["bases"][domain]
                if row["layer"] == layer
            }
            curves.append(lru_miss_curve(routes, base_experts))
        allocation, optimized_misses = exact_allocate(curves)
        allocations[domain] = allocation
        training[domain] = {
            "optimized_misses": optimized_misses,
            "zero_slot_misses": sum(curve[0] for curve in curves),
            "one_slot_each_then_8_lowest_tie_slots_misses": None,
            "nonzero_layers": sum(capacity > 0 for capacity in allocation),
            "maximum_layer_capacity": max(allocation),
            "miss_curves_by_layer": curves,
        }
        print(
            json.dumps(
                {
                    "domain": domain,
                    "allocation": allocation,
                    "optimized_misses": optimized_misses,
                }
            ),
            flush=True,
        )
    payload = {
        "kind": "ldhera_moe_p0a_layer_allocation_lock",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "total_slots_each_domain": TOTAL_SLOTS,
        "objective": "minimize exact training cold LRU misses",
        "tie_break": "lexicographically more slots at lower layer indices",
        "preregistration_sha256": sha256(PREREG),
        "domain_base_lock_sha256": sha256(DOMAIN_LOCK),
        "training_route_artifact_sha256": route_hashes,
        "validation_routes_used": False,
        "allocations": allocations,
        "training_diagnostics": training,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "locked", "domains": len(allocations)}, indent=2))
