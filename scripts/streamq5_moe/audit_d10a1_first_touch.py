#!/usr/bin/env python3
"""CPU-only exact first-touch inventory for the frozen D10A1 component trace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "streamq5_moe"
ROUTES = ROOT / "reports" / "runs" / "streamq5_moe" / "p4d_routes"
OUT = R / "d10a1_first_touch_independent_source_audit.json"
RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a_next_component_composition.py"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, TOP_K, PREFIX = 48, 10, 499
EXPERT_BYTES, PAGES_PER_RECORD = 2_027_520, 495
MASK64 = (1 << 64) - 1
LIFT_SEED = 0xD10A_499D_1308_2026


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


routes = {domain: np.empty((1024, LAYERS, 8), dtype=np.int16) for domain in DOMAINS}
route_hashes: dict[str, str] = {}
for layer in range(LAYERS):
    path = ROUTES / f"layer_{layer:02d}.safetensors"
    route_hashes[str(layer)] = sha256(path)
    with safe_open(path, framework="np") as handle:
        for domain in DOMAINS:
            routes[domain][:, layer] = handle.get_tensor(f"{domain}_router_ids").astype(np.int16)


def lift(source: np.ndarray, domain_index: int, token: int) -> np.ndarray:
    output = np.empty((LAYERS, TOP_K), dtype=np.int16)
    for layer in range(LAYERS):
        used: set[int] = set()
        for rank, raw in enumerate(source[layer]):
            state = LIFT_SEED ^ (domain_index << 56) ^ (token << 24) ^ (layer << 8) ^ rank
            expert = int(raw) * 4 + int(splitmix64(state) & 3)
            output[layer, rank] = expert
            used.add(expert)
        state = LIFT_SEED ^ (domain_index << 52) ^ (token << 20) ^ layer
        rank = 8
        while rank < TOP_K:
            state = splitmix64(state)
            expert = int(state % 512)
            if expert not in used:
                output[layer, rank] = expert
                used.add(expert)
                rank += 1
    return output


def select(bounds: tuple[int, int], limit: int | None = None) -> list[np.ndarray]:
    cases: list[np.ndarray] = []
    for domain_index, domain in enumerate(DOMAINS):
        for token in range(bounds[0], bounds[1]):
            cases.append(lift(routes[domain][token], domain_index, token))
            if limit is not None and len(cases) >= limit:
                return cases
    return cases


def records(cases: list[np.ndarray]) -> set[tuple[int, int]]:
    return {(layer, int(expert)) for route in cases for layer, row in enumerate(route) for expert in row}


correctness_cases = select((0, 8))
validation_cases = select((512, 576), 32)
correctness = records(correctness_cases)
validation = records(validation_cases)
first = correctness_cases[0].reshape(-1)
injection_index = next(index for index, expert in enumerate(first) if int(expert) < PREFIX - 1)
injection_layer = injection_index // TOP_K
injection_expert = int(first[injection_index])
negative = {(injection_layer, injection_expert + 1), ((injection_layer + 1) % LAYERS, injection_expert)}
shared = {(layer, 512) for layer in range(LAYERS)}
reference = {(0, expert) for expert in range(10)}
union = correctness | validation | negative | shared | reference


def summary(values: set[tuple[int, int]]) -> dict[str, int | float]:
    return {
        "records": len(values),
        "pages_4k": len(values) * PAGES_PER_RECORD,
        "bytes": len(values) * EXPERT_BYTES,
        "gib": len(values) * EXPERT_BYTES / 2**30,
        "hot": sum(expert < 499 for _, expert in values),
        "cold": sum(499 <= expert < 512 for _, expert in values),
        "shared": sum(expert == 512 for _, expert in values),
    }


runner_source = RUNNER.read_text(encoding="utf-8")
checks = {
    "correctness_case_count_40": len(correctness_cases) == 40,
    "validation_case_count_32": len(validation_cases) == 32,
    "complete_union_14452": len(union) == 14_452,
    "complete_union_bytes": len(union) * EXPERT_BYTES == 29_301_719_040,
    "complete_union_pages": len(union) * PAGES_PER_RECORD == 7_153_740,
    "reference_already_overlaps_routes": reference <= (correctness | validation),
    "negative_sources_already_overlap_routes": negative <= (correctness | validation),
    "no_bank_hash_call": "sha256(BANK)" not in runner_source,
    "only_two_bank_memmaps": runner_source.count("np.memmap(BANK") == 2,
}

unique_bytes = len(union) * EXPERT_BYTES
result = {
    "kind": "d10a1_first_touch_independent_source_audit",
    "cpu_only": True,
    "overall_pass": all(checks.values()),
    "checks": checks,
    "checks_passed": sum(checks.values()),
    "checks_total": len(checks),
    "inputs": {"runner_sha256": sha256(RUNNER), "route_hashes": route_hashes},
    "inventory": {
        "correctness_40": summary(correctness),
        "validation_32": summary(validation),
        "correctness_validation_union": summary(correctness | validation),
        "negative_override_sources": summary(negative),
        "shared_48": summary(shared),
        "reference_10": summary(reference),
        "complete_unique_union": summary(union),
        "correctness_validation_overlap_records": len(correctness & validation),
    },
    "logical_read_occurrences": {
        "correctness_stage_plus_oracle": 40 * 480 * 2,
        "negative_baseline_oracle": 480,
        "negative_two_stages": 2 * 480,
        "shared": 48,
        "reference": 10,
        "validation_warmups": 8 * 480,
        "validation_measured": 32 * 480,
        "total": 59_098,
    },
    "ram_gate": {
        "unique_first_touch_bytes": unique_bytes,
        "required_final_reserve_bytes": 2 * 2**30,
        "mathematical_minimum_start_bytes": unique_bytes + 2 * 2**30,
        "recommended_os_margin_bytes": 1 * 2**30,
        "recommended_start_bytes": unique_bytes + 3 * 2**30,
    },
    "hidden_full_bank_scan": False,
    "claim_boundary": "Source-derived footprint for the frozen component trace only; no GPU, bank scan, timing or endurance evidence.",
}
OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"overall_pass": result["overall_pass"], "checks": f'{result["checks_passed"]}/{result["checks_total"]}', "union": result["inventory"]["complete_unique_union"], "ram_gate": result["ram_gate"]}, indent=2))
raise SystemExit(0 if result["overall_pass"] else 1)
