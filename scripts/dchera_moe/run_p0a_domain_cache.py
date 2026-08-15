from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from moe_lab.dhera_moe.cache import BudgetedVictimCache
from moe_lab.reporting import ROOT


DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS = 48
EXPERTS = 128
TOP_K = 8
TOKENS = 32_768
CONTEXT_TOKENS = 1_024
CAPACITY = 4_280
EXPERT_MIB = 9
PARAMETERS_PER_EXPERT = 4_718_592
NONEXPERT_PARAMETERS = 1_541_093_376
RATE_BPP = 1.930708991156684

LOCK = ROOT / "reports/dchera_moe/p0a_domain_base_lock.json"
PREREG = ROOT / "reports/dchera_moe/P0A_DOMAIN_CACHE_PREREGISTRATION.md"
ROUTE_CAPTURE = ROOT / "reports/dhera_moe/p0_route_capture.json"
RESULT = ROOT / "reports/dchera_moe/p0a_domain_cache_result.json"
REPORT = ROOT / "reports/dchera_moe/P0A_DOMAIN_CACHE_REPORT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(values)
    return float(ordered[math.ceil(probability * len(ordered)) - 1])


def simulate(
    routes: np.ndarray,
    base: frozenset[tuple[int, int]],
    base_switch_mib: float,
) -> dict[str, object]:
    base_mask = np.zeros((LAYERS, EXPERTS), dtype=np.bool_)
    for layer, expert in base:
        base_mask[layer, expert] = True
    layer_axis = np.arange(LAYERS)[None, :, None]
    is_base = base_mask[layer_axis, routes]
    flat_routes = routes.reshape(TOKENS, LAYERS * TOP_K)
    cold = (~is_base).reshape(TOKENS, LAYERS * TOP_K)

    cache = BudgetedVictimCache(base, LAYERS, 8)
    events: Counter[str] = Counter()
    misses_per_token = np.zeros(TOKENS, dtype=np.int16)
    for context_start in range(0, TOKENS, CONTEXT_TOKENS):
        cache.reset()
        for token in range(context_start, context_start + CONTEXT_TOKENS):
            misses = 0
            for flat_index in np.flatnonzero(cold[token]):
                index = int(flat_index)
                layer = index // TOP_K
                event = cache.access(layer, int(flat_routes[token, index]))
                events[event] += 1
                misses += event == "miss"
            misses_per_token[token] = misses

    base_calls = int(is_base.sum(dtype=np.int64))
    total_calls = TOKENS * LAYERS * TOP_K
    cold_calls = total_calls - base_calls
    if sum(events.values()) != cold_calls:
        raise AssertionError("cold event conservation failed")
    cold_h2d = misses_per_token.astype(np.float64) * EXPERT_MIB
    total_h2d = cold_h2d.copy()
    total_h2d[::CONTEXT_TOKENS] += base_switch_mib
    return {
        "base_invocations": base_calls,
        "base_invocation_fraction": base_calls / total_calls,
        "cold_invocations": cold_calls,
        "events": {
            "primary_hits": int(events["primary_hit"]),
            "victim_hits": int(events["victim_hit"]),
            "misses": int(events["miss"]),
            "cold_hit_fraction": (
                (events["primary_hit"] + events["victim_hit"]) / cold_calls
                if cold_calls
                else 1.0
            ),
        },
        "cold_miss_h2d_mib_per_token": {
            "mean": float(cold_h2d.mean()),
            "p95": nearest_rank(cold_h2d, 0.95),
            "p99": nearest_rank(cold_h2d, 0.99),
            "maximum": float(cold_h2d.max()),
        },
        "total_h2d_mib_per_token_including_base_switch": {
            "mean": float(total_h2d.mean()),
            "p50": nearest_rank(total_h2d, 0.50),
            "p95": nearest_rank(total_h2d, 0.95),
            "p99": nearest_rank(total_h2d, 0.99),
            "maximum": float(total_h2d.max()),
        },
        "base_switches": TOKENS // CONTEXT_TOKENS,
        "base_switch_mib_each": base_switch_mib,
        "total_base_switch_mib": TOKENS // CONTEXT_TOKENS * base_switch_mib,
        "total_cold_miss_h2d_mib": int(events["miss"]) * EXPERT_MIB,
    }


if __name__ == "__main__":
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite DCHERA P0A result")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    capture = json.loads(ROUTE_CAPTURE.read_text(encoding="utf-8"))
    if lock["preregistration_sha256"] != sha256(PREREG):
        raise ValueError("preregistration hash mismatch")

    entropy_base_gib = CAPACITY * PARAMETERS_PER_EXPERT * RATE_BPP / 8 / 2**30
    trunk_gib = NONEXPERT_PARAMETERS * 4 / 8 / 2**30
    cache_gib = 56 * EXPERT_MIB / 1024
    resident_gib = entropy_base_gib + trunk_gib + cache_gib
    cold_gib = (LAYERS * EXPERTS - CAPACITY) * EXPERT_MIB / 1024
    base_switch_mib = entropy_base_gib * 1024

    routes = {domain: [] for domain in DOMAINS}
    integrity = True
    for layer in range(LAYERS):
        manifest = capture["artifacts"][str(layer)]
        path = ROOT / manifest["artifact"]
        integrity &= sha256(path) == manifest["artifact_sha256"]
        tensors = load_file(path)
        for domain in DOMAINS:
            routes[domain].append(tensors[f"{domain}_router_ids"].numpy())

    domain_results = {}
    gates = {}
    timer = time.perf_counter()
    for domain in DOMAINS:
        selected = lock["bases"][domain]
        base = frozenset((row["layer"], row["expert"]) for row in selected)
        if len(base) != CAPACITY:
            raise ValueError(f"invalid base for {domain}")
        domain_routes = np.stack(routes[domain], axis=1)
        domain_results[domain] = simulate(domain_routes, base, base_switch_mib)
        traffic = domain_results[domain][
            "total_h2d_mib_per_token_including_base_switch"
        ]
        gates[domain] = {
            "mean_le_64": traffic["mean"] <= 64.0,
            "p95_le_144": traffic["p95"] <= 144.0,
            "p99_le_288": traffic["p99"] <= 288.0,
        }
        gates[domain]["all_traffic_gates"] = all(gates[domain].values())
        print(
            json.dumps({"domain": domain, "traffic": traffic, "gate": gates[domain]}),
            flush=True,
        )

    memory_gate = resident_gib <= 5.75 and cold_gib <= 24.0
    all_traffic = all(row["all_traffic_gates"] for row in gates.values())
    positive = bool(memory_gate and all_traffic and integrity)
    verdict = (
        "p0a_exploratory_positive_pending_verification"
        if positive
        else "p0a_exploratory_negative_pending_verification"
    )
    base_sets = {
        domain: {(row["layer"], row["expert"]) for row in lock["bases"][domain]}
        for domain in DOMAINS
    }
    pairwise_jaccard = {}
    for index, left in enumerate(DOMAINS):
        for right in DOMAINS[index + 1 :]:
            union = base_sets[left] | base_sets[right]
            pairwise_jaccard[f"{left}__{right}"] = len(
                base_sets[left] & base_sets[right]
            ) / len(union)
    payload = {
        "kind": "dchera_moe_p0a_domain_cache_result",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "exploratory_opened_routes": True,
        "p0b_authorized": positive,
        "p1_authorized": False,
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "domain_base_lock_sha256": sha256(LOCK),
            "route_capture_sha256": sha256(ROUTE_CAPTURE),
            "route_artifact_hashes_valid": integrity,
        },
        "policy": {
            "domain_label": "externally_supplied",
            "base_experts_per_domain": CAPACITY,
            "primary_slots": 48,
            "victim_slots": 8,
            "context_tokens": CONTEXT_TOKENS,
            "full_base_switch_charged_every_context": True,
            "base_switch_mib": base_switch_mib,
        },
        "memory_projection": {
            "entropy_base_gib": entropy_base_gib,
            "nonexpert_int4_gib": trunk_gib,
            "exact_cache_gib": cache_gib,
            "resident_weight_gib": resident_gib,
            "active_cold_bf16_host_gib": cold_gib,
            "memory_gate_pass": memory_gate,
        },
        "domains": domain_results,
        "gates": {
            "memory": memory_gate,
            "traffic_by_domain": gates,
            "all_traffic": all_traffic,
            "independent_verification": "pending",
        },
        "base_union_experts": len(set().union(*base_sets.values())),
        "pairwise_base_jaccard": pairwise_jaccard,
        "elapsed_seconds": time.perf_counter() - timer,
        "claim_boundary": (
            "Exploration on already-opened validation routes. Positive results "
            "authorize only a fresh blind confirmation, not P1 or an Eureka claim."
        ),
    }
    RESULT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for domain in DOMAINS:
        row = domain_results[domain]
        traffic = row["total_h2d_mib_per_token_including_base_switch"]
        rows.append(
            f"| {domain} | {row['base_invocation_fraction']:.3%} | "
            f"{row['events']['cold_hit_fraction']:.3%} | {traffic['mean']:.3f} | "
            f"{traffic['p95']:.0f} | {traffic['p99']:.0f} | "
            f"{'PASS' if gates[domain]['all_traffic_gates'] else 'FAIL'} |"
        )
    REPORT.write_text(
        "\n".join(
            [
                "# DCHERA-MoE P0A — domeingeconditioneerde cache",
                "",
                f"Voorlopige exploratieve uitkomst: **{verdict}**.",
                "",
                "| Domein | Base-calls | Cold hitrate | Gem. MiB/token | p95 | p99 | Gate |",
                "|---|---:|---:|---:|---:|---:|:---:|",
                *rows,
                "",
                f"Iedere 1.024-tokencontext draagt conservatief een volledige "
                f"basewissel van {base_switch_mib:.3f} MiB. Resident: "
                f"{resident_gib:.6f} GiB; actieve host-cold: {cold_gib:.6f} GiB.",
                "",
                "De routes waren al geopend voor DHERA. Ook bij een positieve "
                "uitkomst is dit geen bevestiging: P0B moet nieuwe routes gebruiken.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict, "p0b_authorized": positive}, indent=2))
