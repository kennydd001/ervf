from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from moe_lab.dhera_moe.cache import BudgetedVictimCache
from moe_lab.reporting import ROOT


DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, EXPERTS, TOP_K = 48, 128, 8
TOKENS, CONTEXT_TOKENS, WARMUP_TOKENS = 32_768, 1_024, 64
CAPACITY, EXPERT_MIB = 4_280, 9
PARAMETERS_PER_EXPERT = 4_718_592
NONEXPERT_PARAMETERS = 1_541_093_376
RATE_BPP = 1.930708991156684

PREREG = ROOT / "reports/adhera_moe/P0A_CAUSAL_WARMUP_PREREGISTRATION.md"
DOMAIN_LOCK = ROOT / "reports/dchera_moe/p0a_domain_base_lock.json"
ROUTE_CAPTURE = ROOT / "reports/dhera_moe/p0_route_capture.json"
RESULT = ROOT / "reports/adhera_moe/p0a_causal_warmup_result.json"
REPORT = ROOT / "reports/adhera_moe/P0A_CAUSAL_WARMUP_REPORT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(values: np.ndarray, probability: float) -> float:
    return float(np.sort(values)[math.ceil(probability * len(values)) - 1])


def process_token(
    cache: BudgetedVictimCache,
    routes: np.ndarray,
    token: int,
    events: Counter[str],
) -> int:
    misses = 0
    for layer in range(LAYERS):
        for rank in range(TOP_K):
            event = cache.access(layer, int(routes[token, layer, rank]))
            events[event] += 1
            misses += event == "miss"
    return misses


def simulate(
    routes: np.ndarray,
    domain_rows: list[dict[str, object]],
    switch_mib: float,
) -> dict[str, object]:
    domain_base = frozenset((row["layer"], row["expert"]) for row in domain_rows)
    domain_rank = {
        (row["layer"], row["expert"]): rank
        for rank, row in enumerate(domain_rows)
    }
    all_keys = [(layer, expert) for layer in range(LAYERS) for expert in range(EXPERTS)]
    fallback_rank = {
        key: domain_rank.get(key, CAPACITY + key[0] * EXPERTS + key[1])
        for key in all_keys
    }
    misses = np.zeros(TOKENS, dtype=np.int16)
    events: Counter[str] = Counter()
    adaptive_overlap = []
    adaptive_observed = []
    for start in range(0, TOKENS, CONTEXT_TOKENS):
        cache = BudgetedVictimCache(domain_base, LAYERS, 8)
        warmup_counts: Counter[tuple[int, int]] = Counter()
        for token in range(start, start + WARMUP_TOKENS):
            for layer in range(LAYERS):
                for expert in routes[token, layer]:
                    warmup_counts[(layer, int(expert))] += 1
            misses[token] = process_token(cache, routes, token, events)
        ordered = sorted(
            all_keys,
            key=lambda key: (
                -warmup_counts[key],
                fallback_rank[key],
                key[0],
                key[1],
            ),
        )
        context_base = frozenset(ordered[:CAPACITY])
        adaptive_overlap.append(len(context_base & domain_base))
        adaptive_observed.append(sum(key in context_base for key in warmup_counts))
        cache = BudgetedVictimCache(context_base, LAYERS, 8)
        for token in range(start + WARMUP_TOKENS, start + CONTEXT_TOKENS):
            misses[token] = process_token(cache, routes, token, events)

    total_calls = TOKENS * LAYERS * TOP_K
    if sum(events.values()) != total_calls:
        raise AssertionError("event conservation failed")
    traffic = misses.astype(np.float64) * EXPERT_MIB
    traffic[::CONTEXT_TOKENS] += switch_mib
    traffic[WARMUP_TOKENS::CONTEXT_TOKENS] += switch_mib
    cold_events = events["primary_hit"] + events["victim_hit"] + events["miss"]
    return {
        "events": {
            "base_invocations": int(events["base"]),
            "primary_hits": int(events["primary_hit"]),
            "victim_hits": int(events["victim_hit"]),
            "misses": int(events["miss"]),
            "cold_hit_fraction": (
                (events["primary_hit"] + events["victim_hit"]) / cold_events
                if cold_events
                else 1.0
            ),
        },
        "traffic_mib_per_token": {
            "mean": float(traffic.mean()),
            "p50": nearest_rank(traffic, 0.50),
            "p95": nearest_rank(traffic, 0.95),
            "p99": nearest_rank(traffic, 0.99),
            "maximum": float(traffic.max()),
        },
        "base_switches": 2 * TOKENS // CONTEXT_TOKENS,
        "base_switch_mib_each": switch_mib,
        "adaptive_base_domain_overlap": {
            "mean": float(np.mean(adaptive_overlap)),
            "minimum": min(adaptive_overlap),
            "maximum": max(adaptive_overlap),
        },
        "warmup_observed_experts_selected": {
            "mean": float(np.mean(adaptive_observed)),
            "minimum": min(adaptive_observed),
            "maximum": max(adaptive_observed),
        },
    }


if __name__ == "__main__":
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite ADHERA result")
    domain_lock = json.loads(DOMAIN_LOCK.read_text(encoding="utf-8"))
    capture = json.loads(ROUTE_CAPTURE.read_text(encoding="utf-8"))
    entropy_gib = CAPACITY * PARAMETERS_PER_EXPERT * RATE_BPP / 8 / 2**30
    trunk_gib = NONEXPERT_PARAMETERS * 4 / 8 / 2**30
    cache_gib = 56 * EXPERT_MIB / 1024
    resident_gib = entropy_gib + trunk_gib + cache_gib
    cold_gib = (LAYERS * EXPERTS - CAPACITY) * EXPERT_MIB / 1024
    switch_mib = entropy_gib * 1024

    routes = {domain: [] for domain in DOMAINS}
    integrity = True
    for layer in range(LAYERS):
        item = capture["artifacts"][str(layer)]
        path = ROOT / item["artifact"]
        integrity &= sha256(path) == item["artifact_sha256"]
        tensors = load_file(path)
        for domain in DOMAINS:
            routes[domain].append(tensors[f"{domain}_router_ids"].numpy())
    domains = {}
    gates = {}
    for domain in DOMAINS:
        domains[domain] = simulate(
            np.stack(routes[domain], axis=1), domain_lock["bases"][domain], switch_mib
        )
        traffic = domains[domain]["traffic_mib_per_token"]
        gates[domain] = {
            "mean_le_64": traffic["mean"] <= 64.0,
            "p95_le_144": traffic["p95"] <= 144.0,
            "p99_le_288": traffic["p99"] <= 288.0,
        }
        gates[domain]["all_traffic_gates"] = all(gates[domain].values())
        print(json.dumps({"domain": domain, "traffic": traffic, "gate": gates[domain]}), flush=True)
    memory_gate = resident_gib <= 5.75 and cold_gib <= 24.0
    all_traffic = all(row["all_traffic_gates"] for row in gates.values())
    positive = bool(memory_gate and all_traffic and integrity)
    verdict = (
        "p0a_exploratory_positive_pending_verification"
        if positive
        else "p0a_exploratory_negative_pending_verification"
    )
    payload = {
        "kind": "adhera_moe_p0a_causal_warmup_result",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "exploratory_opened_routes": True,
        "p0b_authorized": positive,
        "p1_authorized": False,
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "domain_base_lock_sha256": sha256(DOMAIN_LOCK),
            "route_capture_sha256": sha256(ROUTE_CAPTURE),
            "route_artifact_hashes_valid": integrity,
        },
        "policy": {
            "warmup_tokens": WARMUP_TOKENS,
            "context_tokens": CONTEXT_TOKENS,
            "base_experts": CAPACITY,
            "primary_slots": 48,
            "victim_slots": 8,
            "selection": "warmup count descending, then locked domain training rank",
            "full_base_switches_per_context": 2,
            "base_switch_mib_each": switch_mib,
        },
        "memory_projection": {
            "resident_weight_gib": resident_gib,
            "active_cold_bf16_host_gib": cold_gib,
            "memory_gate_pass": memory_gate,
        },
        "domains": domains,
        "gates": {
            "memory": memory_gate,
            "traffic_by_domain": gates,
            "all_traffic": all_traffic,
            "independent_verification": "pending",
        },
        "claim_boundary": (
            "Exploration on opened routes. Positive permits only fresh blind P0B; "
            "no pack, quality, latency, classifier, or speed claim."
        ),
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for domain in DOMAINS:
        row = domains[domain]
        traffic = row["traffic_mib_per_token"]
        rows.append(
            f"| {domain} | {row['events']['cold_hit_fraction']:.3%} | "
            f"{traffic['mean']:.3f} | {traffic['p95']:.0f} | {traffic['p99']:.0f} | "
            f"{'PASS' if gates[domain]['all_traffic_gates'] else 'FAIL'} |"
        )
    REPORT.write_text(
        "\n".join(
            [
                "# ADHERA-MoE P0A — causale warmupcache",
                "",
                f"Voorlopige exploratieve uitkomst: **{verdict}**.",
                "",
                "| Domein | Cold hitrate | Gem. MiB/token | p95 | p99 | Gate |",
                "|---|---:|---:|---:|---:|:---:|",
                *rows,
                "",
                "Het verkeer bevat twee volledige geprojecteerde basewissels per context.",
                "De routes waren al geopend; een positieve P0A opent alleen verse P0B.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict, "p0b_authorized": positive}, indent=2))
