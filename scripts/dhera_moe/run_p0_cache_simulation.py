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
BASE_EXPERTS = 4_280
VICTIM_SLOTS = 8
PRIMARY_SLOTS = 48
PARAMETERS_PER_EXPERT = 4_718_592
NONEXPERT_PARAMETERS = 1_541_093_376
HOT_RATE_BPP = 1.930708991156684
EXPERT_MIB = 9

BASE_LOCK = ROOT / "reports/dhera_moe/p0_base_lock.json"
ROUTE_CAPTURE = ROOT / "reports/dhera_moe/p0_route_capture.json"
ROUTE_DIR = ROOT / "reports/runs/dhera_moe/p0_routes"
LAYER_DIR = ROOT / "reports/dhera_moe/p0_route_layers"
PREREG = ROOT / "reports/dhera_moe/P0_BUDGET_CACHE_PREREGISTRATION.md"
CLARIFICATION = ROOT / "reports/dhera_moe/P0_PROTOCOL_CLARIFICATION_001.md"
RESULT = ROOT / "reports/dhera_moe/p0_cache_result.json"
REPORT = ROOT / "reports/dhera_moe/P0_CACHE_TRACE.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(values: np.ndarray, probability: float) -> int:
    ordered = np.sort(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return int(ordered[rank - 1])


def load_and_validate_routes() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    capture = json.loads(ROUTE_CAPTURE.read_text(encoding="utf-8"))
    if capture["status"] != "complete" or len(capture["artifacts"]) != LAYERS:
        raise ValueError("route capture is incomplete")
    routes = {domain: [] for domain in DOMAINS}
    route_hashes = {}
    all_official = True
    for layer in range(LAYERS):
        manifest = capture["artifacts"][str(layer)]
        artifact = ROOT / manifest["artifact"]
        report_path = ROOT / manifest["report"]
        if sha256(artifact) != manifest["artifact_sha256"]:
            raise ValueError(f"route artifact hash mismatch at layer {layer}")
        if sha256(report_path) != manifest["report_sha256"]:
            raise ValueError(f"route report hash mismatch at layer {layer}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        all_official &= report["official_topk_captured_exactly_once_per_chunk"]
        tensors = load_file(artifact)
        for domain in DOMAINS:
            tensor = tensors[f"{domain}_router_ids"]
            if tuple(tensor.shape) != (TOKENS, TOP_K):
                raise ValueError(f"route shape mismatch: {domain}, layer {layer}")
            routes[domain].append(tensor.numpy())
        route_hashes[str(layer)] = manifest["artifact_sha256"]
    return (
        {domain: np.stack(layers, axis=1) for domain, layers in routes.items()},
        {
            "route_capture_sha256": sha256(ROUTE_CAPTURE),
            "layer_artifact_sha256": route_hashes,
            "official_topk_capture_complete": bool(all_official),
        },
    )


def simulate_domain(
    routes: np.ndarray, base: frozenset[tuple[int, int]]
) -> dict[str, object]:
    base_mask = np.zeros((LAYERS, EXPERTS), dtype=np.bool_)
    for layer, expert in base:
        base_mask[layer, expert] = True
    layer_index = np.arange(LAYERS)[None, :, None]
    is_base = base_mask[layer_index, routes]
    flat_routes = routes.reshape(TOKENS, LAYERS * TOP_K)
    flat_cold = (~is_base).reshape(TOKENS, LAYERS * TOP_K)

    cache = BudgetedVictimCache(
        base=base, layers=LAYERS, victim_capacity=VICTIM_SLOTS
    )
    event_counts: Counter[str] = Counter()
    misses_per_token = np.zeros(TOKENS, dtype=np.int16)
    misses_per_layer = np.zeros(LAYERS, dtype=np.int64)
    context_misses = []
    for context_start in range(0, TOKENS, CONTEXT_TOKENS):
        cache.reset()
        for token in range(context_start, context_start + CONTEXT_TOKENS):
            misses = 0
            for flat_index in np.flatnonzero(flat_cold[token]):
                layer = int(flat_index) // TOP_K
                expert = int(flat_routes[token, flat_index])
                event = cache.access(layer, expert)
                event_counts[event] += 1
                if event == "miss":
                    misses += 1
                    misses_per_layer[layer] += 1
            misses_per_token[token] = misses
        context_misses.append(
            int(misses_per_token[context_start : context_start + CONTEXT_TOKENS].sum())
        )

    total_calls = TOKENS * LAYERS * TOP_K
    base_calls = int(is_base.sum(dtype=np.int64))
    cold_calls = total_calls - base_calls
    misses = int(event_counts["miss"])
    if sum(event_counts.values()) != cold_calls:
        raise AssertionError("cold cache event conservation failed")
    if misses != int(misses_per_token.sum(dtype=np.int64)):
        raise AssertionError("miss conservation failed")
    return {
        "tokens": TOKENS,
        "contexts": TOKENS // CONTEXT_TOKENS,
        "total_router_invocations": total_calls,
        "base_invocations": base_calls,
        "base_invocation_fraction": base_calls / total_calls,
        "cold_invocations": cold_calls,
        "events": {
            "primary_hits": int(event_counts["primary_hit"]),
            "victim_hits": int(event_counts["victim_hit"]),
            "misses": misses,
            "cold_hit_fraction": (
                (event_counts["primary_hit"] + event_counts["victim_hit"])
                / cold_calls
                if cold_calls
                else 1.0
            ),
        },
        "misses_per_token": {
            "mean": float(misses_per_token.mean()),
            "p50": nearest_rank(misses_per_token, 0.50),
            "p95": nearest_rank(misses_per_token, 0.95),
            "p99": nearest_rank(misses_per_token, 0.99),
            "maximum": int(misses_per_token.max()),
        },
        "h2d_mib_per_token": {
            "mean": float(misses_per_token.mean()) * EXPERT_MIB,
            "p50": nearest_rank(misses_per_token, 0.50) * EXPERT_MIB,
            "p95": nearest_rank(misses_per_token, 0.95) * EXPERT_MIB,
            "p99": nearest_rank(misses_per_token, 0.99) * EXPERT_MIB,
            "maximum": int(misses_per_token.max()) * EXPERT_MIB,
        },
        "total_h2d_mib": misses * EXPERT_MIB,
        "context_misses": context_misses,
        "misses_per_layer": misses_per_layer.tolist(),
    }


if __name__ == "__main__":
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite DHERA cache result")
    for required in (BASE_LOCK, ROUTE_CAPTURE, PREREG, CLARIFICATION):
        if not required.is_file():
            raise FileNotFoundError(required)
    base_lock = json.loads(BASE_LOCK.read_text(encoding="utf-8"))
    selected = base_lock["selected"]
    base = frozenset((row["layer"], row["expert"]) for row in selected)
    if len(base) != BASE_EXPERTS:
        raise ValueError("base lock does not contain 4,280 unique experts")

    timer = time.perf_counter()
    routes, integrity = load_and_validate_routes()
    domain_results = {}
    for domain in DOMAINS:
        domain_timer = time.perf_counter()
        domain_results[domain] = simulate_domain(routes[domain], base)
        print(
            json.dumps(
                {
                    "domain": domain,
                    "elapsed_seconds": time.perf_counter() - domain_timer,
                    "h2d_mib_per_token": domain_results[domain][
                        "h2d_mib_per_token"
                    ],
                }
            ),
            flush=True,
        )

    entropy_base_gib = (
        BASE_EXPERTS * PARAMETERS_PER_EXPERT * HOT_RATE_BPP / 8 / 2**30
    )
    trunk_gib = NONEXPERT_PARAMETERS * 4 / 8 / 2**30
    cache_gib = (PRIMARY_SLOTS + VICTIM_SLOTS) * EXPERT_MIB / 1024
    cold_experts = LAYERS * EXPERTS - BASE_EXPERTS
    host_cold_gib = cold_experts * EXPERT_MIB / 1024
    resident_gib = entropy_base_gib + trunk_gib + cache_gib
    memory_gate = resident_gib <= 5.75 and host_cold_gib <= 24.0

    gate_results = {}
    for domain, result in domain_results.items():
        traffic = result["h2d_mib_per_token"]
        gate_results[domain] = {
            "mean_le_64": traffic["mean"] <= 64.0,
            "p95_le_144": traffic["p95"] <= 144.0,
            "p99_le_288": traffic["p99"] <= 288.0,
        }
        gate_results[domain]["all_traffic_gates"] = all(
            gate_results[domain].values()
        )
    all_traffic = all(row["all_traffic_gates"] for row in gate_results.values())
    trace_positive = bool(
        memory_gate and all_traffic and integrity["official_topk_capture_complete"]
    )
    verdict = (
        "cache_trace_positive_pending_independent_verification"
        if trace_positive
        else "cache_trace_negative_pending_independent_verification"
    )
    payload = {
        "kind": "dhera_moe_p0_budget_cache_result",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "policy": {
            "base_experts": BASE_EXPERTS,
            "base_order": "descending training aggregate router_weight_squared, then count, layer, expert",
            "primary_slots": PRIMARY_SLOTS,
            "victim_slots": VICTIM_SLOTS,
            "victim_hit_transition": "swap_with_same_layer_primary",
            "context_tokens": CONTEXT_TOKENS,
            "expert_transfer_mib": EXPERT_MIB,
            "percentile_method": "discrete_nearest_rank",
        },
        "inputs": {
            "base_lock_sha256": sha256(BASE_LOCK),
            "preregistration_sha256": sha256(PREREG),
            "clarification_sha256": sha256(CLARIFICATION),
            **integrity,
        },
        "memory_projection": {
            "entropy_base_gib": entropy_base_gib,
            "entropy_rate_bpp_assumption": HOT_RATE_BPP,
            "nonexpert_int4_gib": trunk_gib,
            "exact_cache_gib": cache_gib,
            "resident_weight_gib": resident_gib,
            "resident_gate_gib": 5.75,
            "cold_experts": cold_experts,
            "cold_bf16_host_gib": host_cold_gib,
            "cold_host_gate_gib": 24.0,
            "memory_gate_pass": memory_gate,
        },
        "domains": domain_results,
        "gates": {
            "memory": memory_gate,
            "official_routes": integrity["official_topk_capture_complete"],
            "traffic_by_domain": gate_results,
            "all_traffic": all_traffic,
            "independent_verification": "pending",
        },
        "p1_authorized": False,
        "elapsed_seconds": time.perf_counter() - timer,
        "claim_boundary": (
            "Trace simulation only. No measured PCIe latency, overlap, entropy "
            "pack, model quality, runtime, or tokens-per-second claim."
        ),
    }
    RESULT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = []
    for domain, result in domain_results.items():
        traffic = result["h2d_mib_per_token"]
        gate = gate_results[domain]["all_traffic_gates"]
        rows.append(
            f"| {domain} | {result['base_invocation_fraction']:.3%} | "
            f"{result['events']['cold_hit_fraction']:.3%} | {traffic['mean']:.3f} | "
            f"{traffic['p95']:.0f} | {traffic['p99']:.0f} | "
            f"{'PASS' if gate else 'FAIL'} |"
        )
    report_lines = [
        "# DHERA-MoE P0 — vaste budgetcachetrace",
        "",
        f"Voorlopige uitkomst: **{verdict}**. Onafhankelijke verificatie is nog vereist.",
        "",
        f"De geprojecteerde resident weights zijn **{resident_gib:.6f} GiB**; "
        f"de exacte cold bank in host-RAM is **{host_cold_gib:.6f} GiB**.",
        "",
        "| Domein | Base-calls | Cold hitrate | Gem. MiB/token | p95 | p99 | Gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
        *rows,
        "",
        "De cachepolicy, basis van 4.280 experts, 56 slots, contextreset, "
        "verkeersgates en out-of-sample inputsets zijn niet gesweept.",
        "",
        "Dit is een routesimulatie. Zij bewijst geen echte PCIe-latency of "
        "overlap, geen entropy-packgrootte, geen modelkwaliteit en geen 10 tokens/s.",
        "",
    ]
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "resident_weight_gib": resident_gib,
                "all_traffic_gates": all_traffic,
            },
            indent=2,
        )
    )
