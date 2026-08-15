from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from moe_lab.reporting import ROOT


DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, EXPERTS, TOP_K = 48, 128, 8
TOKENS, CONTEXT_TOKENS, WARMUP = 32_768, 1_024, 64
CAPACITY, EXPERT_MIB = 4_280, 9
PARAMETERS_PER_EXPERT = 4_718_592
NONEXPERT_PARAMETERS = 1_541_093_376
RATE_BPP = 1.930708991156684

PREREG = ROOT / "reports/adhera_moe/P0A_CAUSAL_WARMUP_PREREGISTRATION.md"
DOMAIN_LOCK = ROOT / "reports/dchera_moe/p0a_domain_base_lock.json"
PARENT_VERIFY = ROOT / "reports/dchera_moe/p0a_domain_cache_verification.json"
ROUTE_CAPTURE = ROOT / "reports/dhera_moe/p0_route_capture.json"
RESULT = ROOT / "reports/adhera_moe/p0a_causal_warmup_result.json"
OUTPUT = ROOT / "reports/adhera_moe/p0a_causal_warmup_verification.json"
REPORT = ROOT / "reports/adhera_moe/P0A_CAUSAL_WARMUP_VERIFICATION.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-10)


def nearest(values: np.ndarray, probability: float) -> float:
    return float(np.sort(values)[math.ceil(probability * len(values)) - 1])


def trace(
    routes: np.ndarray, domain_rows: list[dict[str, object]], switch_mib: float
) -> dict[str, object]:
    domain_base = {(row["layer"], row["expert"]) for row in domain_rows}
    rank = {
        (row["layer"], row["expert"]): index
        for index, row in enumerate(domain_rows)
    }
    keys = [(layer, expert) for layer in range(LAYERS) for expert in range(EXPERTS)]
    fallback = {
        key: rank.get(key, CAPACITY + key[0] * EXPERTS + key[1]) for key in keys
    }
    event: Counter[str] = Counter()
    misses = np.zeros(TOKENS, dtype=np.int16)
    overlaps, observed_selected = [], []

    def consume(token: int, base: set[tuple[int, int]], primary, victim) -> None:
        for layer in range(LAYERS):
            for rank_index in range(TOP_K):
                expert = int(routes[token, layer, rank_index])
                pair = (layer, expert)
                if pair in base:
                    event["base_invocations"] += 1
                    continue
                key = layer * EXPERTS + expert
                if primary[layer] == key:
                    event["primary_hits"] += 1
                    continue
                if key in victim:
                    victim.pop(key)
                    old = primary[layer]
                    primary[layer] = key
                    if old is not None:
                        victim.pop(old, None)
                        victim[old] = None
                    event["victim_hits"] += 1
                    continue
                old = primary[layer]
                primary[layer] = key
                if old is not None:
                    victim.pop(old, None)
                    victim[old] = None
                    if len(victim) > 8:
                        victim.popitem(last=False)
                event["misses"] += 1
                misses[token] += 1

    for start in range(0, TOKENS, CONTEXT_TOKENS):
        primary: list[int | None] = [None] * LAYERS
        victim: OrderedDict[int, None] = OrderedDict()
        counts: Counter[tuple[int, int]] = Counter()
        for token in range(start, start + WARMUP):
            for layer in range(LAYERS):
                for expert in routes[token, layer]:
                    counts[(layer, int(expert))] += 1
            consume(token, domain_base, primary, victim)
        ordered = sorted(
            keys,
            key=lambda key: (-counts[key], fallback[key], key[0], key[1]),
        )
        adaptive = set(ordered[:CAPACITY])
        overlaps.append(len(adaptive & domain_base))
        observed_selected.append(sum(key in adaptive for key in counts))
        primary = [None] * LAYERS
        victim = OrderedDict()
        for token in range(start + WARMUP, start + CONTEXT_TOKENS):
            consume(token, adaptive, primary, victim)
    traffic = misses.astype(np.float64) * EXPERT_MIB
    traffic[::CONTEXT_TOKENS] += switch_mib
    traffic[WARMUP::CONTEXT_TOKENS] += switch_mib
    cold = event["primary_hits"] + event["victim_hits"] + event["misses"]
    return {
        "events": dict(event),
        "cold_hit_fraction": (
            (event["primary_hits"] + event["victim_hits"]) / cold if cold else 1.0
        ),
        "traffic": {
            "mean": float(traffic.mean()),
            "p50": nearest(traffic, 0.50),
            "p95": nearest(traffic, 0.95),
            "p99": nearest(traffic, 0.99),
            "maximum": float(traffic.max()),
        },
        "overlap": {
            "mean": float(np.mean(overlaps)),
            "minimum": min(overlaps),
            "maximum": max(overlaps),
        },
        "observed_selected": {
            "mean": float(np.mean(observed_selected)),
            "minimum": min(observed_selected),
            "maximum": max(observed_selected),
        },
    }


if __name__ == "__main__":
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite ADHERA verification")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    lock = json.loads(DOMAIN_LOCK.read_text(encoding="utf-8"))
    capture = json.loads(ROUTE_CAPTURE.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_VERIFY.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "preregistration_hash": result["inputs"]["preregistration_sha256"]
        == sha256(PREREG),
        "domain_base_lock_hash": result["inputs"]["domain_base_lock_sha256"]
        == sha256(DOMAIN_LOCK),
        "route_capture_hash": result["inputs"]["route_capture_sha256"]
        == sha256(ROUTE_CAPTURE),
        "parent_base_verification": parent["verification_pass"] is True
        and parent["source_hashes"]["base_lock_sha256"] == sha256(DOMAIN_LOCK),
        "opened_route_disclosure": result["exploratory_opened_routes"] is True,
        "fixed_policy_constants": result["policy"]["warmup_tokens"] == WARMUP
        and result["policy"]["context_tokens"] == CONTEXT_TOKENS
        and result["policy"]["base_experts"] == CAPACITY
        and result["policy"]["full_base_switches_per_context"] == 2,
    }
    routes = {domain: [] for domain in DOMAINS}
    hashes = shapes = True
    for layer in range(LAYERS):
        item = capture["artifacts"][str(layer)]
        path = ROOT / item["artifact"]
        hashes &= sha256(path) == item["artifact_sha256"]
        tensors = load_file(path)
        for domain in DOMAINS:
            tensor = tensors[f"{domain}_router_ids"]
            shapes &= tuple(tensor.shape) == (TOKENS, TOP_K)
            routes[domain].append(tensor.numpy())
    checks["all_48_route_hashes"] = hashes
    checks["all_route_shapes"] = shapes

    entropy_gib = CAPACITY * PARAMETERS_PER_EXPERT * RATE_BPP / 8 / 2**30
    trunk_gib = NONEXPERT_PARAMETERS * 4 / 8 / 2**30
    cache_gib = 56 * EXPERT_MIB / 1024
    resident_gib = entropy_gib + trunk_gib + cache_gib
    cold_gib = (LAYERS * EXPERTS - CAPACITY) * EXPERT_MIB / 1024
    switch_mib = entropy_gib * 1024
    checks["independent_memory"] = (
        close(result["memory_projection"]["resident_weight_gib"], resident_gib)
        and close(
            result["memory_projection"]["active_cold_bf16_host_gib"], cold_gib
        )
        and resident_gib <= 5.75
        and cold_gib <= 24.0
    )
    checks["independent_switch_bytes"] = close(
        result["policy"]["base_switch_mib_each"], switch_mib
    )

    reproduced = {}
    events_exact = traffic_exact = selector_exact = conservation = gates_exact = True
    pass_pattern = {}
    for domain in DOMAINS:
        observed = trace(
            np.stack(routes[domain], axis=1), lock["bases"][domain], switch_mib
        )
        reproduced[domain] = observed
        expected = result["domains"][domain]
        events_exact &= all(
            observed["events"].get(key, 0) == expected["events"][key]
            for key in ("base_invocations", "primary_hits", "victim_hits", "misses")
        ) and close(
            observed["cold_hit_fraction"], expected["events"]["cold_hit_fraction"]
        )
        conservation &= sum(observed["events"].values()) == TOKENS * LAYERS * TOP_K
        traffic_exact &= all(
            close(observed["traffic"][key], expected["traffic_mib_per_token"][key])
            for key in ("mean", "p50", "p95", "p99", "maximum")
        )
        selector_exact &= all(
            close(
                observed["overlap"][key],
                expected["adaptive_base_domain_overlap"][key],
            )
            and close(
                observed["observed_selected"][key],
                expected["warmup_observed_experts_selected"][key],
            )
            for key in ("mean", "minimum", "maximum")
        )
        gate = {
            "mean_le_64": observed["traffic"]["mean"] <= 64.0,
            "p95_le_144": observed["traffic"]["p95"] <= 144.0,
            "p99_le_288": observed["traffic"]["p99"] <= 288.0,
        }
        pass_pattern[domain] = all(gate.values())
        recorded = result["gates"]["traffic_by_domain"][domain]
        gates_exact &= all(gate[key] == recorded[key] for key in gate)
        gates_exact &= pass_pattern[domain] == recorded["all_traffic_gates"]
    checks["event_conservation"] = conservation
    checks["independent_event_totals"] = events_exact
    checks["independent_selector_summaries"] = selector_exact
    checks["independent_traffic_percentiles"] = traffic_exact
    checks["independent_gates"] = gates_exact
    checks["exact_pass_pattern"] = pass_pattern == {
        "general": True,
        "code": False,
        "math": False,
        "multilingual": True,
        "instruction": False,
    }
    checks["negative_result_required"] = (
        not all(pass_pattern.values())
        and result["gates"]["all_traffic"] is False
        and result["p0b_authorized"] is False
        and result["p1_authorized"] is False
    )
    passed = sum(checks.values())
    verification_pass = passed == len(checks)
    final_verdict = (
        "p0a_exploratory_negative_verified"
        if verification_pass and not all(pass_pattern.values())
        else "verification_failed"
    )
    payload = {
        "kind": "adhera_moe_p0a_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verification_pass": verification_pass,
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "final_verdict": final_verdict,
        "domain_gate_passes": pass_pattern,
        "p0b_authorized": False,
        "p1_authorized": False,
        "reproduced": reproduced,
        "source_hashes": {
            "result_sha256": sha256(RESULT),
            "domain_base_lock_sha256": sha256(DOMAIN_LOCK),
            "route_capture_sha256": sha256(ROUTE_CAPTURE),
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for domain in DOMAINS:
        row = reproduced[domain]["traffic"]
        rows.append(
            f"| {domain} | {row['mean']:.3f} | {row['p95']:.0f} | "
            f"{row['p99']:.0f} | {'PASS' if pass_pattern[domain] else 'FAIL'} |"
        )
    REPORT.write_text(
        "\n".join(
            [
                "# ADHERA-MoE P0A — onafhankelijke verificatie",
                "",
                f"Uitkomst: **{final_verdict}**; **{passed}/{len(checks)}** controles slagen.",
                "",
                "| Domein | Gem. MiB/token | p95 | p99 | Gate |",
                "|---|---:|---:|---:|:---:|",
                *rows,
                "",
                "De 64-tokenwarmup redt de code-p99 niet en verslechtert math en instruction.",
                "P0B en P1 blijven gesloten.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"verification_pass": verification_pass, "checks": f"{passed}/{len(checks)}", "final_verdict": final_verdict}, indent=2))
