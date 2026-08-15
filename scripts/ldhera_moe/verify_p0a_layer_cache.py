from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from moe_lab.reporting import ROOT


DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, EXPERTS, TOP_K = 48, 128, 8
TOKENS, CONTEXT_TOKENS, SLOTS = 32_768, 1_024, 56
CAPACITY, EXPERT_MIB = 4_280, 9
PARAMETERS_PER_EXPERT = 4_718_592
NONEXPERT_PARAMETERS = 1_541_093_376
RATE_BPP = 1.930708991156684

PREREG = ROOT / "reports/ldhera_moe/P0A_LAYER_CACHE_PREREGISTRATION.md"
DOMAIN_LOCK = ROOT / "reports/dchera_moe/p0a_domain_base_lock.json"
PARENT_VERIFY = ROOT / "reports/dchera_moe/p0a_domain_cache_verification.json"
ALLOCATION_LOCK = ROOT / "reports/ldhera_moe/p0a_layer_allocation_lock.json"
ROUTE_CAPTURE = ROOT / "reports/dhera_moe/p0_route_capture.json"
RESULT = ROOT / "reports/ldhera_moe/p0a_layer_cache_result.json"
OUTPUT = ROOT / "reports/ldhera_moe/p0a_layer_cache_verification.json"
REPORT = ROOT / "reports/ldhera_moe/P0A_LAYER_CACHE_VERIFICATION.md"


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


def independent_curve(routes: np.ndarray, base: set[int]) -> list[int]:
    cold_count = 0
    reuse = np.zeros(EXPERTS, dtype=np.int64)
    for start in range(0, TOKENS, CONTEXT_TOKENS):
        recency: list[int] = []
        for raw in routes[start : start + CONTEXT_TOKENS].reshape(-1):
            expert = int(raw)
            if expert in base:
                continue
            if expert not in recency:
                cold_count += 1
            else:
                position = recency.index(expert)
                reuse[len(recency) - position - 1] += 1
                recency.pop(position)
            recency.append(expert)
    return [
        cold_count + int(reuse[capacity:].sum()) if capacity else cold_count + int(reuse.sum())
        for capacity in range(SLOTS + 1)
    ]


def independent_dp(curves: list[list[int]]) -> tuple[list[int], int]:
    table: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for layer_curve in curves:
        updated = {}
        for used, (misses, prefix) in table.items():
            for capacity in range(SLOTS - used + 1):
                candidate = (misses + layer_curve[capacity], prefix + (capacity,))
                key = used + capacity
                current = updated.get(key)
                if current is None or candidate[0] < current[0] or (
                    candidate[0] == current[0]
                    and tuple(-value for value in candidate[1])
                    < tuple(-value for value in current[1])
                ):
                    updated[key] = candidate
        table = updated
    misses, allocation = table[SLOTS]
    return list(allocation), misses


def independent_trace(
    routes: np.ndarray,
    base_rows: list[dict[str, object]],
    allocation: list[int],
    switch_mib: float,
) -> dict[str, object]:
    base = {(row["layer"], row["expert"]) for row in base_rows}
    hits = np.zeros(LAYERS, dtype=np.int64)
    layer_misses = np.zeros(LAYERS, dtype=np.int64)
    token_misses = np.zeros(TOKENS, dtype=np.int16)
    base_calls = 0
    for start in range(0, TOKENS, CONTEXT_TOKENS):
        cache = [OrderedDict() for _ in range(LAYERS)]
        for token in range(start, start + CONTEXT_TOKENS):
            for layer in range(LAYERS):
                for rank in range(TOP_K):
                    expert = int(routes[token, layer, rank])
                    if (layer, expert) in base:
                        base_calls += 1
                    elif expert in cache[layer]:
                        cache[layer].move_to_end(expert)
                        hits[layer] += 1
                    else:
                        token_misses[token] += 1
                        layer_misses[layer] += 1
                        if allocation[layer]:
                            cache[layer][expert] = None
                            if len(cache[layer]) > allocation[layer]:
                                cache[layer].popitem(last=False)
    traffic = token_misses.astype(np.float64) * EXPERT_MIB
    traffic[::CONTEXT_TOKENS] += switch_mib
    total_calls = TOKENS * LAYERS * TOP_K
    return {
        "base_invocations": base_calls,
        "cold_invocations": total_calls - base_calls,
        "hits": int(hits.sum()),
        "misses": int(layer_misses.sum()),
        "hits_by_layer": hits.tolist(),
        "misses_by_layer": layer_misses.tolist(),
        "traffic": {
            "mean": float(traffic.mean()),
            "p50": nearest(traffic, 0.50),
            "p95": nearest(traffic, 0.95),
            "p99": nearest(traffic, 0.99),
            "maximum": float(traffic.max()),
        },
    }


if __name__ == "__main__":
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite LDHERA verification")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    domain_lock = json.loads(DOMAIN_LOCK.read_text(encoding="utf-8"))
    allocation_lock = json.loads(ALLOCATION_LOCK.read_text(encoding="utf-8"))
    capture = json.loads(ROUTE_CAPTURE.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_VERIFY.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "preregistration_hash": result["inputs"]["preregistration_sha256"] == sha256(PREREG),
        "domain_base_lock_hash": result["inputs"]["domain_base_lock_sha256"] == sha256(DOMAIN_LOCK),
        "allocation_lock_hash": result["inputs"]["allocation_lock_sha256"] == sha256(ALLOCATION_LOCK),
        "route_capture_hash": result["inputs"]["route_capture_sha256"] == sha256(ROUTE_CAPTURE),
        "parent_base_verification": parent["verification_pass"] is True
        and parent["source_hashes"]["base_lock_sha256"] == sha256(DOMAIN_LOCK),
        "opened_route_disclosure": result["exploratory_opened_routes"] is True,
    }

    train_allocations, train_misses = {}, {}
    train_hashes = train_shapes = True
    for domain in DOMAINS:
        curves = []
        for layer in range(LAYERS):
            path = ROOT / f"reports/runs/hera_moe/p0_routes/layer_{layer:02d}.safetensors"
            train_hashes &= sha256(path) == allocation_lock["training_route_artifact_sha256"][str(layer)]
            tensor = load_file(path)[f"{domain}_router_ids"]
            train_shapes &= tuple(tensor.shape) == (TOKENS, TOP_K)
            base = {
                row["expert"]
                for row in domain_lock["bases"][domain]
                if row["layer"] == layer
            }
            curves.append(independent_curve(tensor.numpy(), base))
        train_allocations[domain], train_misses[domain] = independent_dp(curves)
    checks["all_training_route_hashes"] = train_hashes
    checks["all_training_route_shapes"] = train_shapes
    checks["independent_training_miss_curves"] = all(
        allocation_lock["training_diagnostics"][domain]["miss_curves_by_layer"]
        == [
            independent_curve(
                load_file(ROOT / f"reports/runs/hera_moe/p0_routes/layer_{layer:02d}.safetensors")[f"{domain}_router_ids"].numpy(),
                {
                    row["expert"]
                    for row in domain_lock["bases"][domain]
                    if row["layer"] == layer
                },
            )
            for layer in range(LAYERS)
        ]
        for domain in DOMAINS
    )
    checks["independent_exact_dp_allocations"] = all(
        train_allocations[domain] == allocation_lock["allocations"][domain]
        and train_misses[domain]
        == allocation_lock["training_diagnostics"][domain]["optimized_misses"]
        and sum(train_allocations[domain]) == SLOTS
        for domain in DOMAINS
    )

    validation = {domain: [] for domain in DOMAINS}
    validation_hashes = validation_shapes = True
    for layer in range(LAYERS):
        item = capture["artifacts"][str(layer)]
        path = ROOT / item["artifact"]
        validation_hashes &= sha256(path) == item["artifact_sha256"]
        tensors = load_file(path)
        for domain in DOMAINS:
            tensor = tensors[f"{domain}_router_ids"]
            validation_shapes &= tuple(tensor.shape) == (TOKENS, TOP_K)
            validation[domain].append(tensor.numpy())
    checks["all_validation_route_hashes"] = validation_hashes
    checks["all_validation_route_shapes"] = validation_shapes

    entropy_gib = CAPACITY * PARAMETERS_PER_EXPERT * RATE_BPP / 8 / 2**30
    resident_gib = entropy_gib + NONEXPERT_PARAMETERS * 4 / 8 / 2**30 + SLOTS * EXPERT_MIB / 1024
    cold_gib = (LAYERS * EXPERTS - CAPACITY) * EXPERT_MIB / 1024
    switch_mib = entropy_gib * 1024
    checks["independent_memory_and_switch_bytes"] = (
        close(result["memory_projection"]["resident_weight_gib"], resident_gib)
        and close(result["memory_projection"]["active_cold_bf16_host_gib"], cold_gib)
        and close(result["policy"]["base_switch_mib_each"], switch_mib)
        and resident_gib <= 5.75
        and cold_gib <= 24.0
    )

    reproduced = {}
    events_exact = layers_exact = conservation = traffic_exact = gates_exact = True
    pass_pattern = {}
    for domain in DOMAINS:
        observed = independent_trace(
            np.stack(validation[domain], axis=1),
            domain_lock["bases"][domain],
            train_allocations[domain],
            switch_mib,
        )
        reproduced[domain] = observed
        expected = result["domains"][domain]
        events_exact &= (
            observed["base_invocations"] == expected["base_invocations"]
            and observed["cold_invocations"] == expected["cold_invocations"]
            and observed["hits"] == expected["layer_local_hits"]
            and observed["misses"] == expected["misses"]
        )
        layers_exact &= observed["hits_by_layer"] == expected["hits_by_layer"] and observed["misses_by_layer"] == expected["misses_by_layer"]
        conservation &= observed["base_invocations"] + observed["hits"] + observed["misses"] == TOKENS * LAYERS * TOP_K
        traffic_exact &= all(
            close(observed["traffic"][key], expected["traffic_mib_per_token"][key])
            for key in ("mean", "p50", "p95", "p99", "maximum")
        )
        gate = {
            "mean_le_64": observed["traffic"]["mean"] <= 64.0,
            "p95_le_144": observed["traffic"]["p95"] <= 144.0,
            "p99_le_288": observed["traffic"]["p99"] <= 288.0,
        }
        pass_pattern[domain] = all(gate.values())
        recorded = result["gates"]["traffic_by_domain"][domain]
        gates_exact &= all(gate[key] == recorded[key] for key in gate) and pass_pattern[domain] == recorded["all_traffic_gates"]
    checks["event_conservation"] = conservation
    checks["independent_validation_events"] = events_exact
    checks["independent_layer_events"] = layers_exact
    checks["independent_traffic_percentiles"] = traffic_exact
    checks["independent_validation_gates"] = gates_exact
    checks["exact_pass_pattern"] = pass_pattern == {"general": True, "code": False, "math": True, "multilingual": True, "instruction": False}
    checks["negative_result_required"] = not all(pass_pattern.values()) and result["gates"]["all_traffic"] is False and result["p0b_authorized"] is False and result["p1_authorized"] is False

    passed = sum(checks.values())
    verification_pass = passed == len(checks)
    final_verdict = "p0a_exploratory_negative_verified" if verification_pass else "verification_failed"
    payload = {
        "kind": "ldhera_moe_p0a_independent_verification",
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
        "source_hashes": {"result_sha256": sha256(RESULT), "allocation_lock_sha256": sha256(ALLOCATION_LOCK), "route_capture_sha256": sha256(ROUTE_CAPTURE)},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [
        f"| {domain} | {reproduced[domain]['traffic']['mean']:.3f} | {reproduced[domain]['traffic']['p95']:.0f} | {reproduced[domain]['traffic']['p99']:.0f} | {'PASS' if pass_pattern[domain] else 'FAIL'} |"
        for domain in DOMAINS
    ]
    REPORT.write_text(
        "\n".join([
            "# LDHERA-MoE P0A — onafhankelijke verificatie", "",
            f"Uitkomst: **{final_verdict}**; **{passed}/{len(checks)}** controles slagen.", "",
            "| Domein | Gem. MiB/token | p95 | p99 | Gate |", "|---|---:|---:|---:|:---:|", *rows, "",
            "De training-misscurves, exacte DP-allocaties en validation-LRU zijn onafhankelijk gereproduceerd.",
            "Code en instruction falen; P0B en P1 blijven gesloten.", "",
        ]), encoding="utf-8"
    )
    print(json.dumps({"verification_pass": verification_pass, "checks": f"{passed}/{len(checks)}", "final_verdict": final_verdict}, indent=2))
