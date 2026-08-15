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
TOKENS, CONTEXT_TOKENS = 32_768, 1_024
CAPACITY, EXPERT_MIB = 4_280, 9
PARAMETERS_PER_EXPERT = 4_718_592
NONEXPERT_PARAMETERS = 1_541_093_376
RATE_BPP = 1.930708991156684

LOCK = ROOT / "reports/dchera_moe/p0a_domain_base_lock.json"
PREREG = ROOT / "reports/dchera_moe/P0A_DOMAIN_CACHE_PREREGISTRATION.md"
ROUTE_CAPTURE = ROOT / "reports/dhera_moe/p0_route_capture.json"
RESULT = ROOT / "reports/dchera_moe/p0a_domain_cache_result.json"
OUTPUT = ROOT / "reports/dchera_moe/p0a_domain_cache_verification.json"
REPORT = ROOT / "reports/dchera_moe/P0A_DOMAIN_CACHE_VERIFICATION.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-10)


def percentile(values: np.ndarray, probability: float) -> float:
    return float(np.sort(values)[math.ceil(probability * len(values)) - 1])


def reconstruct_bases() -> dict[str, list[dict[str, object]]]:
    candidates = {domain: [] for domain in DOMAINS}
    for layer in range(LAYERS):
        path = ROOT / f"reports/hera_moe/p0_route_layers/layer_{layer:02d}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        for domain in DOMAINS:
            for expert in range(EXPERTS):
                candidates[domain].append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "router_weight_squared_sum": report["domains"][domain][
                            "router_weight_squared_sum"
                        ][expert],
                        "count": report["domains"][domain]["counts"][expert],
                    }
                )
    return {
        domain: sorted(
            candidates[domain],
            key=lambda row: (
                -row["router_weight_squared_sum"],
                -row["count"],
                row["layer"],
                row["expert"],
            ),
        )
        for domain in DOMAINS
    }


def independent_trace(
    routes: np.ndarray, selected: list[dict[str, object]], switch_mib: float
) -> dict[str, object]:
    base = np.zeros((LAYERS, EXPERTS), dtype=np.bool_)
    for row in selected:
        base[row["layer"], row["expert"]] = True
    layer_axis = np.arange(LAYERS)[None, :, None]
    is_base = base[layer_axis, routes]
    cold = (~is_base).reshape(TOKENS, LAYERS * TOP_K)
    flat = routes.reshape(TOKENS, LAYERS * TOP_K)
    misses = np.zeros(TOKENS, dtype=np.int16)
    event: Counter[str] = Counter()
    for start in range(0, TOKENS, CONTEXT_TOKENS):
        primary: list[int | None] = [None] * LAYERS
        victim: OrderedDict[int, None] = OrderedDict()
        for token in range(start, start + CONTEXT_TOKENS):
            for flat_index in np.flatnonzero(cold[token]):
                index = int(flat_index)
                layer = index // TOP_K
                key = layer * EXPERTS + int(flat[token, index])
                if primary[layer] == key:
                    event["primary_hits"] += 1
                elif key in victim:
                    victim.pop(key)
                    old = primary[layer]
                    primary[layer] = key
                    if old is not None:
                        victim.pop(old, None)
                        victim[old] = None
                    event["victim_hits"] += 1
                else:
                    old = primary[layer]
                    primary[layer] = key
                    if old is not None:
                        victim.pop(old, None)
                        victim[old] = None
                        if len(victim) > 8:
                            victim.popitem(last=False)
                    event["misses"] += 1
                    misses[token] += 1
    h2d = misses.astype(np.float64) * EXPERT_MIB
    h2d[::CONTEXT_TOKENS] += switch_mib
    base_calls = int(is_base.sum(dtype=np.int64))
    cold_calls = TOKENS * LAYERS * TOP_K - base_calls
    return {
        "base_invocations": base_calls,
        "cold_invocations": cold_calls,
        "primary_hits": int(event["primary_hits"]),
        "victim_hits": int(event["victim_hits"]),
        "misses": int(event["misses"]),
        "mean": float(h2d.mean()),
        "p50": percentile(h2d, 0.50),
        "p95": percentile(h2d, 0.95),
        "p99": percentile(h2d, 0.99),
        "maximum": float(h2d.max()),
    }


if __name__ == "__main__":
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite DCHERA verification")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    capture = json.loads(ROUTE_CAPTURE.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "preregistration_hash": result["inputs"]["preregistration_sha256"]
        == sha256(PREREG),
        "base_lock_hash": result["inputs"]["domain_base_lock_sha256"]
        == sha256(LOCK),
        "route_capture_hash": result["inputs"]["route_capture_sha256"]
        == sha256(ROUTE_CAPTURE),
        "opened_route_disclosure": result["exploratory_opened_routes"] is True,
    }
    reconstructed = reconstruct_bases()
    checks["all_five_base_capacities"] = all(
        len(lock["bases"][domain]) == CAPACITY for domain in DOMAINS
    )
    checks["all_five_base_selections_exact"] = all(
        lock["bases"][domain] == reconstructed[domain][:CAPACITY]
        for domain in DOMAINS
    )
    checks["all_five_base_boundaries_exact"] = all(
        close(
            lock["boundaries"][domain][
                "minimum_selected_router_weight_squared_sum"
            ],
            reconstructed[domain][CAPACITY - 1]["router_weight_squared_sum"],
        )
        and close(
            lock["boundaries"][domain][
                "maximum_rejected_router_weight_squared_sum"
            ],
            reconstructed[domain][CAPACITY]["router_weight_squared_sum"],
        )
        for domain in DOMAINS
    )

    routes = {domain: [] for domain in DOMAINS}
    artifact_hashes = shapes = True
    for layer in range(LAYERS):
        item = capture["artifacts"][str(layer)]
        path = ROOT / item["artifact"]
        artifact_hashes &= sha256(path) == item["artifact_sha256"]
        tensors = load_file(path)
        for domain in DOMAINS:
            tensor = tensors[f"{domain}_router_ids"]
            shapes &= tuple(tensor.shape) == (TOKENS, TOP_K)
            routes[domain].append(tensor.numpy())
    checks["all_48_route_artifact_hashes"] = artifact_hashes
    checks["all_route_shapes"] = shapes

    entropy_gib = CAPACITY * PARAMETERS_PER_EXPERT * RATE_BPP / 8 / 2**30
    trunk_gib = NONEXPERT_PARAMETERS * 4 / 8 / 2**30
    cache_gib = 56 * EXPERT_MIB / 1024
    resident_gib = entropy_gib + trunk_gib + cache_gib
    cold_gib = (LAYERS * EXPERTS - CAPACITY) * EXPERT_MIB / 1024
    switch_mib = entropy_gib * 1024
    memory = result["memory_projection"]
    checks["independent_memory_formulas"] = (
        close(memory["entropy_base_gib"], entropy_gib)
        and close(memory["nonexpert_int4_gib"], trunk_gib)
        and close(memory["exact_cache_gib"], cache_gib)
        and close(memory["resident_weight_gib"], resident_gib)
        and close(memory["active_cold_bf16_host_gib"], cold_gib)
    )
    checks["base_switch_formula"] = close(
        result["policy"]["base_switch_mib"], switch_mib
    )
    checks["memory_gate"] = (
        resident_gib <= 5.75
        and cold_gib <= 24.0
        and result["gates"]["memory"] is True
    )

    reproduced = {}
    events_exact = traffic_exact = event_conservation = gates_exact = True
    independent_domain_passes = {}
    for domain in DOMAINS:
        observed = independent_trace(
            np.stack(routes[domain], axis=1),
            reconstructed[domain][:CAPACITY],
            switch_mib,
        )
        reproduced[domain] = observed
        expected = result["domains"][domain]
        events_exact &= (
            observed["base_invocations"] == expected["base_invocations"]
            and observed["cold_invocations"] == expected["cold_invocations"]
            and observed["primary_hits"] == expected["events"]["primary_hits"]
            and observed["victim_hits"] == expected["events"]["victim_hits"]
            and observed["misses"] == expected["events"]["misses"]
        )
        event_conservation &= (
            observed["primary_hits"]
            + observed["victim_hits"]
            + observed["misses"]
            == observed["cold_invocations"]
        )
        expected_traffic = expected[
            "total_h2d_mib_per_token_including_base_switch"
        ]
        traffic_exact &= all(
            close(observed[key], expected_traffic[key])
            for key in ("mean", "p50", "p95", "p99", "maximum")
        )
        independent_gate = {
            "mean_le_64": observed["mean"] <= 64.0,
            "p95_le_144": observed["p95"] <= 144.0,
            "p99_le_288": observed["p99"] <= 288.0,
        }
        independent_domain_passes[domain] = all(independent_gate.values())
        recorded_gate = result["gates"]["traffic_by_domain"][domain]
        gates_exact &= all(
            independent_gate[key] == recorded_gate[key] for key in independent_gate
        ) and independent_domain_passes[domain] == recorded_gate["all_traffic_gates"]
    checks["cold_event_conservation"] = event_conservation
    checks["independent_event_totals"] = events_exact
    checks["independent_total_traffic_percentiles"] = traffic_exact
    checks["independent_domain_gates"] = gates_exact
    checks["exact_domain_pass_pattern"] = independent_domain_passes == {
        "general": True,
        "code": False,
        "math": True,
        "multilingual": True,
        "instruction": False,
    }
    checks["negative_result_required"] = (
        not all(independent_domain_passes.values())
        and result["gates"]["all_traffic"] is False
        and result["p0b_authorized"] is False
        and result["p1_authorized"] is False
    )

    passed = sum(checks.values())
    verification_pass = passed == len(checks)
    verdict = (
        "p0a_exploratory_negative_verified"
        if verification_pass and not all(independent_domain_passes.values())
        else "verification_failed"
    )
    payload = {
        "kind": "dchera_moe_p0a_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verification_pass": verification_pass,
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "final_verdict": verdict,
        "domain_gate_passes": independent_domain_passes,
        "p0b_authorized": False,
        "p1_authorized": False,
        "reproduced": reproduced,
        "source_hashes": {
            "result_sha256": sha256(RESULT),
            "base_lock_sha256": sha256(LOCK),
            "route_capture_sha256": sha256(ROUTE_CAPTURE),
        },
        "claim_boundary": (
            "Verified exploratory result on opened routes; only the fixed "
            "known-domain policy is closed."
        ),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for domain in DOMAINS:
        row = reproduced[domain]
        rows.append(
            f"| {domain} | {row['mean']:.3f} | {row['p95']:.0f} | "
            f"{row['p99']:.0f} | {'PASS' if independent_domain_passes[domain] else 'FAIL'} |"
        )
    REPORT.write_text(
        "\n".join(
            [
                "# DCHERA-MoE P0A — onafhankelijke verificatie",
                "",
                f"Uitkomst: **{verdict}**; **{passed}/{len(checks)}** controles slagen.",
                "",
                "| Domein | Gem. MiB/token | p95 | p99 | Gate |",
                "|---|---:|---:|---:|:---:|",
                *rows,
                "",
                "General, math en multilingual passeren. Code en instruction "
                "falen de staartgates, zodat P0B en P1 gesloten blijven.",
                "",
                "De conclusie geldt voor de vaste bekende-domeinbasis en niet "
                "voor alle mogelijke contextadaptieve caches.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verification_pass": verification_pass,
                "checks": f"{passed}/{len(checks)}",
                "final_verdict": verdict,
            },
            indent=2,
        )
    )
