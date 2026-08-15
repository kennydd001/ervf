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
LAYERS = 48
EXPERTS = 128
TOP_K = 8
TOKENS = 32_768
CONTEXT_TOKENS = 1_024
BASE_EXPERTS = 4_280
VICTIM_CAPACITY = 8
EXPERT_MIB = 9
PARAMETERS_PER_EXPERT = 4_718_592
NONEXPERT_PARAMETERS = 1_541_093_376
RATE_BPP = 1.930708991156684

BASE_LOCK = ROOT / "reports/dhera_moe/p0_base_lock.json"
RESULT = ROOT / "reports/dhera_moe/p0_cache_result.json"
ROUTE_CAPTURE = ROOT / "reports/dhera_moe/p0_route_capture.json"
INPUT_LOCK = ROOT / "reports/dhera_moe/p0_input_lock.json"
PREREG = ROOT / "reports/dhera_moe/P0_BUDGET_CACHE_PREREGISTRATION.md"
CLARIFICATION = ROOT / "reports/dhera_moe/P0_PROTOCOL_CLARIFICATION_001.md"
OUTPUT = ROOT / "reports/dhera_moe/p0_cache_verification.json"
REPORT = ROOT / "reports/dhera_moe/P0_CACHE_VERIFICATION.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(values: np.ndarray, probability: float) -> int:
    ordered = np.sort(values)
    return int(ordered[math.ceil(probability * len(ordered)) - 1])


def independently_select_base() -> list[dict[str, object]]:
    candidates = []
    for layer in range(LAYERS):
        path = ROOT / f"reports/hera_moe/p0_route_layers/layer_{layer:02d}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        for expert in range(EXPERTS):
            squared = sum(
                report["domains"][domain]["router_weight_squared_sum"][expert]
                for domain in DOMAINS
            )
            count = sum(
                report["domains"][domain]["counts"][expert]
                for domain in DOMAINS
            )
            candidates.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "router_weight_squared_sum": squared,
                    "count": count,
                }
            )
    return sorted(
        candidates,
        key=lambda row: (
            -row["router_weight_squared_sum"],
            -row["count"],
            row["layer"],
            row["expert"],
        ),
    )


def independently_simulate(
    routes: np.ndarray, base_mask: np.ndarray
) -> dict[str, object]:
    flat_routes = routes.reshape(TOKENS, LAYERS * TOP_K)
    layer_axis = np.arange(LAYERS)[None, :, None]
    base_calls = int(base_mask[layer_axis, routes].sum(dtype=np.int64))
    cold = (~base_mask[layer_axis, routes]).reshape(TOKENS, LAYERS * TOP_K)

    events: Counter[str] = Counter()
    misses_per_token = np.zeros(TOKENS, dtype=np.int16)
    misses_per_layer = np.zeros(LAYERS, dtype=np.int64)
    context_misses = []
    for context_start in range(0, TOKENS, CONTEXT_TOKENS):
        primary: list[int | None] = [None] * LAYERS
        victim: OrderedDict[int, None] = OrderedDict()
        for token in range(context_start, context_start + CONTEXT_TOKENS):
            token_misses = 0
            for flat_index in np.flatnonzero(cold[token]):
                index = int(flat_index)
                layer = index // TOP_K
                expert = int(flat_routes[token, index])
                key = layer * EXPERTS + expert
                if primary[layer] == key:
                    events["primary_hits"] += 1
                    continue
                if key in victim:
                    victim.pop(key)
                    old = primary[layer]
                    primary[layer] = key
                    if old is not None:
                        victim.pop(old, None)
                        victim[old] = None
                        if len(victim) > VICTIM_CAPACITY:
                            victim.popitem(last=False)
                    events["victim_hits"] += 1
                    continue
                old = primary[layer]
                primary[layer] = key
                if old is not None:
                    victim.pop(old, None)
                    victim[old] = None
                    if len(victim) > VICTIM_CAPACITY:
                        victim.popitem(last=False)
                events["misses"] += 1
                token_misses += 1
                misses_per_layer[layer] += 1
            misses_per_token[token] = token_misses
        context_misses.append(
            int(misses_per_token[context_start : context_start + CONTEXT_TOKENS].sum())
        )
    total = TOKENS * LAYERS * TOP_K
    cold_calls = total - base_calls
    return {
        "base_invocations": base_calls,
        "cold_invocations": cold_calls,
        "primary_hits": int(events["primary_hits"]),
        "victim_hits": int(events["victim_hits"]),
        "misses": int(events["misses"]),
        "context_misses": context_misses,
        "misses_per_layer": misses_per_layer.tolist(),
        "misses_mean": float(misses_per_token.mean()),
        "misses_p50": nearest_rank(misses_per_token, 0.50),
        "misses_p95": nearest_rank(misses_per_token, 0.95),
        "misses_p99": nearest_rank(misses_per_token, 0.99),
        "misses_maximum": int(misses_per_token.max()),
    }


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


if __name__ == "__main__":
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite DHERA verification")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    base_lock = json.loads(BASE_LOCK.read_text(encoding="utf-8"))
    capture = json.loads(ROUTE_CAPTURE.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}

    checks["result_preregistration_hash"] = (
        result["inputs"]["preregistration_sha256"] == sha256(PREREG)
    )
    checks["result_clarification_hash"] = (
        result["inputs"]["clarification_sha256"] == sha256(CLARIFICATION)
    )
    checks["result_base_lock_hash"] = (
        result["inputs"]["base_lock_sha256"] == sha256(BASE_LOCK)
    )
    checks["result_route_capture_hash"] = (
        result["inputs"]["route_capture_sha256"] == sha256(ROUTE_CAPTURE)
    )
    checks["capture_input_lock_hash"] = (
        capture["input_lock_sha256"] == sha256(INPUT_LOCK)
    )

    ordered = independently_select_base()
    checks["base_capacity"] = (
        base_lock["capacity_experts"] == BASE_EXPERTS
        and len(base_lock["selected"]) == BASE_EXPERTS
    )
    checks["base_selection_exact"] = base_lock["selected"] == ordered[:BASE_EXPERTS]
    checks["base_boundary_exact"] = (
        close(
            base_lock["minimum_selected_router_weight_squared_sum"],
            ordered[BASE_EXPERTS - 1]["router_weight_squared_sum"],
        )
        and close(
            base_lock["maximum_rejected_router_weight_squared_sum"],
            ordered[BASE_EXPERTS]["router_weight_squared_sum"],
        )
    )

    base_mask = np.zeros((LAYERS, EXPERTS), dtype=np.bool_)
    for row in ordered[:BASE_EXPERTS]:
        base_mask[row["layer"], row["expert"]] = True
    routes_by_domain = {domain: [] for domain in DOMAINS}
    artifact_hashes = report_hashes = shapes = counts = official = True
    for layer in range(LAYERS):
        item = capture["artifacts"][str(layer)]
        artifact = ROOT / item["artifact"]
        layer_report_path = ROOT / item["report"]
        artifact_hashes &= sha256(artifact) == item["artifact_sha256"]
        report_hashes &= sha256(layer_report_path) == item["report_sha256"]
        layer_report = json.loads(layer_report_path.read_text(encoding="utf-8"))
        official &= layer_report["official_topk_captured_exactly_once_per_chunk"]
        tensors = load_file(artifact)
        for domain in DOMAINS:
            tensor = tensors[f"{domain}_router_ids"]
            shapes &= tuple(tensor.shape) == (TOKENS, TOP_K)
            observed_counts = np.bincount(
                tensor.numpy().reshape(-1).astype(np.int64), minlength=EXPERTS
            ).tolist()
            counts &= observed_counts == layer_report["domains"][domain]["counts"]
            routes_by_domain[domain].append(tensor.numpy())
    checks["all_48_artifact_hashes"] = artifact_hashes
    checks["all_48_report_hashes"] = report_hashes
    checks["all_official_topk_calls"] = official
    checks["all_route_shapes"] = shapes
    checks["all_route_counts"] = counts

    reproduced = {}
    events_exact = summaries_exact = contexts_exact = layers_exact = True
    conservation = gates_exact = True
    independently_negative = False
    for domain in DOMAINS:
        routes = np.stack(routes_by_domain[domain], axis=1)
        observed = independently_simulate(routes, base_mask)
        reproduced[domain] = observed
        expected = result["domains"][domain]
        events_exact &= (
            observed["base_invocations"] == expected["base_invocations"]
            and observed["cold_invocations"] == expected["cold_invocations"]
            and observed["primary_hits"] == expected["events"]["primary_hits"]
            and observed["victim_hits"] == expected["events"]["victim_hits"]
            and observed["misses"] == expected["events"]["misses"]
        )
        conservation &= (
            observed["primary_hits"]
            + observed["victim_hits"]
            + observed["misses"]
            == observed["cold_invocations"]
            and observed["base_invocations"] + observed["cold_invocations"]
            == TOKENS * LAYERS * TOP_K
        )
        contexts_exact &= observed["context_misses"] == expected["context_misses"]
        layers_exact &= observed["misses_per_layer"] == expected["misses_per_layer"]
        exp_misses = expected["misses_per_token"]
        summaries_exact &= (
            close(observed["misses_mean"], exp_misses["mean"])
            and observed["misses_p50"] == exp_misses["p50"]
            and observed["misses_p95"] == exp_misses["p95"]
            and observed["misses_p99"] == exp_misses["p99"]
            and observed["misses_maximum"] == exp_misses["maximum"]
        )
        h2d = {
            "mean": observed["misses_mean"] * EXPERT_MIB,
            "p95": observed["misses_p95"] * EXPERT_MIB,
            "p99": observed["misses_p99"] * EXPERT_MIB,
        }
        independent_gate = {
            "mean_le_64": h2d["mean"] <= 64.0,
            "p95_le_144": h2d["p95"] <= 144.0,
            "p99_le_288": h2d["p99"] <= 288.0,
        }
        recorded_gate = result["gates"]["traffic_by_domain"][domain]
        gates_exact &= all(
            independent_gate[key] == recorded_gate[key]
            for key in independent_gate
        ) and (all(independent_gate.values()) == recorded_gate["all_traffic_gates"])
    independently_negative = any(
        not result["gates"]["traffic_by_domain"][domain]["all_traffic_gates"]
        for domain in DOMAINS
    )
    checks["cold_event_conservation"] = conservation
    checks["independent_event_totals"] = events_exact
    checks["independent_context_totals"] = contexts_exact
    checks["independent_layer_totals"] = layers_exact
    checks["independent_nearest_rank_summaries"] = summaries_exact
    checks["independent_traffic_gates"] = gates_exact

    entropy_gib = BASE_EXPERTS * PARAMETERS_PER_EXPERT * RATE_BPP / 8 / 2**30
    trunk_gib = NONEXPERT_PARAMETERS * 4 / 8 / 2**30
    cache_gib = 56 * EXPERT_MIB / 1024
    cold_gib = (LAYERS * EXPERTS - BASE_EXPERTS) * EXPERT_MIB / 1024
    resident_gib = entropy_gib + trunk_gib + cache_gib
    memory = result["memory_projection"]
    checks["independent_byte_formulas"] = (
        close(memory["entropy_base_gib"], entropy_gib)
        and close(memory["nonexpert_int4_gib"], trunk_gib)
        and close(memory["exact_cache_gib"], cache_gib)
        and close(memory["cold_bf16_host_gib"], cold_gib)
        and close(memory["resident_weight_gib"], resident_gib)
    )
    checks["memory_gate"] = (
        resident_gib <= 5.75
        and cold_gib <= 24.0
        and result["gates"]["memory"] is True
    )
    checks["negative_verdict_required"] = (
        independently_negative
        and result["gates"]["all_traffic"] is False
        and result["verdict"]
        == "cache_trace_negative_pending_independent_verification"
        and result["p1_authorized"] is False
    )

    passed = sum(checks.values())
    verification_pass = passed == len(checks)
    final_verdict = (
        "cache_trace_negative_verified"
        if verification_pass and independently_negative
        else "verification_failed"
    )
    payload = {
        "kind": "dhera_moe_p0_cache_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verification_pass": verification_pass,
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "final_verdict": final_verdict,
        "p1_authorized": False,
        "reproduced": reproduced,
        "memory": {
            "resident_weight_gib": resident_gib,
            "cold_bf16_host_gib": cold_gib,
        },
        "source_hashes": {
            "result_sha256": sha256(RESULT),
            "route_capture_sha256": sha256(ROUTE_CAPTURE),
            "base_lock_sha256": sha256(BASE_LOCK),
            "input_lock_sha256": sha256(INPUT_LOCK),
        },
        "claim_boundary": (
            "Verified trace-level falsification of this fixed policy only; no "
            "measured transfer latency, model quality, or general cache impossibility."
        ),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for domain in DOMAINS:
        row = result["domains"][domain]["h2d_mib_per_token"]
        rows.append(
            f"| {domain} | {row['mean']:.3f} | {row['p95']:.0f} | "
            f"{row['p99']:.0f} | FAIL |"
        )
    REPORT.write_text(
        "\n".join(
            [
                "# DHERA-MoE P0 — onafhankelijke cacheverificatie",
                "",
                f"Uitkomst: **{final_verdict}**. Alle **{passed}/{len(checks)}** "
                "controles slagen.",
                "",
                "| Domein | Gem. MiB/token | p95 | p99 | Verkeersgate |",
                "|---|---:|---:|---:|:---:|",
                *rows,
                "",
                f"De geheugengate slaagt op **{resident_gib:.6f} GiB** resident "
                f"en **{cold_gib:.6f} GiB** host-cold. De verkeersgate faalt "
                "voor ieder domein; P1 blijft gesloten.",
                "",
                "Deze conclusie falsifieert uitsluitend de vooraf geregistreerde "
                "4.280-base + 48-primary + 8-victimpolicy. Zij bewijst niet dat "
                "iedere dynamische of domeingeconditioneerde cache onmogelijk is.",
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
                "final_verdict": final_verdict,
            },
            indent=2,
        )
    )
