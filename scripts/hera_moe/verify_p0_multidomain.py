from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file

from moe_lab.reporting import ROOT


DOMAINS = ("general", "code", "math", "multilingual", "instruction")
RESULT = ROOT / "reports/hera_moe/p0_multidomain_tier_result.json"
OUTPUT = ROOT / "reports/hera_moe/p0_multidomain_verification.json"
REPORT = ROOT / "reports/hera_moe/P0_MULTIDOMAIN_VERIFICATION.md"
THRESHOLD = 128
PARAMETERS_PER_EXPERT = 4_718_592
NONEXPERT_PARAMETERS = 1_541_093_376
RATE = 1.930708991156684


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean()), "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)), "p99": float(torch.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


if __name__ == "__main__":
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite HERA P0 verification")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = {
        "verdict_static_tier_negative": result["verdict"] == "static_tier_negative",
        "p1_not_authorized": result["p1_authorized"] is False,
        "execution_controls_pass": all(result["controls"].values()),
        "all_48_artifact_hashes_match": True,
        "all_48_report_hashes_match": True,
        "all_route_shapes_match": True,
        "all_route_counts_recompute": True,
        "all_domain_invocation_totals_match": True,
        "general_exactly_reproduces_e2gq": True,
        "domain_hot_counts_match": True,
        "hot_union_and_growth_match": True,
        "cold_call_distributions_match": True,
        "maximum_layer_cold_fractions_match": True,
        "memory_projection_matches": True,
        "attempts_and_addenda_preserved": True,
    }
    hot_masks = {domain: torch.zeros((48, 128), dtype=torch.bool) for domain in DOMAINS}
    all_routes = {domain: [] for domain in DOMAINS}
    for layer in range(48):
        manifest = result["artifacts"][str(layer)]
        artifact = ROOT / manifest["artifact"]
        report_path = ROOT / manifest["report"]
        checks["all_48_artifact_hashes_match"] &= sha256(artifact) == manifest["artifact_sha256"]
        checks["all_48_report_hashes_match"] &= sha256(report_path) == manifest["report_sha256"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        routes = load_file(artifact)
        old = json.loads((ROOT / f"reports/e2gq_moe/p0_capture_layers/layer_{layer:02d}.json").read_text(encoding="utf-8"))
        for domain in DOMAINS:
            ids = routes[f"{domain}_router_ids"]
            checks["all_route_shapes_match"] &= tuple(ids.shape) == (32768, 8)
            counts = torch.bincount(ids.long().reshape(-1), minlength=128)
            stored = report["domains"][domain]["counts"]
            checks["all_route_counts_recompute"] &= counts.tolist() == stored
            checks["all_domain_invocation_totals_match"] &= int(counts.sum()) == 262144 == report["domains"][domain]["total_invocations"]
            hot_masks[domain][layer] = counts >= THRESHOLD
            all_routes[domain].append(ids)
            if domain == "general":
                checks["general_exactly_reproduces_e2gq"] &= counts.tolist() == old["router_counts"]

    recomputed_domain_hot = {domain: int(hot_masks[domain].sum()) for domain in DOMAINS}
    checks["domain_hot_counts_match"] &= recomputed_domain_hot == result["hot_experts_by_domain"]
    union = torch.stack([hot_masks[domain] for domain in DOMAINS]).any(dim=0)
    growth = []
    cumulative = torch.zeros_like(union)
    for domain in DOMAINS:
        before = int(cumulative.sum()); cumulative |= hot_masks[domain]
        growth.append({"domain": domain, "new_hot_experts": int(cumulative.sum()) - before, "cumulative_hot_experts": int(cumulative.sum())})
    checks["hot_union_and_growth_match"] &= int(union.sum()) == result["hot_union_experts"] and growth == result["union_growth"]

    for domain in DOMAINS:
        cold_calls = torch.zeros(32768, dtype=torch.int16)
        maximum_fraction = 0.0
        for layer in range(48):
            calls = (~union[layer])[all_routes[domain][layer].long()].sum(dim=1).to(torch.int16)
            cold_calls += calls
            maximum_fraction = max(maximum_fraction, float(calls.sum()) / (32768 * 8))
        measured = quantiles(cold_calls)
        checks["cold_call_distributions_match"] &= all(close(measured[key], result["cold_calls_per_token"][domain][key]) for key in measured)
        checks["maximum_layer_cold_fractions_match"] &= close(maximum_fraction, result["maximum_layer_cold_invocation_fraction"][domain])

    hot = int(union.sum()); cold = 6144 - hot
    hot_gib = hot * PARAMETERS_PER_EXPERT * RATE / 8 / 2**30
    trunk_gib = NONEXPERT_PARAMETERS * 4 / 8 / 2**30
    cold_gib = cold * PARAMETERS_PER_EXPERT * 16 / 8 / 2**30
    memory = result["memory_projection"]
    checks["memory_projection_matches"] &= (
        close(hot_gib, memory["hot_entropy_gib"]) and close(trunk_gib, memory["nonexpert_int4_gib"])
        and close(hot_gib + trunk_gib, memory["resident_weight_gib"])
        and close(cold_gib, memory["cold_bf16_host_gib"])
        and memory["memory_gate_pass"] is False and hot_gib + trunk_gib > 5.75
    )
    preserved = {
        "reports/hera_moe/p0_multidomain_tier_result_attempt_001.json": "5ed9cfc1c411b6e2d75dca51ba404edac3e6cee6c0250e96e294a0b1b7a74066",
        "reports/hera_moe/P0_MULTIDOMAIN_TIER_AUDIT_ATTEMPT_001.md": "f502ba7f5bfac5e440a895e75eb657e66f352aab2b7d4c9416cf9011423fbd2f",
        "reports/hera_moe/p0_multidomain_tier_result_attempt_002.json": "b7833d21454362ce90478626f3302868a1417cf3b0257ae0d66251d524dbebba",
        "reports/hera_moe/P0_MULTIDOMAIN_TIER_AUDIT_ATTEMPT_002.md": "4458cdf57dab61a400c43f01034d9e88470b1a0deef95c8076b5e6f3f49d0df3",
    }
    checks["attempts_and_addenda_preserved"] &= all(sha256(ROOT / path) == digest for path, digest in preserved.items())
    checks["attempts_and_addenda_preserved"] &= all((ROOT / f"reports/hera_moe/P0_PROTOCOL_ADDENDUM_{i:03d}.md").is_file() for i in (1, 2))

    payload = {
        "kind": "hera_moe_p0_independent_verification", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "result_sha256": sha256(RESULT), "checks": checks,
        "checks_passed": sum(checks.values()), "checks_total": len(checks),
        "verification_pass": all(checks.values()), "hot_union_experts": hot,
        "cold_experts": cold, "resident_weight_gib": hot_gib + trunk_gib,
        "gate_excess_gib": hot_gib + trunk_gib - 5.75,
        "verdict": "static_multidomain_tier_falsified_before_quality",
        "claim_boundary": "The count>=128 union rule is falsified; dynamic/domain-conditioned tiers are not tested.",
    }
    if not payload["verification_pass"]:
        raise AssertionError([name for name, passed in checks.items() if not passed])
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# HERA-MoE P0 — onafhankelijke verificatie", "",
        f"Uitkomst: **PASS ({payload['checks_passed']}/{payload['checks_total']})**.", "",
        f"De vijf vooraf gelockte domeinen maken **{hot:,}/6.144** experts hot. De geprojecteerde resident weights zijn **{hot_gib + trunk_gib:.3f} GiB**, oftewel **{payload['gate_excess_gib']:.3f} GiB boven** de 5,75-GiB-gate.", "",
        "Alle 48 artifact- en rapporthashes, 240 routedatasets, counts, invocationtotalen, uniongroei, cold-callpercentielen en geheugenformules zijn onafhankelijk herberekend. General reproduceert E2GQ exact.", "",
        "Daarmee is uitsluitend de statische multidomain `count>=128`-union gefalsificeerd vóór kwaliteitstuning. Een dynamische of domeingeconditioneerde cachearchitectuur is niet getest en mag niet als gered resultaat worden gepresenteerd.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("verification_pass", "checks_passed", "checks_total", "hot_union_experts", "cold_experts", "resident_weight_gib", "gate_excess_gib")}, indent=2))
