from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from moe_lab.reporting import ROOT

R = ROOT / "reports/streamq5_moe"
REGISTRY = R / "BREAKTHROUGH_PHASE_REGISTRY_2026-08-12.yaml"
FINAL_REPORT = R / "BREAKTHROUGH_RESEARCH_FINAL_REPORT_2026-08-12.md"
OUT = R / "breakthrough_phase_independent_verification_2026-08-12.json"
REPORT = R / "BREAKTHROUGH_PHASE_INDEPENDENT_VERIFICATION_REPORT_2026-08-12.md"
BANK = ROOT / "reports/runs/streamq5_moe/port80b_p0/port80b_p0_full_q5_bank.bin"


def load(name: str) -> dict:
    return json.loads((R / name).read_text(encoding="utf-8"))


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


checks: list[dict] = []


def check(name: str, passed: bool, detail=None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    if OUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite final verification")
    registry = REGISTRY.read_text(encoding="utf-8")
    ids = re.findall(r"^  - id: (B\d{3})$", registry, flags=re.MULTILINE)
    statuses = re.findall(r"^    status: ([a-z_]+)$", registry, flags=re.MULTILINE)
    evidence = re.findall(r"^      - (reports/[^\r\n]+)$", registry, flags=re.MULTILINE)
    check("registry_ids_contiguous", ids == [f"B{i:03d}" for i in range(1, 11)], ids)
    check("registry_all_terminal", "in_progress" not in statuses and len(statuses) == 10, statuses)
    missing = [path for path in evidence if not (ROOT / path).is_file()]
    check("registry_evidence_exists", not missing and len(evidence) >= 20, {"count": len(evidence), "missing": missing})
    check("final_report_exists", FINAL_REPORT.is_file() and FINAL_REPORT.stat().st_size > 5_000, FINAL_REPORT.stat().st_size if FINAL_REPORT.exists() else 0)

    audit = load("gaugepack_p9d1_p9b_mutation_audit.json")
    check("p9b_noop_audit", audit["status"] == "p9b_pruning_noop_proven" and all(audit["checks"].values()), audit["checks"])
    for name, rel, top1 in (
        ("p9br_corrected_structured_wanda_validation.json", 0.4780387471409896, 0.608661413192749),
        ("p9er_group_balanced_corrected_validation.json", 0.4718605527685509, 0.608661413192749),
        ("p9f_quarter_pruning_validation.json", 0.22845561347740692, 0.748031497001648),
    ):
        result = load(name)
        mutation = all(row["mutation"]["effective_mutation"] and row["mutation"]["remaining_masked_nonzero"] == 0 for row in result["layers"])
        valid = result["status"] == "validation_closed" and not result["overall_pass"] and len(result["layers"]) == 48 and mutation and close(result["relative_cross_entropy_increase"], rel) and close(result["top1_agreement"], top1)
        check(f"quality_close_{name.removesuffix('.json')}", valid, {"relative_ce": result["relative_cross_entropy_increase"], "top1": result["top1_agreement"]})

    gauge = load("gaugepack_p9d1_result.json")
    check("gaugepack_blocked_before_kernel", gauge["status"] == "blocked_invalid_p9b_premise" and not gauge.get("gpu_used", False), gauge.get("status"))

    c0 = load("ergv_compiler_cpu_tests.json")
    c1 = load("ergv_c1_generated_gpu_gate.json")
    c2 = load("ergv_c2_independent_verification.json")
    check("ergv_c0", c0["overall_pass"] and c0["tests_passed"] == c0["tests_total"] == 7)
    check("ergv_c1", c1["overall_pass"] and c1["elements_compared"] == 115_496 and c1["different_bits"] == 0 and c1["all_finite"])
    check("ergv_c2_verifier", c2["overall_pass"] and c2["checks_passed"] == c2["checks_total"] == 63 and c2["raw_event_values_verified"] == 960)
    ratios = c2["recomputed_ratios"]
    check("ergv_c2_beats_uniform_p7", ratios["versus_manual_p7"]["q8"]["p50_ratio"] <= 0.98 and ratios["versus_manual_p7"]["q5"]["p50_ratio"] <= 0.98, ratios["versus_manual_p7"])
    check("ergv_c2_n1c_parity_not_breakthrough", all(0.995 <= ratios["versus_manual_n1c"][bank][percentile] <= 1.005 for bank in ("q8", "q5") for percentile in ("p50_ratio", "p95_ratio")), ratios["versus_manual_n1c"])

    tier = load("tierflow_f0_independent_verification.json")
    tier_metrics = tier["recomputed"]["test"]["aggregate"]
    check("tierflow_verifier", tier["status"] == "independent_verification_pass" and tier["checks_passed"] == tier["checks_total"] == 20)
    check("tierflow_traffic_gates", tier_metrics["critical_expert_bytes_reduction_x"] >= 4 and tier_metrics["worst_case_new_load_reduction_x"] >= 8, tier_metrics)
    check("tierflow_quality_boundary", tier_metrics["router_output_substitution_rate"] > 0.30 and tier_metrics["exact_route_set_match_rate"] < 0.10, {"substitution": tier_metrics["router_output_substitution_rate"], "exact_sets": tier_metrics["exact_route_set_match_rate"]})

    port = load("port80b_p0_physical_host_bank_result.json")
    port_verify = load("port80b_p0_independent_verification.json")
    check("port80b_independent_verifier", port_verify["verification_pass"] and port_verify["checks_passed"] == port_verify["checks_total"] == 10)
    check("port80b_physical_file_size", BANK.is_file() and BANK.stat().st_size == 49_925_652_480, BANK.stat().st_size if BANK.exists() else None)
    check("port80b_one_hour", port["gates"]["one_hour_uninterrupted"] and port["elapsed_seconds"] >= 3_600, port["elapsed_seconds"])
    zero = port["scenarios"]["zero_cache"]
    check("port80b_zero_cache_raw_count", len(zero["h2d_ms"]) == 10_000 and zero["tokens"] == 10_000, len(zero["h2d_ms"]))
    check("port80b_h2d_gate_failed", not port["gates"]["zero_cache_h2d_p95_le_45ms"] and close(zero["h2d"]["p95"], 73.54437026977537), zero["h2d"])
    telemetry = port["hard_page_read_telemetry"]
    check("port80b_page_read_gate_failed", not port["gates"]["no_system_page_reads_after_warmup"] and len(telemetry["samples"]) == 3569 and telemetry["page_reads_per_sec"]["max"] > 0, telemetry["page_reads_per_sec"])
    check("port80b_commit_pass", port["gates"]["peak_process_commit_le_58gib"] and port["memory_final"]["peak_pagefile"] == 6_074_916_864, port["memory_final"]["peak_pagefile"])
    check("port80b_overall_negative", port["status"] == "fail" and not port["overall_pass"] and sum(not value for value in port["gates"].values()) == 2, port["gates"])

    check("industrial_breakthrough_not_claimed", "industrial_breakthrough_proven: false" in registry and "not an industrial LLM" in FINAL_REPORT.read_text(encoding="utf-8"))
    failures = [row for row in checks if not row["passed"]]
    payload = {
        "kind": "breakthrough_phase_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": not failures,
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "checks": checks,
        "failures": failures,
        "gpu_used": False,
        "network_used": False,
        "physical_bank_rehashed": False,
        "claim_boundary": "Cross-artifact consistency verifier; relies on the dedicated PORT80B full-hash verifier and does not repeat the one-hour GPU experiment.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# Breakthrough-phase independent verification\n\n"
        f"Verdict: **{'PASS' if payload['overall_pass'] else 'FAIL'}** — {payload['checks_passed']}/{payload['checks_total']} checks.\n\n"
        "De cross-artifactcontrole bevestigt de GaugePack-correctie en quality-closes, ERGV C0/C1/C2, TierFlow-F0 en de fysieke PORT80B-P0-gates. "
        "Er is geen GPU- of netwerkwerk herhaald; de volledige bankhash wordt gedekt door de afzonderlijke 10/10 PORT80B-verifier.\n",
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in ("overall_pass", "checks_passed", "checks_total", "failures")}, indent=2))


if __name__ == "__main__":
    main()
