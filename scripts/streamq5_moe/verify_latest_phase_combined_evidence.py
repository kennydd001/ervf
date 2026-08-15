#!/usr/bin/env python3
"""CPU-only verifier for the latest STREAMQ5-MoE combined evidence registry.

This program never imports a GPU runtime and never reruns a model.  It hashes
every evidence reference in the registry and independently recomputes the
decisive numerical/status claims from the frozen JSON artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "streamq5_moe"
REGISTRY = REPORT_DIR / "LATEST_PHASE_COMBINED_EVIDENCE_REGISTRY_2026-08-12.json"
OUTPUT = REPORT_DIR / "latest_phase_combined_evidence_verification.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def collect_evidence(node: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            found.append((node["path"], node["sha256"].lower()))
        for value in node.values():
            found.extend(collect_evidence(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(collect_evidence(value))
    return found


checks: list[dict[str, Any]] = []


def check(name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


registry = load_json(REGISTRY)
check("registry_kind", registry.get("registry_kind") == "streamq5_moe_latest_phase_combined_evidence")
check("central_registry_not_mutated", registry.get("central_registry_mutated") is False)
check("registry_final", str(registry.get("status", "")).startswith("final_"), registry.get("status"))

evidence = collect_evidence(registry)
evidence_results: list[dict[str, Any]] = []
for relative, expected in evidence:
    path = ROOT / relative
    exists = path.is_file()
    actual = sha256(path) if exists else None
    matched = exists and actual == expected
    evidence_results.append(
        {"path": relative, "exists": exists, "expected_sha256": expected, "actual_sha256": actual, "pass": matched}
    )
check("all_registry_evidence_hashes", bool(evidence_results) and all(item["pass"] for item in evidence_results),
      {"checked": len(evidence_results), "failed": [item["path"] for item in evidence_results if not item["pass"]]})

# Immutable prior state.
prior = load_json(REPORT_DIR / "breakthrough_phase_independent_verification_2026-08-12.json")
check("immutable_prior_26_of_26", prior.get("overall_pass") is True and prior.get("checks_passed") == 26
      and prior.get("checks_total") == 26)
ergv = load_json(REPORT_DIR / "ergv_c2_independent_verification.json")
check("immutable_ergv_63_of_63", ergv.get("overall_pass") is True and ergv.get("checks_passed") == 63
      and ergv.get("checks_total") == 63)
tier_f0 = load_json(REPORT_DIR / "tierflow_f0_independent_verification.json")
check("immutable_tierflow_f0_20_of_20", tier_f0.get("status") == "independent_verification_pass"
      and tier_f0.get("checks_passed") == 20 and tier_f0.get("checks_total") == 20)
p0 = load_json(REPORT_DIR / "port80b_p0_independent_verification.json")
check("immutable_port80b_p0_verified", p0.get("verification_pass") is True)

# DirectPath D1-D4R independent verdicts.
d1v = load_json(REPORT_DIR / "port80b_d1_token_batch_directpath_independent_verification.json")
check("d1_verified_negative", d1v.get("all_replayable_checks_pass") is True
      and d1v.get("independent_verdict") == "verified_negative")
d2v = load_json(REPORT_DIR / "port80b_d2_registered_scatter_independent_verification.json")
check("d2_verified_negative", d2v.get("all_replayable_checks_pass") is True
      and d2v.get("independent_verdict") == "verified_negative_with_protocol_findings")
d34v = load_json(REPORT_DIR / "port80b_d3_d4_combined_independent_verification.json")
check("d3_d4r_verified_negative", d34v.get("all_replayable_checks_pass") is True
      and d34v.get("independent_verdict") == "all_four_negative_confirmed_with_evidence_limits")

# D5: independently audited strong byte-transport component pass, with its
# explicit post-hoc replay limitation retained rather than hidden.
d5 = load_json(REPORT_DIR / "port80b_d5_cp_async_host_smem.json")
d5v = load_json(REPORT_DIR / "port80b_d5_cp_async_host_smem_independent_verification.json")
d5_times = d5["test"]["raw_ms"]
d5_p95 = percentile(d5_times, 0.95)
d5_payload_bytes = d5["protocol"]["total_tiles"] * d5["protocol"]["tile_bytes"]
d5_bw = d5_payload_bytes / (d5_p95 / 1000.0) / 1e9
check("d5_test_count_and_stats", len(d5_times) == 120
      and close(percentile(d5_times, 0.50), d5["test"]["stats"]["p50"])
      and close(d5_p95, d5["test"]["stats"]["p95"])
      and close(d5_bw, d5["effective_gb_s_at_p95"]))
check("d5_strong_transport_gates", d5.get("strong_pass") is True and d5_p95 <= 45.0 and d5_bw >= 21.627
      and d5["full_destination_mismatch_count"] == 0
      and not d5.get("unregister_failures") and d5.get("error") is None)
check("d5_independent_audit_with_replay_limit", d5v.get("independent_verdict") == "verified_strong_transport_component_pass"
      and d5v.get("all_replayable_checks_pass") is True)

# D6: exact but too slow; validation hard stop.
d6 = load_json(REPORT_DIR / "port80b_d6_exact_host_q5_fusion.json")
d6_times = d6["validation"]["raw_ms"]
d6_p95 = percentile(d6_times, 0.95)
check("d6_exact_q5", d6["correctness"]["different_bits"] == 0 and d6["correctness"]["bitwise_equal"] is True)
check("d6_negative_hard_stop", close(d6_p95, d6["validation"]["stats"]["p95"])
      and d6_p95 > 65.0 and d6["validation"]["open"] is False and d6.get("status") == "exact_host_q5_negative")

# D7: staged exact Q5 strong component pass.  The registry must keep the
# 60%-bank/973-MB-work-buffer/projection boundary.
d7 = load_json(REPORT_DIR / "port80b_d7_staged_exact_q5_plane.json")
d7_times = d7["test"]["raw_ms"]
d7_p95 = percentile(d7_times, 0.95)
d7_p50 = percentile(d7_times, 0.50)
d7_bw = d7["physical"]["remote_payload_bytes"] / (d7_p95 / 1000.0) / 1e9
d7_projected = d7_p95 + d7["dense_projection"]["frozen_dense_shell_p95_ms"]
check("d7_exact_output_and_digest", d7["correctness"]["different_bits"] == 0
      and d7["correctness"]["bitwise_equal"] is True
      and d7["output_digests"]["resident"] == d7["output_digests"]["staged"])
check("d7_test_stats", len(d7_times) == 120 and close(d7_p50, d7["test"]["stats"]["p50"])
      and close(d7_p95, d7["test"]["stats"]["p95"]) and close(d7_bw, d7["effective_remote_payload_gb_s_at_p95"])
      and close(d7_projected, d7["dense_projection"]["projected_total_p95_ms"]))
check("d7_strong_component_gates", d7.get("strong_pass") is True and d7_p95 <= 55.0
      and d7_bw >= 15.0 and d7_projected <= 90.0 and d7.get("full_bank_pass") is False
      and d7["physical"]["registered_experts_per_layer"] == 307
      and d7["physical"]["hbm_work_buffer_bytes"] == 973_209_600)

# Optional D7 independent verifier becomes mandatory when the final registry
# references one; the recursive hash check above also freezes it.
d7_phase = registry["routes"]["directpath"]["phases"]["D7_STAGED_EXACT_Q5"]
d7_verification = d7_phase.get("independent_verification")
if isinstance(d7_verification, dict):
    d7v = load_json(ROOT / d7_verification["path"])
    check("d7_independent_verification", d7v.get("all_replayable_checks_pass") is True
          and d7v.get("verdict") == "d6_negative_and_d7_strong_component_pass_verified_with_exactness_scope_limit",
          d7v.get("verdict"))
    check("d7_exactness_scope_limit_preserved", d7v["exactness_scope_limit"]["payload_is_invariant"] is True
          and d7v["exactness_scope_limit"]["headers_are_ignored_by_q5_compute"] is True
          and "invariant synthetic expert payloads" in d7_phase["claim_boundary"])
else:
    check("d7_independent_verification", False, "registry still marks it pending")

# Other user-provided routes.
co = load_json(REPORT_DIR / "co_route_physical_ordering_trace_independent_verification.json")
check("co_route_verified_negative", co.get("status") == "independent_verification_pass"
      and co.get("checks_passed") == 17 and co.get("checks_total") == 17
      and co.get("verified_phase") == "validation")
nemotron = load_json(REPORT_DIR / "nemotron_n0_metadata_gate.json")
check("nemotron_metadata_only", nemotron.get("status") == "public_checkpoint_metadata_pass_alias_unproven")
nemotron_n1 = load_json(REPORT_DIR / "nemotron_n1_header_inventory.json")
check("nemotron_n1_header_inventory", nemotron_n1.get("status") == "header_inventory_pass"
      and nemotron_n1.get("overall_pass") is True
      and nemotron_n1["inventory"]["tensor_count"] == 24_147
      and nemotron_n1["inventory"]["routed_records"] == 23 * 128
      and nemotron_n1["inventory"]["routed_record_bytes"] == 5_612_560)
st2 = load_json(REPORT_DIR / "st2_mini_independent_verification.json")
check("st2_41_of_41_negative", st2.get("all_pass") is True and st2.get("checks_passed") == 41
      and st2.get("checks_total") == 41
      and st2.get("reconstructed_verdict") == "exact_host_usm_q5_pass_but_21_63_gbps_p95_side_fails_for_source_tree_and_ergv_width8")
tier = load_json(REPORT_DIR / "tierflow_persistent_set_functional_span_independent_verification.json")
check("tierflow_span_16_of_16_negative", tier.get("passed") is True and len(tier.get("checks", [])) == 16
      and all(item.get("pass") is True for item in tier["checks"])
      and tier["recomputed_gates"]["overall_pass"] is False)

d8 = load_json(REPORT_DIR / "port80b_d8_registration_capacity_knee.json")
d8_clean = [arm for arm in d8["sweep"] if arm["success"] is True and not arm["unregister_failures"]]
d8_raw_full = next(arm for arm in d8["sweep"] if arm["experts_per_layer"] == 512)
check("d8_clean_capacity_reinterpreted", max(arm["experts_per_layer"] for arm in d8_clean) == 499
      and len(d8_raw_full["unregister_failures"]) > 0
      and d8["entropy_pin_theoretical_gib"] < max(arm["registered_gib"] for arm in d8_clean))

# D9 bridges the clean 499-prefix capacity boundary with an explicit pageable
# 13-expert tail. Pass/fail timing is inclusive wall time, not CUDA events.
d9 = load_json(REPORT_DIR / "port80b_d9_capacity_aware_bank_bridge.json")
d9v = load_json(REPORT_DIR / "port80b_d9_capacity_aware_bank_bridge_independent_verification.json")
d9_expected = {
    "all_hot": (48.49565000040457, 49.1163649916416, 65.0),
    "mixed_5_hot_5_cold": (61.900600005174056, 68.6695049967966, 100.0),
    "all_cold_tail": (80.95004998904187, 88.13604997849325, 135.0),
}
d9_timing_ok = True
for case, (expected_p50, expected_p95, gate) in d9_expected.items():
    wall = d9["test"]["cases"][case]["wall_ms"]
    p50 = percentile(wall, 0.50)
    p95 = percentile(wall, 0.95)
    d9_timing_ok &= (len(wall) == 60 and close(p50, expected_p50) and close(p95, expected_p95)
                     and close(p50, d9["test"]["cases"][case]["wall_stats"]["p50"])
                     and close(p95, d9["test"]["cases"][case]["wall_stats"]["p95"])
                     and p95 <= gate)
check("d9_inclusive_wall_timings", d9_timing_ok)
d9_digest = "d91c3dd5c4464d11fa0688a29e8b471480a21779ff69f278ea1c946e91cf79f0"
check("d9_exactness_and_differentiated_route_integrity",
      all(item["different_bits"] == 0 and item["bitwise_equal"] is True
          and item["expected_sha256"] == d9_digest and item["observed_sha256"] == d9_digest
          for item in d9["correctness"].values())
      and all(item["full_image_byte_mismatches"] == 0 and item["pass"] is True
              for item in d9["integrity"]["positive"].values())
      and d9["integrity"]["negative_controls"]["wrong_expert"]["detected_mismatches"] == 3
      and d9["integrity"]["negative_controls"]["wrong_layer"]["detected_mismatches"] == 150)
check("d9_capacity_bridge_strong_pass_and_clean_cleanup",
      d9.get("status") == "capacity_bridge_strong_pass" and d9.get("primary_pass") is True
      and d9.get("strong_pass") is True and d9.get("full_bank_registration_pass") is False
      and d9["physical"]["registered_experts_per_layer"] == 499
      and d9["physical"]["cold_escape_experts_per_layer"] == 13
      and d9["physical"]["registered_ranges"] == 48
      and not d9["unregister_failures"] and d9["error"] is None)
check("d9_independent_16_of_16", d9v.get("all_checks_pass") is True
      and d9v.get("verdict") == "capacity_bridge_strong_pass_independently_verified"
      and len(d9v.get("checks", [])) == 16 and all(item.get("pass") is True for item in d9v["checks"]))
check("d9_ram_first_touch_caveat_preserved",
      d9["physical"]["available_ram_before_registration"] == 52_887_109_632
      and d9["physical"]["available_ram_after_registration"] == 52_777_537_536
      and d9["physical"]["available_ram_after_unregister"] == 3_122_561_024
      and "does not prove prompt OS reclaim" in registry["routes"]["directpath"]["phases"]
          ["D9_CAPACITY_AWARE_BANK_BRIDGE"]["ram_first_touch_caveat"]["interpretation"])

# D10A1-R is immutable negative evidence.  Three, and only three, component
# gates failed; later source analysis cannot retroactively promote this run.
d10a1r = load_json(REPORT_DIR / "port80b_d10a1r_conservative_resource_retry.json")
d10a1r_failed = {name for name, value in d10a1r["gates"].items() if not value}
check("d10a1r_immutable_three_gate_negative",
      d10a1r.get("status") == "component_composition_negative_endurance_closed"
      and d10a1r.get("overall_pass") is False
      and d10a1r.get("endurance_authorized_by_evidence") is False
      and d10a1r_failed == {
          "all_correctness_headers_zero_mismatch",
          "gdn_reference_abs_rel_le_2e_5",
          "shared_q5_bitexact",
      }
      and sum(bool(value) for value in d10a1r["gates"].values()) == 18)
check("d10a1r_failure_signatures_preserved",
      len(d10a1r["correctness"]) == 40
      and sum(item["header_mismatches"] for item in d10a1r["correctness"]) == 491
      and all(item["comparison"]["different_bits"] == 0
              and item["comparison"]["bitwise_equal"] is True
              for item in d10a1r["correctness"])
      and d10a1r["components"]["conv_nonzero"] == 0
      and d10a1r["components"]["shared_vs_resident"]["different_bits"] == 96_256
      and d10a1r["components"]["shared_vs_resident"]["elements"] == 98_304
      and not d10a1r["unregister_failures"] and d10a1r.get("error") is None)

# D10A2-R repaired stream ordering but its new 48-vs-36-layer CPU conv oracle
# asserted before validation.  It remains experimental-plumbing evidence only.
d10a2r = load_json(REPORT_DIR / "port80b_d10a2r_single_stream_repair_revision.json")
check("d10a2r_immutable_oracle_assertion",
      d10a2r.get("status") == "component_composition_negative_endurance_closed"
      and d10a2r.get("overall_pass") is False
      and d10a2r.get("endurance_authorized_by_evidence") is False
      and str(d10a2r.get("error", "")).startswith("AssertionError")
      and len(d10a2r.get("gates", {})) == 3
      and d10a2r["gates"]["registration_attempt_rows_48_all_success"] is True
      and d10a2r["gates"]["clean_unregister_48_ranges"] is True
      and d10a2r["gates"]["no_cuda_or_runner_error"] is False)
check("d10a2r_lifecycle_was_clean",
      len(d10a2r["registration_attempts"]) == 48
      and all(item["attempted"] and item["success"] for item in d10a2r["registration_attempts"])
      and len(d10a2r["unregister_attempts"]) == 48
      and all(item["attempted"] and item["success"] for item in d10a2r["unregister_attempts"])
      and not d10a2r["unregister_failures"])

# D10A2-R2 is the corrected, independently verified component composition.
# Its canonical JSON deliberately keeps endurance authorization false; the
# immutable Markdown interpolation error is recorded in a separate erratum.
d10a2r2 = load_json(REPORT_DIR / "port80b_d10a2r2_gdn36_oracle_repair.json")
d10a2r2v = load_json(REPORT_DIR / "port80b_d10a2r2_component_independent_verification.json")
d10a2_phase = registry["routes"]["directpath"]["phases"]["D10A2R2_GDN36_COMPONENT"]
check("d10a2r2_component_28_of_28_endurance_closed",
      d10a2r2.get("status") == "component_composition_pass_endurance_closed_pending_new_authorization"
      and d10a2r2.get("overall_pass") is True
      and d10a2r2.get("endurance_authorized_by_evidence") is False
      and len(d10a2r2["gates"]) == 28 and all(d10a2r2["gates"].values())
      and d10a2r2.get("error") is None)
check("d10a2r2_routed_q5_exact_and_fully_written",
      len(d10a2r2["correctness"]) == 40
      and all(item["header_mismatches"] == 0 and item["canary_mismatches"] == 0
              and item["comparison"]["different_bits"] == 0
              and item["comparison"]["bitwise_equal"] is True
              and item["comparison"]["finite"] is True
              and item["comparison"].get("poison_count", 0) == 0
              for item in d10a2r2["correctness"]))
d10a2_conv = d10a2r2["components"]["conv_full_bf16_comparison"]
check("d10a2r2_conv_oracle_exact",
      d10a2r2["components"]["conv_nonzero"] == 292_608
      and d10a2r2["components"]["conv_expected_nonzero"] == 292_608
      and d10a2_conv["elements"] == 1_179_648
      and d10a2_conv["different_words"] == 0 and d10a2_conv["bitwise_equal"] is True
      and d10a2_conv["left_sha256"] == d10a2_conv["right_sha256"]
      == "cedf5736557919b023d6f7cce73d0064df07236ff1e18b5d8b3fec49d658fa1e")
d10a2_shared = d10a2r2["components"]["shared_vs_resident"]
d10a2_payload = d10a2r2["components"]["shared_payload_comparison"]
check("d10a2r2_shared_exact_and_payload_audited",
      d10a2_shared["elements"] == 98_304 and d10a2_shared["different_bits"] == 0
      and d10a2_shared["bitwise_equal"] is True
      and len(d10a2_payload["layers"]) == 48
      and all(layer["matches_reference"] is True for layer in d10a2_payload["layers"]))
check("d10a2r2_validation_evidence_complete",
      len(d10a2r2["validation"]["wall_ms"]) == 32
      and len(d10a2r2["validation_output_evidence"]) == 32
      and all(len(item["arrays"]) == 9
              and all(array["finite"] is True and array["poison_count"] == 0
                      and len(array["sha256"]) == 64 for array in item["arrays"].values())
              for item in d10a2r2["validation_output_evidence"]))
check("d10a2r2_lifecycle_clean_48",
      len(d10a2r2["registration_attempts"]) == 48
      and all(item["success"] for item in d10a2r2["registration_attempts"])
      and len(d10a2r2["unregister_attempts"]) == 48
      and all(item["success"] for item in d10a2r2["unregister_attempts"])
      and not d10a2r2["unregister_failures"])
check("d10a2r2_independent_26_of_26",
      d10a2r2v.get("verification_pass") is True
      and len(d10a2r2v["independent_checks"]) == 26
      and all(d10a2r2v["independent_checks"].values())
      and d10a2r2v["gate_replay"]["count"] == 28
      and d10a2r2v["gate_replay"]["passed"] == 28)
check("d10a2r2_report_erratum_preserved",
      "canonical JSON keeps endurance_authorized_by_evidence=false" in d10a2_phase["report_erratum"]
      and d10a2r2v["endurance_decision"]["canonical_json_endurance_authorized_by_evidence"] is False
      and "incorrectly prints Endurance evidence-authorized: True"
          in d10a2r2v["endurance_decision"]["report_bug"])

# The original D10B preregistration failed CPU preflight.  All physical-action
# flags stayed false and no result artifact exists.
d10b_preflight = load_json(REPORT_DIR / "port80b_d10b_heldout_10000_endurance_preflight.json")
check("d10b_original_preflight_immutable_cpu_only_fail",
      d10b_preflight.get("status") == "compile_preflight_fail"
      and d10b_preflight.get("pass") is False
      and d10b_preflight.get("error") == "RuntimeError: preflight evidence failed"
      and d10b_preflight.get("component_opened") is False
      and d10b_preflight.get("endurance_opened") is False
      and d10b_preflight["physical_actions"]
      and not any(d10b_preflight["physical_actions"].values()))
check("d10b_original_has_no_gpu_result",
      not (REPORT_DIR / "port80b_d10b_heldout_10000_endurance.json").exists())

# D10B-R is the final synthetic 10,000-step endurance arm.  Recompute timing,
# drift, checkpoint-summary structure, resource gates and lifecycle from raw.
d10br = load_json(REPORT_DIR / "port80b_d10br_heldout_10000_endurance_revision.json")
d10brv1 = load_json(REPORT_DIR / "port80b_d10br_heldout_10000_endurance_independent_verification.json")
d10brv2 = load_json(REPORT_DIR / "port80b_d10br_endurance_independent_verification.json")
d10br_preflight = load_json(REPORT_DIR / "port80b_d10br_heldout_10000_endurance_revision_preflight.json")
d10br_phase = registry["routes"]["directpath"]["phases"]["D10BR_HELDOUT_10000_ENDURANCE"]
d10br_wall = d10br["latency"]["wall_ms"]
d10br_event = d10br["latency"]["cuda_event_ms"]
d10br_wall_stats = d10br["latency"]["wall_stats"]
check("d10br_status_and_19_of_19_gates",
      d10br.get("status") == "heldout_10000_endurance_pass"
      and d10br.get("overall_pass") is True
      and len(d10br["gates"]) == 19 and all(d10br["gates"].values())
      and d10br.get("error") is None)
check("d10br_heldout_route_contract",
      d10br["route_contract"] == {
          "label": "p4d_shaped_synthetic_proxy",
          "partition": [768, 1024],
          "route_sha256": "85f12fb0020bb8568dfc3683662e8251b29bf83684beb296dbb6d8734f5ffd20",
          "steps": 10_000,
          "warmups": 8,
      })
check("d10br_latency_stats_recomputed",
      len(d10br_wall) == 10_000 and len(d10br_event) == 10_000
      and all(math.isfinite(value) and value > 0 for value in d10br_wall + d10br_event)
      and close(percentile(d10br_wall, 0.50), d10br_wall_stats["p50"])
      and close(percentile(d10br_wall, 0.95), d10br_wall_stats["p95"])
      and close(percentile(d10br_wall, 0.99), d10br_wall_stats["p99"])
      and close(max(d10br_wall), d10br_wall_stats["max"])
      and d10br_wall_stats["p95"] <= 150.0 and d10br_wall_stats["p99"] <= 200.0)
d10br_first_p95 = percentile(d10br_wall[:1000], 0.95)
d10br_last_p95 = percentile(d10br_wall[-1000:], 0.95)
check("d10br_no_latency_degradation_gate",
      close(d10br_first_p95, d10br["latency"]["first_1000_wall_stats"]["p95"])
      and close(d10br_last_p95, d10br["latency"]["last_1000_wall_stats"]["p95"])
      and close(d10br_last_p95 / d10br_first_p95, d10br["latency"]["last_first_p95_ratio"])
      and d10br["latency"]["last_first_p95_ratio"] <= 1.20)
check("d10br_10000_state_and_telemetry_records",
      len(d10br["state_checks"]) == 10_000 and all(d10br["state_checks"])
      and len(d10br["telemetry"]) == 10_000
      and [item["step"] for item in d10br["telemetry"]] == list(range(10_000))
      and all(item["state_finite_and_written"] is True for item in d10br["telemetry"]))
d10br_schedule = [0] + list(range(99, 10_000, 100))
d10br_arrays = [array for row in d10br["checkpoint_evidence"] for array in row["arrays"].values()]
d10br_composed = [row["arrays"]["composed_state"]["sha256"] for row in d10br["checkpoint_evidence"]]
check("d10br_checkpoint_summary_contract",
      len(d10br["checkpoint_evidence"]) == 101
      and [row["step"] for row in d10br["checkpoint_evidence"]] == d10br_schedule
      and len(d10br_arrays) == 909
      and all(array["finite"] is True and array["poison_count"] == 0
              and len(array["sha256"]) == 64
              and all(character in "0123456789abcdef" for character in array["sha256"])
              for array in d10br_arrays)
      and len(set(d10br_composed)) == 101)
d10br_min_ram = min(item["available_ram"] for item in d10br["telemetry"])
d10br_min_vram = min(item["free_vram"] for item in d10br["telemetry"])
d10br_ram_loss = d10br["telemetry"][0]["available_ram"] - d10br["telemetry"][-1]["available_ram"]
check("d10br_resource_and_paging_gates",
      d10br_min_ram >= 2 * 1024**3
      and d10br_min_vram >= 512 * 1024**2
      and d10br_ram_loss <= 1024**3
      and d10br["physical"]["available_ram_after_first_touch"] >= 2 * 1024**3
      and len(d10br["page_reads"]["samples"]) == 703
      and max(item["page_reads_per_sec"] for item in d10br["page_reads"]["samples"]) <= 2048
      and d10br["page_reads"]["error"] is None)
check("d10br_dense_runtime_and_lifecycle",
      d10br["dense_runtime"]["dense_checksum_observed"]
      == d10br["dense_runtime"]["dense_checksum_expected"]
      and d10br["dense_runtime"]["runtime_sentinels"] == [165, 165]
      and len(d10br["registration_attempts"]) == 48
      and all(item["attempted"] and item["success"] for item in d10br["registration_attempts"])
      and len(d10br["unregister_attempts"]) == 48
      and all(item["attempted"] and item["success"] for item in d10br["unregister_attempts"])
      and not d10br["unregister_failures"])
check("d10br_first_independent_audit_19_of_19",
      d10brv1.get("verification_pass") is True and d10brv1.get("check_count") == 19
      and len(d10brv1["checks"]) == 19 and all(d10brv1["checks"].values()))
check("d10br_second_independent_audit_49_of_49_and_19_gates",
      d10brv2.get("status") == "independent_endurance_pass_with_summary_replayability_boundary"
      and d10brv2.get("pass") is True
      and d10brv2.get("check_count") == 49 and d10brv2.get("checks_passed") == 49
      and len(d10brv2["checks"]) == 49 and all(d10brv2["checks"].values())
      and len(d10brv2["recomputed_gates"]) == 19
      and all(d10brv2["recomputed_gates"].values()))
check("d10br_checkpoint_byte_replay_limit_preserved",
      d10brv2["replayability_boundary"]["checkpoint_summaries_persisted"] is True
      and d10brv2["replayability_boundary"]["checkpoint_underlying_arrays_persisted"] is False
      and d10brv2["replayability_boundary"]["checkpoint_sha_recomputed_from_underlying_arrays"] is False
      and d10brv2["replayability_boundary"]["second_endurance_replay_digest_available"] is False
      and "not the underlying array bytes" in d10br_phase["checkpoint_replayability_boundary"]
      and "cannot recompute the 909 hashes" in d10br_phase["checkpoint_replayability_boundary"])

# The second audit's final 49th check is a stat-only active-bank check.  The
# manifest-declared hash is frozen, but this combined verifier intentionally
# does not spend another 49.9 GB read to rescan the payload.
d10br_bank = Path(d10br_preflight["audit"]["required_bulk_files"][0]["path"])
check("d10br_active_bank_exists_with_exact_frozen_size_no_rescan",
      d10br_bank.is_file() and d10br_bank.stat().st_size == 49_925_652_480
      and d10brv2["notes"]["manifest_bank_hash_field_seen"]
          == "4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462"
      and d10brv2["notes"]["bulk_bank_rehashed_by_this_verifier"] is False
      and d10brv2["notes"]["bulk_bank_size_bytes"] == 49_925_652_480)
check("d10br_audit_count_correction_preserved",
      d10br_phase["independent_verification"]["audit_one"] == "19/19"
      and d10br_phase["independent_verification"]["audit_two"].startswith("49/49")
      and "49th check" in d10br_phase["independent_verification"]["audit_two"])

# Cleanup removed only reproducible bulk.  The unique evidence and active bank
# remain, and the registry exposes exact removed/retained byte counts.
cleanup = registry["disk_cleanup_manifest"]
check("disk_cleanup_manifest_and_active_bank_preserved",
      cleanup["status"] == "reproducible_bulk_removed_active_bank_retained"
      and cleanup["removed_bytes"] == 86_968_817_216
      and cleanup["retained_active_bank_bytes"] == 49_925_652_480
      and d10br_bank.is_file() and d10br_bank.stat().st_size == cleanup["retained_active_bank_bytes"])
check("d10_real_checkpoint_gate_remains_blocked",
      registry["routes"]["directpath"]["phases"]["D10_REAL_CHECKPOINT_INTEGRATION"]["status"]
      == "blocked_missing_real_payload_natural_routes_quality_and_raw_checkpoint_bytes")

check("industrial_claim_remains_closed", registry["current_synthesis"].get("industrial_breakthrough_claim") == "not_yet_supported")

failures = [item for item in checks if not item["pass"]]
result = {
    "kind": "latest_phase_combined_evidence_independent_cpu_verification",
    "cpu_only": True,
    "gpu_runtime_imported": False,
    "registry_path": str(REGISTRY.relative_to(ROOT)).replace("\\", "/"),
    "registry_sha256": sha256(REGISTRY),
    "checks_passed": len(checks) - len(failures),
    "checks_total": len(checks),
    "overall_pass": not failures,
    "failures": failures,
    "evidence_hashes": evidence_results,
    "claim_boundary": (
        "CPU-only provenance, arithmetic and status verification of frozen artifacts. "
        "No GPU context, model rerun, checkpoint download, quality evaluation or physical end-to-end run."
    ),
    "checks": checks,
}
with OUTPUT.open("w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
    handle.write("\n")

print(json.dumps({key: result[key] for key in ("overall_pass", "checks_passed", "checks_total", "failures")}, indent=2))
raise SystemExit(0 if result["overall_pass"] else 1)
