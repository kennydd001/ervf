from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from moe_lab.craft_moe.repro_audit import (  # noqa: E402
    AUDIT_SEED,
    DATASET_REVISION,
    MODEL_REVISION,
    AuditCollector,
    contains_value,
    disjoint,
    finite_tree,
    gap_closure,
    load_json,
    load_top_level_members,
    paired_gap_bootstrap_independent,
    paired_load_bootstrap_independent,
    ratio_reduction,
    sha256_file,
    upper_empirical_quantile,
    values_for_key,
)


REPORT_DIR = ROOT / "reports" / "craft_moe"
RUN_DIR = ROOT / "reports" / "runs" / "craft_moe"
DEFAULT_OUTPUT = REPORT_DIR / "repro_audit.json"


RESULT_MEMBERS: dict[str, list[str]] = {
    "H7_ROUTE_CORESET": [
        "schema_version", "kind", "status", "model", "dataset", "configuration",
        "controls", "gates", "results", "reproducibility", "verdict",
    ],
    "H1_CRCQ_SCREEN": [
        "schema_version", "kind", "status", "model", "dataset", "configuration",
        "controls", "gates", "reproducibility", "verdict",
    ],
    "H1_CRCQ_FULL": [
        "schema_version", "kind", "status", "model", "dataset", "configuration",
        "controls", "gates", "reproducibility", "verdict", "layer23_eligible",
    ],
    "H1_CRCQ_LAYER23": [
        "schema_version", "kind", "status", "model", "dataset", "configuration",
        "final", "gates", "reproducibility", "routing_control", "verdict",
        "candidate_validation_eligible",
    ],
    "H3_ATOMIC_LAYER26": [
        "schema_version", "kind", "status", "model", "dataset", "configuration",
        "controls", "gates", "reproducibility", "verdict",
    ],
    "H3_ATOMIC_LAYER23": [
        "schema_version", "kind", "status", "model", "dataset", "configuration",
        "decomposition_control", "gates", "reproducibility", "routing_control",
        "verdict",
    ],
    "H3_ATOMIC_SPREAD": [
        "schema_version", "kind", "status", "model", "dataset", "configuration",
        "gates", "reproducibility", "verdict", "simultaneous_full_depth_eligible",
    ],
    "H3_ATOMIC_FULL_DEPTH": [
        "schema_version", "kind", "status", "model", "dataset", "configuration",
        "exact_control_by_layer", "gates", "reproducibility", "support_artifact",
        "verdict", "candidate_validation_eligible",
    ],
    "H4_SKETCHGATE_REPLICATION": [
        "schema_version", "kind", "status", "model", "dataset", "configuration",
        "controls", "gates", "hardware_model", "metadata_accounting",
        "reproducibility", "results", "verdict",
    ],
    "H2_BLOCK_COALESCING": [
        "schema_version", "kind", "status", "model", "dataset", "configuration",
        "controls", "gates", "reproducibility", "verdict",
    ],
    "H6_QERC": [
        "schema_version", "kind", "status", "model", "dataset", "controls",
        "gates", "phase_a", "phase_b", "reproducibility", "verdict",
    ],
    "H8_CACHE_SPAN": [
        "schema_version", "kind", "status", "model", "dataset", "controls",
        "gates", "hardware_model", "heldout_test", "protocol", "reproducibility",
        "validation", "verdict", "capture_artifact", "screen_positive",
    ],
    "H10_REDUCTION_ORDER": [
        "schema_version", "kind", "status", "model", "dataset", "controls",
        "exact_quality", "gates", "protocol", "protected_fp32_order_invariance",
        "raw_sweep_artifact", "capture_artifact", "reproducibility",
        "validation_selection", "validation_selected_fp32_control", "verdict",
        "content_positive",
    ],
}


RESULT_PATHS = {
    "H7_ROUTE_CORESET": REPORT_DIR / "route_coreset_oracle.json",
    "H1_CRCQ_SCREEN": REPORT_DIR / "crcq_oracle.json",
    "H1_CRCQ_FULL": REPORT_DIR / "crcq_full_oracle.json",
    "H1_CRCQ_LAYER23": REPORT_DIR / "crcq_layer23_downstream.json",
    "H3_ATOMIC_LAYER26": REPORT_DIR / "atomic_oracle.json",
    "H3_ATOMIC_LAYER23": REPORT_DIR / "atomic_layer23_downstream.json",
    "H3_ATOMIC_SPREAD": REPORT_DIR / "atomic_spread_oracle.json",
    "H3_ATOMIC_FULL_DEPTH": REPORT_DIR / "atomic_full_depth_oracle.json",
    "H4_SKETCHGATE_REPLICATION": REPORT_DIR / "sketchgate_trace_anchored_replication.json",
    "H2_BLOCK_COALESCING": REPORT_DIR / "block_route_coalescing.json",
    "H6_QERC": REPORT_DIR / "qerc.json",
    "H8_CACHE_SPAN": REPORT_DIR / "cache_span.json",
    "H10_REDUCTION_ORDER": REPORT_DIR / "reduction_order.json",
}


PUBLISHED_HASHES = {
    "reports/craft_moe/crcq_oracle.json": "f3ef034a0f336d433f988f0b9b7feba38c18f5a6c76d2f84090a02e1da670e7a",
    "reports/craft_moe/atomic_oracle.json": "8af20192684b5427b293a858a90b711a9e1d364c85daa6395058bb1acd592bb9",
    "reports/craft_moe/atomic_layer23_downstream.json": "398a4bf126f14a4cf583324eac10ac6a55e1128ae512fa84569b4f57b6aee588",
    "reports/craft_moe/atomic_spread_oracle.json": "2e49426c58d0132a130e67ffcedb0e616673124198beea7036187a0fd51df66b",
    "reports/craft_moe/atomic_full_depth_oracle.json": "bc681cea5c03d2407686fe438ef63b8cff67950fa70296c5f779ba786d3bb60c",
    "reports/craft_moe/atomic_full_depth_supports.safetensors": "cf09d9096be20efedac1eca429e6d452c372211e8ff5c6b69ed72dff17288258",
    "reports/craft_moe/sketchgate_trace_anchored_replication.json": "cae54755ad40a4dd1e46a95075f0331a82b58ff45eedaa15e98de3b1459f6f90",
    "reports/craft_moe/sketchgate.json": "98fa42ed2987c31ed89a2c2d00f05aecbf1d8fcbac922ce16bb63613b4bbf0b9",
    "reports/craft_moe/block_route_coalescing.json": "63e80464823a7c696230e9e4d87c4d889a583901296fca50d005e70e9ba9a09d",
    "reports/craft_moe/block_route_coalescing_control_audit_v2.json": "79bddd85cafe9fdb420ac716f8934e05ccdeb156d1cf727cd4d24e9112543942",
    "reports/craft_moe/block_route_coalescing_control_audit.json": "11be8316cdb583ce73ef327325734e2886d0667f41d0c54c85ed5a2167899083",
    "reports/craft_moe/qerc.json": "681284583baf0b08d39dd5c153e184b008f514d16ecf86c543c520816db42cc3",
    "reports/runs/craft_moe/qerc_covariance_layer26.json": "e26d802a68b7f106dd5157d623eabbdcbd1eade26725408f2f8c6b86def90946",
    "reports/runs/craft_moe/qerc_layer26_components.safetensors": "4ecb801589b6221567edaaab390ea2373665708d4ad815f83e15a0feedfcc1f0",
    "reports/craft_moe/cache_span.json": "60b39e4aa5717221c561ad5cbb286b412c04c40d2ab34b5f6435e43a1d5b63bc",
    "reports/runs/craft_moe/cache_span_layer26_capture.safetensors": "260c241b4513a45e9c7165df996847a2b7ef4481f3d52c2420c3ba76eb771a93",
    "reports/craft_moe/cache_span_block_bootstrap_audit.json": "b051af8a2d383d7f56423f440bb30e3a867ebb5f01294d643a7b7709f117229a",
    "reports/craft_moe/reduction_order.json": "59497a28a0bacf791bb47cf1ef13f216caac6fb4bc6a1d17da874ad089f370ef",
    "reports/runs/craft_moe/reduction_order_capture.safetensors": "c2cfcedf1e147ee7fec5a73adece667fb45bea9a4ba88a03582284890e0c5079",
    "reports/runs/craft_moe/reduction_order_raw.safetensors": "b47ac1cc872e43f8c43bb0a2d8d162faf48cd45ec201a15fd0d278421f97e7c9",
    "reports/craft_moe/reduction_order_bootstrap_audit.json": "7735559957f79ea24a99b0959f33f4922c1c9e9bfbdc822ff746c90b6c73cb39",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently verify all preregistered CRAFT-MoE gates."
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="compare current artifacts with the append-only output manifest",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="run checks without writing output"
    )
    parser.add_argument(
        "--skip-artifact-hashes",
        action="store_true",
        help="development-only shortcut; recorded as a warning",
    )
    return parser.parse_args()


def load_reports() -> dict[str, dict[str, Any]]:
    return {
        experiment: load_top_level_members(RESULT_PATHS[experiment], members)
        for experiment, members in RESULT_MEMBERS.items()
    }


def check_reported_bool(
    audit: AuditCollector,
    experiment: str,
    check_id: str,
    reported: Any,
    recalculated: bool,
    evidence: str,
) -> None:
    audit.equal(
        check_id,
        experiment,
        "gate_recalculation",
        reported,
        bool(recalculated),
        evidence,
    )


def check_reported_float(
    audit: AuditCollector,
    experiment: str,
    check_id: str,
    reported: float,
    recalculated: float,
    evidence: str,
    *,
    tolerance: float = 1e-12,
) -> None:
    audit.close(
        check_id,
        experiment,
        "metric_recalculation",
        float(reported),
        float(recalculated),
        evidence,
        abs_tol=tolerance,
        rel_tol=tolerance,
    )


def audit_common_contract(
    audit: AuditCollector, reports: Mapping[str, Mapping[str, Any]]
) -> None:
    for experiment, report in reports.items():
        evidence = str(RESULT_PATHS[experiment].relative_to(ROOT))
        audit.equal(
            f"{experiment}.schema_version",
            experiment,
            "schema",
            report["schema_version"],
            1,
            evidence,
        )
        audit.equal(
            f"{experiment}.status",
            experiment,
            "schema",
            report["status"],
            "complete",
            evidence,
        )
        audit.add(
            f"{experiment}.model_revision",
            experiment,
            "provenance",
            contains_value(report["model"], MODEL_REVISION),
            report["model"].get("revision"),
            MODEL_REVISION,
            evidence,
        )
        audit.add(
            f"{experiment}.dataset_revision",
            experiment,
            "provenance",
            contains_value(report["dataset"], DATASET_REVISION),
            "revision present" if contains_value(report["dataset"], DATASET_REVISION) else report["dataset"],
            DATASET_REVISION,
            evidence,
        )
        audit.add(
            f"{experiment}.finite_gate_tree",
            experiment,
            "finite",
            finite_tree(report["gates"]),
            finite_tree(report["gates"]),
            True,
            evidence,
        )


def audit_h7(audit: AuditCollector, report: Mapping[str, Any]) -> None:
    experiment = "H7_ROUTE_CORESET"
    for split in ("validation", "test"):
        raw_by_k = {
            k: report["results"][split]["methods"]["nnls"][str(k)]["raw"][
                "teacher_to_candidate_kl"
            ]
            for k in range(1, 6)
        }
        token_count = len(raw_by_k[1])
        minimum_k = [
            next(
                (k for k in range(1, 6) if raw_by_k[k][token] <= 0.001),
                6,
            )
            for token in range(token_count)
        ]
        gate = report["gates"][split]
        audit.equal(
            f"h7.{split}.minimum_k_raw",
            experiment,
            "raw_recalculation",
            gate["minimum_k_at_kl_le_0_001"],
            minimum_k,
            f"results.{split}.methods.nnls[*].raw.teacher_to_candidate_kl",
        )
        distribution = {str(k): minimum_k.count(k) for k in range(1, 7)}
        audit.equal(
            f"h7.{split}.distribution",
            experiment,
            "raw_recalculation",
            gate["minimum_k_distribution"],
            distribution,
            "per-token reconstructed minimum k",
        )
        median = float(upper_empirical_quantile(minimum_k, 0.5))
        p95 = float(upper_empirical_quantile(minimum_k, 0.95))
        check_reported_float(
            audit, experiment, f"h7.{split}.median", gate["minimum_k_median_higher_empirical"], median,
            "higher empirical median of reconstructed minimum k",
        )
        check_reported_float(
            audit, experiment, f"h7.{split}.p95", gate["minimum_k_p95_higher_empirical"], p95,
            "higher empirical p95 of reconstructed minimum k",
        )
        primary = median <= 3 or p95 <= 4
        falsification_fraction = sum(value > 0.003 for value in raw_by_k[5]) / token_count
        check_reported_float(
            audit, experiment, f"h7.{split}.falsification_fraction",
            gate["falsification_fraction_k5_kl_gt_0_003"], falsification_fraction,
            "raw NNLS k=5 KL series",
        )
        check_reported_bool(
            audit, experiment, f"h7.{split}.primary_gate",
            gate["primary_criterion_passed"], primary,
            "median<=3 or p95<=4 at KL<=0.001",
        )
        check_reported_bool(
            audit, experiment, f"h7.{split}.falsification",
            gate["falsification_triggered"], falsification_fraction > 0.25,
            "fraction of NNLS k=5 tokens above KL 0.003",
        )
        expected_split_verdict = (
            "falsified" if falsification_fraction > 0.25 else
            "oracle_positive" if primary else "inconclusive_negative"
        )
        audit.equal(
            f"h7.{split}.verdict",
            experiment,
            "verdict",
            gate["verdict"],
            expected_split_verdict,
            "preregistered H7 decision tree",
        )
    audit.equal(
        "h7.overall_verdict",
        experiment,
        "verdict",
        report["verdict"],
        report["gates"]["validation"]["verdict"],
        "validation is the preregistered primary split",
    )
    audit.equal(
        "h7.original_control",
        experiment,
        "exact_control",
        report["controls"]["exact_top6"],
        "required and passed on every evaluated split",
        "controls.exact_top6",
    )


def audit_h1_screen(audit: AuditCollector, report: Mapping[str, Any]) -> None:
    experiment = "H1_CRCQ_SCREEN"
    for split in ("validation", "test"):
        gate = report["gates"][split]
        fraction = gate["joint_minimum_upgrade_fraction"]
        closure = gate["all_q3_alternative_mean_gap_closure"]
        expected = {
            "joint_upgrade_fraction_le_0_15": fraction <= 0.15,
            "average_active_bits_le_3_15": 3.0 + fraction <= 3.15,
            "all_q3_gap_closure_ge_0_50": closure >= 0.50,
        }
        audit.equal(
            f"h1_screen.{split}.criteria",
            experiment,
            "gate_recalculation",
            gate["strong_criteria"],
            expected,
            "numeric upgrade fraction, active bits and Q3-gap closure",
        )
        check_reported_float(
            audit, experiment, f"h1_screen.{split}.active_bits",
            gate["joint_average_active_bits"], 3.0 + fraction,
            "Q3 base plus Q4-upgrade fraction",
        )
        check_reported_bool(
            audit, experiment, f"h1_screen.{split}.any_criterion",
            gate["any_strong_criterion"], any(expected.values()),
            "any preregistered strong criterion",
        )
    shared = any(
        report["gates"]["validation"]["strong_criteria"][name]
        and report["gates"]["test"]["strong_criteria"][name]
        for name in report["gates"]["validation"]["strong_criteria"]
    )
    both_negative = all(
        row["route_axis_negative_gap_closure_lt_0_10"]
        and row["joint_axis_negative_upgrade_fraction_gt_0_25"]
        for row in report["gates"].values()
    )
    expected_verdict = "strong_positive" if shared else "screen_negative" if both_negative else "inconclusive"
    audit.equal(
        "h1_screen.verdict", experiment, "verdict", report["verdict"], expected_verdict,
        "shared validation/test criterion decision tree",
    )
    audit.add(
        "h1_screen.original_control", experiment, "exact_control",
        "passed" in report["controls"]["natural_bf16_exact"],
        report["controls"]["natural_bf16_exact"], "passed",
        "controls.natural_bf16_exact",
    )
    audit.equal(
        "h1_screen.router_weights_not_renormalized", experiment, "routing",
        report["configuration"]["router_weights_renormalized"], False,
        "configuration.router_weights_renormalized",
    )


def audit_h1_full(
    audit: AuditCollector,
    report: Mapping[str, Any],
    screen: Mapping[str, Any],
) -> None:
    experiment = "H1_CRCQ_FULL"
    for split in ("validation", "test"):
        gate = report["gates"][split]
        fraction = gate["minimum_upgrade_fraction"]
        top32 = screen["gates"][split]["joint_minimum_upgrade_fraction"]
        expected = fraction <= 0.15 and fraction <= top32 and gate["direct_dp_abs_error"] <= 1e-6
        check_reported_bool(
            audit, experiment, f"h1_full.{split}.fraction_gate",
            gate["full_upgrade_fraction_le_0_15"], fraction <= 0.15,
            "minimum_upgrade_fraction <= 0.15",
        )
        check_reported_bool(
            audit, experiment, f"h1_full.{split}.superset_gate",
            gate["no_worse_than_top32"], fraction <= top32,
            "full search cannot be worse than frozen top-32 value",
        )
        check_reported_bool(
            audit, experiment, f"h1_full.{split}.direct_dp_gate",
            gate["direct_dp_abs_error_le_1e_6"], gate["direct_dp_abs_error"] <= 1e-6,
            "direct schedule versus dynamic-program KL",
        )
        check_reported_bool(
            audit, experiment, f"h1_full.{split}.passed", gate["passed"], expected,
            "conjunction of all full-oracle gates",
        )
        check_reported_float(
            audit, experiment, f"h1_full.{split}.top32_reproduction",
            gate["top32_upgrade_fraction"], top32,
            "frozen H1 screen result",
        )
        check_reported_float(
            audit, experiment, f"h1_full.{split}.improvement",
            gate["absolute_fraction_improvement_vs_top32"], top32 - fraction,
            "top32 fraction minus full-search fraction",
        )
        check_reported_float(
            audit, experiment, f"h1_full.{split}.active_bits",
            gate["average_active_bits"], 3.0 + fraction,
            "Q3 base plus Q4-upgrade fraction",
        )
    positive = all(report["gates"][split]["passed"] for split in ("validation", "test"))
    expected_verdict = "full_oracle_positive" if positive else "full_oracle_gate_failed"
    audit.equal(
        "h1_full.verdict", experiment, "verdict", report["verdict"], expected_verdict,
        "both fixed splits must pass",
    )
    audit.equal(
        "h1_full.layer23_eligibility", experiment, "stop_go",
        report["layer23_eligible"], positive,
        "layer 23 opens only after full oracle positive",
    )


def exact_quality_control(row: Mapping[str, Any]) -> bool:
    raw = row["raw"]
    return (
        max(raw["teacher_to_candidate_kl"]) == 0.0
        and all(raw["top1_agreement"])
        and row["aggregate"]["cross_entropy_delta"] == 0.0
    )


def audit_h1_downstream(audit: AuditCollector, report: Mapping[str, Any]) -> None:
    experiment = "H1_CRCQ_LAYER23"
    for split in ("validation", "test"):
        rows = report["final"][split]
        gate = report["gates"][split]
        control_exact = exact_quality_control(rows["natural_bf16_patch_control"])
        natural_q4 = rows["natural_all_q4"]["aggregate"]["teacher_to_candidate_kl"]
        joint_min = rows["joint_minimum_local_q4_quality"]["aggregate"]
        natural15 = rows["natural_budget_0_15"]["aggregate"]["teacher_to_candidate_kl"]
        joint15 = rows["joint_budget_0_15"]["aggregate"]["teacher_to_candidate_kl"]
        fraction = gate["local_joint_upgrade_fraction"]
        criteria = {
            "local_joint_upgrade_fraction_le_0_15": fraction <= 0.15,
            "joint_min_final_kl_le_1_10x_natural_q4": joint_min["teacher_to_candidate_kl"] <= 1.10 * natural_q4,
            "joint_min_abs_relative_ce_lt_0_02": abs(joint_min["relative_cross_entropy_delta"]) < 0.02,
            "joint_15pct_final_kl_le_natural_15pct": joint15 <= natural15,
            "bf16_control_exact": control_exact,
        }
        audit.equal(
            f"h1_downstream.{split}.criteria", experiment, "gate_recalculation",
            gate["criteria"], criteria, f"final.{split} policy metrics",
        )
        check_reported_bool(
            audit, experiment, f"h1_downstream.{split}.passed",
            gate["passed"], all(criteria.values()), "conjunction of downstream criteria",
        )
        hard = (
            joint_min["teacher_to_candidate_kl"] > 1.25 * natural_q4
            or abs(joint_min["relative_cross_entropy_delta"]) >= 0.02
            or fraction > 0.25
        )
        check_reported_bool(
            audit, experiment, f"h1_downstream.{split}.hard_falsification",
            gate["hard_falsification"], hard, "preregistered downstream hard-stop disjunction",
        )
        for key, expected in (
            ("natural_q4_final_kl", natural_q4),
            ("joint_min_final_kl", joint_min["teacher_to_candidate_kl"]),
            ("natural_15pct_final_kl", natural15),
            ("joint_15pct_final_kl", joint15),
        ):
            check_reported_float(
                audit, experiment, f"h1_downstream.{split}.{key}", gate[key], expected,
                f"final.{split} aggregate metric",
            )
    positive = all(report["gates"][split]["passed"] for split in ("validation", "test"))
    hard = any(report["gates"][split]["hard_falsification"] for split in ("validation", "test"))
    expected = "downstream_positive" if positive else "downstream_falsified" if hard else "inconclusive"
    audit.equal("h1_downstream.verdict", experiment, "verdict", report["verdict"], expected, "downstream decision tree")
    audit.equal(
        "h1_downstream.no_candidate_validation", experiment, "stop_go",
        report["candidate_validation_eligible"], False,
        "hard-falsified downstream gate",
    )
    audit.add(
        "h1_downstream.route_control", experiment, "exact_control",
        report["routing_control"]["top6_id_sets_exact"]
        and report["routing_control"]["top6_router_weight_max_abs"] == 0.0,
        report["routing_control"], "exact IDs and zero router-weight error",
        "routing_control",
    )


def audit_atomic(audit: AuditCollector, reports: Mapping[str, Mapping[str, Any]]) -> None:
    layer26 = reports["H3_ATOMIC_LAYER26"]
    gates = layer26["gates"]
    for split in ("validation", "test"):
        primary = gates["primary_by_split"][split]
        moonshot = gates["moonshot_by_split"][split]
        tile = gates["tile64_by_split"][split]
        check_reported_bool(audit, "H3_ATOMIC_LAYER26", f"atomic26.{split}.primary", primary["passes_lt_0_02"], primary["relative_cross_entropy_delta"] < 0.02, "25% relative CE")
        check_reported_bool(audit, "H3_ATOMIC_LAYER26", f"atomic26.{split}.moonshot", moonshot["passes_lt_0_03"], moonshot["relative_cross_entropy_delta"] < 0.03, "10% relative CE")
        check_reported_float(audit, "H3_ATOMIC_LAYER26", f"atomic26.{split}.tile_ratio", tile["kl_ratio"], tile["tile64_mean_kl"] / tile["global_neuron_mean_kl"], "tile64 KL divided by global-neuron KL")
        check_reported_bool(audit, "H3_ATOMIC_LAYER26", f"atomic26.{split}.tile_gate", tile["passes_le_1_20x"], tile["tile64_mean_kl"] <= 1.20 * tile["global_neuron_mean_kl"], "tile64 <=1.20x global KL")
    primary_all = all(gates["primary_by_split"][split]["passes_lt_0_02"] for split in ("validation", "test"))
    audit.equal("atomic26.primary_all", "H3_ATOMIC_LAYER26", "gate_recalculation", gates["primary_global_25pct_relative_ce_lt_2pct_both_splits"], primary_all, "both splits")
    expected26 = "oracle_positive_opens_depth_and_domain_expansion" if primary_all else "oracle_negative_hard_stop" if gates["hard_stop"]["triggered"] else "inconclusive_negative_no_expansion"
    audit.equal("atomic26.verdict", "H3_ATOMIC_LAYER26", "verdict", layer26["verdict"], expected26, "atomic layer-26 decision tree")
    audit.add("atomic26.original_control", "H3_ATOMIC_LAYER26", "exact_control", "passed" in layer26["controls"]["official_teacher_exact_delta"], layer26["controls"]["official_teacher_exact_delta"], "passed", "controls.official_teacher_exact_delta")

    layer23 = reports["H3_ATOMIC_LAYER23"]
    gates23 = layer23["gates"]
    for split, row in gates23["splits"].items():
        metric = row["primary_25pct"]
        criteria = {
            "relative_ce_increase_lt_0_02": metric["relative_cross_entropy_delta"] < 0.02,
            "mean_kl_le_0_01": metric["teacher_to_candidate_kl"] <= 0.01,
            "top1_agreement_ge_0_95": metric["top1_agreement"] >= 0.95,
            "exact_control": row["criteria"]["exact_control"],
        }
        audit.equal(f"atomic23.{split}.criteria", "H3_ATOMIC_LAYER23", "gate_recalculation", row["criteria"], criteria, "25% downstream metrics")
        check_reported_bool(audit, "H3_ATOMIC_LAYER23", f"atomic23.{split}.passed", row["passed"], all(criteria.values()), "conjunction of downstream criteria")
        moonshot = row["moonshot_10pct"]
        check_reported_bool(audit, "H3_ATOMIC_LAYER23", f"atomic23.{split}.moonshot", row["moonshot_10pct_relative_ce_lt_0_03"], moonshot["relative_cross_entropy_delta"] < 0.03, "10% relative CE")
    primary23 = all(row["passed"] for row in gates23["splits"].values())
    expected23 = "downstream_positive_opens_spread_layers" if primary23 else "downstream_falsified" if any(row["hard_falsification"] for row in gates23["splits"].values()) else "inconclusive"
    audit.equal("atomic23.verdict", "H3_ATOMIC_LAYER23", "verdict", layer23["verdict"], expected23, "atomic layer-23 decision tree")
    audit.add("atomic23.route_control", "H3_ATOMIC_LAYER23", "exact_control", all((layer23["routing_control"]["set_ids_exact"], layer23["routing_control"]["slot_order_ids_exact"], layer23["routing_control"]["router_weight_max_absolute_error"] == 0.0)), layer23["routing_control"], "exact IDs/order/weights", "routing_control")

    spread = reports["H3_ATOMIC_SPREAD"]
    spread_gates = spread["gates"]
    audit.equal("atomic_spread.cell_count", "H3_ATOMIC_SPREAD", "coverage", len(spread_gates["cells"]), 12, "3 layers x 4 domains")
    for key, row in spread_gates["cells"].items():
        metric = row["primary_25pct"]
        criteria = {
            "relative_ce_increase_lt_0_02": metric["relative_cross_entropy_delta"] < 0.02,
            "mean_kl_le_0_01": metric["teacher_to_candidate_kl"] <= 0.01,
            "top1_agreement_ge_0_95": metric["top1_agreement"] >= 0.95,
            "exact_control": row["criteria"]["exact_control"],
        }
        audit.equal(f"atomic_spread.{key}.criteria", "H3_ATOMIC_SPREAD", "gate_recalculation", row["criteria"], criteria, "fixed 25% cell metrics")
        check_reported_bool(audit, "H3_ATOMIC_SPREAD", f"atomic_spread.{key}.passed", row["passed"], all(criteria.values()), "cell criterion conjunction")
    spread_positive = all(row["passed"] for row in spread_gates["cells"].values())
    expected_spread = "spread_positive_opens_simultaneous_full_depth" if spread_positive else "spread_falsified" if any(row["hard_falsification"] for row in spread_gates["cells"].values()) else "inconclusive"
    audit.equal("atomic_spread.verdict", "H3_ATOMIC_SPREAD", "verdict", spread["verdict"], expected_spread, "12-cell decision tree")

    full = reports["H3_ATOMIC_FULL_DEPTH"]
    full_gates = full["gates"]
    for domain, row in full_gates["domains"].items():
        metric = row["primary_25pct"]
        primary_criteria = {
            "relative_ce_increase_lt_0_02": metric["relative_cross_entropy_delta"] < 0.02,
            "mean_kl_le_0_03": metric["teacher_to_candidate_kl"] <= 0.03,
            "top1_agreement_ge_0_90": metric["top1_agreement"] >= 0.90,
            "exact_control": row["primary_criteria"]["exact_control"],
        }
        audit.equal(f"atomic_full.{domain}.criteria", "H3_ATOMIC_FULL_DEPTH", "gate_recalculation", row["primary_criteria"], primary_criteria, "simultaneous 25% domain metrics")
        check_reported_bool(audit, "H3_ATOMIC_FULL_DEPTH", f"atomic_full.{domain}.passed", row["primary_passed"], all(primary_criteria.values()), "domain criterion conjunction")
        moon = row["moonshot_10pct"]
        moon_criteria = {
            "relative_ce_increase_lt_0_03": moon["relative_cross_entropy_delta"] < 0.03,
            "mean_kl_le_0_05_safety": moon["teacher_to_candidate_kl"] <= 0.05,
            "top1_agreement_ge_0_85_safety": moon["top1_agreement"] >= 0.85,
        }
        audit.equal(f"atomic_full.{domain}.moonshot_criteria", "H3_ATOMIC_FULL_DEPTH", "gate_recalculation", row["moonshot_criteria"], moon_criteria, "simultaneous 10% domain metrics")
    full_primary = all(row["primary_passed"] for row in full_gates["domains"].values())
    full_hard = any(row["hard_falsification"] for row in full_gates["domains"].values())
    expected_full = "full_depth_positive_opens_candidate_validation" if full_primary else "full_depth_falsified" if full_hard else "inconclusive"
    audit.equal("atomic_full.verdict", "H3_ATOMIC_FULL_DEPTH", "verdict", full["verdict"], expected_full, "four-domain simultaneous decision tree")
    audit.equal("atomic_full.layer_controls", "H3_ATOMIC_FULL_DEPTH", "exact_control", all(full["exact_control_by_layer"].values()), True, "all 26 intervened MoE layers")
    audit.equal("atomic_full.no_candidate_validation", "H3_ATOMIC_FULL_DEPTH", "stop_go", full["candidate_validation_eligible"], False, "failed 25% full-depth gate")


def audit_h4(audit: AuditCollector, report: Mapping[str, Any]) -> None:
    experiment = "H4_SKETCHGATE_REPLICATION"
    gates = report["gates"]
    down_gate = all(value >= 0.70 for value in gates["down_attribution_fraction_by_split"].values())
    check_reported_bool(audit, experiment, "h4.down_attribution", gates["down_attribution_ge_0_70_both_splits"], down_gate, "down attribution >=70% on both splits")
    primary_gate = True
    stability_gate = True
    for split in ("validation", "test"):
        primary = gates["primary_by_split"][split]
        primary_split = primary["oracle_recovery"] >= 0.80 and primary["high_damage_false_negative_rate"] <= 0.01
        primary_gate &= primary_split
        stability = gates["stability_by_split"][split]
        stability_split = stability["minimum_recovery"] >= 0.80 and stability["maximum_false_negative_rate"] <= 0.01
        stability_gate &= stability_split
        check_reported_bool(audit, experiment, f"h4.{split}.stability", stability["all_five_seeds_pass"], stability_split, "minimum recovery and maximum FN over five seeds")
    check_reported_bool(audit, experiment, "h4.primary_seed_gate", gates["primary_seed_recovery_ge_0_80_and_fn_le_0_01_both_splits"], primary_gate, "primary fixed seed on both splits")
    check_reported_bool(audit, experiment, "h4.five_seed_gate", gates["all_five_seeds_stable_both_splits"], stability_gate, "all five seeds on both splits")
    expected_seeds = list(range(AUDIT_SEED, AUDIT_SEED + 5))
    audit.equal("h4.seed_bank", experiment, "randomness", report["configuration"]["probe_seeds"], expected_seeds, "preregistered five-seed bank")
    for split in ("validation", "test"):
        primary = gates["primary_by_split"][split]
        audit.equal(f"h4.{split}.primary_seed", experiment, "randomness", primary["seed"], AUDIT_SEED, "test never selects a seed")
    audit.equal(
        "h4.fixed_configuration_on_test", experiment, "split_discipline",
        (gates["primary_by_split"]["validation"]["distribution"], gates["primary_by_split"]["validation"]["rank"]),
        (gates["primary_by_split"]["test"]["distribution"], gates["primary_by_split"]["test"]["rank"]),
        "validation-selected distribution/rank applied unchanged to test",
    )
    positive = all((down_gate, primary_gate, stability_gate, gates["metadata_lt_0_1_effective_bit"], gates["hardware_model_compute_lt_0_10_avoided_transfer"], gates["exact_controls_pass"]))
    gate_up_dominates = bool(report["results"]["validation"]["gate_up_dominates_down_only"])
    expected = "layer26_positive_opens_spread_preregistration" if positive else "falsified_gate_up_dominates_and_down_sketch_misses" if gate_up_dominates and (not primary_gate or not stability_gate) else "falsified_layer26_gate_no_spread"
    audit.equal("h4.verdict", experiment, "verdict", report["verdict"], expected, "preregistered H4 decision tree")
    controls = report["controls"]
    controls_expected = controls["official_teacher_exact_delta_bit_exact"] and controls["route_recomputation"]["slot_order_ids_exact"] and report["configuration"]["trace_anchor"]["stored_sources_used_bit_exact"]
    check_reported_bool(audit, experiment, "h4.exact_controls", gates["exact_controls_pass"], controls_expected, "replication trace anchor and official control")

    initial = load_top_level_members(
        REPORT_DIR / "sketchgate.json",
        ["status", "controls", "gates", "verdict"],
    )
    audit.equal("h4.initial_retained", experiment, "failed_artifact_retention", initial["status"], "complete", "sketchgate.json")
    audit.equal("h4.initial_control_failure_retained", experiment, "failed_artifact_retention", initial["gates"]["exact_controls_pass"], False, "original control-failing run remains unchanged")


def audit_h2(audit: AuditCollector, report: Mapping[str, Any]) -> None:
    experiment = "H2_BLOCK_COALESCING"
    gates = report["gates"]
    for split, row in gates["primary_by_split"].items():
        natural = ratio_reduction(row["exact_total_union"], row["natural_total_union"])
        mass = ratio_reduction(row["exact_total_union"], row["mass_budget_total_union"])
        check_reported_float(audit, experiment, f"h2.{split}.natural_reduction", row["union_reduction_vs_natural"], natural, "integer union totals")
        check_reported_float(audit, experiment, f"h2.{split}.mass_reduction", row["additional_union_reduction_vs_mass_budget"], mass, "integer union totals")
    natural_gate = all(row["union_reduction_vs_natural"] >= 0.40 for row in gates["primary_by_split"].values())
    mass_gate = all(row["additional_union_reduction_vs_mass_budget"] >= 0.25 for row in gates["primary_by_split"].values())
    kl_gate = all(row["mean_local_kl"] <= 0.001 for row in gates["primary_by_split"].values())
    check_reported_bool(audit, experiment, "h2.natural_gate", gates["union_reduction_ge_0_40_both_splits"], natural_gate, "both primary splits")
    check_reported_bool(audit, experiment, "h2.mass_gate", gates["additional_reduction_vs_mass_budget_ge_0_25_both_splits"], mass_gate, "both primary splits")
    check_reported_bool(audit, experiment, "h2.kl_gate", gates["mean_local_kl_le_0_001_both_splits"], kl_gate, "both primary splits")
    hard_union = any(row["union_reduction_vs_natural"] < 0.25 for row in gates["primary_by_split"].values())
    check_reported_bool(audit, experiment, "h2.hard_union", gates["hard_falsification"]["any_split_union_reduction_lt_0_25"], hard_union, "hard union ceiling")

    audit_v1 = load_json(REPORT_DIR / "block_route_coalescing_control_audit.json")
    audit_v2 = load_json(REPORT_DIR / "block_route_coalescing_control_audit_v2.json")
    audit.equal("h2.audit_v1_retained", experiment, "failed_artifact_retention", audit_v1["all_exact_records_pass"], False, "first erroneous audit retained")
    audit.equal("h2.audit_v2_exact", experiment, "exact_control", audit_v2["all_exact_records_pass"], True, "corrected empty/cold-cache objective audit")
    audit.equal("h2.audit_v2_failures", experiment, "exact_control", audit_v2["failed_records"], [], "all 1,280 exact records")
    audit.add("h2.audit_v2_tolerances", experiment, "exact_control", audit_v2["maximum_absolute_mip_gap"] <= 1e-12 and audit_v2["maximum_objective_absolute_error"] <= 1e-6, {"mip_gap": audit_v2["maximum_absolute_mip_gap"], "objective_error": audit_v2["maximum_objective_absolute_error"]}, {"mip_gap_le": 1e-12, "objective_error_le": 1e-6}, "preregistered numerical tolerances")
    expected = "oracle_negative_hard_falsification" if hard_union else "layer26_positive_opens_layer23_preregistration" if natural_gate and mass_gate and kl_gate else "inconclusive_negative_no_downstream"
    audit.equal("h2.corrected_verdict", experiment, "verdict", audit_v2["recalculated_verdict"], expected, "v2 controls plus numeric gates")
    audit.equal("h2.main_verdict_unchanged", experiment, "verdict", report["verdict"], expected, "content verdict unaffected by control-adjudicator bug")


def audit_h6(audit: AuditCollector, report: Mapping[str, Any]) -> None:
    experiment = "H6_QERC"
    phase = report["phase_a"]
    for split in ("validation", "test"):
        energy = phase["energy_sums"][split]
        cancellation = (energy["diagonal"] - energy["aggregate"]) / energy["diagonal"]
        check_reported_float(audit, experiment, f"h6.{split}.cancellation", phase["global_cancellation_fraction"][split], cancellation, "(diagonal-aggregate)/diagonal")
        check_reported_bool(audit, experiment, f"h6.{split}.near_zero", phase["near_zero_by_split"][split], abs(cancellation) < phase["absolute_near_zero_threshold"], "absolute cancellation below 2%")
        check_reported_float(audit, experiment, f"h6.{split}.cross_sum", energy["cross"], energy["aggregate"] - energy["diagonal"], "aggregate minus diagonal energy")
    hard = all(phase["near_zero_by_split"].values())
    check_reported_bool(audit, experiment, "h6.hard_stop", report["gates"]["cross_terms_near_zero_both_splits_hard_falsification"], hard, "near-zero on both fixed splits")
    audit.equal("h6.phase_b_closed", experiment, "stop_go", report["phase_b"]["status"], "not_opened_preregistered_phase_a_hard_stop", "phase A hard stop")
    same = report["gates"]["same_byte_layout"]
    audit.add("h6.same_bytes", experiment, "accounting", same["additional_bytes"] == 0 and same["additional_values"] == 0 and not same["integer_codes_changed"] and not same["new_kernel_operand"] and not same["tensor_shape_changed"], same, "zero extra bytes/values and unchanged codes/layout", "same_byte_layout")
    controls = report["controls"]
    audit.add("h6.exact_controls", experiment, "exact_control", controls["official_teacher_delta_bit_exact"] and controls["route_recomputation"]["slot_order_ids_exact"] and controls["route_recomputation"]["router_weight_maximum_absolute_error"] == 0.0, controls, "exact teacher/routes/weights", "controls")
    expected = "falsified_phase_a_cross_terms_near_zero" if hard and report["gates"]["exact_controls_pass"] else "invalid_exact_control_failure"
    audit.equal("h6.verdict", experiment, "verdict", report["verdict"], expected, "phase-A decision tree")


def select_h8_validation(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [
        row for row in rows
        if row["avoided_misses"] > 0
        and row["quality"]["aggregate"]["teacher_to_candidate_kl"] <= 0.001
        and row["extra_computations_per_avoided_load"] is not None
        and row["extra_computations_per_avoided_load"] <= 2
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -float(row["miss_reduction_fraction"]),
            float(row["extra_computations_per_avoided_load"]),
            float(row["quality"]["aggregate"]["teacher_to_candidate_kl"]),
            int(row["configuration"]["tie_order"]),
        ),
    )


def audit_h8(audit: AuditCollector, report: Mapping[str, Any]) -> None:
    experiment = "H8_CACHE_SPAN"
    winner = select_h8_validation(report["validation"]["all_span_configurations"])
    selected = report["validation"]["selected"]
    audit.equal("h8.validation_selection", experiment, "selection", selected["configuration"], winner["configuration"] if winner else None, "independent fixed tie-order selection over all 12 configurations")
    test = report["heldout_test"]["selected_configuration"]
    audit.equal("h8.fixed_test_configuration", experiment, "split_discipline", test["configuration"], selected["configuration"], "validation winner applied unchanged to held-out test")
    audit.equal("h8.no_test_selection", experiment, "split_discipline", report["protocol"]["test_selection"], "none; fixed validation winner only", "protocol.test_selection")
    for split, primary, zero in (
        ("validation", selected, report["validation"]["zero_fill"]),
        ("test", test, report["heldout_test"]["zero_fill"]),
    ):
        fraction = primary["avoided_misses"] / primary["baseline_misses"]
        zero_fraction = zero["avoided_misses"] / zero["baseline_misses"]
        check_reported_float(audit, experiment, f"h8.{split}.miss_fraction", primary["miss_reduction_fraction"], fraction, "integer avoided/baseline miss totals")
        check_reported_float(audit, experiment, f"h8.{split}.zero_fraction", zero["miss_reduction_fraction"], zero_fraction, "zero-fill integer totals")
        check_reported_bool(audit, experiment, f"h8.{split}.miss_gate", report["gates"]["miss_reduction_ge_0_50"][split], fraction >= 0.50, "miss reduction >=50%")
        kl = primary["quality"]["aggregate"]["teacher_to_candidate_kl"]
        check_reported_bool(audit, experiment, f"h8.{split}.kl_gate", report["gates"]["mean_kl_le_0_001"][split], kl <= 0.001, "mean full-vocabulary KL <=0.001")
        extra = primary["extra_computations_per_avoided_load"]
        check_reported_bool(audit, experiment, f"h8.{split}.compute_gate", report["gates"]["extra_compute_per_avoided_le_2"][split], extra <= 2, "extra resident forwards per avoided load")
        check_reported_bool(audit, experiment, f"h8.{split}.uplift_gate", report["gates"]["span_uplift_over_zero_fill_ge_0_10"][split], fraction - zero_fraction >= 0.10, "span uplift over zero-fill")
        audit.equal(f"h8.{split}.block_boundaries", experiment, "bootstrap_unit", [row["block_start"] for row in primary["per_block"]], [0, 128], "two non-overlapping sequence blocks")
    controls = report["controls"]
    controls_pass = all((controls["all_adjudicated_candidates_finite"], controls["all_outputs_finite"], controls["dense_router_weight_maximum_absolute_error"] == 0.0, controls["official_router"]["slot_order_ids_exact"], controls["official_router"]["router_weight_maximum_absolute_error"] == 0.0, controls["original_teacher_delta_bit_exact"], controls["post_capture_original_teacher_delta_bit_exact"], controls["sorted_top6_set_exact"], all(row["expert_loads_exact"] and row["route_ids_exact"] for row in controls["mass_budget_independent_audit"].values())))
    check_reported_bool(audit, experiment, "h8.exact_controls", report["gates"]["exact_controls_pass"], controls_pass, "teacher/route/cache/finite controls")
    hardware = report["hardware_model"]
    transfer = float(np.median(hardware["transfer_milliseconds_raw"]))
    combine = float(np.median(hardware["span_combine_milliseconds_raw"]))
    expert = float(np.median(hardware["resident_expert_forward_milliseconds_raw"]))
    projected = hardware["extra_computations_per_avoided_load"] * expert + combine
    check_reported_float(audit, experiment, "h8.hardware.transfer_median", hardware["transfer_milliseconds_median"], transfer, "median of seven repetitions")
    check_reported_float(audit, experiment, "h8.hardware.projected", hardware["projected_compute_milliseconds_per_avoided_load"], projected, "extra forwards plus span combine")
    check_reported_float(audit, experiment, "h8.hardware.ratio", hardware["compute_over_avoided_transfer_time"], projected / transfer, "projected compute divided by measured transfer")
    check_reported_bool(audit, experiment, "h8.hardware.gate", report["gates"]["projected_compute_lt_avoided_transfer"], projected < transfer, "microbenchmark model only")
    hard = test["miss_reduction_fraction"] < 0.40 or test["quality"]["aggregate"]["teacher_to_candidate_kl"] > 0.001 or test["miss_reduction_fraction"] - report["heldout_test"]["zero_fill"]["miss_reduction_fraction"] <= 0.0
    positive = all((report["gates"]["exact_controls_pass"], report["gates"]["validation_candidate_selected_without_test"], *(report["gates"][name][split] for name in ("miss_reduction_ge_0_50", "mean_kl_le_0_001", "extra_compute_per_avoided_le_2", "span_uplift_over_zero_fill_ge_0_10") for split in ("validation", "test")), report["gates"]["projected_compute_lt_avoided_transfer"]))
    expected = "oracle_positive_requires_causal_cache_and_predictor" if positive else "invalid_exact_control" if not controls_pass else "falsified_optimistic_ghost_cache_oracle" if hard else "inconclusive_negative_layer26_screen"
    audit.equal("h8.verdict", experiment, "verdict", report["verdict"], expected, "preregistered optimistic-screen decision tree")

    recorded = load_json(REPORT_DIR / "cache_span_block_bootstrap_audit.json")
    for split, primary, zero in (
        ("validation", selected, report["validation"]["zero_fill"]),
        ("test", test, report["heldout_test"]["zero_fill"]),
    ):
        recomputed = paired_load_bootstrap_independent(
            [row["baseline_misses"] for row in primary["per_block"]],
            [row["avoided_misses"] for row in primary["per_block"]],
            [row["avoided_misses"] for row in zero["per_block"]],
            seed=AUDIT_SEED + (200 if split == "validation" else 201),
        )
        actual = recorded["splits"][split]["bootstrap"]
        audit.equal(f"h8.bootstrap.{split}.metadata", experiment, "bootstrap_unit", {k: actual[k] for k in ("method", "seed", "resamples", "sampling_units")}, {k: recomputed[k] for k in ("method", "seed", "resamples", "sampling_units")}, "independent paired sequence-block bootstrap")
        for metric in recomputed["point_estimates"]:
            check_reported_float(audit, experiment, f"h8.bootstrap.{split}.{metric}.point", actual["point_estimates"][metric], recomputed["point_estimates"][metric], "independent block-total reconciliation")
            for bound in ("low", "high"):
                check_reported_float(audit, experiment, f"h8.bootstrap.{split}.{metric}.{bound}", actual["intervals_95"][metric][bound], recomputed["intervals_95"][metric][bound], "independent 10,000x block bootstrap")


def audit_h10(audit: AuditCollector, report: Mapping[str, Any]) -> None:
    experiment = "H10_REDUCTION_ORDER"
    for split in ("validation", "test"):
        quality = report["exact_quality"][split]
        gap = quality["gap_analysis"]
        names = {
            "q3": "q3_reference_vectorized_fp32",
            "q4": "q4_reference_vectorized_fp32",
            "fixed": "q3_fixed_validation_order",
            "fp32": "q3_validation_selected_fp32_control",
        }
        means = {
            key: float(np.mean(quality[name]["raw"]["teacher_to_candidate_kl"]))
            for key, name in names.items()
        }
        closure = gap_closure(means["q3"], means["q4"], means["fixed"])
        fp32_closure = gap_closure(means["q3"], means["q4"], means["fp32"])
        for key, expected in (
            ("q3_reference_kl", means["q3"]),
            ("q4_reference_kl", means["q4"]),
            ("fixed_q3_kl", means["fixed"]),
            ("fixed_gap_closure", closure),
            ("fp32_control_gap_closure", fp32_closure),
        ):
            check_reported_float(audit, experiment, f"h10.{split}.{key}", gap[key], expected, "mean of raw per-token exact-KL series", tolerance=2e-12)
        check_reported_bool(audit, experiment, f"h10.{split}.denominator", report["gates"]["q3_to_q4_denominator_positive"][split], means["q3"] > means["q4"], "raw Q3 and Q4 mean KL")
        check_reported_bool(audit, experiment, f"h10.{split}.closure_gate", report["gates"]["fixed_gap_closure_ge_0_20"][split], closure >= 0.20, "fixed gap closure >=20%")
        check_reported_bool(audit, experiment, f"h10.{split}.q3_not_worse", report["gates"]["fixed_q3_not_worse_than_reference"][split], means["fixed"] <= means["q3"], "fixed Q3 versus vectorized reference")
        check_reported_bool(audit, experiment, f"h10.{split}.fp32_hard_control", report["gates"]["fp32_control_closure_ge_0_10"][split], fp32_closure >= 0.10, "validation-selected FP32 control closure")
    audit.equal("h10.no_test_tuning", experiment, "split_discipline", report["protocol"]["test_tuning"], "none", "protocol.test_tuning")
    audit.equal("h10.selection_criterion", experiment, "split_discipline", report["protocol"]["selection_uses"], "validation Q3 routed MSE only", "protocol.selection_uses")
    audit.add("h10.protected_invariance", experiment, "exact_control", all(value == 0.0 for bit in report["protected_fp32_order_invariance"].values() for split in bit.values() for value in split.values()), report["protected_fp32_order_invariance"], "all protected BF16-operands-to-FP32 spreads equal zero", "protected order-invariance table")
    controls = report["controls"]
    controls_pass = all((controls["capture_finite"], controls["original_teacher_delta_bit_exact"], controls["post_capture_original_teacher_delta_bit_exact"], controls["route_slot_ids_exact"], controls["router_weight_maximum_absolute_error"] <= 1e-6))
    check_reported_bool(audit, experiment, "h10.exact_controls", report["gates"]["exact_controls_pass"], controls_pass, "capture/teacher/route/weight controls")

    raw_path = Path(report["raw_sweep_artifact"]["path"])
    try:
        from safetensors.torch import load_file
        import torch

        raw = load_file(raw_path, device="cpu")
        scores = raw["q3_validation_mse"].double().sum(dim=2)
        flat = int(torch.argmin(scores.reshape(-1)).item())
        scheme_index, permutation_index = divmod(flat, scores.shape[1])
        selected = report["validation_selection"]
        audit.equal("h10.raw_argmin", experiment, "raw_recalculation", [selected["scheme_index"], selected["permutation_index"]], [scheme_index, permutation_index], "lossless q3_validation_mse tensor")
    except Exception as error:  # pragma: no cover - converted to a fatal audit record
        audit.add("h10.raw_argmin", experiment, "raw_recalculation", False, repr(error), "loadable lossless sweep and exact argmin", str(raw_path))

    recorded = load_json(REPORT_DIR / "reduction_order_bootstrap_audit.json")
    for split_index, split in enumerate(("validation", "test")):
        quality = report["exact_quality"][split]
        candidates = {
            "fixed": quality["q3_fixed_validation_order"]["raw"]["teacher_to_candidate_kl"],
            "fp32_control": quality["q3_validation_selected_fp32_control"]["raw"]["teacher_to_candidate_kl"],
            "per_token_local_mse_oracle": quality["q3_per_token_local_mse_oracle"]["raw"]["teacher_to_candidate_kl"],
        }
        recomputed = paired_gap_bootstrap_independent(
            quality["q3_reference_vectorized_fp32"]["raw"]["teacher_to_candidate_kl"],
            quality["q4_reference_vectorized_fp32"]["raw"]["teacher_to_candidate_kl"],
            candidates,
            block_size=128,
            seed=AUDIT_SEED + 300 + split_index,
        )
        actual = recorded["splits"][split]["bootstrap"]
        audit.equal(f"h10.bootstrap.{split}.metadata", experiment, "bootstrap_unit", {k: actual[k] for k in ("method", "seed", "resamples", "sampling_units", "block_size")}, {k: recomputed[k] for k in ("method", "seed", "resamples", "sampling_units", "block_size")}, "independent paired sequence-block bootstrap")
        for candidate in recomputed["point_closure"]:
            check_reported_float(audit, experiment, f"h10.bootstrap.{split}.{candidate}.point", actual["point_closure"][candidate], recomputed["point_closure"][candidate], "raw per-token series")
            for bound in ("low", "high"):
                check_reported_float(audit, experiment, f"h10.bootstrap.{split}.{candidate}.{bound}", actual["intervals_95"][candidate][bound], recomputed["intervals_95"][candidate][bound], "independent 10,000x sequence-block bootstrap")
    test_gap = report["exact_quality"]["test"]["gap_analysis"]
    hard = (not test_gap["denominator_positive"] or test_gap["fixed_gap_closure"] is None or test_gap["fixed_gap_closure"] < 0.10 or test_gap["fixed_q3_kl"] > test_gap["q3_reference_kl"] or test_gap["fp32_control_gap_closure"] is None or test_gap["fp32_control_gap_closure"] < 0.10)
    content = controls_pass and all(report["gates"]["q3_to_q4_denominator_positive"].values()) and all(report["gates"]["fixed_gap_closure_ge_0_20"].values()) and all(report["gates"]["fixed_q3_not_worse_than_reference"].values()) and report["gates"]["same_weight_bytes_and_no_metadata"]
    expected = "content_positive_requires_physical_reducer_benchmark" if content else "invalid_exact_control" if not controls_pass else "falsified_fixed_heldout_reduction_order" if hard else "inconclusive_negative_layer26_screen"
    audit.equal("h10.verdict", experiment, "verdict", report["verdict"], expected, "preregistered H10 decision tree")
    audit.equal("h10.throughput_not_opened", experiment, "stop_go", report["gates"]["throughput_ratio_le_1_05"]["evaluated"], False, "content gate failed before physical benchmark")
    audit.add("h10.q4_catastrophic_threshold", experiment, "preregistration_precision", False, "qualitative 'not catastrophic' with finite outputs only", "numeric preregistered threshold", "H10 preregistration item 3 did not numerically define catastrophic Q4 degradation; it cannot support a positive claim", severity="warning")


def audit_split_discipline(
    audit: AuditCollector, reports: Mapping[str, Mapping[str, Any]]
) -> None:
    found = 0
    for experiment, report in reports.items():
        for index, trace_indices in enumerate(values_for_key(report, "trace_indices")):
            if not isinstance(trace_indices, Mapping) or not {"validation", "test"}.issubset(trace_indices):
                continue
            found += 1
            audit.add(
                f"{experiment}.trace_disjoint.{index}", experiment, "split_discipline",
                disjoint(trace_indices["validation"], trace_indices["test"]),
                {split: [min(values), max(values), len(values)] for split, values in trace_indices.items() if isinstance(values, list) and values},
                "disjoint validation/test trace indices",
                "reproducibility trace_indices",
            )
    audit.add("global.trace_index_evidence", "GLOBAL", "split_discipline", found >= 6, found, ">=6 explicit trace-index maps; remaining studies use separately named/hash-identified dataset splits", "cross-experiment provenance")
    qerc = reports["H6_QERC"]["phase_b"]["fixed_candidate_definition_retained"]
    fit = range(*qerc["validation_fit_positions"])
    selection = range(*qerc["validation_selection_positions"])
    audit.add("h6.validation_fit_selection_disjoint", "H6_QERC", "split_discipline", disjoint(fit, selection), [qerc["validation_fit_positions"], qerc["validation_selection_positions"]], "non-overlapping validation slices", "fixed Phase-B definition retained even though phase did not open")


def audit_bootstrap_metadata(
    audit: AuditCollector, reports: Mapping[str, Mapping[str, Any]]
) -> None:
    inspected = 0
    selected_reports = (
        "H7_ROUTE_CORESET",
        "H4_SKETCHGATE_REPLICATION",
        "H2_BLOCK_COALESCING",
        "H8_CACHE_SPAN",
        "H10_REDUCTION_ORDER",
    )
    for experiment in selected_reports:
        for bootstrap in values_for_key(reports[experiment], "bootstrap_95"):
            if not isinstance(bootstrap, Mapping) or "method" not in bootstrap:
                continue
            inspected += 1
            method = str(bootstrap["method"]).lower()
            audit.add(f"bootstrap.{experiment}.{inspected}.unit", experiment, "bootstrap_unit", "block" in method, bootstrap["method"], "sequence/verification block", "bootstrap method metadata")
            audit.equal(f"bootstrap.{experiment}.{inspected}.resamples", experiment, "bootstrap_unit", bootstrap.get("resamples"), 10_000, "fixed resample count")
    audit.add("global.bootstrap_metadata_coverage", "GLOBAL", "bootstrap_unit", inspected >= 10, inspected, ">=10 recorded bootstrap objects plus independent H8/H10 audits", "selected raw reports")


def registry_section(text: str, experiment: str) -> str:
    match = re.search(
        rf"^  - id: {re.escape(experiment)}\s*$([\s\S]*?)(?=^  - id: |\Z)",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise KeyError(experiment)
    return match.group(1)


def audit_registry_and_retention(audit: AuditCollector) -> None:
    text = (REPORT_DIR / "EXPERIMENT_REGISTRY.yaml").read_text(encoding="utf-8")
    expected_status = {
        "H7_ROUTE_CORESET": "complete",
        "H1_CRCQ": "falsified",
        "H3_ATOMIC_ORACLE": "falsified",
        "H4_SKETCHGATE": "falsified",
        "H2_BLOCK_COALESCING": "falsified",
        "H5_ATOMIC_INDEX": "blocked",
        "H6_QERC": "falsified",
        "H8_CACHE_SPAN": "complete",
        "H9_BISPARSE": "blocked",
        "H10_REDUCTION_ORDER": "falsified",
        "PACKED_RUNTIME": "blocked",
    }
    for experiment, expected in expected_status.items():
        section = registry_section(text, experiment)
        match = re.search(r"^    status: (.+)$", section, flags=re.MULTILINE)
        audit.equal(f"registry.{experiment}.status", experiment, "registry", match.group(1).strip() if match else None, expected, "EXPERIMENT_REGISTRY.yaml")
    audit.add("registry.h5_dependency", "H5_ATOMIC_INDEX", "dependency", "H3 simultaneous full-depth primary gate falsified" in registry_section(text, "H5_ATOMIC_INDEX"), registry_section(text, "H5_ATOMIC_INDEX").strip(), "H3 falsification dependency", "registry blocked_reason")
    audit.add("registry.h9_dependency", "H9_BISPARSE", "dependency", "H3 simultaneous full-depth primary gate falsified" in registry_section(text, "H9_BISPARSE"), registry_section(text, "H9_BISPARSE").strip(), "H3 falsification dependency", "registry blocked_reason")
    audit.add("retention.h4_initial", "H4_SKETCHGATE_REPLICATION", "failed_artifact_retention", (REPORT_DIR / "sketchgate.json").exists(), str(REPORT_DIR / "sketchgate.json"), "file exists", "append-only initial failure")
    audit.add("retention.h2_audit_v1", "H2_BLOCK_COALESCING", "failed_artifact_retention", (REPORT_DIR / "block_route_coalescing_control_audit.json").exists(), str(REPORT_DIR / "block_route_coalescing_control_audit.json"), "file exists", "append-only erroneous v1")
    selector_files = list(RUN_DIR.glob("*selector*.safetensors")) + list(REPORT_DIR.glob("*selector*.json"))
    audit.equal("retention.no_unopened_trained_selector", "GLOBAL", "stop_go", [str(path) for path in selector_files], [], "H5/H8/H9 predictors never opened")


def audit_benchmarks(audit: AuditCollector, reports: Mapping[str, Mapping[str, Any]]) -> None:
    h8 = reports["H8_CACHE_SPAN"]["hardware_model"]
    audit.add("benchmark.h8.repetitions", "H8_CACHE_SPAN", "benchmark", h8["repetitions"] >= 7 and len(h8["transfer_milliseconds_raw"]) == h8["repetitions"], {"repetitions": h8["repetitions"], "raw": len(h8["transfer_milliseconds_raw"])}, ">=7 aligned repetitions", "hardware_model")
    h8_source = (ROOT / "scripts/craft_moe/evaluate_cache_span_oracle.py").read_text(encoding="utf-8")
    h4_source = (ROOT / "scripts/craft_moe/evaluate_sketchgate.py").read_text(encoding="utf-8")
    audit.add("benchmark.h8.warmup", "H8_CACHE_SPAN", "benchmark", "for _ in range(15):" in h8_source and "torch.cuda.synchronize()" in h8_source, "15 synchronized warmup iterations", "warmup present", "source inspection")
    audit.add("benchmark.h4.warmup", "H4_SKETCHGATE_REPLICATION", "benchmark", "for _ in range(20):" in h4_source and "torch.cuda.synchronize()" in h4_source, "20 synchronized warmup iterations", "warmup present", "source inspection")
    audit.equal("benchmark.h8.no_runtime_claim", "H8_CACHE_SPAN", "claims", h8["not_a_runtime_claim"], True, "hardware_model.not_a_runtime_claim")
    for experiment in ("H8_CACHE_SPAN", "H4_SKETCHGATE_REPLICATION"):
        audit.add(f"benchmark.{experiment}.thermal_telemetry", experiment, "benchmark", False, "warmup and repeated CUDA events; no clock/temperature trace", "thermal steady-state telemetry", "microbenchmark cannot support a wall-clock speedup claim", severity="warning")


def path_key(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_declared_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def declared_hashes(value: Any) -> list[tuple[Path, str, str]]:
    found: list[tuple[Path, str, str]] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            found.append((resolve_declared_path(value["path"]), value["sha256"], "path+sha256"))
        sha_map = value.get("sha256")
        if isinstance(sha_map, Mapping):
            for raw_path, digest in sha_map.items():
                if isinstance(raw_path, str) and isinstance(digest, str):
                    found.append((resolve_declared_path(raw_path), digest, "sha256 map"))
        for path_field, hash_field in (
            ("source", "source_sha256"),
            ("component_artifact", "component_sha256"),
        ):
            if isinstance(value.get(path_field), str) and isinstance(value.get(hash_field), str):
                found.append((resolve_declared_path(value[path_field]), value[hash_field], f"{path_field}+{hash_field}"))
        for item in value.values():
            found.extend(declared_hashes(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(declared_hashes(item))
    return found


def locked_artifact_paths() -> list[Path]:
    paths = set(RESULT_PATHS.values())
    paths.update(
        path for path in REPORT_DIR.glob("*.json")
        if path.name not in {"repro_audit.json", "novelty_matrix.json"}
    )
    paths.update(
        path for path in REPORT_DIR.glob("*.md")
        if path.name not in {"REPRO_AUDIT.md", "NOVELTY_VERDICT.md", "CRAFT_MOE_MASTER_VERDICT.md"}
    )
    paths.update(
        {
            RUN_DIR / "qerc_covariance_layer26.json",
            RUN_DIR / "qerc_layer26_components.safetensors",
            RUN_DIR / "cache_span_layer26_capture.safetensors",
            RUN_DIR / "reduction_order_capture.safetensors",
            RUN_DIR / "reduction_order_raw.safetensors",
            REPORT_DIR / "atomic_full_depth_supports.safetensors",
            ROOT / "scripts/craft_moe/verify_all_gates.py",
            ROOT / "src/moe_lab/craft_moe/repro_audit.py",
            ROOT / "tests/craft_moe/test_repro_audit.py",
        }
    )
    return sorted(paths, key=lambda path: path_key(path))


def audit_hashes(
    audit: AuditCollector,
    reports: Mapping[str, Mapping[str, Any]],
    *,
    reference_manifest: Mapping[str, Any] | None,
    skip: bool,
) -> dict[str, dict[str, Any]]:
    if skip:
        audit.add("hashes.skipped", "GLOBAL", "artifact_hash", False, "skipped by development flag", "full SHA-256 verification", "--skip-artifact-hashes", severity="warning")
        return {}

    expected: dict[str, tuple[Path, str, str]] = {}
    for report in reports.values():
        for path, digest, source in declared_hashes(report):
            key = path_key(path)
            previous = expected.get(key)
            if previous and previous[1] != digest:
                audit.add(f"hash.declaration_conflict.{key}", "GLOBAL", "artifact_hash", False, [previous[1], digest], "one digest per path", source)
            expected[key] = (path, digest, source)
    for key, digest in PUBLISHED_HASHES.items():
        expected[key] = (ROOT / key, digest, "published Markdown hash")

    paths = {path_key(path): path for path in locked_artifact_paths()}
    paths.update({key: row[0] for key, row in expected.items()})
    manifest: dict[str, dict[str, Any]] = {}
    hash_cache: dict[Path, str] = {}
    for key, path in sorted(paths.items()):
        resolved = path.resolve()
        if not resolved.exists():
            audit.add(f"hash.missing.{key}", "GLOBAL", "artifact_hash", False, "missing", "file exists", key)
            continue
        digest = hash_cache.setdefault(resolved, sha256_file(resolved))
        manifest[key] = {"bytes": resolved.stat().st_size, "sha256": digest}
        if key in expected:
            audit.equal(f"hash.declared.{key}", "GLOBAL", "artifact_hash", digest, expected[key][1], expected[key][2])

    if reference_manifest is not None:
        for key, expected_row in reference_manifest.items():
            actual = manifest.get(key)
            audit.equal(f"hash.manifest.{key}", "GLOBAL", "artifact_manifest", actual, expected_row, "append-only repro_audit.json manifest")
    return manifest


def git_state() -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)

    revision_result = run("git", "rev-parse", "HEAD")
    status_result = run("git", "status", "--porcelain")
    return {
        "revision": revision_result.stdout.strip() if revision_result.returncode == 0 else None,
        "no_commits_yet": revision_result.returncode != 0,
        "dirty": bool(status_result.stdout.strip()),
        "status_returncode": status_result.returncode,
    }


def run_audit(
    *,
    reference_manifest: Mapping[str, Any] | None = None,
    skip_artifact_hashes: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    reports = load_reports()
    audit = AuditCollector()
    audit_common_contract(audit, reports)
    audit_h7(audit, reports["H7_ROUTE_CORESET"])
    audit_h1_screen(audit, reports["H1_CRCQ_SCREEN"])
    audit_h1_full(audit, reports["H1_CRCQ_FULL"], reports["H1_CRCQ_SCREEN"])
    audit_h1_downstream(audit, reports["H1_CRCQ_LAYER23"])
    audit_atomic(audit, reports)
    audit_h4(audit, reports["H4_SKETCHGATE_REPLICATION"])
    audit_h2(audit, reports["H2_BLOCK_COALESCING"])
    audit_h6(audit, reports["H6_QERC"])
    audit_h8(audit, reports["H8_CACHE_SPAN"])
    audit_h10(audit, reports["H10_REDUCTION_ORDER"])
    audit_split_discipline(audit, reports)
    audit_bootstrap_metadata(audit, reports)
    audit_registry_and_retention(audit)
    audit_benchmarks(audit, reports)
    manifest = audit_hashes(
        audit,
        reports,
        reference_manifest=reference_manifest,
        skip=skip_artifact_hashes,
    )
    summary = audit.summary()
    return {
        "schema_version": 1,
        "kind": "craft_moe_reproducibility_and_statistical_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if summary["all_required_checks_pass"] else "failed",
        "verdict": "reproducibility_audit_passed_with_declared_warnings" if summary["all_required_checks_pass"] else "reproducibility_audit_failed",
        "scope": {
            "technical_results": list(RESULT_PATHS),
            "model_revision": MODEL_REVISION,
            "dataset_revision": DATASET_REVISION,
            "gate_source": "numeric/raw result fields evaluated against preregistered thresholds",
            "selected_raw_recomputations": [
                "H7 per-token minimum k",
                "H8 validation configuration and paired bootstrap",
                "H10 lossless validation sweep argmin, exact-KL gap and paired bootstrap",
            ],
        },
        "summary": summary,
        "checks": audit.serializable_checks(),
        "artifact_manifest": manifest,
        "limitations": [
            "The audit independently recomputes all gate decisions but does not rerun every multi-gigabyte GPU enumeration.",
            "The H8 and H4 hardware measurements are component microbenchmarks, not packed end-to-end runtimes.",
            "Thermal/clock telemetry was not recorded; therefore no wall-clock speedup claim is admissible.",
            "Two-block confidence intervals describe window heterogeneity and are not confirmatory evidence.",
            "The repository has no commit; immutable artifact hashes substitute for a Git revision for this audit.",
        ],
        "reproducibility": {
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "cwd": str(ROOT),
            "repository": git_state(),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "ram_total_bytes": psutil.virtual_memory().total,
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.check_only and args.dry_run:
        raise SystemExit("--check-only and --dry-run are mutually exclusive")
    reference_manifest = None
    if args.check_only:
        if not args.output_json.exists():
            raise FileNotFoundError(f"reference audit does not exist: {args.output_json}")
        reference = load_json(args.output_json)
        reference_manifest = reference["artifact_manifest"]
    result = run_audit(
        reference_manifest=reference_manifest,
        skip_artifact_hashes=args.skip_artifact_hashes,
    )
    if not args.check_only and not args.dry_run:
        write_json_once(args.output_json, result)
    print(json.dumps(result["summary"], sort_keys=True))
    print(f"verdict={result['verdict']}")
    if not result["summary"]["all_required_checks_pass"]:
        for check in result["checks"]:
            if not check["passed"] and check["severity"] == "error":
                print(f"FAILED {check['check_id']}: observed={check['observed']!r} expected={check['expected']!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
