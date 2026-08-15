from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .subspace import image_storage_elements, select_validation_candidate


@dataclass
class Audit:
    checks: int = 0
    passed: int = 0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def require(self, condition: bool, label: str) -> None:
        self.checks += 1
        if condition:
            self.passed += 1
        else:
            self.failures.append(label)

    def equal(self, actual: Any, expected: Any, label: str) -> None:
        self.require(actual == expected, f"{label}: {actual!r} != {expected!r}")

    def close(self, actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
        self.require(
            math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=atol),
            f"{label}: {actual!r} != {expected!r}",
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_metric(metric: dict[str, Any], audit: Audit, label: str) -> None:
    invocations = int(metric["invocations"])
    audit.require(invocations > 0, f"{label}.invocations_positive")
    for name in (
        "x_fast_fraction",
        "z_fast_fraction",
        "double_gate_fast_fraction",
        "router_mass_double_gate_fast_fraction",
        "projected_cold_byte_fraction",
        "router_mass_projected_cold_byte_fraction",
    ):
        value = float(metric[name])
        audit.require(0.0 <= value <= 1.0, f"{label}.{name}_range")
    audit.require(
        metric["double_gate_fast_fraction"] <= metric["x_fast_fraction"] + 1e-15,
        f"{label}.double_le_x",
    )
    audit.require(
        metric["double_gate_fast_fraction"] <= metric["z_fast_fraction"] + 1e-15,
        f"{label}.double_le_z",
    )
    cold = float(metric["projected_cold_byte_fraction"])
    reduction = metric["projected_routed_cold_byte_reduction"]
    if cold == 0.0:
        audit.equal(reduction, None, f"{label}.zero_cold_reduction")
    else:
        audit.close(reduction, 1.0 / cold, f"{label}.cold_reciprocal")
    mass_cold = float(metric["router_mass_projected_cold_byte_fraction"])
    mass_reduction = metric["router_mass_projected_cold_byte_reduction"]
    if mass_cold == 0.0:
        audit.equal(mass_reduction, None, f"{label}.zero_mass_cold_reduction")
    else:
        audit.close(
            mass_reduction, 1.0 / mass_cold, f"{label}.mass_cold_reciprocal"
        )


def _verify_grid(rows: list[dict[str, Any]], audit: Audit, label: str) -> None:
    audit.equal(len(rows), 42, f"{label}.candidate_count")
    pairs = {(int(row["rank_cap"]), float(row["threshold"])) for row in rows}
    audit.equal(len(pairs), 42, f"{label}.unique_candidates")
    for row_index, row in enumerate(rows):
        for evaluation in ("offline", "causal_prefix_future"):
            value = row[evaluation]
            _verify_metric(
                value["aggregate"], audit, f"{label}.{row_index}.{evaluation}.aggregate"
            )
            audit.equal(
                set(value["layers"]), {"1", "13", "26"},
                f"{label}.{row_index}.{evaluation}.layers",
            )
            for layer, metric in value["layers"].items():
                _verify_metric(
                    metric,
                    audit,
                    f"{label}.{row_index}.{evaluation}.layer_{layer}",
                )


def _verify_census(census: dict[str, Any], audit: Audit, label: str) -> None:
    for layer_name, layer in census.items():
        experts = layer["experts"]
        audit.equal(len(experts), 64, f"{label}.{layer_name}.expert_count")
        counts = [int(row["count"]) for row in experts]
        input_ranks = [int(row["input_stored_rank"]) for row in experts]
        intermediate_ranks = [int(row["intermediate_stored_rank"]) for row in experts]
        audit.equal(
            sum(counts), int(layer["expected_expert_invocations"]),
            f"{label}.{layer_name}.count_sum",
        )
        audit.require(
            all(rank <= count for rank, count in zip(input_ranks, counts)),
            f"{label}.{layer_name}.input_rank_bound",
        )
        audit.require(
            all(rank <= count for rank, count in zip(intermediate_ranks, counts)),
            f"{label}.{layer_name}.z_rank_bound",
        )
        elements = image_storage_elements(
            2048, 1408, input_ranks, intermediate_ranks
        )
        bound = (2 * 2048 + 3 * 1408) * int(layer["expert_invocations"])
        audit.equal(elements, int(layer["full_rank_image_elements"]),
                    f"{label}.{layer_name}.image_elements")
        audit.equal(bound, int(layer["expert_count_cancellation_bound_elements"]),
                    f"{label}.{layer_name}.bound_elements")
        audit.require(elements <= bound, f"{label}.{layer_name}.storage_bound")
        audit.require(all(layer["controls"].values()), f"{label}.{layer_name}.controls")


def build_verification(root: Path) -> dict[str, Any]:
    root = root.resolve()
    result_path = root / "reports/rsiv_moe/routed_subspace_rank.json"
    report_path = root / "reports/rsiv_moe/ROUTED_SUBSPACE_RANK.md"
    analysis_path = root / "reports/rsiv_moe/P1_V2_ANALYSIS.md"
    capture_path = root / "reports/runs/rsiv_moe/routed_subspace_pilot.safetensors"
    capture_report_path = root / "reports/rsiv_moe/routed_subspace_capture.json"
    selection_path = root / "reports/rsiv_moe/p1_validation_selection_v2.json"
    invalid_selection_path = root / "reports/rsiv_moe/p1_validation_selection.json"
    invalid_verification_json = root / "reports/rsiv_moe/p1_verification.json"
    invalid_verification_md = root / "reports/rsiv_moe/P1_VERIFICATION.md"
    addendum_path = root / "reports/rsiv_moe/RSIV_MOE_P1_CONTROL_ADDENDUM_001.md"
    preregistration_path = root / "reports/rsiv_moe/RSIV_MOE_PREREGISTRATION.md"
    paths = (
        result_path,
        report_path,
        analysis_path,
        capture_path,
        capture_report_path,
        selection_path,
        invalid_selection_path,
        invalid_verification_json,
        invalid_verification_md,
        addendum_path,
        preregistration_path,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    result = _load(result_path)
    selection = _load(selection_path)
    capture_report = _load(capture_report_path)
    audit = Audit()
    capture_hash = sha256_file(capture_path)
    selection_hash = sha256_file(selection_path)
    audit.equal(capture_hash, result["capture_sha256"], "hash.capture.result")
    audit.equal(capture_hash, selection["capture_sha256"], "hash.capture.selection")
    audit.equal(capture_hash, capture_report["capture_sha256"], "hash.capture.report")
    audit.equal(selection_hash, result["selection_sha256"], "hash.selection.result")
    audit.equal(selection["test_data_opened"], False, "selection.test_closed")
    audit.equal(result["test_opened_once"], True, "result.test_opened_once")
    audit.equal(result["verdict"], "screen_negative_v2", "result.verdict")
    audit.equal(
        result["claim_boundaries"]["runtime"], "not measured", "claim.runtime"
    )
    audit.equal(result["claim_boundaries"]["eureka"], False, "claim.eureka")

    recomputed = select_validation_candidate(selection["validation_grid"])
    for key in ("rank_cap", "threshold", "selection_kind"):
        audit.equal(
            recomputed[key], selection["selected_candidate"][key],
            f"selection.recomputed.{key}",
        )
    audit.equal(
        selection["selected_candidate"]["selection_kind"],
        "diagnostic_validation_failure",
        "selection.diagnostic_failure",
    )
    _verify_grid(selection["validation_grid"], audit, "validation_grid")
    _verify_grid(
        result["test_confirmation"]["full_post_lock_grid_for_diagnostics_only"],
        audit,
        "test_grid",
    )
    _verify_census(selection["rank_census"]["train"], audit, "census.train")
    _verify_census(
        selection["rank_census"]["validation"], audit, "census.validation"
    )
    _verify_census(
        result["test_confirmation"]["rank_census"], audit, "census.test"
    )

    selected = result["test_confirmation"]["selected_candidate_metrics"]
    audit.equal(selected["rank_cap"], 4, "test.selected.rank")
    audit.close(selected["threshold"], 0.001, "test.selected.threshold")
    audit.close(selected["offline_double_fast_fraction"], 0.0,
                "test.selected.offline_fast")
    audit.close(selected["causal_double_fast_fraction"], 0.0,
                "test.selected.causal_fast")
    audit.equal(
        result["test_confirmation"]["selected_candidate_pass"],
        False,
        "test.selected.pass",
    )

    operator = result["controls"]["operator_image_controls"]
    audit.equal(operator["all_required_controls_pass"], True,
                "operator.required_controls")
    for scope, metrics in [("global", operator["global"]), *operator["layers"].items()]:
        values = metrics if scope == "global" else metrics["operator_images"]
        for name, regression in values.items():
            audit.require(
                regression["relative_l2"] <= operator["relative_l2_tolerance"],
                f"operator.{scope}.{name}.relative",
            )
            audit.require(
                regression["maximum_absolute_error"]
                <= operator["maximum_absolute_tolerance"],
                f"operator.{scope}.{name}.maximum",
            )
            audit.equal(regression["pass"], True, f"operator.{scope}.{name}.pass")
    if not operator["stored_bf16_z_batch_shape_bit_exact_diagnostic"]:
        audit.warnings.append(
            "BF16 z regrouping is not bit-exact across GEMM batch shapes; "
            "Control Addendum 001 correctly keeps this diagnostic outside required gates."
        )

    with safe_open(capture_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        audit.equal(metadata.get("model_revision"),
                    "604d5664dddd88a0433dbae533b7fe9472482de0",
                    "capture.model_revision")
        audit.equal(metadata.get("dataset_revision"),
                    "b08601e04326c79dfdd32d625aee71d232d685c3",
                    "capture.dataset_revision")
        for layer in (1, 13, 26):
            prefix = f"layer_{layer:02d}"
            expected_shapes = {
                f"{prefix}_moe_input": (2048, 2048),
                f"{prefix}_router_ids": (2048, 6),
                f"{prefix}_router_weights": (2048, 6),
                f"{prefix}_intermediate_z": (2048, 6, 1408),
            }
            for name, shape in expected_shapes.items():
                audit.equal(tuple(handle.get_slice(name).get_shape()), shape,
                            f"capture.{name}.shape")
            ids = handle.get_slice(f"{prefix}_router_ids")[:].long()
            weights = handle.get_slice(f"{prefix}_router_weights")[:].float()
            audit.require(bool(((ids >= 0) & (ids < 64)).all()),
                          f"capture.layer_{layer}.id_range")
            audit.require(bool(torch.isfinite(weights).all()),
                          f"capture.layer_{layer}.weights_finite")
            audit.equal(ids.numel(), 2048 * 6, f"capture.layer_{layer}.count")

    report_text = report_path.read_text(encoding="utf-8")
    analysis_text = analysis_path.read_text(encoding="utf-8")
    audit.require("screen_negative_v2" in report_text, "report.verdict_present")
    audit.require("0,749%" in analysis_text, "analysis.rank128_test_present")
    audit.require("Eureka-verdict" in analysis_text, "analysis.claim_boundary_present")
    audit.equal(
        sha256_file(invalid_selection_path),
        "1ff70f0657b873b9cebbed19dc93b6c95c4a0ad739bf1da129ab3dd0713e3eb1",
        "hash.invalid_selection_v1",
    )
    audit.equal(
        sha256_file(invalid_verification_json),
        "1f3a42af8ce397654a28a36cf1846865352ecc63d0d5992168705f12ed3e7230",
        "hash.invalid_verification_v1_json",
    )
    audit.equal(
        sha256_file(invalid_verification_md),
        "1dbccaec28084a378bcc9483fa6bdfc54e2a043b2067907ec634ce72b467ea74",
        "hash.invalid_verification_v1_md",
    )

    return {
        "kind": "rsiv_moe_p1_independent_verification",
        "verdict": (
            "p1_verification_passed_with_declared_warning"
            if not audit.failures
            else "p1_verification_failed"
        ),
        "all_required_checks_pass": not audit.failures,
        "checks": audit.checks,
        "passed": audit.passed,
        "failed": len(audit.failures),
        "warnings": len(audit.warnings),
        "failure_messages": audit.failures,
        "warning_messages": audit.warnings,
        "artifacts": {
            "result_sha256": sha256_file(result_path),
            "report_sha256": sha256_file(report_path),
            "analysis_sha256": sha256_file(analysis_path),
            "capture_sha256": capture_hash,
            "capture_report_sha256": sha256_file(capture_report_path),
            "selection_v2_sha256": selection_hash,
            "invalid_selection_v1_sha256": sha256_file(invalid_selection_path),
            "invalid_verification_v1_json_sha256": sha256_file(
                invalid_verification_json
            ),
            "invalid_verification_v1_md_sha256": sha256_file(
                invalid_verification_md
            ),
            "control_addendum_sha256": sha256_file(addendum_path),
            "preregistration_sha256": sha256_file(preregistration_path),
        },
        "confirmed_result": {
            "verdict": result["verdict"],
            "selected_rank": selected["rank_cap"],
            "selected_threshold": selected["threshold"],
            "validation_selection_kind": selection["selected_candidate"]["selection_kind"],
            "test_offline_double_fast_fraction": selected[
                "offline_double_fast_fraction"
            ],
            "test_causal_double_fast_fraction": selected[
                "causal_double_fast_fraction"
            ],
        },
    }


def render_verification(payload: dict[str, Any]) -> str:
    warning = payload["warning_messages"][0] if payload["warning_messages"] else "Geen."
    return "\n".join(
        [
            "# RSIV-MoE P1 onafhankelijke verificatie",
            "",
            f"**Verdict: `{payload['verdict']}`.**",
            "",
            f"- Controles: {payload['checks']}.",
            f"- Geslaagd: {payload['passed']}.",
            f"- Fouten: {payload['failed']}.",
            f"- Gedeclareerde waarschuwingen: {payload['warnings']}.",
            "",
            "De verifier herberekent hashes, validationselectie, grididentiteiten, koude-byte-reciproken, rank/count/opslagbounds, operatorimagetoleranties en raw capturevormen. Het bevestigde empirische besluit is `screen_negative_v2`; er volgt geen runtime- of Eureka-claim.",
            "",
            "## Waarschuwing",
            "",
            warning,
            "",
        ]
    )
