from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .p1_verify import Audit, _verify_metric, sha256_file
from .subspace import image_storage_elements, select_single_evaluation_candidate


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_grid(rows: list[dict[str, Any]], audit: Audit, label: str) -> None:
    audit.equal(len(rows), 42, f"{label}.candidate_count")
    audit.equal(
        len({(row["rank_cap"], row["threshold"]) for row in rows}),
        42,
        f"{label}.candidate_unique",
    )
    for index, row in enumerate(rows):
        _verify_metric(row["aggregate"], audit, f"{label}.{index}.aggregate")
        audit.equal(set(row["layers"]), {"1", "13", "26"},
                    f"{label}.{index}.layers")
        for layer, metric in row["layers"].items():
            _verify_metric(metric, audit, f"{label}.{index}.layer_{layer}")


def _verify_census(census: dict[str, Any], audit: Audit, label: str) -> None:
    audit.equal(set(census), {"1", "13", "26"}, f"{label}.layers")
    for layer, payload in census.items():
        audit.equal(len(payload["contexts"]), 2, f"{label}.{layer}.contexts")
        for context in payload["contexts"]:
            audit.equal(context["expert_invocations"], 1024 * 6,
                        f"{label}.{layer}.{context['context']}.invocations")
            # The stored matrices are full row-rank in this result. Verify the
            # exact cancellation-bound arithmetic without trusting the summary.
            input_sum = int(context["input_rank_sum"])
            z_sum = int(context["intermediate_rank_sum"])
            elements = (2048 + 2 * 1408) * input_sum + (1408 + 2048) * z_sum
            audit.equal(elements, context["full_rank_image_elements"],
                        f"{label}.{layer}.{context['context']}.elements")
            audit.equal(context["bound_elements"],
                        (2 * 2048 + 3 * 1408) * 1024 * 6,
                        f"{label}.{layer}.{context['context']}.bound")
            audit.require(elements <= context["bound_elements"],
                          f"{label}.{layer}.{context['context']}.bound_pass")
            audit.require(all(context["controls"].values()),
                          f"{label}.{layer}.{context['context']}.controls")
        audit.equal(payload["all_required_controls_pass"], True,
                    f"{label}.{layer}.all_controls")


def build_p1b_verification(root: Path) -> dict[str, Any]:
    root = root.resolve()
    capture = root / "reports/runs/rsiv_moe/p1b_v2_long_prefix.safetensors"
    capture_report = root / "reports/rsiv_moe/p1b_v2_long_prefix_capture.json"
    selection = root / "reports/rsiv_moe/p1b_long_prefix_validation_selection_v2.json"
    invalid_selection = root / "reports/rsiv_moe/p1b_long_prefix_validation_selection.json"
    result_path = root / "reports/rsiv_moe/p1b_long_prefix.json"
    report_path = root / "reports/rsiv_moe/P1B_V2_LONG_PREFIX.md"
    preregistration = root / "reports/rsiv_moe/P1B_V2_LONG_PREFIX_PREREGISTRATION.md"
    addendum = root / "reports/rsiv_moe/P1B_CONTROL_ADDENDUM_001.md"
    for path in (
        capture,
        capture_report,
        selection,
        invalid_selection,
        result_path,
        report_path,
        preregistration,
        addendum,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    capture_report_data = _load(capture_report)
    selection_data = _load(selection)
    result = _load(result_path)
    audit = Audit()
    capture_hash = sha256_file(capture)
    selection_hash = sha256_file(selection)
    audit.equal(capture_hash, capture_report_data["capture_sha256"],
                "hash.capture_report")
    audit.equal(capture_hash, selection_data["capture_sha256"],
                "hash.selection")
    audit.equal(capture_hash, result["capture_sha256"], "hash.result")
    audit.equal(selection_hash, result["selection_sha256"], "hash.selection_result")
    audit.equal(selection_data["test_data_opened"], False, "selection.test_closed")
    audit.equal(result["test_opened_once"], True, "result.test_once")
    audit.equal(result["verdict"], "long_prefix_screen_negative_v2", "verdict")
    audit.equal(result["claim_boundaries"]["runtime"], "not measured", "claim.runtime")
    audit.equal(result["claim_boundaries"]["eureka"], False, "claim.eureka")

    recomputed = select_single_evaluation_candidate(selection_data["validation_grid"])
    for key in ("rank_cap", "threshold", "selection_kind"):
        audit.equal(recomputed[key], selection_data["selected_candidate"][key],
                    f"selection.{key}")
    audit.equal(recomputed["rank_cap"], 32, "selection.rank32")
    audit.close(recomputed["threshold"], 0.1, "selection.threshold")
    audit.equal(recomputed["selection_kind"], "diagnostic_validation_failure",
                "selection.kind")
    _verify_grid(selection_data["validation_grid"], audit, "validation_grid")
    _verify_grid(result["test"]["post_lock_grid"], audit, "test_grid")
    _verify_census(selection_data["rank_census"], audit, "validation_census")
    _verify_census(result["test"]["rank_census"], audit, "test_census")

    selected = result["test"]["selected_candidate"]
    audit.equal(selected["rank_cap"], 32, "test.selected.rank")
    audit.close(selected["threshold"], 0.1, "test.selected.threshold")
    audit.close(selected["double_gate_fast_fraction"], 0.004340277777777778,
                "test.selected.fast")
    audit.close(selected["cold_byte_reduction"], 1.0074333187581985,
                "test.selected.reduction")
    audit.equal(result["test"]["selected_candidate_pass"], False,
                "test.selected.pass")
    diagnostic = next(
        row for row in result["test"]["post_lock_grid"]
        if row["rank_cap"] == 128 and row["threshold"] == 0.1
    )
    audit.close(diagnostic["double_gate_fast_fraction"], 0.018663194444444444,
                "test.rank128.fast")
    audit.close(diagnostic["cold_byte_reduction"], 1.033261080798266,
                "test.rank128.reduction")
    audit.equal(result["controls"]["all_required_controls_pass"], True,
                "controls.required")
    audit.equal(result["controls"]["long_prefix_operator_diagnostic_pass"], False,
                "controls.extra_diagnostic_false")
    audit.warnings.append(
        "The extra long-prefix FP32 maximum-absolute diagnostic exceeds the old "
        "sample-extreme tolerance; P1B Control Addendum 001 keeps it outside the "
        "preregistered required controls while preserving all values."
    )

    with safe_open(capture, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        audit.equal(metadata["prefix_tokens"], "1024", "capture.prefix")
        audit.equal(metadata["future_tokens"], "128", "capture.future")
        audit.equal(metadata["contexts_per_split"], "2", "capture.contexts")
        for layer in (1, 13, 26):
            prefix = f"layer_{layer:02d}"
            shapes = {
                f"{prefix}_moe_input": (4608, 2048),
                f"{prefix}_router_ids": (4608, 6),
                f"{prefix}_router_weights": (4608, 6),
                f"{prefix}_intermediate_z": (4608, 6, 1408),
            }
            for name, expected in shapes.items():
                audit.equal(tuple(handle.get_slice(name).get_shape()), expected,
                            f"capture.{name}.shape")
            ids = handle.get_slice(f"{prefix}_router_ids")[:].long()
            weights = handle.get_slice(f"{prefix}_router_weights")[:].float()
            audit.require(bool(((ids >= 0) & (ids < 64)).all()),
                          f"capture.layer_{layer}.ids")
            audit.require(bool(torch.isfinite(weights).all()),
                          f"capture.layer_{layer}.weights")

    audit.equal(
        sha256_file(invalid_selection),
        "192be7621be51d3679f0e65613b5aae5539b4e75779ad5534f81083594a8de84",
        "hash.invalid_selection_v1",
    )
    report_text = report_path.read_text(encoding="utf-8")
    audit.require("long_prefix_screen_negative_v2" in report_text,
                  "report.verdict")
    return {
        "kind": "rsiv_moe_p1b_independent_verification",
        "verdict": (
            "p1b_verification_passed_with_declared_warning"
            if not audit.failures else "p1b_verification_failed"
        ),
        "all_required_checks_pass": not audit.failures,
        "checks": audit.checks,
        "passed": audit.passed,
        "failed": len(audit.failures),
        "warnings": len(audit.warnings),
        "failure_messages": audit.failures,
        "warning_messages": audit.warnings,
        "artifacts": {
            "capture_sha256": capture_hash,
            "capture_report_sha256": sha256_file(capture_report),
            "selection_v2_sha256": selection_hash,
            "invalid_selection_v1_sha256": sha256_file(invalid_selection),
            "result_sha256": sha256_file(result_path),
            "report_sha256": sha256_file(report_path),
            "preregistration_sha256": sha256_file(preregistration),
            "control_addendum_sha256": sha256_file(addendum),
        },
        "confirmed_result": {
            "verdict": result["verdict"],
            "selected_rank": selected["rank_cap"],
            "selected_threshold": selected["threshold"],
            "validation_double_fast_fraction": selection_data["selected_candidate"][
                "double_gate_fast_fraction"
            ],
            "test_double_fast_fraction": selected["double_gate_fast_fraction"],
            "test_cold_byte_reduction": selected["cold_byte_reduction"],
            "test_rank128_threshold_0_1_double_fast_fraction": diagnostic[
                "double_gate_fast_fraction"
            ],
        },
    }


def render_p1b_verification(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# RSIV-MoE P1B onafhankelijke verificatie",
            "",
            f"**Verdict: `{payload['verdict']}`.**",
            "",
            f"- Controles: {payload['checks']}.",
            f"- Geslaagd: {payload['passed']}.",
            f"- Fouten: {payload['failed']}.",
            f"- Gedeclareerde waarschuwingen: {payload['warnings']}.",
            "",
            "De audit bevestigt de 1.024→128-splits, hashes, validationselectie, beide grids, koude-byteboekhouding, rank/count/opslagbounds en raw capturevormen. Het resultaat blijft `long_prefix_screen_negative_v2`; kwaliteit en runtime zijn niet gemeten.",
            "",
            "## Waarschuwing",
            "",
            payload["warning_messages"][0],
            "",
        ]
    )

