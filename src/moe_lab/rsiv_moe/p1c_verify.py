from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .p1_verify import Audit, _verify_metric, sha256_file


LAYERS = {"0", "23", "47"}
RANKS = (4, 8, 16, 32, 64, 128)
THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)
BF16_UNIT_ROUNDOFF = torch.finfo(torch.bfloat16).eps / 2
INVALID_VERIFICATION_JSON_SHA256 = (
    "79e193decedd954f43ab0a4c80cc13ccb2451af7c341ee5194860357f0d70eb1"
)
INVALID_VERIFICATION_MD_SHA256 = (
    "044a0b8bdc15b41bdef3c6b8b9fd31493b9de67f7f215643a92fa072180cc50b"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if int(row["rank_cap"]) <= 32]
    primary = [
        row
        for row in eligible
        if float(row["double_gate_fast_fraction"]) >= 0.92
        and float(row["cold_byte_reduction"]) >= 10.0
    ]
    if primary:
        return min(
            primary,
            key=lambda row: (
                int(row["rank_cap"]),
                float(row["threshold"]),
                -float(row["double_gate_fast_fraction"]),
                -float(row["cold_byte_reduction"]),
            ),
        )

    def score(row: dict[str, Any]) -> float:
        return min(
            float(row["double_gate_fast_fraction"]) / 0.92,
            float(row["cold_byte_reduction"]) / 10.0,
        )

    return min(
        eligible,
        key=lambda row: (
            -score(row),
            int(row["rank_cap"]),
            float(row["threshold"]),
            -float(row["double_gate_fast_fraction"]),
            -float(row["cold_byte_reduction"]),
        ),
    )


def _verify_grid(rows: list[dict[str, Any]], audit: Audit, label: str) -> None:
    audit.equal(len(rows), 42, f"{label}.candidate_count")
    audit.equal(
        {(int(row["rank_cap"]), float(row["threshold"])) for row in rows},
        {(rank, threshold) for rank in RANKS for threshold in THRESHOLDS},
        f"{label}.candidate_grid",
    )
    for index, row in enumerate(rows):
        _verify_metric(row["aggregate"], audit, f"{label}.{index}.aggregate")
        audit.equal(set(row["layers"]), LAYERS, f"{label}.{index}.layers")
        layer_invocations = 0
        layer_double_hits = 0.0
        layer_cold_sum = 0.0
        for layer, metric in row["layers"].items():
            _verify_metric(metric, audit, f"{label}.{index}.layer_{layer}")
            count = int(metric["invocations"])
            layer_invocations += count
            layer_double_hits += count * float(metric["double_gate_fast_fraction"])
            layer_cold_sum += count * float(metric["projected_cold_byte_fraction"])
        audit.equal(layer_invocations, int(row["aggregate"]["invocations"]),
                    f"{label}.{index}.invocations_sum")
        audit.close(layer_double_hits / layer_invocations,
                    float(row["double_gate_fast_fraction"]),
                    f"{label}.{index}.fast_aggregate")
        audit.close(layer_cold_sum / layer_invocations,
                    float(row["aggregate"]["projected_cold_byte_fraction"]),
                    f"{label}.{index}.cold_aggregate")


def _verify_census(census: dict[str, Any], audit: Audit, label: str) -> None:
    audit.equal(set(census), LAYERS, f"{label}.layers")
    for layer, payload in census.items():
        audit.equal(len(payload["contexts"]), 2, f"{label}.{layer}.contexts")
        for context in payload["contexts"]:
            suffix = f"{label}.{layer}.{context['context']}"
            experts = context["experts"]
            audit.equal(len(experts), 128, f"{suffix}.experts")
            counts = [int(row["prefix_count"]) for row in experts]
            x_ranks = [int(row["input_rank"]) for row in experts]
            z_ranks = [int(row["intermediate_rank"]) for row in experts]
            audit.equal(sum(counts), 1024 * 8, f"{suffix}.count_sum")
            audit.equal(context["expert_invocations"], sum(counts), f"{suffix}.invocations")
            audit.require(all(rank <= count for rank, count in zip(x_ranks, counts)),
                          f"{suffix}.x_rank_bound")
            audit.require(all(rank <= count for rank, count in zip(z_ranks, counts)),
                          f"{suffix}.z_rank_bound")
            elements = (2048 + 2 * 768) * sum(x_ranks) + (768 + 2048) * sum(z_ranks)
            bound = (2 * 2048 + 3 * 768) * 1024 * 8
            audit.equal(context["full_rank_image_elements"], elements, f"{suffix}.elements")
            audit.equal(context["bound_elements"], bound, f"{suffix}.bound")
            audit.close(context["bound_utilization"], elements / bound,
                        f"{suffix}.utilization")
            for checkpoint in (128, 512, 1024):
                growth = context["growth_checkpoints"][str(checkpoint)]
                observations = sum(
                    int(row["rank_growth"][str(checkpoint)]["observations"])
                    for row in experts
                )
                x_sum = sum(
                    int(row["rank_growth"][str(checkpoint)]["input_rank"])
                    for row in experts
                )
                z_sum = sum(
                    int(row["rank_growth"][str(checkpoint)]["intermediate_rank"])
                    for row in experts
                )
                audit.equal(observations, checkpoint * 8, f"{suffix}.growth_{checkpoint}.observations")
                audit.equal(growth["observations"], observations, f"{suffix}.growth_{checkpoint}.stored_observations")
                audit.equal(growth["input_rank_sum"], x_sum, f"{suffix}.growth_{checkpoint}.x_sum")
                audit.equal(growth["intermediate_rank_sum"], z_sum, f"{suffix}.growth_{checkpoint}.z_sum")
                audit.require(x_sum <= observations and z_sum <= observations,
                              f"{suffix}.growth_{checkpoint}.rank_bounds")
            audit.require(all(context["controls"].values()), f"{suffix}.controls")
        audit.equal(payload["all_required_controls_pass"], True, f"{label}.{layer}.all_controls")


def _verify_capture(path: Path, split: str, audit: Audit) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        audit.equal(metadata.get("split"), split, f"capture.{split}.metadata_split")
        audit.equal(metadata.get("prefix_tokens"), "1024", f"capture.{split}.prefix")
        audit.equal(metadata.get("future_tokens"), "128", f"capture.{split}.future")
        audit.equal(tuple(handle.get_slice("input_ids").get_shape()), (2, 1152),
                    f"capture.{split}.input_ids_shape")
        for layer in (0, 23, 47):
            prefix = f"layer_{layer:02d}"
            expected = {
                f"{prefix}_moe_input": (2304, 2048),
                f"{prefix}_router_ids": (2304, 8),
                f"{prefix}_router_weights": (2304, 8),
                f"{prefix}_intermediate_z": (18432, 768),
            }
            for name, shape in expected.items():
                audit.equal(tuple(handle.get_slice(name).get_shape()), shape,
                            f"capture.{split}.{name}.shape")
            ids = handle.get_tensor(f"{prefix}_router_ids").long()
            weights = handle.get_tensor(f"{prefix}_router_weights").float()
            x = handle.get_tensor(f"{prefix}_moe_input").float()
            z = handle.get_tensor(f"{prefix}_intermediate_z").float()
            audit.require(bool(((ids >= 0) & (ids < 128)).all()),
                          f"capture.{split}.{layer}.ids")
            audit.require(bool(torch.isfinite(weights).all()),
                          f"capture.{split}.{layer}.weights_finite")
            audit.require(bool(torch.isfinite(x).all()), f"capture.{split}.{layer}.x_finite")
            audit.require(bool(torch.isfinite(z).all()), f"capture.{split}.{layer}.z_finite")
            # Qwen normalizes routing weights in FP32 and stores the selected
            # values in BF16. For non-negative weights summing to one before
            # the cast, the total first-order cast error is bounded by BF16
            # unit roundoff u = eps/2; it does not grow as top_k*u.
            audit.require(bool(torch.allclose(
                weights.sum(1),
                torch.ones(2304),
                atol=BF16_UNIT_ROUNDOFF,
                rtol=0,
            )), f"capture.{split}.{layer}.weights_normalized")


def build_p1c_verification(root: Path) -> dict[str, Any]:
    root = root.resolve()
    paths = {
        "acquisition": root / "reports/rsiv_moe/qwen_checkpoint_acquisition.json",
        "preregistration": root / "reports/rsiv_moe/P1C_QWEN3_30B_A3B_PREREGISTRATION.md",
        "validation_capture": root / "reports/runs/rsiv_moe/p1c_qwen_validation.safetensors",
        "validation_capture_report": root / "reports/rsiv_moe/p1c_qwen_validation_capture.json",
        "selection": root / "reports/rsiv_moe/p1c_qwen_validation_selection.json",
        "test_capture": root / "reports/runs/rsiv_moe/p1c_qwen_test.safetensors",
        "test_capture_report": root / "reports/rsiv_moe/p1c_qwen_test_capture.json",
        "result": root / "reports/rsiv_moe/p1c_qwen_result.json",
        "report": root / "reports/rsiv_moe/P1C_QWEN3_30B_A3B.md",
        "invalid_verification_json": root / "reports/rsiv_moe/p1c_verification_attempt_001.json",
        "invalid_verification_md": root / "reports/rsiv_moe/P1C_VERIFICATION_ATTEMPT_001.md",
        "verifier_addendum": root / "reports/rsiv_moe/P1C_VERIFIER_ADDENDUM_001.md",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    acquisition = _load(paths["acquisition"])
    validation_report = _load(paths["validation_capture_report"])
    selection = _load(paths["selection"])
    test_report = _load(paths["test_capture_report"])
    result = _load(paths["result"])
    audit = Audit()

    hashes = {name: sha256_file(path) for name, path in paths.items()}
    audit.equal(
        hashes["invalid_verification_json"],
        INVALID_VERIFICATION_JSON_SHA256,
        "hash.invalid_verification_json",
    )
    audit.equal(
        hashes["invalid_verification_md"],
        INVALID_VERIFICATION_MD_SHA256,
        "hash.invalid_verification_md",
    )
    audit.equal(acquisition["status"], "complete_verified", "acquisition.status")
    audit.equal(acquisition["revision"], "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
                "acquisition.revision")
    audit.equal(acquisition["local_weight_bytes"], 61_066_575_648, "acquisition.bytes")
    audit.equal(acquisition["local_sha256_verified"], True, "acquisition.sha_verified")
    audit.equal(hashes["validation_capture"], validation_report["capture_sha256"],
                "hash.validation_capture_report")
    audit.equal(hashes["validation_capture"], selection["validation_capture_sha256"],
                "hash.validation_selection")
    audit.equal(hashes["selection"], test_report["validation_lock_sha256"],
                "hash.test_lock")
    audit.equal(hashes["selection"], result["selection_sha256"], "hash.result_lock")
    audit.equal(hashes["test_capture"], test_report["capture_sha256"],
                "hash.test_capture_report")
    audit.equal(hashes["test_capture"], result["test_capture_sha256"],
                "hash.test_result")
    audit.equal(selection["status"], "locked", "selection.status")
    audit.equal(selection["test_data_opened"], False, "selection.test_closed")
    audit.equal(selection["test_open_authorized"], True, "selection.test_authorized")
    audit.equal(result["test"]["opened_once"], True, "result.test_once")

    _verify_grid(selection["validation_grid"], audit, "validation_grid")
    _verify_grid(result["test"]["post_lock_grid"], audit, "test_grid")
    _verify_census(selection["rank_census"], audit, "validation_census")
    _verify_census(result["test"]["rank_census"], audit, "test_census")
    _verify_capture(paths["validation_capture"], "validation", audit)
    _verify_capture(paths["test_capture"], "test", audit)

    recomputed = _select(selection["validation_grid"])
    stored = selection["selected_candidate"]
    for key in ("rank_cap", "threshold", "double_gate_fast_fraction", "cold_byte_reduction"):
        audit.close(float(recomputed[key]), float(stored[key]), f"selection.{key}")
    expected_kind = (
        "primary_gate_pass"
        if recomputed["double_gate_fast_fraction"] >= 0.92 and recomputed["cold_byte_reduction"] >= 10.0
        else "diagnostic_validation_failure"
    )
    audit.equal(stored["selection_kind"], expected_kind, "selection.kind")
    selected_test = result["test"]["selected_candidate"]
    audit.equal(selected_test["rank_cap"], stored["rank_cap"], "test.selected_rank")
    audit.close(selected_test["threshold"], stored["threshold"], "test.selected_threshold")

    validation_rank32 = result["validation_rank32_threshold_0_10"]
    test_rank32 = result["test"]["rank32_threshold_0_10"]
    primary_pass = (
        stored["selection_kind"] == "primary_gate_pass"
        and selected_test["double_gate_fast_fraction"] >= 0.92
        and selected_test["cold_byte_reduction"] >= 10.0
    )
    hard_failure = (
        validation_rank32["double_gate_fast_fraction"] < 0.80
        and test_rank32["double_gate_fast_fraction"] < 0.80
    )
    expected_verdict = (
        "higher_e_screen_positive"
        if primary_pass
        else "falsified_rank_working_set"
        if hard_failure
        else "higher_e_screen_negative_nonterminal"
    )
    audit.equal(result["verdict"], expected_verdict, "result.verdict")
    audit.equal(result["controls"]["all_required_controls_pass"], True,
                "result.controls")
    audit.equal(result["claim_boundaries"]["runtime"], "not measured", "claim.runtime")
    audit.equal(result["claim_boundaries"]["eureka"], False, "claim.eureka")
    report_text = paths["report"].read_text(encoding="utf-8")
    audit.require(result["verdict"] in report_text, "report.verdict")
    audit.require("geen Eureka" in report_text, "report.no_eureka")
    audit.warnings.append(
        "Verifierpoging 001 gebruikte atol=0,002 voor de som van naar BF16 "
        "teruggecastte routergewichten. Dat was strenger dan de analytische "
        "BF16-eenheidsafronding eps/2=0,00390625. Addendum 001 vervangt alleen "
        "deze extra verifiercontrole; captures, selectie, grids, gates en "
        "onderzoeksresultaat zijn ongewijzigd."
    )

    return {
        "kind": "rsiv_moe_p1c_independent_verification",
        "verdict": (
            "p1c_verification_passed_with_declared_warning"
            if not audit.failures else "p1c_verification_failed"
        ),
        "all_required_checks_pass": not audit.failures,
        "checks": audit.checks,
        "passed": audit.passed,
        "failed": len(audit.failures),
        "warnings": len(audit.warnings),
        "failure_messages": audit.failures,
        "warning_messages": audit.warnings,
        "artifacts": hashes,
        "confirmed_result": {
            "verdict": result["verdict"],
            "selected_rank": stored["rank_cap"],
            "selected_threshold": stored["threshold"],
            "validation_double_fast_fraction": stored["double_gate_fast_fraction"],
            "test_double_fast_fraction": selected_test["double_gate_fast_fraction"],
            "test_cold_byte_reduction": selected_test["cold_byte_reduction"],
            "validation_rank32_threshold_0_1_double_fast_fraction": validation_rank32[
                "double_gate_fast_fraction"
            ],
            "test_rank32_threshold_0_1_double_fast_fraction": test_rank32[
                "double_gate_fast_fraction"
            ],
        },
    }


def render_p1c_verification(payload: dict[str, Any]) -> str:
    confirmed = payload["confirmed_result"]
    return "\n".join([
        "# RSIV-MoE P1C onafhankelijke verificatie",
        "",
        f"**Verdict: `{payload['verdict']}`.**",
        "",
        f"- Controles: {payload['checks']}.",
        f"- Geslaagd: {payload['passed']}.",
        f"- Fouten: {payload['failed']}.",
        f"- Gedeclareerde waarschuwingen: {payload['warnings']}.",
        "",
        f"Bevestigd onderzoeksverdict: `{confirmed['verdict']}`. De audit "
        "controleert checkpointidentiteit, capturehashes en -vormen, "
        "validation→test-slot, beide grids, rankgroei, opslagbounds, "
        "selectieregel en de hard-falsificatielogica.",
        "",
        "## Waarschuwing",
        "",
        payload["warning_messages"][0],
        "",
    ])
