from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

from safetensors.torch import load_file

from moe_lab.fleq_moe.expert_quant import QuantizedProjection, output_metrics
from moe_lab.reporting import ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a: float, b: float, tolerance: float = 2e-6) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def candidate(tensors: dict, prefix: str) -> dict[str, QuantizedProjection]:
    return {
        name: QuantizedProjection(
            tensors[f"{prefix}_{name}_weight"].cuda(),
            tensors[f"{prefix}_{name}_scales"].cuda(),
        )
        for name in ("gate", "up", "down")
    }


def verify(*, recompute_anchors: bool = True) -> dict:
    # Imported here to keep the verifier's arithmetic and artifact walk
    # independent while reusing only the frozen checkpoint/capture loaders.
    runner_path = ROOT / "scripts/fleq_moe/run_p1_expert.py"
    spec = importlib.util.spec_from_file_location("fleq_p1_frozen_runner", runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen runner {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    load_original, split_rows = runner.load_original, runner.split_rows

    result_path = ROOT / "reports/fleq_moe/p1_smoke_result.json"
    lock_path = ROOT / "reports/fleq_moe/p1_smoke_expert_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}

    checks["verdict_is_smoke_negative"] = result["verdict"] == "smoke_negative"
    checks["p2_is_not_authorized"] = result["p2_authorized"] is False
    checks["no_eureka_claim"] = result["claim_boundaries"]["eureka"] is False
    checks["all_execution_controls_pass"] = (
        result["all_required_controls_pass"] is True
        and all(result["controls"].values())
    )
    checks["two_bit_gate_fails"] = result["two_bit_gate_pass"] is False
    checks["selection_lock_hash_matches"] = (
        result["selection_lock_sha256"] == sha256(lock_path)
    )
    checks["all_three_addenda_hash_match"] = len(result["protocol_addenda_sha256"]) == 3 and all(
        sha256(ROOT / "reports/fleq_moe" / name) == digest
        for name, digest in result["protocol_addenda_sha256"].items()
    )

    selected_match = True
    formulas_match = True
    artifact_hashes_match = True
    report_hashes_match = True
    code_ranges_match = True
    primary_improved = 0
    primary_p95_regressions = 0
    for layer_text, layer_lock in lock["layers"].items():
        selected_match &= (
            result["layers"][layer_text]["selected_experts"]
            == layer_lock["selected_experts"]
        )
        layer_improvements = []
        for expert in layer_lock["selected_experts"]:
            for kind, report_dir, baseline, gsq, improvement_key in (
                ("2bit", "p1_experts", "gptq_2bit", "gsq_2bit", "heldout_gsq_improvement_over_gptq"),
                ("ternary", "p1_ternary_experts", "rtn_ternary", "gsq_ternary", "heldout_gsq_improvement_over_rtn"),
            ):
                report_path = ROOT / "reports/fleq_moe" / report_dir / f"layer_{int(layer_text):02d}_expert_{expert:03d}.json"
                row = json.loads(report_path.read_text(encoding="utf-8"))
                key = f"layer_{int(layer_text):02d}_expert_{expert:03d}_{kind}"
                manifest = result["artifacts"][key]
                report_hashes_match &= sha256(report_path) == manifest["report_sha256"]
                artifact_path = ROOT / row["artifact"]
                artifact_hashes_match &= (
                    sha256(artifact_path) == row["artifact_sha256"] == manifest["artifact_sha256"]
                )
                base_mse = row["metrics"][baseline]["heldout"]["router_weighted_relative_mse"]
                gsq_mse = row["metrics"][gsq]["heldout"]["router_weighted_relative_mse"]
                calculated = (base_mse - gsq_mse) / max(base_mse, 1e-30)
                formulas_match &= close(calculated, row[improvement_key], 1e-12)
                code_ranges_match &= all(
                    projection["codes_in_range"]
                    for method in row["code_summaries"].values()
                    for projection in method.values()
                )
                if kind == "2bit":
                    layer_improvements.append(calculated)
                    primary_improved += calculated > 0
                    primary_p95_regressions += (
                        row["metrics"][gsq]["heldout"]["relative_row_p95"]
                        > row["metrics"][baseline]["heldout"]["relative_row_p95"]
                    )
        summary = result["layers"][layer_text]["two_bit"]
        formulas_match &= close(summary["mean_improvement"], sum(layer_improvements) / 8, 1e-12)

    checks["selected_experts_match_lock"] = selected_match
    checks["all_32_report_hashes_match"] = report_hashes_match
    checks["all_32_artifact_hashes_match"] = artifact_hashes_match
    checks["all_improvement_formulas_match"] = formulas_match
    checks["all_code_ranges_match"] = code_ranges_match
    checks["primary_has_zero_of_16_improvements"] = primary_improved == 0
    checks["primary_has_16_of_16_p95_regressions"] = primary_p95_regressions == 16
    checks["two_bit_effective_bpp_is_2_125"] = close(
        result["storage"]["two_bit_codes_plus_bf16_group128_scales_bpp"], 2.125, 1e-15
    )
    checks["ternary_bounds_match"] = close(
        result["storage"]["ternary_ideal_log2_3_plus_bf16_group128_scales_bpp"],
        math.log2(3) + 16 / 128,
        1e-15,
    ) and close(
        result["storage"]["ternary_two_bit_pack_plus_bf16_group128_scales_bpp"],
        2.125,
        1e-15,
    )

    expected_attempts = {
        "reports/runs/fleq_moe/p1_ternary_attempt_004/layer_00_expert_046.safetensors": "5afb6789c8652d2258e91806d3a623cd4dda0e44da2fad854d8ee669d4534e6d",
        "reports/fleq_moe/p1_ternary_experts_attempt_004/layer_00_expert_046.json": "38e091c3d78f1befdbf2dbb07a86a489862b963e96f6296eabe61afb49af718d",
    }
    checks["attempt_004_preserved"] = all(sha256(ROOT / path) == digest for path, digest in expected_attempts.items())

    anchor_details = []
    if recompute_anchors:
        for layer, expert in ((0, 46), (47, 78)):
            _calibration, heldout = split_rows(layer, expert)
            original = load_original(layer, expert, __import__("torch").device("cuda"))
            for kind, run_dir, methods in (
                ("2bit", "p1", (("gptq", "gptq_2bit"), ("gsq", "gsq_2bit"))),
                ("ternary", "p1_ternary", (("rtn", "rtn_ternary"), ("gsq", "gsq_ternary"))),
            ):
                artifact_path = ROOT / "reports/runs/fleq_moe" / run_dir / f"layer_{layer:02d}_expert_{expert:03d}.safetensors"
                report_path = ROOT / "reports/fleq_moe" / ("p1_experts" if kind == "2bit" else "p1_ternary_experts") / f"layer_{layer:02d}_expert_{expert:03d}.json"
                stored = json.loads(report_path.read_text(encoding="utf-8"))
                tensors = load_file(artifact_path)
                for prefix, method_name in methods:
                    measured = output_metrics(heldout[0], heldout[2], original, candidate(tensors, prefix))
                    expected = stored["metrics"][method_name]["heldout"]
                    matched = all(
                        measured[name] == expected[name] if isinstance(expected[name], bool)
                        else close(measured[name], expected[name])
                        for name in expected
                    )
                    anchor_details.append({
                        "layer": layer, "expert": expert, "method": method_name,
                        "metrics_match": matched,
                    })
    checks["eight_anchor_metric_sets_recompute"] = (
        len(anchor_details) == 8 and all(item["metrics_match"] for item in anchor_details)
    ) if recompute_anchors else True

    return {
        "kind": "fleq_moe_p1_independent_verification",
        "result_sha256": sha256(result_path),
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "verification_pass": all(checks.values()),
        "anchors": anchor_details,
        "declared_warnings": [],
    }
