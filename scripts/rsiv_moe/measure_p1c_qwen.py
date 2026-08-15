from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from measure_routed_subspace_rank import metric_summary
from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.subspace import (
    fit_origin_subspace,
    image_storage_elements,
    relative_residual_ratio,
)


LAYERS = (0, 23, 47)
RANKS = (4, 8, 16, 32, 64, 128)
THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)
GROWTH_CHECKPOINTS = (128, 512, 1024)
EXPERTS = 128
TOP_K = 8
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
CONTEXTS = 2
PREFIX_TOKENS = 1024
FUTURE_TOKENS = 128
CONTEXT_TOKENS = PREFIX_TOKENS + FUTURE_TOKENS
PREREGISTRATION = ROOT / "reports/rsiv_moe/P1C_QWEN3_30B_A3B_PREREGISTRATION.md"
ACQUISITION = ROOT / "reports/rsiv_moe/qwen_checkpoint_acquisition.json"
VALIDATION_CAPTURE = ROOT / "reports/runs/rsiv_moe/p1c_qwen_validation.safetensors"
VALIDATION_CAPTURE_REPORT = ROOT / "reports/rsiv_moe/p1c_qwen_validation_capture.json"
SELECTION = ROOT / "reports/rsiv_moe/p1c_qwen_validation_selection.json"
TEST_CAPTURE = ROOT / "reports/runs/rsiv_moe/p1c_qwen_test.safetensors"
TEST_CAPTURE_REPORT = ROOT / "reports/rsiv_moe/p1c_qwen_test_capture.json"
RESULT = ROOT / "reports/rsiv_moe/p1c_qwen_result.json"
REPORT = ROOT / "reports/rsiv_moe/P1C_QWEN3_30B_A3B.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure preregistered P1C Qwen rank screen.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--select-validation", action="store_true")
    group.add_argument("--open-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 2**20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_capture(path: Path, layer: int) -> dict[str, torch.Tensor]:
    prefix = f"layer_{layer:02d}"
    with safe_open(path, framework="pt", device="cpu") as handle:
        x = handle.get_tensor(f"{prefix}_moe_input")
        ids = handle.get_tensor(f"{prefix}_router_ids").long()
        weights = handle.get_tensor(f"{prefix}_router_weights").float()
        z_flat = handle.get_tensor(f"{prefix}_intermediate_z")
    expected_tokens = CONTEXTS * CONTEXT_TOKENS
    if x.shape != (expected_tokens, HIDDEN_SIZE):
        raise RuntimeError(f"layer {layer} x shape {tuple(x.shape)}")
    if ids.shape != (expected_tokens, TOP_K) or weights.shape != ids.shape:
        raise RuntimeError(f"layer {layer} route shape mismatch")
    if z_flat.shape != (expected_tokens * TOP_K, INTERMEDIATE_SIZE):
        raise RuntimeError(f"layer {layer} z shape {tuple(z_flat.shape)}")
    return {
        "x": x,
        "ids": ids,
        "weights": weights,
        "z": z_flat.reshape(expected_tokens, TOP_K, INTERMEDIATE_SIZE),
    }


def slice_tokens(data: dict[str, torch.Tensor], start: int, stop: int):
    return {key: value[start:stop] for key, value in data.items()}


def expert_matrices(data: dict[str, torch.Tensor], expert: int):
    positions = (data["ids"] == expert).nonzero(as_tuple=False)
    token, slot = positions[:, 0], positions[:, 1]
    return (
        data["x"][token],
        data["z"][token, slot],
        data["weights"][token, slot],
        token * TOP_K + slot,
    )


def weighted_summary(values: list[int], counts: list[int]) -> dict[str, float]:
    expanded = torch.tensor(
        [value for value, count in zip(values, counts) for _ in range(count)],
        dtype=torch.float64,
    )
    if expanded.numel() == 0:
        return {key: 0.0 for key in ("mean", "p50", "p95", "p99", "maximum")}
    return {
        "mean": float(expanded.mean()),
        "p50": float(torch.quantile(expanded, 0.50)),
        "p95": float(torch.quantile(expanded, 0.95)),
        "p99": float(torch.quantile(expanded, 0.99)),
        "maximum": float(expanded.max()),
    }


def analyze_layer(layer: int, split: str, data: dict[str, torch.Tensor]):
    invocations = CONTEXTS * FUTURE_TOKENS * TOP_K
    residuals = {
        rank: {
            "x": torch.full((invocations,), float("inf"), dtype=torch.float64),
            "z": torch.full((invocations,), float("inf"), dtype=torch.float64),
            "weights": torch.empty(invocations, dtype=torch.float64),
        }
        for rank in RANKS
    }
    contexts: list[dict[str, Any]] = []
    output_offset = 0
    for context in range(CONTEXTS):
        start = context * CONTEXT_TOKENS
        full_context = slice_tokens(data, start, start + CONTEXT_TOKENS)
        prefix = slice_tokens(full_context, 0, PREFIX_TOKENS)
        future = slice_tokens(full_context, PREFIX_TOKENS, CONTEXT_TOKENS)
        block_invocations = FUTURE_TOKENS * TOP_K
        for rank in RANKS:
            residuals[rank]["weights"][output_offset : output_offset + block_invocations] = future[
                "weights"
            ].reshape(-1).double()

        counts: list[int] = []
        x_ranks: list[int] = []
        z_ranks: list[int] = []
        expert_rows: list[dict[str, Any]] = []
        maximum_x_reconstruction = 0.0
        maximum_z_reconstruction = 0.0
        for expert in range(EXPERTS):
            prefix_x, prefix_z, prefix_weights, _prefix_flat = expert_matrices(prefix, expert)
            future_x, future_z, _future_weights, future_flat = expert_matrices(future, expert)
            x_fit = fit_origin_subspace(prefix_x)
            z_fit = fit_origin_subspace(prefix_z)
            count = int(prefix_x.shape[0])
            counts.append(count)
            x_ranks.append(x_fit.stored_rank)
            z_ranks.append(z_fit.stored_rank)
            maximum_x_reconstruction = max(maximum_x_reconstruction, x_fit.reconstruction_relative_l2)
            maximum_z_reconstruction = max(maximum_z_reconstruction, z_fit.reconstruction_relative_l2)

            growth: dict[str, Any] = {}
            for checkpoint in GROWTH_CHECKPOINTS:
                if checkpoint == PREFIX_TOKENS:
                    growth[str(checkpoint)] = {
                        "observations": count,
                        "input_rank": x_fit.stored_rank,
                        "intermediate_rank": z_fit.stored_rank,
                    }
                    continue
                checkpoint_data = slice_tokens(full_context, 0, checkpoint)
                checkpoint_x, checkpoint_z, _w, _flat = expert_matrices(checkpoint_data, expert)
                growth[str(checkpoint)] = {
                    "observations": int(checkpoint_x.shape[0]),
                    "input_rank": fit_origin_subspace(checkpoint_x).stored_rank,
                    "intermediate_rank": fit_origin_subspace(checkpoint_z).stored_rank,
                }
            expert_rows.append(
                {
                    "expert": expert,
                    "prefix_count": count,
                    "router_mass": float(prefix_weights.double().sum()),
                    "input_rank": x_fit.stored_rank,
                    "intermediate_rank": z_fit.stored_rank,
                    "input_effective_rank": x_fit.effective_rank,
                    "intermediate_effective_rank": z_fit.effective_rank,
                    "rank_growth": growth,
                }
            )
            destination = future_flat + output_offset
            for rank in RANKS:
                q = x_fit.basis[:, : min(rank, x_fit.basis.shape[1])]
                p = z_fit.basis[:, : min(rank, z_fit.basis.shape[1])]
                residuals[rank]["x"][destination] = relative_residual_ratio(future_x, q)
                residuals[rank]["z"][destination] = relative_residual_ratio(future_z, p)

        expected = PREFIX_TOKENS * TOP_K
        elements = image_storage_elements(HIDDEN_SIZE, INTERMEDIATE_SIZE, x_ranks, z_ranks)
        bound = (2 * HIDDEN_SIZE + 3 * INTERMEDIATE_SIZE) * expected
        checkpoint_summaries = {}
        for checkpoint in GROWTH_CHECKPOINTS:
            checkpoint_counts = [row["rank_growth"][str(checkpoint)]["observations"] for row in expert_rows]
            checkpoint_x_ranks = [row["rank_growth"][str(checkpoint)]["input_rank"] for row in expert_rows]
            checkpoint_z_ranks = [row["rank_growth"][str(checkpoint)]["intermediate_rank"] for row in expert_rows]
            checkpoint_summaries[str(checkpoint)] = {
                "observations": sum(checkpoint_counts),
                "input_rank_sum": sum(checkpoint_x_ranks),
                "intermediate_rank_sum": sum(checkpoint_z_ranks),
                "input_rank_to_observation_fraction": sum(checkpoint_x_ranks) / max(1, sum(checkpoint_counts)),
                "intermediate_rank_to_observation_fraction": sum(checkpoint_z_ranks) / max(1, sum(checkpoint_counts)),
            }
        controls = {
            "count_sum_exact": sum(counts) == expected,
            "input_ranks_le_counts": all(rank <= count for rank, count in zip(x_ranks, counts)),
            "intermediate_ranks_le_counts": all(rank <= count for rank, count in zip(z_ranks, counts)),
            "storage_bound_pass": elements <= bound,
            "full_rank_input_reconstruction_pass": maximum_x_reconstruction <= 2e-12,
            "full_rank_intermediate_reconstruction_pass": maximum_z_reconstruction <= 2e-12,
        }
        contexts.append(
            {
                "context": context,
                "expert_invocations": sum(counts),
                "minimum_expert_count": min(counts),
                "maximum_expert_count": max(counts),
                "rare_experts_count_lt_4": sum(count < 4 for count in counts),
                "unseen_experts": sum(count == 0 for count in counts),
                "input_rank_invocation_weighted": weighted_summary(x_ranks, counts),
                "intermediate_rank_invocation_weighted": weighted_summary(z_ranks, counts),
                "full_rank_image_elements": elements,
                "bound_elements": bound,
                "bound_utilization": elements / bound,
                "maximum_full_rank_input_reconstruction_relative_l2": maximum_x_reconstruction,
                "maximum_full_rank_intermediate_reconstruction_relative_l2": maximum_z_reconstruction,
                "growth_checkpoints": checkpoint_summaries,
                "experts": expert_rows,
                "controls": controls,
            }
        )
        output_offset += block_invocations
    if output_offset != invocations:
        raise RuntimeError("future residual alignment failed")
    return residuals, {
        "layer": layer,
        "split": split,
        "contexts": contexts,
        "all_required_controls_pass": all(all(row["controls"].values()) for row in contexts),
    }


def make_grid(sources: dict[int, dict[int, dict[str, torch.Tensor]]]):
    rows = []
    for rank in RANKS:
        x = torch.cat([sources[layer][rank]["x"] for layer in LAYERS])
        z = torch.cat([sources[layer][rank]["z"] for layer in LAYERS])
        weights = torch.cat([sources[layer][rank]["weights"] for layer in LAYERS])
        for threshold in THRESHOLDS:
            aggregate = metric_summary(x, z, weights, threshold)
            reduction = aggregate["projected_routed_cold_byte_reduction"]
            rows.append(
                {
                    "rank_cap": rank,
                    "threshold": threshold,
                    "double_gate_fast_fraction": aggregate["double_gate_fast_fraction"],
                    "cold_byte_reduction": reduction if reduction is not None else 1e300,
                    "aggregate": aggregate,
                    "layers": {
                        str(layer): metric_summary(
                            sources[layer][rank]["x"],
                            sources[layer][rank]["z"],
                            sources[layer][rank]["weights"],
                            threshold,
                        )
                        for layer in LAYERS
                    },
                }
            )
    return rows


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["rank_cap"] <= 32]
    primary = [
        row
        for row in eligible
        if row["double_gate_fast_fraction"] >= 0.92 and row["cold_byte_reduction"] >= 10.0
    ]
    if primary:
        row = min(
            primary,
            key=lambda item: (
                item["rank_cap"],
                item["threshold"],
                -item["double_gate_fast_fraction"],
                -item["cold_byte_reduction"],
            ),
        )
        return {**row, "selection_kind": "primary_gate_pass"}

    def bottleneck(item: dict[str, Any]) -> float:
        return min(item["double_gate_fast_fraction"] / 0.92, item["cold_byte_reduction"] / 10.0)

    row = min(
        eligible,
        key=lambda item: (
            -bottleneck(item),
            item["rank_cap"],
            item["threshold"],
            -item["double_gate_fast_fraction"],
            -item["cold_byte_reduction"],
        ),
    )
    return {**row, "selection_kind": "diagnostic_validation_failure", "normalized_bottleneck_score": bottleneck(row)}


def capture_controls(report: dict[str, Any]) -> bool:
    return all(
        row["route_ids_exact"]
        and row["router_weight_maximum_absolute_error"] == 0.0
        and row["router_logits_maximum_absolute_error"] == 0.0
        and row["sum_expert_invocations"] == row["expected_expert_invocations"]
        and row["finite_x"]
        and row["finite_z"]
        for row in report["controls"].values()
    )


def analyze(path: Path, split: str):
    sources = {}
    census = {}
    for layer in LAYERS:
        sources[layer], census[str(layer)] = analyze_layer(layer, split, load_capture(path, layer))
        print(f"P1C {split}: analyzed layer {layer}", flush=True)
    return make_grid(sources), census


def validation_phase() -> None:
    if SELECTION.exists():
        raise FileExistsError("P1C validation lock already exists")
    for path in (PREREGISTRATION, ACQUISITION, VALIDATION_CAPTURE, VALIDATION_CAPTURE_REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    capture_sha = sha256_file(VALIDATION_CAPTURE)
    report = json.loads(VALIDATION_CAPTURE_REPORT.read_text(encoding="utf-8"))
    if report["capture_sha256"] != capture_sha or report["split"] != "validation":
        raise RuntimeError("P1C validation capture identity failed")
    grid, census = analyze(VALIDATION_CAPTURE, "validation")
    selected = select_candidate(grid)
    controls = {
        "capture_controls_pass": capture_controls(report),
        "rank_controls_pass": all(row["all_required_controls_pass"] for row in census.values()),
        "checkpoint_acquisition_verified": json.loads(ACQUISITION.read_text(encoding="utf-8"))["status"] == "complete_verified",
    }
    valid = all(controls.values())
    payload = {
        "kind": "rsiv_moe_p1c_qwen_validation_lock",
        "status": "locked" if valid else "invalid_controls_failed",
        "test_open_authorized": valid,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_opened": False,
        "validation_capture_sha256": capture_sha,
        "validation_capture_report_sha256": sha256_file(VALIDATION_CAPTURE_REPORT),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "acquisition_sha256": sha256_file(ACQUISITION),
        "selected_candidate": selected,
        "validation_grid": grid,
        "rank_census": census,
        "controls": controls,
    }
    write_json_once(SELECTION, payload)
    print(json.dumps({"status": payload["status"], "selection_sha256": sha256_file(SELECTION), "selected": selected}, indent=2))


def render(result: dict[str, Any]) -> str:
    validation = result["selection"]["selected_candidate"]
    test = result["test"]["selected_candidate"]
    rank32 = result["test"]["rank32_threshold_0_10"]
    rank128 = result["test"]["rank128_threshold_0_10"]
    return "\n".join(
        [
            "# RSIV-MoE P1C — Qwen3-30B-A3B hogere-E rankscreen",
            "",
            f"**Verdict: `{result['verdict']}`.**",
            "",
            result["verdict_explanation"],
            "",
            "## Vergrendelde kandidaat",
            "",
            f"- Rank/threshold: `{validation['rank_cap']}` / `{validation['threshold']}`.",
            f"- Validation: `{validation['double_gate_fast_fraction']:.3%}` double-fast, `{validation['cold_byte_reduction']:.3f}×` cold-byte reductie.",
            f"- Test: `{test['double_gate_fast_fraction']:.3%}` double-fast, `{test['cold_byte_reduction']:.3f}×` cold-byte reductie.",
            "",
            "## Hard-falsificatiediagnostiek",
            "",
            f"- Rank 32 / threshold 0,10: validation `{result['validation_rank32_threshold_0_10']['double_gate_fast_fraction']:.3%}`, test `{rank32['double_gate_fast_fraction']:.3%}`.",
            f"- Rank 128 / threshold 0,10 test: `{rank128['double_gate_fast_fraction']:.3%}`, `{rank128['cold_byte_reduction']:.3f}×`.",
            "",
            "## Controles en claimgrens",
            "",
            f"Alle vereiste controls: `{result['controls']['all_required_controls_pass']}`.",
            "Dit is een teacher-state rank/page-faultscreen. Kwaliteit, echte cold I/O, latency en decode-snelheid zijn niet gemeten; dit is geen Eureka.",
            "",
            "## Volgende actie",
            "",
            result["next_action"],
            "",
        ]
    )


def test_phase() -> None:
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite P1C result")
    for path in (SELECTION, TEST_CAPTURE, TEST_CAPTURE_REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection_sha = sha256_file(SELECTION)
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection["status"] != "locked" or selection["test_open_authorized"] is not True:
        raise RuntimeError("P1C validation lock does not authorize test")
    capture_sha = sha256_file(TEST_CAPTURE)
    capture_report = json.loads(TEST_CAPTURE_REPORT.read_text(encoding="utf-8"))
    if capture_report["capture_sha256"] != capture_sha or capture_report["split"] != "test":
        raise RuntimeError("P1C test capture identity failed")
    if capture_report["validation_lock_sha256"] != selection_sha:
        raise RuntimeError("P1C test capture used a different validation lock")
    started = time.perf_counter()
    test_grid, census = analyze(TEST_CAPTURE, "test")
    rank = selection["selected_candidate"]["rank_cap"]
    threshold = selection["selected_candidate"]["threshold"]
    selected = next(row for row in test_grid if row["rank_cap"] == rank and row["threshold"] == threshold)
    validation_rank32 = next(row for row in selection["validation_grid"] if row["rank_cap"] == 32 and row["threshold"] == 0.10)
    test_rank32 = next(row for row in test_grid if row["rank_cap"] == 32 and row["threshold"] == 0.10)
    test_rank128 = next(row for row in test_grid if row["rank_cap"] == 128 and row["threshold"] == 0.10)
    validation_pass = selection["selected_candidate"]["selection_kind"] == "primary_gate_pass"
    test_pass = selected["double_gate_fast_fraction"] >= 0.92 and selected["cold_byte_reduction"] >= 10.0
    hard_rank_failure = (
        validation_rank32["double_gate_fast_fraction"] < 0.80
        and test_rank32["double_gate_fast_fraction"] < 0.80
    )
    controls = {
        "validation_controls_pass": all(selection["controls"].values()),
        "test_capture_controls_pass": capture_controls(capture_report),
        "test_rank_controls_pass": all(row["all_required_controls_pass"] for row in census.values()),
    }
    all_controls = all(controls.values())
    if all_controls and validation_pass and test_pass:
        verdict = "higher_e_screen_positive"
        explanation = "De vooraf geselecteerde hogere-E kandidaat haalt op validation en test beide P1-gates."
        next_action = "Preregistreer P2 met echte A/B/C-operatorimages; maak nog geen Eureka- of runtimeclaim."
    elif all_controls and hard_rank_failure:
        verdict = "falsified_rank_working_set"
        explanation = "Zelfs rank 32 bij de ruimste geregistreerde threshold blijft op validation én test onder 80%; hard-falsificatieregel 1 is nu gereproduceerd op V2 en Qwen3."
        next_action = "Sluit RSIV_MOE_V1 en zoek alleen onder een nieuwe, onafhankelijk vooraf geregistreerde hypothese verder."
    elif all_controls:
        verdict = "higher_e_screen_negative_nonterminal"
        explanation = "De primaire P1C-gates falen, maar de vooraf geregistreerde harde rank-32-grens is niet op beide splits overschreden."
        next_action = "Rapporteer inconclusive en beslis vóór nieuwe data of één aanvullende mechanistische controle gerechtvaardigd is."
    else:
        verdict = "invalid"
        explanation = "Minstens één verplichte P1C-control faalde; de numerieke test is niet interpreteerbaar."
        next_action = "Diagnosticeer alleen de controlfout en open geen nieuwe evaluatie."
    payload = {
        "kind": "rsiv_moe_p1c_qwen_result",
        "status": "complete",
        "verdict": verdict,
        "verdict_explanation": explanation,
        "next_action": next_action,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "selection_sha256": selection_sha,
        "test_capture_sha256": capture_sha,
        "selection": selection,
        "validation_rank32_threshold_0_10": validation_rank32,
        "test": {
            "opened_once": True,
            "selected_candidate": selected,
            "selected_candidate_pass": test_pass,
            "rank32_threshold_0_10": test_rank32,
            "rank128_threshold_0_10": test_rank128,
            "post_lock_grid": test_grid,
            "rank_census": census,
        },
        "controls": {**controls, "all_required_controls_pass": all_controls},
        "claim_boundaries": {
            "quality": "not measured",
            "runtime": "not measured",
            "cold_bytes": "analytical optimistic packed-int4 miss projection",
            "eureka": False,
        },
    }
    write_json_once(RESULT, payload)
    REPORT.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "result_sha256": sha256_file(RESULT), "report_sha256": sha256_file(REPORT)}, indent=2))


if __name__ == "__main__":
    args = parse_args()
    if args.select_validation:
        validation_phase()
    else:
        test_phase()
