from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from measure_routed_subspace_rank import (
    environment,
    full_rank_operator_controls,
    metric_summary,
    sha256_file,
    write_json_once,
)
from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.subspace import (
    fit_origin_subspace,
    image_storage_elements,
    relative_residual_ratio,
    select_single_evaluation_candidate,
)


LAYERS = (1, 13, 26)
RANKS = (4, 8, 16, 32, 64, 128)
THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)
EXPERTS = 64
TOP_K = 6
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 1408
CONTEXTS_PER_SPLIT = 2
PREFIX_TOKENS = 1024
FUTURE_TOKENS = 128
CONTEXT_TOKENS = PREFIX_TOKENS + FUTURE_TOKENS
CAPTURE = ROOT / "reports/runs/rsiv_moe/p1b_v2_long_prefix.safetensors"
CAPTURE_REPORT = ROOT / "reports/rsiv_moe/p1b_v2_long_prefix_capture.json"
PREREGISTRATION = ROOT / "reports/rsiv_moe/P1B_V2_LONG_PREFIX_PREREGISTRATION.md"
UPSTREAM_P1 = ROOT / "reports/rsiv_moe/routed_subspace_rank.json"
SELECTION = ROOT / "reports/rsiv_moe/p1b_long_prefix_validation_selection_v2.json"
INVALID_SELECTION_V1 = ROOT / "reports/rsiv_moe/p1b_long_prefix_validation_selection.json"
CONTROL_ADDENDUM = ROOT / "reports/rsiv_moe/P1B_CONTROL_ADDENDUM_001.md"
RESULT = ROOT / "reports/rsiv_moe/p1b_long_prefix.json"
REPORT = ROOT / "reports/rsiv_moe/P1B_V2_LONG_PREFIX.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSIV P1B 1024->128 V2 census.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--select-validation", action="store_true")
    group.add_argument("--open-test", action="store_true")
    return parser.parse_args()


def offsets() -> dict[str, tuple[int, int]]:
    with safe_open(CAPTURE, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    raw = json.loads(metadata["split_offsets"])
    return {key: (int(value[0]), int(value[1])) for key, value in raw.items()}


def load_split(layer: int, split: str) -> dict[str, torch.Tensor]:
    if split not in {"validation", "test"}:
        raise ValueError("P1B has only validation and test")
    start, stop = offsets()[split]
    prefix = f"layer_{layer:02d}"
    with safe_open(CAPTURE, framework="pt", device="cpu") as handle:
        return {
            "x": handle.get_slice(f"{prefix}_moe_input")[start:stop],
            "ids": handle.get_slice(f"{prefix}_router_ids")[start:stop].long(),
            "weights": handle.get_slice(f"{prefix}_router_weights")[start:stop].float(),
            "z": handle.get_slice(f"{prefix}_intermediate_z")[start:stop],
        }


def subset(data: dict[str, torch.Tensor], selection: slice) -> dict[str, torch.Tensor]:
    return {key: value[selection] for key, value in data.items()}


def expert_matrices(
    data: dict[str, torch.Tensor], expert: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = (data["ids"] == expert).nonzero(as_tuple=False)
    token_ids, slots = positions[:, 0], positions[:, 1]
    return (
        data["x"][token_ids],
        data["z"][token_ids, slots],
        data["weights"][token_ids, slots],
        token_ids * TOP_K + slots,
    )


def prefix_only(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    parts: dict[str, list[torch.Tensor]] = {key: [] for key in data}
    for context in range(CONTEXTS_PER_SPLIT):
        start = context * CONTEXT_TOKENS
        for key, value in data.items():
            parts[key].append(value[start : start + PREFIX_TOKENS])
    return {key: torch.cat(values, dim=0) for key, values in parts.items()}


def context_census(layer: int, split: str, data: dict[str, torch.Tensor]) -> dict[str, Any]:
    context_rows: list[dict[str, Any]] = []
    all_controls = True
    for context in range(CONTEXTS_PER_SPLIT):
        start = context * CONTEXT_TOKENS
        prefix = subset(data, slice(start, start + PREFIX_TOKENS))
        counts: list[int] = []
        x_ranks: list[int] = []
        z_ranks: list[int] = []
        maximum_x_reconstruction = 0.0
        maximum_z_reconstruction = 0.0
        for expert in range(EXPERTS):
            x, z, _weights, _flat = expert_matrices(prefix, expert)
            x_fit = fit_origin_subspace(x)
            z_fit = fit_origin_subspace(z)
            count = int(x.shape[0])
            counts.append(count)
            x_ranks.append(x_fit.stored_rank)
            z_ranks.append(z_fit.stored_rank)
            maximum_x_reconstruction = max(
                maximum_x_reconstruction, x_fit.reconstruction_relative_l2
            )
            maximum_z_reconstruction = max(
                maximum_z_reconstruction, z_fit.reconstruction_relative_l2
            )
        count_sum = sum(counts)
        expected = PREFIX_TOKENS * TOP_K
        elements = image_storage_elements(
            HIDDEN_SIZE, INTERMEDIATE_SIZE, x_ranks, z_ranks
        )
        bound = (2 * HIDDEN_SIZE + 3 * INTERMEDIATE_SIZE) * expected
        controls = {
            "count_sum_exact": count_sum == expected,
            "input_ranks_le_counts": all(
                rank <= count for rank, count in zip(x_ranks, counts)
            ),
            "intermediate_ranks_le_counts": all(
                rank <= count for rank, count in zip(z_ranks, counts)
            ),
            "storage_bound_pass": elements <= bound,
            "full_rank_input_reconstruction_pass": maximum_x_reconstruction <= 1e-10,
            "full_rank_intermediate_reconstruction_pass": maximum_z_reconstruction <= 1e-10,
        }
        all_controls = all_controls and all(controls.values())
        context_rows.append(
            {
                "context": context,
                "prefix_tokens": PREFIX_TOKENS,
                "expert_invocations": count_sum,
                "minimum_expert_count": min(counts),
                "maximum_expert_count": max(counts),
                "input_rank_sum": sum(x_ranks),
                "intermediate_rank_sum": sum(z_ranks),
                "input_rank_mean": sum(x_ranks) / EXPERTS,
                "intermediate_rank_mean": sum(z_ranks) / EXPERTS,
                "full_rank_image_elements": elements,
                "bound_elements": bound,
                "bound_utilization": elements / bound,
                "maximum_full_rank_input_reconstruction_relative_l2": maximum_x_reconstruction,
                "maximum_full_rank_intermediate_reconstruction_relative_l2": maximum_z_reconstruction,
                "controls": controls,
            }
        )
    return {
        "layer": layer,
        "split": split,
        "contexts": context_rows,
        "all_required_controls_pass": all_controls,
    }


def prompt_transfer_residuals(
    data: dict[str, torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    invocations = CONTEXTS_PER_SPLIT * FUTURE_TOKENS * TOP_K
    result = {
        rank: {
            "x": torch.full((invocations,), float("inf"), dtype=torch.float64),
            "z": torch.full((invocations,), float("inf"), dtype=torch.float64),
            "weights": torch.empty(invocations, dtype=torch.float64),
        }
        for rank in RANKS
    }
    output_offset = 0
    for context in range(CONTEXTS_PER_SPLIT):
        start = context * CONTEXT_TOKENS
        prefix = subset(data, slice(start, start + PREFIX_TOKENS))
        future = subset(
            data,
            slice(start + PREFIX_TOKENS, start + PREFIX_TOKENS + FUTURE_TOKENS),
        )
        block_invocations = FUTURE_TOKENS * TOP_K
        for rank in RANKS:
            result[rank]["weights"][output_offset : output_offset + block_invocations] = (
                future["weights"].reshape(-1).double()
            )
        for expert in range(EXPERTS):
            prefix_x, prefix_z, _prefix_weights, _prefix_flat = expert_matrices(
                prefix, expert
            )
            future_x, future_z, _future_weights, future_flat = expert_matrices(
                future, expert
            )
            x_fit = fit_origin_subspace(prefix_x)
            z_fit = fit_origin_subspace(prefix_z)
            destination = future_flat + output_offset
            for rank in RANKS:
                q = x_fit.basis[:, : min(rank, x_fit.basis.shape[1])]
                p = z_fit.basis[:, : min(rank, z_fit.basis.shape[1])]
                result[rank]["x"][destination] = relative_residual_ratio(future_x, q)
                result[rank]["z"][destination] = relative_residual_ratio(future_z, p)
        output_offset += block_invocations
    if output_offset != invocations:
        raise RuntimeError("P1B residual alignment failed")
    return result


def grid(
    sources: dict[int, dict[int, dict[str, torch.Tensor]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in RANKS:
        x = torch.cat([sources[layer][rank]["x"] for layer in LAYERS])
        z = torch.cat([sources[layer][rank]["z"] for layer in LAYERS])
        weights = torch.cat([sources[layer][rank]["weights"] for layer in LAYERS])
        for threshold in THRESHOLDS:
            aggregate = metric_summary(x, z, weights, threshold)
            per_layer = {
                str(layer): metric_summary(
                    sources[layer][rank]["x"],
                    sources[layer][rank]["z"],
                    sources[layer][rank]["weights"],
                    threshold,
                )
                for layer in LAYERS
            }
            reduction = aggregate["projected_routed_cold_byte_reduction"]
            rows.append(
                {
                    "rank_cap": rank,
                    "threshold": threshold,
                    "double_gate_fast_fraction": aggregate[
                        "double_gate_fast_fraction"
                    ],
                    "cold_byte_reduction": reduction if reduction is not None else 1e300,
                    "aggregate": aggregate,
                    "layers": per_layer,
                }
            )
    return rows


def render(result: dict[str, Any]) -> str:
    selected = result["selection"]["selected_candidate"]
    test = result["test"]["selected_candidate"]
    diagnostic = next(
        row
        for row in result["test"]["post_lock_grid"]
        if row["rank_cap"] == 128 and row["threshold"] == 0.10
    )
    return "\n".join(
        [
            "# RSIV-MoE P1B — V2 1.024→128 long-prefixscreen",
            "",
            f"**Verdict: `{result['verdict']}`.**",
            "",
            result["verdict_explanation"],
            "",
            "## Bevroren validationkandidaat",
            "",
            f"- Rank: `{selected['rank_cap']}`.",
            f"- Threshold: `{selected['threshold']}`.",
            f"- Selectietype: `{selected['selection_kind']}`.",
            f"- Validation double-fast: `{selected['double_gate_fast_fraction']:.3%}`.",
            f"- Validation cold reduction: `{selected['cold_byte_reduction']:.3f}×`.",
            f"- Test double-fast: `{test['aggregate']['double_gate_fast_fraction']:.3%}`.",
            f"- Test cold reduction: `{test['aggregate']['projected_routed_cold_byte_reduction']:.3f}×`.",
            "",
            "## Ruimste diagnostiek na lock",
            "",
            f"Rank 128 / threshold 0,10 bereikt op test `{diagnostic['double_gate_fast_fraction']:.3%}` double-fast en `{diagnostic['cold_byte_reduction']:.3f}×` koude-bytereductie.",
            "",
            "## Controles",
            "",
            f"- Capture: `{result['controls']['capture_controls_pass']}`.",
            f"- Rank/count/bound/full-rankprojectie: `{result['controls']['rank_controls_pass']}`.",
            f"- Vereiste upstream operatoridentiteit: `{result['controls']['operator_controls_pass']}`.",
            f"- Extra long-prefix FP32-extreemcheck (diagnostisch): `{result['controls']['long_prefix_operator_diagnostic_pass']}`.",
            "",
            "## Claimgrens",
            "",
            "Dit blijft een teacher-state rank/page-faultscreen. Kwaliteit, packed runtime, SSD-latency en snelheid zijn niet gemeten; geen P1B-uitkomst is op zichzelf Eureka.",
            "",
            "## Volgende actie",
            "",
            result["next_action"],
            "",
        ]
    )


def validation_phase() -> None:
    if SELECTION.exists():
        raise FileExistsError("P1B validation lock already exists")
    for path in (
        CAPTURE,
        CAPTURE_REPORT,
        PREREGISTRATION,
        UPSTREAM_P1,
        INVALID_SELECTION_V1,
        CONTROL_ADDENDUM,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    capture_hash = sha256_file(CAPTURE)
    capture_report = json.loads(CAPTURE_REPORT.read_text(encoding="utf-8"))
    if capture_report["capture_sha256"] != capture_hash:
        raise RuntimeError("P1B capture hash mismatch")
    capture_controls = all(
        row["route_ids_exact"]
        and row["router_weight_maximum_absolute_error"] <= 1e-6
        and row["sum_expert_invocations"] == row["expected_expert_invocations"]
        and row["finite_x"]
        and row["finite_z"]
        for row in capture_report["controls"].values()
    )
    upstream = json.loads(UPSTREAM_P1.read_text(encoding="utf-8"))
    upstream_controls = upstream["controls"]["all_required_controls_pass"]
    validation_data: dict[int, dict[str, torch.Tensor]] = {}
    censuses: dict[str, Any] = {}
    sources: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    for layer in LAYERS:
        data = load_split(layer, "validation")
        validation_data[layer] = data
        censuses[str(layer)] = context_census(layer, "validation", data)
        sources[layer] = prompt_transfer_residuals(data)
        print(f"P1B validation layer={layer} complete", flush=True)
    validation_grid = grid(sources)
    selected = select_single_evaluation_candidate(validation_grid)
    operator_controls = full_rank_operator_controls(
        {layer: prefix_only(data) for layer, data in validation_data.items()}
    )
    rank_controls = all(row["all_required_controls_pass"] for row in censuses.values())
    valid = capture_controls and upstream_controls and rank_controls
    payload = {
        "kind": "rsiv_moe_p1b_v2_long_prefix_validation_lock",
        "status": "validation_selected" if valid else "invalid_controls_failed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "capture_sha256": capture_hash,
        "capture_bytes": CAPTURE.stat().st_size,
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "upstream_p1_sha256": sha256_file(UPSTREAM_P1),
        "control_addendum_sha256": sha256_file(CONTROL_ADDENDUM),
        "preserved_invalid_selection_v1": {
            "path": str(INVALID_SELECTION_V1.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(INVALID_SELECTION_V1),
        },
        "test_data_opened": False,
        "selected_candidate": selected,
        "validation_grid": validation_grid,
        "rank_census": censuses,
        "controls": {
            "capture_controls_pass": capture_controls,
            "upstream_p1_controls_pass": upstream_controls,
            "rank_controls_pass": rank_controls,
            "required_operator_identity_pass": upstream_controls,
            "long_prefix_operator_image_diagnostic": operator_controls,
        },
        "environment": environment(),
    }
    write_json_once(SELECTION, payload)
    print(
        json.dumps(
            {
                "selection": str(SELECTION),
                "selection_sha256": sha256_file(SELECTION),
                "status": payload["status"],
                "selected": selected,
            },
            indent=2,
        )
    )


def test_phase() -> None:
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite P1B result")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection["status"] != "validation_selected":
        raise RuntimeError("P1B validation controls invalid; test stays closed")
    if selection["capture_sha256"] != sha256_file(CAPTURE):
        raise RuntimeError("P1B capture changed after lock")
    rank = int(selection["selected_candidate"]["rank_cap"])
    threshold = float(selection["selected_candidate"]["threshold"])
    started = time.perf_counter()
    sources: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    censuses: dict[str, Any] = {}
    for layer in LAYERS:
        data = load_split(layer, "test")
        censuses[str(layer)] = context_census(layer, "test", data)
        sources[layer] = prompt_transfer_residuals(data)
        print(f"P1B test layer={layer} complete", flush=True)
    test_grid = grid(sources)
    selected = next(
        row
        for row in test_grid
        if int(row["rank_cap"]) == rank and float(row["threshold"]) == threshold
    )
    test_reduction = selected["aggregate"]["projected_routed_cold_byte_reduction"]
    test_pass = (
        selected["double_gate_fast_fraction"] >= 0.92
        and (test_reduction is None or test_reduction >= 10.0)
    )
    validation_pass = selection["selected_candidate"]["selection_kind"] == "primary_gate_pass"
    rank_controls = all(row["all_required_controls_pass"] for row in censuses.values())
    base_controls = selection["controls"]
    all_controls = (
        base_controls["capture_controls_pass"]
        and base_controls["upstream_p1_controls_pass"]
        and base_controls["rank_controls_pass"]
        and base_controls["required_operator_identity_pass"]
        and rank_controls
    )
    if all_controls and validation_pass and test_pass:
        verdict = "long_prefix_screen_positive"
        explanation = (
            "Een prompt-specifieke 1.024-tokenbasis haalt op validation en test de "
            "bevroren rank/page-faultgates. Dit opent alleen een nieuwe V2-P2-preregistratie."
        )
        next_action = "Preregistreer één-laag A/B/C-operatorimages; claim nog geen runtime of Eureka."
    elif all_controls:
        verdict = "long_prefix_screen_negative_v2"
        explanation = (
            "Ook na 1.024 prompttokens haalt dezelfde validation→testdiscipline de "
            "rank-32/page-faultgates niet. Promptlengte verklaart P1A dus niet."
        )
        next_action = (
            "Houd V2-P2 gesloten en preregistreer de hogere-E-rankcensus op "
            "Qwen3-30B-A3B-Base vóór de checkpointdownload."
        )
    else:
        verdict = "invalid"
        explanation = "Minstens één verplichte P1B-control faalde."
        next_action = "Diagnosticeer alleen de controlfout; interpreteer de test niet."
    payload = {
        "kind": "rsiv_moe_p1b_v2_long_prefix_result",
        "status": "complete",
        "verdict": verdict,
        "verdict_explanation": explanation,
        "next_action": next_action,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "capture_sha256": sha256_file(CAPTURE),
        "selection_sha256": sha256_file(SELECTION),
        "selection": selection,
        "test_opened_once": True,
        "test": {
            "selected_candidate": selected,
            "selected_candidate_pass": test_pass,
            "post_lock_grid": test_grid,
            "rank_census": censuses,
        },
        "controls": {
            "capture_controls_pass": base_controls["capture_controls_pass"],
            "rank_controls_pass": base_controls["rank_controls_pass"] and rank_controls,
            "operator_controls_pass": (
                base_controls["upstream_p1_controls_pass"]
                and base_controls["required_operator_identity_pass"]
            ),
            "long_prefix_operator_diagnostic_pass": base_controls[
                "long_prefix_operator_image_diagnostic"
            ]["all_required_controls_pass"],
            "long_prefix_operator_diagnostic": base_controls[
                "long_prefix_operator_image_diagnostic"
            ],
            "all_required_controls_pass": all_controls,
        },
        "claim_boundaries": {
            "quality": "not measured",
            "runtime": "not measured",
            "cold_bytes": "analytical packed-int4 projection",
            "eureka": False,
        },
    }
    write_json_once(RESULT, payload)
    REPORT.write_text(render(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "result_sha256": sha256_file(RESULT),
                "report_sha256": sha256_file(REPORT),
                "selection_sha256": sha256_file(SELECTION),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.select_validation:
        validation_phase()
    else:
        test_phase()
