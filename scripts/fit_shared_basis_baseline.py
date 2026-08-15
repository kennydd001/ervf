from __future__ import annotations

import argparse

import torch
from safetensors.torch import save_file

from moe_lab.metrics import regression_metrics
from moe_lab.reporting import ROOT, envelope, write_json
from moe_lab.shared_basis import (
    fit_expert_maps,
    fit_output_basis,
    oracle_projected_routed,
    predict_routed,
    shared_basis_parameter_count,
)
from moe_lab.trace import load_trace, slice_trace


ROUTED_EXPERT_PARAMETERS = 64 * 8_650_752


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rank", type=int, default=256)
    parser.add_argument("--tokens-per-split", type=int, default=2048)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ranks = [rank for rank in (4, 8, 16, 32, 64, 128, 256) if rank <= args.max_rank]
    if not ranks:
        raise ValueError("max-rank must be at least 4")
    ridge_values = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    traces = {
        split: slice_trace(
            load_trace(ROOT / "data" / "traces" / f"wikitext_{split}_layer_1.safetensors"),
            args.tokens_per_split,
        )
        for split in ("train", "validation", "test")
    }
    torch.manual_seed(20260809)
    basis = fit_output_basis(traces["train"], max(ranks), device)

    oracle = {
        split: {
            str(rank): regression_metrics(
                oracle_projected_routed(trace, basis, rank), trace.routed_output
            )
            for rank in ranks
        }
        for split, trace in traces.items()
    }

    validation_sweep: dict[str, dict[str, dict[str, float]]] = {}
    selected: dict[int, dict[str, object]] = {}
    expert_counts: list[int] | None = None
    for ridge in ridge_values:
        maps, counts = fit_expert_maps(traces["train"], basis, ridge, 64)
        expert_counts = counts
        ridge_key = str(ridge)
        validation_sweep[ridge_key] = {}
        for rank in ranks:
            metrics = regression_metrics(
                predict_routed(traces["validation"], basis, maps, rank),
                traces["validation"].routed_output,
            )
            validation_sweep[ridge_key][str(rank)] = metrics
            if rank not in selected or metrics["nrmse"] < selected[rank]["validation"]["nrmse"]:
                selected[rank] = {"ridge_relative": ridge, "validation": metrics}
        del maps

    final_rows = []
    fitted_by_ridge: dict[float, torch.Tensor] = {}
    for rank in ranks:
        ridge = float(selected[rank]["ridge_relative"])
        if ridge not in fitted_by_ridge:
            fitted_by_ridge[ridge], _ = fit_expert_maps(
                traces["train"], basis, ridge, 64
            )
        maps = fitted_by_ridge[ridge]
        parameters = shared_basis_parameter_count(2048, 64, rank)
        row = {
            "rank": rank,
            "ridge_relative": ridge,
            "parameter_count": parameters,
            "bf16_bytes": parameters * 2,
            "compression_ratio_vs_bf16_routed_bank": ROUTED_EXPERT_PARAMETERS / parameters,
            "validation": selected[rank]["validation"],
            "test": regression_metrics(
                predict_routed(traces["test"], basis, maps, rank),
                traces["test"].routed_output,
            ),
            "oracle_test": oracle["test"][str(rank)],
        }
        final_rows.append(row)

    best = min(final_rows, key=lambda row: row["validation"]["nrmse"])
    best_rank = int(best["rank"])
    best_ridge = float(best["ridge_relative"])
    artifact = ROOT / "data" / "models" / f"layer1_shared_basis_rank{best_rank}.safetensors"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "basis": basis[:, :best_rank].to(torch.bfloat16).cpu().contiguous(),
            "expert_maps": fitted_by_ridge[best_ridge][
                :, :, :best_rank
            ].to(torch.bfloat16).cpu().contiguous(),
        },
        artifact,
        metadata={
            "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
            "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "layer": "1",
            "rank": str(best_rank),
            "ridge_relative": str(best_ridge),
        },
    )
    report = {
        "status": "complete",
        "method": "shared_output_basis_with_expert_specific_linear_activation_codes",
        "basis_fit_split": "train",
        "hyperparameter_selection_split": "validation",
        "final_evaluation_split": "test",
        "ranks": ranks,
        "tokens_per_split": args.tokens_per_split,
        "ridge_values": list(ridge_values),
        "expert_train_sample_counts": expert_counts,
        "oracle": oracle,
        "validation_sweep": validation_sweep,
        "selected_results": final_rows,
        "selected_artifact": str(artifact.resolve()),
        "selected_rank_by_validation": best_rank,
        "device": str(device),
    }
    path = write_json("shared_basis_layer1.json", envelope("compression_baseline", report))
    print(path)
    for row in final_rows:
        print(
            f"rank={row['rank']:3d} ratio={row['compression_ratio_vs_bf16_routed_bank']:.2f}x "
            f"val={row['validation']['nrmse']:.6f} test={row['test']['nrmse']:.6f} "
            f"oracle={row['oracle_test']['nrmse']:.6f} ridge={row['ridge_relative']}"
        )
