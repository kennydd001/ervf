from __future__ import annotations

import gc

import torch
from safetensors.torch import save_file

from moe_lab.metrics import regression_metrics
from moe_lab.moe_layer import load_moe_layer
from moe_lab.quantization import fake_quantize_symmetric_per_row_
from moe_lab.reporting import ROOT, envelope, write_json
from moe_lab.shared_basis import (
    fit_expert_maps,
    fit_output_basis,
    oracle_projected_routed,
    predict_routed,
    shared_basis_parameter_count,
)
from moe_lab.trace import MoETrace, load_trace, slice_trace


ROUTED_PARAMETERS = 64 * 8_650_752
ROUTED_BF16_BYTES = ROUTED_PARAMETERS * 2
QUANT4_BYTES = (ROUTED_PARAMETERS * 4 + 7) // 8 + 64 * (1408 + 1408 + 2048) * 2
RANKS = (4, 8, 16, 32, 64, 128)
RIDGES = (0.01, 0.1, 1.0)


def quantize_layer_four_bit(layer) -> None:
    for expert in layer.experts:
        for weight in (expert.gate, expert.up, expert.down):
            fake_quantize_symmetric_per_row_(weight, 4)


def residual_trace(teacher: MoETrace, quantized: MoETrace) -> MoETrace:
    return MoETrace(
        hidden_states=teacher.hidden_states,
        router_ids=teacher.router_ids,
        router_weights=teacher.router_weights,
        selected_expert_outputs=(
            teacher.selected_expert_outputs.float()
            - quantized.selected_expert_outputs.float()
        ).to(torch.bfloat16),
        routed_output=(teacher.routed_output.float() - quantized.routed_output.float()).to(
            torch.bfloat16
        ),
        shared_output=torch.zeros_like(teacher.shared_output),
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("hybrid residual baseline requires CUDA")
    device = torch.device("cuda")
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    token_limits = {"train": 4096, "validation": 2048, "test": 4096}
    teacher = {
        split: slice_trace(
            load_trace(ROOT / "data" / "traces" / f"wikitext_{split}_layer_1.safetensors"),
            tokens,
        )
        for split, tokens in token_limits.items()
    }
    layer = load_moe_layer(model_dir, 1, device)
    quantize_layer_four_bit(layer)
    quantized = {split: layer.trace(trace.hidden_states) for split, trace in teacher.items()}
    residual = {
        split: residual_trace(teacher[split], quantized[split]) for split in teacher
    }
    del layer
    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(20260809)
    basis = fit_output_basis(residual["train"], max(RANKS), device)
    validation_sweep = {}
    selected = {}
    for ridge in RIDGES:
        maps, _ = fit_expert_maps(residual["train"], basis, ridge, 64)
        validation_sweep[str(ridge)] = {}
        for rank in RANKS:
            corrected = quantized["validation"].routed_output.float() + predict_routed(
                residual["validation"], basis, maps, rank
            ).float()
            metrics = regression_metrics(corrected, teacher["validation"].routed_output)
            validation_sweep[str(ridge)][str(rank)] = metrics
            if rank not in selected or metrics["nrmse"] < selected[rank]["validation"]["nrmse"]:
                selected[rank] = {"ridge": ridge, "validation": metrics}
        del maps

    maps_by_ridge = {}
    rows = []
    for rank in RANKS:
        ridge = float(selected[rank]["ridge"])
        if ridge not in maps_by_ridge:
            maps_by_ridge[ridge], _ = fit_expert_maps(
                residual["train"], basis, ridge, 64
            )
        maps = maps_by_ridge[ridge]
        predicted_residual = predict_routed(residual["test"], basis, maps, rank)
        corrected = quantized["test"].routed_output.float() + predicted_residual.float()
        adapter_parameters = shared_basis_parameter_count(2048, 64, rank)
        storage_bytes = QUANT4_BYTES + adapter_parameters * 2
        oracle_corrected = quantized["test"].routed_output.float() + oracle_projected_routed(
            residual["test"], basis, rank
        ).float()
        rows.append(
            {
                "rank": rank,
                "ridge": ridge,
                "adapter_parameters": adapter_parameters,
                "packed_storage_bytes": storage_bytes,
                "compression_ratio_vs_bf16_routed_bank": ROUTED_BF16_BYTES / storage_bytes,
                "validation": selected[rank]["validation"],
                "test": regression_metrics(corrected, teacher["test"].routed_output),
                "oracle_test": regression_metrics(
                    oracle_corrected, teacher["test"].routed_output
                ),
            }
        )
    best = min(rows, key=lambda row: row["validation"]["nrmse"])
    best_rank = int(best["rank"])
    best_ridge = float(best["ridge"])
    artifact = ROOT / "data" / "models" / f"layer1_quant4_residual_rank{best_rank}.safetensors"
    save_file(
        {
            "basis": basis[:, :best_rank].to(torch.bfloat16).cpu().contiguous(),
            "expert_maps": maps_by_ridge[best_ridge][
                :, :, :best_rank
            ].to(torch.bfloat16).cpu().contiguous(),
        },
        artifact,
        metadata={"rank": str(best_rank), "ridge": str(best_ridge)},
    )
    report = {
        "status": "complete",
        "method": "per-row_4bit_plus_shared_activation_residual_basis",
        "token_limits": token_limits,
        "ranks": list(RANKS),
        "ridges": list(RIDGES),
        "quant4_test": regression_metrics(
            quantized["test"].routed_output, teacher["test"].routed_output
        ),
        "validation_sweep": validation_sweep,
        "selected_results": rows,
        "selected_rank": best_rank,
        "artifact": str(artifact.resolve()),
    }
    path = write_json(
        "quant4_activation_residual_layer1.json", envelope("compression_baseline", report)
    )
    print(path)
    print("quant4", report["quant4_test"])
    for row in rows:
        print(
            f"rank={row['rank']:3d} ratio={row['compression_ratio_vs_bf16_routed_bank']:.3f}x "
            f"val={row['validation']['nrmse']:.6f} test={row['test']['nrmse']:.6f} "
            f"oracle={row['oracle_test']['nrmse']:.6f}"
        )
