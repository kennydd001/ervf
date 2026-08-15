from __future__ import annotations

import argparse
import time

import torch
from safetensors.torch import load_file, save_file

from moe_lab.aggregate_student import ResidualBasisStudent
from moe_lab.metrics import regression_metrics
from moe_lab.reporting import ROOT, envelope, write_json
from moe_lab.trace import MoETrace, load_trace


ROUTED_EXPERT_PARAMETERS = 64 * 8_650_752
SEED = 20260809


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--report-name", default="residual_basis_students_layer1.json")
    return parser.parse_args()


def batches(trace: MoETrace, batch_size: int, order: torch.Tensor | None = None):
    indices = torch.arange(trace.hidden_states.shape[0]) if order is None else order
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        yield (
            trace.hidden_states[selected],
            trace.router_ids[selected],
            trace.router_weights[selected],
            trace.routed_output[selected],
        )


@torch.inference_mode()
def evaluate(model, trace, device, batch_size):
    predictions = []
    for x, ids, weights, _ in batches(trace, batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                x.to(device=device, dtype=torch.float32),
                ids.to(device=device),
                weights.to(device=device, dtype=torch.float32),
            )
        predictions.append(output.float().cpu())
    return regression_metrics(torch.cat(predictions), trace.routed_output)


def initialize_base(model: ResidualBasisStudent) -> str:
    artifact = ROOT / "data" / "models" / "layer1_aggregate_unconditioned_width1408.safetensors"
    state = load_file(artifact, device="cpu")
    model.base.load_state_dict(state, strict=True)
    return str(artifact.resolve())


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this training baseline requires CUDA")
    device = torch.device("cuda")
    traces = {
        split: load_trace(ROOT / "data" / "traces" / f"wikitext_{split}_layer_1.safetensors")
        for split in ("train", "validation", "test")
    }
    results = []
    for rank in args.ranks:
        torch.manual_seed(SEED + rank)
        model = ResidualBasisStudent(2048, 1408, 64, rank)
        base_artifact = initialize_base(model)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        target_scale = traces["train"].routed_output.float().pow(2).mean().clamp_min(1e-12)
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        best_nrmse = float("inf")
        best_state = None
        best_epoch = 0
        stale = 0
        history = []
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(traces["train"].hidden_states.shape[0], generator=generator)
            loss_sum = 0.0
            seen = 0
            for x, ids, weights, target in batches(traces["train"], args.batch_size, order):
                x = x.to(device=device, dtype=torch.float32)
                ids = ids.to(device=device)
                weights = weights.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(x, ids, weights)
                    loss = torch.mean((prediction.float() - target) ** 2) / target_scale.to(device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += float(loss.detach().item()) * x.shape[0]
                seen += x.shape[0]
            model.eval()
            validation = evaluate(model, traces["validation"], device, args.batch_size)
            history.append(
                {
                    "epoch": epoch,
                    "train_normalized_mse": loss_sum / seen,
                    "validation": validation,
                }
            )
            if validation["nrmse"] < best_nrmse:
                best_nrmse = validation["nrmse"]
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        if best_state is None:
            raise RuntimeError("training produced no checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        test = evaluate(model, traces["test"], device, args.batch_size)
        artifact = ROOT / "data" / "models" / f"layer1_residual_basis_rank{rank}.safetensors"
        save_file(
            {name: value.to(torch.bfloat16).contiguous() for name, value in best_state.items()},
            artifact,
            metadata={"rank": str(rank), "best_epoch": str(best_epoch)},
        )
        parameters = model.parameter_count
        row = {
            "rank": rank,
            "parameter_count": parameters,
            "bf16_bytes": parameters * 2,
            "compression_ratio_vs_bf16_routed_bank": ROUTED_EXPERT_PARAMETERS / parameters,
            "best_epoch": best_epoch,
            "best_validation_nrmse": best_nrmse,
            "test": test,
            "history": history,
            "initial_base_artifact": base_artifact,
            "artifact": str(artifact.resolve()),
            "wall_seconds": time.perf_counter() - started,
        }
        results.append(row)
        print(
            f"rank={rank} ratio={row['compression_ratio_vs_bf16_routed_bank']:.2f}x "
            f"epoch={best_epoch} val={best_nrmse:.6f} test={test['nrmse']:.6f}"
        )
    report = {
        "status": "complete",
        "method": "shared_swiglu_plus_expert_specific_low_rank_residual",
        "train_tokens": int(traces["train"].hidden_states.shape[0]),
        "validation_tokens": int(traces["validation"].hidden_states.shape[0]),
        "test_tokens": int(traces["test"].hidden_states.shape[0]),
        "seed": SEED,
        "learning_rate": args.learning_rate,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "results": results,
    }
    path = write_json(args.report_name, envelope("compression_baseline", report))
    print(path)
