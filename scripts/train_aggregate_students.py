from __future__ import annotations

import argparse
import time

import torch
from safetensors.torch import save_file

from moe_lab.aggregate_student import AggregateStudent
from moe_lab.metrics import regression_metrics
from moe_lab.reporting import ROOT, envelope, write_json
from moe_lab.trace import MoETrace, load_trace


ROUTED_EXPERT_PARAMETERS = 64 * 8_650_752
SEED = 20260809


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", type=int, nargs="+", default=[256, 1408])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--report-name", default="aggregate_students_layer1.json")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def batches(trace: MoETrace, batch_size: int, order: torch.Tensor | None = None):
    size = trace.hidden_states.shape[0]
    indices = torch.arange(size) if order is None else order
    for start in range(0, size, batch_size):
        selected = indices[start : start + batch_size]
        yield (
            trace.hidden_states[selected],
            trace.router_ids[selected],
            trace.router_weights[selected],
            trace.routed_output[selected],
        )


@torch.inference_mode()
def evaluate(
    model: AggregateStudent,
    trace: MoETrace,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    predictions = []
    for x, ids, weights, _ in batches(trace, batch_size):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(
                x.to(device=device, dtype=torch.float32),
                ids.to(device=device),
                weights.to(device=device, dtype=torch.float32),
            )
        predictions.append(prediction.float().cpu())
    return regression_metrics(torch.cat(predictions), trace.routed_output)


def train_one(
    train: MoETrace,
    validation: MoETrace,
    test: MoETrace,
    width: int,
    conditioned: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(SEED + width + int(conditioned))
    model = AggregateStudent(2048, width, 64, conditioned).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    target_scale = train.routed_output.float().pow(2).mean().clamp_min(1e-12)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    best_validation = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train.hidden_states.shape[0], generator=generator)
        total_loss = 0.0
        seen = 0
        for x, ids, weights, target in batches(train, args.batch_size, order):
            x = x.to(device=device, dtype=torch.float32)
            ids = ids.to(device=device)
            weights = weights.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(x, ids, weights)
                loss = torch.mean((prediction.float() - target) ** 2) / target_scale.to(device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach().item()) * x.shape[0]
            seen += x.shape[0]
        model.eval()
        validation_metrics = evaluate(model, validation, device, args.batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_normalized_mse": total_loss / seen,
                "validation": validation_metrics,
            }
        )
        if validation_metrics["nrmse"] < best_validation:
            best_validation = validation_metrics["nrmse"]
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
    test_metrics = evaluate(model, test, device, args.batch_size)
    label = f"{'conditioned' if conditioned else 'unconditioned'}_width{width}"
    artifact = ROOT / "data" / "models" / f"layer1_aggregate_{label}.safetensors"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.to(torch.bfloat16).contiguous() for name, value in best_state.items()},
        artifact,
        metadata={
            "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
            "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "layer": "1",
            "width": str(width),
            "route_conditioned": str(conditioned).lower(),
            "best_epoch": str(best_epoch),
        },
    )
    parameters = model.parameter_count
    return {
        "label": label,
        "width": width,
        "route_conditioned": conditioned,
        "parameter_count": parameters,
        "bf16_bytes": parameters * 2,
        "compression_ratio_vs_bf16_routed_bank": ROUTED_EXPERT_PARAMETERS / parameters,
        "best_epoch": best_epoch,
        "best_validation_nrmse": best_validation,
        "test": test_metrics,
        "history": history,
        "artifact": str(artifact.resolve()),
        "wall_seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    traces = {
        split: load_trace(ROOT / "data" / "traces" / f"wikitext_{split}_layer_1.safetensors")
        for split in ("train", "validation", "test")
    }
    results = []
    for width in args.widths:
        for conditioned in (False, True):
            result = train_one(
                traces["train"],
                traces["validation"],
                traces["test"],
                width,
                conditioned,
                args,
                device,
            )
            results.append(result)
            print(
                result["label"],
                f"ratio={result['compression_ratio_vs_bf16_routed_bank']:.2f}x",
                f"epoch={result['best_epoch']}",
                f"val={result['best_validation_nrmse']:.6f}",
                f"test={result['test']['nrmse']:.6f}",
            )
    report = {
        "status": "complete",
        "method": "aggregate_swiglu_student",
        "train_tokens": int(traces["train"].hidden_states.shape[0]),
        "validation_tokens": int(traces["validation"].hidden_states.shape[0]),
        "test_tokens": int(traces["test"].hidden_states.shape[0]),
        "seed": SEED,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "gradient_clip": args.gradient_clip,
        "batch_size": args.batch_size,
        "device": str(device),
        "results": results,
    }
    path = write_json(args.report_name, envelope("compression_baseline", report))
    print(path)
