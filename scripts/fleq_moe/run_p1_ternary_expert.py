from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone

import psutil
import torch
from safetensors.torch import save_file

from moe_lab.fleq_moe.expert_quant import (
    QuantizedProjection,
    hard_gsq_initialization,
    official_rtn_projection,
    optimize_gsq_expert,
    output_metrics,
)
from moe_lab.reporting import ROOT
from run_p1_expert import (
    GSQ_ROOT,
    LOCK,
    SEED,
    file_sha256,
    load_original,
    split_rows,
)


RUN_DIR = ROOT / "reports/runs/fleq_moe/p1_ternary"
REPORT_DIR = ROOT / "reports/fleq_moe/p1_ternary_experts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, choices=(0, 47))
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--concise", action="store_true")
    return parser.parse_args()


def ternary_code_summary(item: QuantizedProjection) -> dict:
    indices = torch.arange(item.weight.shape[1], device=item.weight.device) // 128
    scale = item.scales[:, indices].to(item.weight.dtype)
    codes = torch.round(item.weight / scale).to(torch.int16)
    unique, counts = torch.unique(codes.cpu(), return_counts=True)
    scale_bits = item.scales.numel() * 16
    ideal_code_bits = item.weight.numel() * math.log2(3)
    packed_code_bits = item.weight.numel() * 2
    return {
        "histogram": {str(int(key)): int(value) for key, value in zip(unique, counts)},
        "weights": item.weight.numel(),
        "scales": item.scales.numel(),
        "ideal_cardinality_bits_per_weight_including_bf16_scales": (
            ideal_code_bits + scale_bits
        ) / item.weight.numel(),
        "two_bit_packed_bits_per_weight_including_bf16_scales": (
            packed_code_bits + scale_bits
        ) / item.weight.numel(),
        "codes_in_range": bool(((codes >= -1) & (codes <= 1)).all()),
    }


if __name__ == "__main__":
    args = parse_args()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    selected = lock["layers"][str(args.layer)]["selected_experts"]
    if args.expert not in selected:
        raise ValueError("expert is outside the preregistered lock")
    artifact = RUN_DIR / f"layer_{args.layer:02d}_expert_{args.expert:03d}.safetensors"
    report = REPORT_DIR / f"layer_{args.layer:02d}_expert_{args.expert:03d}.json"
    if artifact.exists() or report.exists():
        raise FileExistsError(f"refusing to overwrite {artifact} or {report}")

    device = torch.device("cuda")
    run_seed = SEED + 500_000 + args.layer * 1000 + args.expert
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    rss_peak = psutil.Process().memory_info().rss
    calibration, heldout = split_rows(args.layer, args.expert)
    calibration_x, _calibration_z, calibration_router = calibration
    original = load_original(args.layer, args.expert, device)
    raw_rtn = {
        name: official_rtn_projection(weight, GSQ_ROOT, "ternary")
        for name, weight in original.items()
    }
    rtn = hard_gsq_initialization(raw_rtn, GSQ_ROOT, kind="ternary", seed=run_seed)
    gsq, losses = optimize_gsq_expert(
        original,
        raw_rtn,
        calibration_x,
        GSQ_ROOT,
        kind="ternary",
        epochs=10,
        batch_size=64,
        seed=run_seed,
    )
    rss_peak = max(rss_peak, psutil.Process().memory_info().rss)
    repeat_required = args.expert == selected[0]
    repeat_exact = None
    repeat_weights_exact = None
    repeat_losses_exact = None
    repeated_losses = None
    if repeat_required:
        repeated, repeated_losses = optimize_gsq_expert(
            original, raw_rtn, calibration_x, GSQ_ROOT, kind="ternary",
            epochs=10, batch_size=64, seed=run_seed,
        )
        repeat_weights_exact = all(
            torch.equal(gsq[name].weight, repeated[name].weight) for name in gsq
        )
        repeat_losses_exact = losses == repeated_losses
        repeat_exact = repeat_weights_exact and repeat_losses_exact
        del repeated

    metrics = {}
    for method, candidate in (("rtn_ternary", rtn), ("gsq_ternary", gsq)):
        metrics[method] = {
            "calibration": output_metrics(calibration_x, calibration_router, original, candidate),
            "heldout": output_metrics(heldout[0], heldout[2], original, candidate),
        }
    rtn_mse = metrics["rtn_ternary"]["heldout"]["router_weighted_relative_mse"]
    gsq_mse = metrics["gsq_ternary"]["heldout"]["router_weighted_relative_mse"]
    improvement = (rtn_mse - gsq_mse) / max(rtn_mse, 1e-30)
    tensors = {}
    for method, source in (("rtn", rtn), ("gsq", gsq)):
        for name, item in source.items():
            tensors[f"{method}_{name}_weight"] = item.weight.cpu().contiguous()
            tensors[f"{method}_{name}_scales"] = item.scales.cpu().contiguous()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    save_file(tensors, artifact, metadata={
        "layer": str(args.layer), "expert": str(args.expert),
        "seed": str(run_seed), "format": "ternary diagnostic",
    })
    payload = {
        "kind": "fleq_moe_p1_ternary_expert_result",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "layer": args.layer,
        "expert": args.expert,
        "selection_lock_sha256": file_sha256(LOCK),
        "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": file_sha256(artifact),
        "calibration_rows": int(calibration_x.shape[0]),
        "heldout_rows": int(heldout[0].shape[0]),
        "metrics": metrics,
        "gsq_losses": losses,
        "heldout_gsq_improvement_over_rtn": improvement,
        "code_summaries": {
            method: {name: ternary_code_summary(item) for name, item in source.items()}
            for method, source in (("rtn_ternary", rtn), ("gsq_ternary", gsq))
        },
        "repeat_required": repeat_required,
        "repeat_exact": repeat_exact,
        "repeat_weights_exact": repeat_weights_exact,
        "repeat_losses_exact": repeat_losses_exact,
        "repeat_losses": repeated_losses,
        "all_finite": all(
            metrics[method][split]["all_finite"]
            for method in metrics for split in ("calibration", "heldout")
        ),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "process_rss_peak_observed_bytes": rss_peak,
        "elapsed_seconds": time.perf_counter() - started,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.concise:
        print(json.dumps({
            "layer": args.layer, "expert": args.expert,
            "improvement": improvement, "rtn_heldout_mse": rtn_mse,
            "gsq_heldout_mse": gsq_mse,
            "elapsed_seconds": payload["elapsed_seconds"],
        }, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2))
