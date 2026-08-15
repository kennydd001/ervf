from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone

import psutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from moe_lab.fleq_moe.expert_quant import (
    QuantizedProjection,
    official_gptq_projection,
    official_rtn_projection,
    optimize_gsq_expert,
    output_metrics,
    routed_expert_rows,
)
from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_checkpoint_tensors


MODEL_DIR = ROOT / "models/qwen3-30b-a3b-base"
CAPTURE = ROOT / "reports/runs/rsiv_moe/p1c_qwen_validation.safetensors"
LOCK = ROOT / "reports/fleq_moe/p1_smoke_expert_lock.json"
GSQ_ROOT = ROOT / "third_party/GSQ"
RUN_DIR = ROOT / "reports/runs/fleq_moe/p1"
REPORT_DIR = ROOT / "reports/fleq_moe/p1_experts"
CONTEXT_TOKENS = 1152
TOP_K = 8
SEED = 260811


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, choices=(0, 47))
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--concise", action="store_true")
    return parser.parse_args()


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_summary(item: QuantizedProjection) -> dict:
    indices = torch.arange(item.weight.shape[1], device=item.weight.device) // 128
    scale = item.scales[:, indices].to(item.weight.dtype)
    codes = torch.round(item.weight / scale).to(torch.int16)
    unique, counts = torch.unique(codes.cpu(), return_counts=True)
    scale_bits = item.scales.numel() * 16
    code_bits = item.weight.numel() * 2
    return {
        "histogram": {str(int(key)): int(value) for key, value in zip(unique, counts)},
        "weights": item.weight.numel(),
        "scales": item.scales.numel(),
        "effective_bits_per_weight_including_bf16_scales": (code_bits + scale_bits) / item.weight.numel(),
        "codes_in_range": bool(((codes >= -2) & (codes <= 1)).all()),
    }


def load_capture(layer: int):
    prefix = f"layer_{layer:02d}"
    with safe_open(CAPTURE, framework="pt", device="cpu") as handle:
        x = handle.get_tensor(f"{prefix}_moe_input")
        ids = handle.get_tensor(f"{prefix}_router_ids").long()
        weights = handle.get_tensor(f"{prefix}_router_weights").float()
        z = handle.get_tensor(f"{prefix}_intermediate_z").reshape(-1, TOP_K, 768)
    return x, ids, weights, z


def split_rows(layer: int, expert: int):
    x, ids, weights, z = load_capture(layer)
    calibration = routed_expert_rows(
        x[:CONTEXT_TOKENS], ids[:CONTEXT_TOKENS], weights[:CONTEXT_TOKENS], z[:CONTEXT_TOKENS], expert
    )
    heldout = routed_expert_rows(
        x[CONTEXT_TOKENS:], ids[CONTEXT_TOKENS:], weights[CONTEXT_TOKENS:], z[CONTEXT_TOKENS:], expert
    )
    if calibration[0].numel() == 0 or heldout[0].numel() == 0:
        raise RuntimeError("selected expert has no calibration or held-out invocations")
    return calibration, heldout


def load_original(layer: int, expert: int, device: torch.device):
    base = f"model.layers.{layer}.mlp.experts.{expert}"
    names = {
        "gate": f"{base}.gate_proj.weight",
        "up": f"{base}.up_proj.weight",
        "down": f"{base}.down_proj.weight",
    }
    loaded = load_checkpoint_tensors(MODEL_DIR, list(names.values()), checkpoint_weight_map(MODEL_DIR))
    return {key: loaded[name].to(device=device).contiguous() for key, name in names.items()}


if __name__ == "__main__":
    args = parse_args()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    selected = lock["layers"][str(args.layer)]["selected_experts"]
    if args.expert not in selected:
        raise ValueError(f"expert {args.expert} was not locked for layer {args.layer}")
    artifact = RUN_DIR / f"layer_{args.layer:02d}_expert_{args.expert:03d}.safetensors"
    report = REPORT_DIR / f"layer_{args.layer:02d}_expert_{args.expert:03d}.json"
    if artifact.exists() or report.exists():
        raise FileExistsError(f"refusing to overwrite {artifact} or {report}")

    device = torch.device("cuda")
    torch.manual_seed(SEED + args.layer * 1000 + args.expert)
    torch.cuda.manual_seed_all(SEED + args.layer * 1000 + args.expert)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    rss_peak = psutil.Process().memory_info().rss
    started = time.perf_counter()
    calibration, heldout = split_rows(args.layer, args.expert)
    original = load_original(args.layer, args.expert, device)
    rss_peak = max(rss_peak, psutil.Process().memory_info().rss)

    with torch.no_grad():
        first_reference = __import__("moe_lab.fleq_moe.gsq_bridge", fromlist=["expert_forward"]).expert_forward(
            heldout[0].to(device), original["gate"], original["up"], original["down"]
        )
        second_reference = __import__("moe_lab.fleq_moe.gsq_bridge", fromlist=["expert_forward"]).expert_forward(
            heldout[0].to(device), original["gate"], original["up"], original["down"]
        )
        fallback_bit_exact = torch.equal(first_reference, second_reference)
    del first_reference, second_reference

    rtn = {
        name: official_rtn_projection(weight, GSQ_ROOT, 2)
        for name, weight in original.items()
    }
    calibration_x, calibration_z, calibration_router = calibration
    gptq = {
        "gate": official_gptq_projection(original["gate"], calibration_x, GSQ_ROOT),
        "up": official_gptq_projection(original["up"], calibration_x, GSQ_ROOT),
        "down": official_gptq_projection(original["down"], calibration_z, GSQ_ROOT),
    }
    rss_peak = max(rss_peak, psutil.Process().memory_info().rss)
    gsq, losses = optimize_gsq_expert(
        original,
        gptq,
        calibration_x,
        GSQ_ROOT,
        kind="2bit",
        epochs=10,
        seed=SEED + args.layer * 1000 + args.expert,
    )
    rss_peak = max(rss_peak, psutil.Process().memory_info().rss)

    repeat_required = args.expert == selected[0]
    repeat_exact = None
    repeat_losses = None
    if repeat_required:
        repeated, repeat_losses = optimize_gsq_expert(
            original,
            gptq,
            calibration_x,
            GSQ_ROOT,
            kind="2bit",
            epochs=10,
            seed=SEED + args.layer * 1000 + args.expert,
        )
        repeat_exact = all(torch.equal(gsq[name].weight, repeated[name].weight) for name in gsq)
        repeat_exact = repeat_exact and losses == repeat_losses
        del repeated

    methods = {
        "rtn_2bit": rtn,
        "gptq_2bit": gptq,
        "gsq_2bit": gsq,
    }
    metrics = {}
    for method, candidate in methods.items():
        metrics[method] = {
            "calibration": output_metrics(calibration_x, calibration_router, original, candidate),
            "heldout": output_metrics(heldout[0], heldout[2], original, candidate),
        }
    gptq_mse = metrics["gptq_2bit"]["heldout"]["router_weighted_relative_mse"]
    gsq_mse = metrics["gsq_2bit"]["heldout"]["router_weighted_relative_mse"]
    improvement = (gptq_mse - gsq_mse) / max(gptq_mse, 1e-30)

    tensors = {}
    for method in ("gptq", "gsq"):
        source = gptq if method == "gptq" else gsq
        for name, item in source.items():
            tensors[f"{method}_{name}_weight"] = item.weight.detach().cpu().contiguous()
            tensors[f"{method}_{name}_scales"] = item.scales.detach().cpu().contiguous()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    save_file(tensors, artifact, metadata={
        "layer": str(args.layer),
        "expert": str(args.expert),
        "seed": str(SEED + args.layer * 1000 + args.expert),
        "gsq_commit": "03fc16484c369e3127225615d5e03e8d3a6043e3",
    })
    torch.cuda.synchronize(device)
    payload = {
        "kind": "fleq_moe_p1_expert_result",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "layer": args.layer,
        "expert": args.expert,
        "selection_lock_sha256": file_sha256(LOCK),
        "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": file_sha256(artifact),
        "calibration_rows": int(calibration_x.shape[0]),
        "heldout_rows": int(heldout[0].shape[0]),
        "fallback_bit_exact": fallback_bit_exact,
        "metrics": metrics,
        "gsq_losses": losses,
        "heldout_gsq_improvement_over_gptq": improvement,
        "code_summaries": {
            method: {name: code_summary(item) for name, item in source.items()}
            for method, source in (("gptq_2bit", gptq), ("gsq_2bit", gsq))
        },
        "original_weight_sha256": {name: tensor_sha256(weight) for name, weight in original.items()},
        "repeat_required": repeat_required,
        "repeat_exact": repeat_exact,
        "repeat_losses": repeat_losses,
        "all_finite": all(
            row[split]["all_finite"]
            for row in metrics.values()
            for split in ("calibration", "heldout")
        ) and all(torch.isfinite(item.weight).all() and torch.isfinite(item.scales).all() for item in gsq.values()),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "process_rss_peak_observed_bytes": rss_peak,
        "elapsed_seconds": time.perf_counter() - started,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.concise:
        print(json.dumps({
            "layer": args.layer,
            "expert": args.expert,
            "improvement": improvement,
            "gptq_heldout_mse": gptq_mse,
            "gsq_heldout_mse": gsq_mse,
            "peak_cuda_allocated_bytes": payload["peak_cuda_allocated_bytes"],
            "elapsed_seconds": payload["elapsed_seconds"],
        }, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2))
