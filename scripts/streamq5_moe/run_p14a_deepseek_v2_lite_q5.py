from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.metrics import regression_metrics, topk_overlap
from moe_lab.moe_layer import loaded_moe_from_official_module
from moe_lab.partial_forward import load_decoder_layer
from moe_lab.reporting import ROOT
from scripts.bitflow_moe.fit_validate_p0_c1_q4 import (
    final_logits,
    final_metrics,
    forward_capture,
    layer_zero,
)


MODEL = ROOT / "models/deepseek-v2-lite"
INPUTS = ROOT / "reports/runs/bitflow_moe/p0_input_ids.safetensors"
INPUT_LOCK = ROOT / "reports/bitflow_moe/p0_input_lock.json"
PREREG = ROOT / "reports/streamq5_moe/P14A_DEEPSEEK_V2_LITE_Q5_REPLICATION_PREREGISTRATION.md"
OUT_DIR = ROOT / "reports/streamq5_moe"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def physical_q5_per_row_(weight: torch.Tensor) -> None:
    work = weight.float()
    maximum = work.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 15.0, torch.ones_like(maximum))
    codes = torch.round(work / scale).clamp(-15, 15)
    stored_scale = scale.to(torch.bfloat16).float()
    weight.copy_((codes * stored_scale).to(weight.dtype))


@torch.no_grad()
def quantize_routed_experts_q5_(layer) -> None:
    moe = loaded_moe_from_official_module(layer.mlp, layer=1)
    for expert in moe.experts:
        for weight in (expert.gate, expert.up, expert.down):
            physical_q5_per_row_(weight)


def tensor_error(candidate: torch.Tensor, teacher: torch.Tensor) -> dict:
    left, right = candidate.float(), teacher.float()
    delta = left - right
    return {
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(right).clamp_min(1e-30)),
        "max_abs": float(delta.abs().max()),
        "finite": bool(torch.isfinite(left).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    output = OUT_DIR / f"p14a_deepseek_v2_lite_q5_{args.split}.json"
    if output.exists():
        raise FileExistsError(output)
    validation_path = OUT_DIR / "p14a_deepseek_v2_lite_q5_validation.json"
    if args.split == "test":
        if not validation_path.exists():
            raise RuntimeError("validation required")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation["status"] != "validation_pass_test_authorized":
            raise RuntimeError("test not authorized")
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    if sha256(INPUTS) != lock["artifact_sha256"]:
        raise ValueError("input provenance mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    ids_source = load_file(INPUTS)[args.split]
    expected = lock["splits"][args.split]["token_ids_sha256"]
    observed = hashlib.sha256(ids_source.contiguous().numpy().tobytes()).hexdigest()
    if observed != expected:
        raise ValueError("split tensor hash mismatch")
    ids = ids_source.long()

    device = torch.device("cuda")
    started = time.perf_counter()
    teacher = layer_zero(MODEL, ids, device)
    student = teacher.clone()
    layers = []
    overlaps = []
    for layer_index in range(1, 27):
        layer_started = time.perf_counter()
        layer, _ = load_decoder_layer(MODEL, layer_index, device)
        teacher, _, teacher_ids, _ = forward_capture(layer, teacher)
        quantize_routed_experts_q5_(layer)
        student, _, student_ids, _ = forward_capture(layer, student)
        overlap = float(topk_overlap(student_ids.cpu(), teacher_ids.cpu()))
        overlaps.append(overlap)
        row = {
            "layer": layer_index,
            "hidden_error": tensor_error(student, teacher),
            "route_overlap": overlap,
            "seconds": time.perf_counter() - layer_started,
        }
        layers.append(row)
        print(json.dumps(row), flush=True)
        del layer, teacher_ids, student_ids
        gc.collect(); torch.cuda.empty_cache()

    teacher_logits = final_logits(teacher)
    student_logits = final_logits(student)
    teacher_metrics = final_metrics(teacher_logits, teacher_logits, ids)
    q5_metrics = final_metrics(student_logits, teacher_logits, ids)
    rel = q5_metrics["relative_cross_entropy_increase"]
    finite = bool(math.isfinite(rel) and torch.isfinite(student_logits).all())
    median_overlap = float(torch.tensor(overlaps).median())
    if args.split == "validation":
        passed = finite and rel <= 0.02 and len(layers) == 26
        status = "validation_pass_test_authorized" if passed else "validation_closed"
        overall = False
    else:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        passed = finite and rel <= 0.02 and validation["q5"]["relative_cross_entropy_increase"] <= 0.02
        overall = passed and median_overlap >= 0.95 and validation["median_route_overlap"] >= 0.95
        status = "quality_pass" if overall else "quality_closed"
    payload = {
        "kind": "streamq5_moe_p14a_deepseek_v2_lite_q5_replication",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "status": status,
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "input_lock_sha256": sha256(INPUT_LOCK),
            "input_artifact_sha256": sha256(INPUTS),
            "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
        },
        "semantics": "per-row symmetric Q5; FP32 code selection; BF16 stored scale; routed experts only",
        "teacher": teacher_metrics,
        "q5": q5_metrics,
        "logit_error": regression_metrics(student_logits.cpu(), teacher_logits.cpu()),
        "median_route_overlap": median_overlap,
        "layers": layers,
        "gates": {
            "finite": finite,
            "all_26_moe_layers": len(layers) == 26,
            "relative_ce_le_2pct": rel <= 0.02,
            "median_route_overlap_ge_95pct": median_overlap >= 0.95,
        },
        "overall_pass": overall,
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": "Full-depth DeepSeek-V2-Lite Q5 routed-expert quality only; reused historical test split is corroborative; no physical bank, cache, kernel or speed claim.",
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": status, "relative_ce": rel, "median_route_overlap": median_overlap, "runtime_seconds": payload["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
