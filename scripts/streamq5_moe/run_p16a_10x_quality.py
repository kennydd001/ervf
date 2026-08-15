from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import (
    checkpoint_weight_map,
    load_checkpoint_tensors,
    load_qwen_decoder_layer,
)
from scripts.streamq5_moe.run_p0c_model_quality import (
    forward_layer,
    quantize_experts_q5_,
    quantize_groupwise_,
    quantize_trunk_,
    selected_embeddings,
    tensor_error,
)


MODEL = ROOT / "models/qwen3-30b-a3b-base"
SOURCE = ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_input_ids.safetensors"
SOURCE_LOCK = ROOT / "reports/qwen_gptq_bank/p0_input_lock.json"
PREREG = ROOT / "reports/streamq5_moe/P16A_10X_QUALITY_BOOTSTRAP_PREREGISTRATION.md"
OUTPUT = ROOT / "reports/streamq5_moe/p16a_10x_quality.json"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
CONTEXTS_PER_DOMAIN = 20
CONTEXT = 128
CHUNK = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forward_chunks(layer, rotary, hidden: torch.Tensor, device: torch.device) -> torch.Tensor:
    parts = []
    for start in range(0, hidden.shape[0], CHUNK):
        parts.append(forward_layer(layer, rotary, hidden[start:start + CHUNK], device))
    return torch.cat(parts, dim=0)


@torch.inference_mode()
def context_metrics(hidden, ids, norm_weight, head, device, teacher_top1=None):
    ce = []
    predictions = []
    for index in range(hidden.shape[0]):
        state = hidden[index:index + 1].to(device)
        normalized = state.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = (normalized * torch.rsqrt(variance + 1e-6)).to(torch.bfloat16) * norm_weight
        logits = F.linear(normalized, head)
        labels = ids[index, 1:].to(device)
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean")
        ce.append(float(loss))
        predictions.append(logits[:, :-1].argmax(-1).cpu())
        del state, normalized, logits, labels, loss
    top1 = torch.cat(predictions, dim=0)
    agreement = None if teacher_top1 is None else float((top1 == teacher_top1).float().mean())
    return np.asarray(ce, dtype=np.float64), top1, agreement


def bootstrap_relative(teacher_ce: np.ndarray, candidate_ce: np.ndarray) -> dict:
    rng = np.random.default_rng(20260812)
    n = teacher_ce.size
    samples = np.empty(10_000, dtype=np.float64)
    for i in range(samples.size):
        indices = rng.integers(0, n, n)
        t = teacher_ce[indices].mean()
        samples[i] = (candidate_ce[indices].mean() - t) / t
    return {
        "seed": 20260812,
        "resamples": 10_000,
        "p2_5": float(np.percentile(samples, 2.5)),
        "p50": float(np.percentile(samples, 50)),
        "p97_5": float(np.percentile(samples, 97.5)),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    if sha256(SOURCE) != lock["artifact_sha256"]:
        raise ValueError("source provenance mismatch")
    source = load_file(SOURCE)
    selected = [source[d][:CONTEXTS_PER_DOMAIN, :CONTEXT].long() for d in DOMAINS]
    ids = torch.cat(selected, dim=0).contiguous()
    selected_sha = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    if ids.shape != (100, 128):
        raise RuntimeError(f"unexpected selected shape {tuple(ids.shape)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda")
    config = Qwen3MoeConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = "sdpa"
    weight_map = checkpoint_weight_map(MODEL)
    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    teacher = selected_embeddings(MODEL, ids, device, weight_map, None)
    candidate = selected_embeddings(MODEL, ids, device, weight_map, 8)
    layers = []
    for layer_index in range(48):
        layer_started = time.perf_counter()
        layer = load_qwen_decoder_layer(MODEL, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        teacher = forward_chunks(layer, rotary, teacher, device)
        quantize_experts_q5_(layer)
        quantize_trunk_(layer, 8)
        candidate = forward_chunks(layer, rotary, candidate, device)
        error = tensor_error(candidate, teacher)
        row = {"layer": layer_index, "error": error, "seconds": time.perf_counter() - layer_started}
        layers.append(row); print(json.dumps(row), flush=True)
        del layer, rotary
        gc.collect(); torch.cuda.empty_cache()

    final = load_checkpoint_tensors(MODEL, ["model.norm.weight", "lm_head.weight"], weight_map)
    norm = final["model.norm.weight"].to(device)
    head = final["lm_head.weight"].to(device)
    teacher_context_ce, teacher_top1, _ = context_metrics(teacher, ids, norm, head, device)
    quantize_groupwise_(head, 8, row_batch=256)
    candidate_context_ce, _candidate_top1, agreement = context_metrics(candidate, ids, norm, head, device, teacher_top1)
    teacher_mean = float(teacher_context_ce.mean())
    candidate_mean = float(candidate_context_ce.mean())
    relative = (candidate_mean - teacher_mean) / teacher_mean
    bootstrap = bootstrap_relative(teacher_context_ce, candidate_context_ce)
    finite = bool(np.isfinite(teacher_context_ce).all() and np.isfinite(candidate_context_ce).all())
    gates = {
        "all_48_layers": len(layers) == 48,
        "contexts_100": ids.shape[0] == 100,
        "labels_12700": ids.shape[0] * (ids.shape[1] - 1) == 12_700,
        "finite": finite,
        "relative_ce_le_2pct": relative <= 0.02,
        "bootstrap_upper_le_2_5pct": bootstrap["p97_5"] <= 0.025,
        "top1_agreement_ge_90pct": agreement >= 0.90,
    }
    result = {
        "kind": "streamq5_moe_p16a_10x_quality_bootstrap",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "source_sha256": sha256(SOURCE),
            "source_lock_sha256": sha256(SOURCE_LOCK),
            "selected_ids_sha256": selected_sha,
            "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
        },
        "data": {"domains": list(DOMAINS), "contexts_per_domain": 20, "context": 128, "contexts": 100, "labels": 12_700},
        "teacher_context_ce": teacher_context_ce.tolist(),
        "candidate_context_ce": candidate_context_ce.tolist(),
        "teacher_ce": teacher_mean,
        "candidate_ce": candidate_mean,
        "relative_cross_entropy_increase": relative,
        "top1_agreement": agreement,
        "bootstrap_relative_ce": bootstrap,
        "final_hidden_error": tensor_error(candidate, teacher),
        "layers": layers,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "runtime": {
            "seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_rss_bytes": int(process.memory_info().rss),
        },
        "claim_boundary": "Large corroborative full-depth quality audit only; the data were previously used in route/GPTQ calibration research and are not a pristine public benchmark.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "teacher_ce": teacher_mean, "candidate_ce": candidate_mean, "relative": relative, "top1": agreement, "bootstrap": bootstrap, "gates": gates, "overall_pass": result["overall_pass"], "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
