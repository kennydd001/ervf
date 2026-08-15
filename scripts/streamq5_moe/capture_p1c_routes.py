from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import psutil
import torch
import transformers
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_qwen_decoder_layer, load_token_embeddings


ORIGINAL_TOPK = torch.topk
MODEL = ROOT / "models/qwen3-30b-a3b-base"
PREREG = ROOT / "reports/streamq5_moe/P1C_CORRECTED_ROUTE_CACHE_PREREGISTRATION.md"
LOCK = ROOT / "reports/streamq5_moe/p1c_route_input_lock.json"
EVALUATOR_LOCK = ROOT / "reports/streamq5_moe/p1c_route_evaluator_lock.json"
INPUT = ROOT / "reports/runs/streamq5_moe/p1c_fresh_route_input_ids.safetensors"
RUN_DIR = ROOT / "reports/runs/streamq5_moe/p1c_routes"
LAYER_DIR = ROOT / "reports/streamq5_moe/p1c_route_layers"
RESULT = ROOT / "reports/streamq5_moe/p1c_route_capture_result.json"
REPORT = ROOT / "reports/streamq5_moe/P1C_ROUTE_CAPTURE.md"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, EXPERTS, TOP_K, TOKENS, EXPERT_BATCH = 48, 128, 8, 1024, 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def quantize_groupwise_(value: torch.Tensor, bits: int, row_batch: int = 512) -> None:
    if value.ndim != 2 or value.shape[1] % 128:
        raise ValueError(f"matrix is not group-128 compatible: {tuple(value.shape)}")
    qmax = 15 if bits == 5 else 127 if bits == 8 else None
    if qmax is None:
        raise ValueError(bits)
    rows, columns = value.shape
    groups = columns // 128
    for start in range(0, rows, row_batch):
        end = min(rows, start + row_batch)
        work = value[start:end].float().reshape(end - start, groups, 128)
        maximum = work.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
        quantized = torch.round(work / scale).clamp(-qmax, qmax)
        stored_scale = scale.to(torch.bfloat16).float()
        value[start:end].copy_((quantized * stored_scale).reshape(end - start, columns).to(value.dtype))


def trunk_parameters(layer):
    return [(name, parameter) for name, parameter in layer.named_parameters() if parameter.ndim == 2 and ".experts." not in name]


@torch.no_grad()
def quantize_candidate_(layer) -> None:
    for _name, parameter in trunk_parameters(layer):
        quantize_groupwise_(parameter, 8)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for start in range(0, len(layer.mlp.experts), EXPERT_BATCH):
            experts = layer.mlp.experts[start : start + EXPERT_BATCH]
            weights = [getattr(expert, projection).weight for expert in experts]
            shape = weights[0].shape
            work = torch.stack(weights).reshape(-1, shape[1])
            quantize_groupwise_(work, 5, row_batch=work.shape[0])
            work = work.reshape(len(weights), *shape)
            for offset, weight in enumerate(weights):
                weight.copy_(work[offset])


class OfficialTopKCapture:
    def __init__(self):
        self.calls = []
        self.patcher = None

    def __enter__(self):
        def intercepted(input, k, *args, **kwargs):
            output = ORIGINAL_TOPK(input, k, *args, **kwargs)
            if input.ndim == 2 and input.shape[-1] == EXPERTS and k == TOP_K:
                self.calls.append(output.indices.detach().to(torch.int16).cpu().contiguous())
            return output
        self.patcher = mock.patch.object(torch, "topk", side_effect=intercepted)
        self.patcher.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.patcher is not None:
            self.patcher.stop()

    def result(self):
        if len(self.calls) != 1:
            raise RuntimeError(f"expected one official router top-k call, got {len(self.calls)}")
        return self.calls[0]


if __name__ == "__main__":
    if RESULT.exists() or REPORT.exists() or RUN_DIR.exists() or LAYER_DIR.exists():
        raise FileExistsError("refusing to overwrite P1C route capture")
    if transformers.__version__ != "4.51.3" or not torch.cuda.is_available():
        raise RuntimeError("pinned transformers 4.51.3 and CUDA required")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    evaluator_lock = json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"))
    if sha256(Path(__file__)) != evaluator_lock["evaluator_sha256"] or sha256(LOCK) != evaluator_lock["input_lock_sha256"]:
        raise ValueError("P1C route evaluator lock mismatch")
    if sha256(PREREG) != lock["preregistration_sha256"] or sha256(INPUT) != lock["artifact_sha256"]:
        raise ValueError("P1C route input provenance mismatch")
    inputs = load_file(INPUT)
    if set(inputs) != set(DOMAINS) or any(tuple(value.shape) != (1, TOKENS) for value in inputs.values()):
        raise ValueError("P1C input contract mismatch")

    device = torch.device("cuda")
    config = Qwen3MoeConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = "sdpa"
    if (config.num_hidden_layers, config.num_experts, config.num_experts_per_tok) != (LAYERS, EXPERTS, TOP_K):
        raise RuntimeError("unexpected Qwen configuration")
    weight_map = checkpoint_weight_map(MODEL)
    hidden = {}
    for domain in DOMAINS:
        value = load_token_embeddings(MODEL, inputs[domain], device, weight_map)
        quantize_groupwise_(value.reshape(-1, value.shape[-1]), 8)
        hidden[domain] = value.cpu().contiguous()
    position_ids = torch.arange(TOKENS, device=device).unsqueeze(0)
    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats(device)
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    RUN_DIR.mkdir(parents=True); LAYER_DIR.mkdir(parents=True)
    manifest = {}

    for layer_index in range(LAYERS):
        layer_started = time.perf_counter()
        layer = load_qwen_decoder_layer(MODEL, config, layer_index, device, weight_map)
        quantize_candidate_(layer)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        routes = {}
        counts = {}
        for domain in DOMAINS:
            batch = hidden[domain].to(device)
            with torch.inference_mode():
                position_embeddings = rotary(batch, position_ids)
                with OfficialTopKCapture() as capture:
                    output = layer(
                        batch, attention_mask=None, position_ids=position_ids,
                        use_cache=False, output_attentions=False, output_router_logits=False,
                        cache_position=position_ids.squeeze(0), position_embeddings=position_embeddings,
                    )[0]
            ids = capture.result()
            if tuple(ids.shape) != (TOKENS, TOP_K) or int(ids.min()) < 0 or int(ids.max()) >= EXPERTS:
                raise RuntimeError(f"invalid route IDs at layer {layer_index}, {domain}")
            routes[f"{domain}_router_ids"] = ids
            counts[domain] = torch.bincount(ids.long().reshape(-1), minlength=EXPERTS).tolist()
            hidden[domain] = output.detach().cpu().contiguous()
            del batch, position_embeddings, output, ids

        artifact = RUN_DIR / f"layer_{layer_index:02d}.safetensors"
        save_file(routes, artifact, metadata={
            "kind": "streamq5_moe_p1c_candidate_routes", "layer": str(layer_index),
            "input_lock_sha256": sha256(LOCK), "evaluator_sha256": sha256(Path(__file__)),
        })
        payload = {
            "kind": "streamq5_moe_p1c_route_layer", "layer": layer_index,
            "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"), "artifact_sha256": sha256(artifact),
            "domain_counts": counts, "seconds": time.perf_counter() - layer_started,
        }
        layer_report = LAYER_DIR / f"layer_{layer_index:02d}.json"
        layer_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest[str(layer_index)] = {"artifact": payload["artifact"], "artifact_sha256": payload["artifact_sha256"], "report_sha256": sha256(layer_report)}
        del layer, rotary, routes
        gc.collect(); torch.cuda.empty_cache()
        peak_rss = max(peak_rss, process.memory_info().rss)
        print(json.dumps({"layer": layer_index, "seconds": payload["seconds"]}), flush=True)

    finite = all(torch.isfinite(value.float()).all().item() for value in hidden.values())
    result = {
        "kind": "streamq5_moe_p1c_route_capture", "status": "route_capture_complete" if finite else "route_capture_nonfinite",
        "started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"preregistration_sha256": sha256(PREREG), "input_lock_sha256": sha256(LOCK), "input_artifact_sha256": sha256(INPUT), "evaluator_lock_sha256": sha256(EVALUATOR_LOCK), "evaluator_sha256": sha256(Path(__file__)), "model_index_sha256": sha256(MODEL / "model.safetensors.index.json")},
        "model_variant": "q5_experts_int8_trunk", "layers": LAYERS, "domains": list(DOMAINS), "tokens_per_domain": TOKENS,
        "scale_semantics": "codes selected with FP32 maxabs scale; scale rounded to BF16 before dequantization; output rounded BF16",
        "manifests": manifest, "controls": {"all_hidden_finite": finite, "all_layers": len(manifest) == LAYERS, "all_route_ids_valid": True, "fresh_inputs": lock["exact_128_context_disjoint_from_prior_decisions"]},
        "runtime": {"seconds": time.perf_counter() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_rss_bytes": peak_rss},
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(f"# STREAMQ5-MoE P1C - corrected-semantics routecapture\n\nUitkomst: **{result['status']}**. Alle 48 lagen, vijf domeinen en 1.024 tokens per domein zijn opgeslagen met de fysieke BF16-schaalsemantiek.\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "runtime": result["runtime"]}, indent=2), flush=True)
    if not finite:
        raise SystemExit(1)
