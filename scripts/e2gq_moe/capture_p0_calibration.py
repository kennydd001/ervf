from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone

import psutil
import pyarrow.parquet as pq
import safetensors
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import (
    checkpoint_weight_map,
    load_qwen_decoder_layer,
    load_token_embeddings,
)


MODEL_REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
INPUT_LOCK = ROOT / "reports/e2gq_moe/p0_input_lock.json"
PREREG = ROOT / "reports/e2gq_moe/P0_FULL_BANK_PREREGISTRATION.md"
RUN_DIR = ROOT / "reports/runs/e2gq_moe/p0_calibration"
LAYER_REPORT_DIR = ROOT / "reports/e2gq_moe/p0_capture_layers"
RESULT = ROOT / "reports/e2gq_moe/p0_capture_result.json"
CONTEXTS = 32
CONTEXT_TOKENS = 1024
CHUNK_CONTEXTS = 2
MINIMUM_ROWS = 128
CUDA_LIMIT = int(7.5 * 2**30)
RSS_LIMIT = 32 * 2**30


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_ids() -> torch.Tensor:
    parquet = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet"
    model = ROOT / "models/qwen3-30b-a3b-base"
    texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    joined = "\n\n".join(text for text in texts if text and text.strip())
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
    ids = tokenizer.encode(
        joined, add_special_tokens=False, truncation=True,
        max_length=CONTEXTS * CONTEXT_TOKENS,
    )
    return torch.tensor(ids, dtype=torch.int64).reshape(CONTEXTS, CONTEXT_TOKENS)


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


class RouteCapture:
    def __init__(self, block):
        self.block = block
        self.x = None
        self.ids = None
        self.ids_exact = False
        self.logit_error = None
        self.handles = []

    def __enter__(self):
        def pre(_module, inputs):
            flat = inputs[0].reshape(-1, inputs[0].shape[-1])
            logits = self.block.gate(flat)
            self.x = flat.detach().to("cpu").contiguous()
            self.ids = torch.topk(torch.softmax(logits, dim=1, dtype=torch.float), self.block.top_k, dim=-1).indices.detach()
            self._logits = logits.detach()

        def post(_module, _inputs, output):
            official_logits = output[1]
            official_ids = torch.topk(torch.softmax(official_logits, dim=1, dtype=torch.float), self.block.top_k, dim=-1).indices
            self.ids_exact = bool(torch.equal(self.ids, official_ids))
            self.logit_error = float((self._logits.float() - official_logits.float()).abs().max())
            self.ids = official_ids.to(torch.int16).cpu().contiguous()

        self.handles = [self.block.register_forward_pre_hook(pre), self.block.register_forward_hook(post)]
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in reversed(self.handles):
            handle.remove()


if __name__ == "__main__":
    if RESULT.exists():
        raise FileExistsError(f"refusing to overwrite {RESULT}")
    if not PREREG.is_file() or not INPUT_LOCK.is_file():
        raise FileNotFoundError("P0 preregistration and input lock are required")
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    ids = input_ids()
    if tensor_sha256(ids) != lock["input_ids_sha256"]:
        raise ValueError("reconstructed P0 input IDs differ from the lock")
    if transformers.__version__ != "4.51.3":
        raise RuntimeError(f"transformers version drift: {transformers.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("P0 capture requires CUDA")

    model_dir = ROOT / "models/qwen3-30b-a3b-base"
    config = Qwen3MoeConfig.from_pretrained(model_dir, local_files_only=True)
    expected = (config.num_hidden_layers, config.num_experts, config.num_experts_per_tok)
    if expected != (48, 128, 8):
        raise RuntimeError(f"unexpected Qwen MoE config {expected}")
    config._attn_implementation = "sdpa"
    device = torch.device("cuda")
    process = psutil.Process()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    torch.cuda.reset_peak_memory_stats(device)
    peak_rss = process.memory_info().rss
    weight_map = checkpoint_weight_map(model_dir)
    hidden = load_token_embeddings(model_dir, ids, "cpu", weight_map).contiguous()
    position_ids = torch.arange(CONTEXT_TOKENS, device=device).unsqueeze(0)
    manifests = {}
    coverage = {}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LAYER_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for layer_index in range(48):
        artifact = RUN_DIR / f"layer_{layer_index:02d}.safetensors"
        layer_report = LAYER_REPORT_DIR / f"layer_{layer_index:02d}.json"
        if artifact.exists() or layer_report.exists():
            raise FileExistsError(f"refusing to overwrite layer {layer_index} capture")
        layer_started = time.perf_counter()
        layer = load_qwen_decoder_layer(model_dir, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        next_hidden_chunks = []
        x_chunks = []
        route_chunks = []
        exact = True
        maximum_logit_error = 0.0
        for start in range(0, CONTEXTS, CHUNK_CONTEXTS):
            batch = hidden[start:start + CHUNK_CONTEXTS].to(device=device, non_blocking=False)
            with torch.inference_mode():
                position_embeddings = rotary(batch, position_ids)
                with RouteCapture(layer.mlp) as capture:
                    output = layer(
                        batch, attention_mask=None, position_ids=position_ids,
                        use_cache=False, output_attentions=False,
                        output_router_logits=False,
                        cache_position=position_ids.squeeze(0),
                        position_embeddings=position_embeddings,
                    )[0]
            if capture.x is None or capture.ids is None:
                raise RuntimeError("MoE route hook did not capture")
            exact &= capture.ids_exact
            maximum_logit_error = max(maximum_logit_error, capture.logit_error)
            x_chunks.append(capture.x)
            route_chunks.append(capture.ids)
            next_hidden_chunks.append(output.detach().cpu().contiguous())
            del batch, output, position_embeddings
        layer_x = torch.cat(x_chunks, dim=0)
        layer_routes = torch.cat(route_chunks, dim=0)
        hidden = torch.cat(next_hidden_chunks, dim=0)
        if layer_x.shape != (CONTEXTS * CONTEXT_TOKENS, 2048):
            raise RuntimeError(f"unexpected layer input shape {tuple(layer_x.shape)}")
        if layer_routes.shape != (CONTEXTS * CONTEXT_TOKENS, 8):
            raise RuntimeError(f"unexpected route shape {tuple(layer_routes.shape)}")
        counts = torch.bincount(layer_routes.long().reshape(-1), minlength=128)
        save_file(
            {"moe_input": layer_x, "router_ids": layer_routes}, artifact,
            metadata={
                "kind": "e2gq_p0_calibration", "layer": str(layer_index),
                "input_lock_sha256": sha256_file(INPUT_LOCK),
                "model_revision": MODEL_REVISION,
            },
        )
        artifact_sha = sha256_file(artifact)
        payload = {
            "kind": "e2gq_p0_calibration_layer", "layer": layer_index,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
            "artifact_sha256": artifact_sha, "input_lock_sha256": sha256_file(INPUT_LOCK),
            "router_counts": counts.tolist(), "minimum_rows": int(counts.min()),
            "maximum_rows": int(counts.max()), "experts_below_128": int((counts < MINIMUM_ROWS).sum()),
            "route_ids_exact": exact, "router_logit_maximum_absolute_error": maximum_logit_error,
            "finite_moe_input": bool(torch.isfinite(layer_x.float()).all()),
            "elapsed_seconds": time.perf_counter() - layer_started,
        }
        layer_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifests[str(layer_index)] = {
            "artifact": payload["artifact"], "artifact_sha256": artifact_sha,
            "report": str(layer_report.relative_to(ROOT)).replace("\\", "/"),
            "report_sha256": sha256_file(layer_report),
        }
        coverage[str(layer_index)] = {
            "minimum_rows": payload["minimum_rows"], "maximum_rows": payload["maximum_rows"],
            "experts_below_128": payload["experts_below_128"],
        }
        del layer, rotary, layer_x, layer_routes, x_chunks, route_chunks, next_hidden_chunks
        gc.collect()
        torch.cuda.empty_cache()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if torch.cuda.max_memory_allocated(device) > CUDA_LIMIT or peak_rss > RSS_LIMIT:
            raise MemoryError("P0 capture exceeded preregistered resource limit")
        print(json.dumps({
            "layer": layer_index, **coverage[str(layer_index)],
            "elapsed_seconds": payload["elapsed_seconds"],
        }), flush=True)

    coverage_pass = all(row["experts_below_128"] == 0 for row in coverage.values())
    result = {
        "kind": "e2gq_p0_calibration_capture", "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "coverage_positive" if coverage_pass else "coverage_negative",
        "coverage_pass": coverage_pass, "input_lock_sha256": sha256_file(INPUT_LOCK),
        "layers": coverage, "artifacts": manifests,
        "elapsed_seconds": time.perf_counter() - started,
        "hardware": {
            "platform": platform.platform(), "python": sys.version,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "safetensors": safetensors.__version__, "device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_process_rss_bytes": peak_rss,
        },
        "claim_boundary": "Calibration coverage only; no GPTQ rate, quality, runtime or Eureka claim.",
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
