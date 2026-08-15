from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import psutil
import safetensors
import torch
import transformers
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import (
    checkpoint_weight_map,
    load_qwen_decoder_layer,
    load_token_embeddings,
)


ORIGINAL_TOPK = torch.topk
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS = 48
EXPERTS = 128
TOP_K = 8
CONTEXTS = 128
CONTEXT_TOKENS = 1_024
CHUNK_CONTEXTS = 2
THRESHOLD = 128
CUDA_LIMIT = int(7.5 * 2**30)
RSS_LIMIT = 32 * 2**30
MODEL = ROOT / "models/qwen3-30b-a3b-base"
PREREG = ROOT / "reports/qwen_gptq_bank/P0_FULL_BANK_PREREGISTRATION.md"
INPUT_LOCK = ROOT / "reports/qwen_gptq_bank/p0_input_lock.json"
INPUT_ARTIFACT = ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_input_ids.safetensors"
HERA_ROUTES = ROOT / "reports/runs/hera_moe/p0_routes"
DHERA_ROUTES = ROOT / "reports/runs/dhera_moe/p0_routes"
RUN_DIR = ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_routes"
LAYER_DIR = ROOT / "reports/qwen_gptq_bank/p0_supplement_route_layers"
RESULT = ROOT / "reports/qwen_gptq_bank/p0_coverage_result.json"
REPORT = ROOT / "reports/qwen_gptq_bank/P0_COVERAGE_REPORT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfficialTopKCapture:
    def __init__(self):
        self.calls: list[torch.Tensor] = []
        self._patcher = None

    def __enter__(self):
        def intercepted(input, k, *args, **kwargs):
            output = ORIGINAL_TOPK(input, k, *args, **kwargs)
            if input.ndim == 2 and input.shape[-1] == EXPERTS and k == TOP_K:
                self.calls.append(output.indices.detach().to(torch.int16).cpu().contiguous())
            return output

        self._patcher = mock.patch.object(torch, "topk", side_effect=intercepted)
        self._patcher.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._patcher is not None:
            self._patcher.stop()

    def result(self) -> torch.Tensor:
        if len(self.calls) != 1:
            raise RuntimeError(f"expected one official router top-k call, got {len(self.calls)}")
        return self.calls[0]


def load_existing(path: Path, expected_lock_hash: str, layer: int) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("input_lock_sha256") != expected_lock_hash or metadata.get("layer") != str(layer):
            raise ValueError(f"existing route checkpoint metadata mismatch at layer {layer}")
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    if set(tensors) != {f"{domain}_router_ids" for domain in DOMAINS}:
        raise ValueError(f"existing route checkpoint key mismatch at layer {layer}")
    return tensors


if __name__ == "__main__":
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite the full-bank coverage result")
    if not PREREG.is_file() or not INPUT_LOCK.is_file():
        raise FileNotFoundError("full-bank preregistration or input lock is missing")
    if transformers.__version__ != "4.51.3" or not torch.cuda.is_available():
        raise RuntimeError("pinned transformers 4.51.3 and CUDA are required")
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    input_lock_hash = sha256(INPUT_LOCK)
    if sha256(INPUT_ARTIFACT) != lock["artifact_sha256"]:
        raise ValueError("supplement input artifact hash mismatch")
    inputs = load_file(INPUT_ARTIFACT)
    if set(inputs) != set(DOMAINS):
        raise ValueError("supplement domain set differs from input lock")

    config = Qwen3MoeConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = "sdpa"
    if (config.num_hidden_layers, config.num_experts, config.num_experts_per_tok) != (48, 128, 8):
        raise RuntimeError("unexpected Qwen MoE configuration")
    device = torch.device("cuda")
    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats(device)
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    weight_map = checkpoint_weight_map(MODEL)
    hidden = {
        domain: load_token_embeddings(MODEL, inputs[domain], "cpu", weight_map).contiguous()
        for domain in DOMAINS
    }
    position_ids = torch.arange(CONTEXT_TOKENS, device=device).unsqueeze(0)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LAYER_DIR.mkdir(parents=True, exist_ok=True)
    manifests = {}

    for layer_index in range(LAYERS):
        layer_started = time.perf_counter()
        artifact = RUN_DIR / f"layer_{layer_index:02d}.safetensors"
        layer_report = LAYER_DIR / f"layer_{layer_index:02d}.json"
        if artifact.exists() != layer_report.exists():
            raise RuntimeError(f"partial layer checkpoint at {layer_index}")
        was_existing = artifact.exists()
        existing = load_existing(artifact, input_lock_hash, layer_index) if was_existing else None
        layer = load_qwen_decoder_layer(MODEL, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        routes = {}
        domain_counts = {}
        for domain in DOMAINS:
            next_chunks, route_chunks = [], []
            for begin in range(0, CONTEXTS, CHUNK_CONTEXTS):
                batch = hidden[domain][begin : begin + CHUNK_CONTEXTS].to(device)
                with torch.inference_mode():
                    position_embeddings = rotary(batch, position_ids)
                    with OfficialTopKCapture() as capture:
                        output = layer(
                            batch,
                            attention_mask=None,
                            position_ids=position_ids,
                            use_cache=False,
                            output_attentions=False,
                            output_router_logits=False,
                            cache_position=position_ids.squeeze(0),
                            position_embeddings=position_embeddings,
                        )[0]
                route_chunks.append(capture.result())
                next_chunks.append(output.detach().cpu().contiguous())
                del batch, output, position_embeddings
            ids = torch.cat(route_chunks)
            hidden[domain] = torch.cat(next_chunks)
            key = f"{domain}_router_ids"
            routes[key] = ids
            counts = torch.bincount(ids.long().reshape(-1), minlength=EXPERTS)
            domain_counts[domain] = counts.tolist()
            if existing is not None and not torch.equal(ids, existing[key]):
                raise RuntimeError(f"resumed route checkpoint is not reproducible: layer {layer_index}, {domain}")
            del route_chunks, next_chunks, ids

        if existing is None:
            save_file(routes, artifact, metadata={
                "kind": "qwen_gptq_bank_p0_supplement_routes",
                "layer": str(layer_index),
                "input_lock_sha256": input_lock_hash,
            })
            payload = {
                "kind": "qwen_gptq_bank_p0_supplement_route_layer",
                "layer": layer_index,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "input_lock_sha256": input_lock_hash,
                "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
                "artifact_sha256": sha256(artifact),
                "domain_counts": domain_counts,
                "elapsed_seconds": time.perf_counter() - layer_started,
            }
            layer_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            payload = json.loads(layer_report.read_text(encoding="utf-8"))
            if payload["artifact_sha256"] != sha256(artifact):
                raise ValueError(f"existing layer report hash mismatch at {layer_index}")
        manifests[str(layer_index)] = {
            "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
            "artifact_sha256": sha256(artifact),
            "report": str(layer_report.relative_to(ROOT)).replace("\\", "/"),
            "report_sha256": sha256(layer_report),
        }
        del layer, rotary, routes, existing
        gc.collect()
        torch.cuda.empty_cache()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if torch.cuda.max_memory_allocated(device) > CUDA_LIMIT or peak_rss > RSS_LIMIT:
            raise MemoryError("supplement route capture exceeded its resource ceiling")
        print(json.dumps({
            "layer": layer_index,
            "seconds": time.perf_counter() - layer_started,
            "resumed": was_existing,
        }), flush=True)

    coverage_layers = []
    deficient = []
    for layer_index in range(LAYERS):
        counts = torch.zeros(EXPERTS, dtype=torch.int64)
        source_totals = {}
        for label, directory in (
            ("hera", HERA_ROUTES), ("dhera", DHERA_ROUTES), ("supplement", RUN_DIR)
        ):
            with safe_open(directory / f"layer_{layer_index:02d}.safetensors", framework="pt", device="cpu") as handle:
                source = torch.zeros(EXPERTS, dtype=torch.int64)
                for key in handle.keys():
                    source += torch.bincount(handle.get_tensor(key).long().reshape(-1), minlength=EXPERTS)
            counts += source
            source_totals[label] = int(source.sum())
        low = torch.where(counts < THRESHOLD)[0].tolist()
        coverage_layers.append({
            "layer": layer_index,
            "minimum_rows": int(counts.min()),
            "maximum_rows": int(counts.max()),
            "experts_below_128": len(low),
            "source_invocations": source_totals,
            "counts": counts.tolist(),
        })
        deficient.extend(
            {"layer": layer_index, "expert": expert, "rows": int(counts[expert])}
            for expert in low
        )

    coverage_pass = not deficient
    payload = {
        "kind": "qwen_gptq_bank_p0_coverage_result",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "coverage_pass" if coverage_pass else "coverage_fail",
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "input_lock_sha256": input_lock_hash,
            "hera_input_sha256": sha256(ROOT / "reports/runs/hera_moe/p0_input_ids.safetensors"),
            "dhera_input_sha256": sha256(ROOT / "reports/runs/dhera_moe/p0_input_ids.safetensors"),
        },
        "manifests": manifests,
        "coverage": {
            "required_layer_expert_pairs": LAYERS * EXPERTS,
            "threshold_rows": THRESHOLD,
            "minimum_rows": min(row["minimum_rows"] for row in coverage_layers),
            "deficient_pairs": deficient,
            "layers": coverage_layers,
            "all_pairs_pass": coverage_pass,
        },
        "resources": {
            "device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_process_rss_bytes": peak_rss,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "software": {
            "platform": platform.platform(), "python": sys.version,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
        },
        "claim_boundary": "Official route coverage only; calibration activations and GPTQ bank are not yet produced.",
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Qwen GPTQ Bank — P0 coverage",
        "",
        f"Uitkomst: **{payload['status']}**.",
        "",
        f"Alle 6.144 layer-expertparen halen 128 echte routed rijen: `{coverage_pass}`. "
        f"Minimum: **{payload['coverage']['minimum_rows']}**; ontbrekende paren: **{len(deficient)}**.",
        "",
        "De meting combineert de immutable HERA- en DHERA-routes met de vooraf fysiek vergrendelde supplementroutes. "
        "Er is nog geen activatiecapture of GPTQ-kwantisatie geopend.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "minimum_rows": payload["coverage"]["minimum_rows"],
        "deficient_pairs": len(deficient),
        "elapsed_seconds": payload["resources"]["elapsed_seconds"],
    }, indent=2))
