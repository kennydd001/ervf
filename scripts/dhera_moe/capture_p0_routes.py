from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from unittest import mock

import psutil
import safetensors
import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_qwen_decoder_layer, load_token_embeddings


DOMAINS = ("general", "code", "math", "multilingual", "instruction")
INPUT_LOCK = ROOT / "reports/dhera_moe/p0_input_lock.json"
INPUT_ARTIFACT = ROOT / "reports/runs/dhera_moe/p0_input_ids.safetensors"
BASE_LOCK = ROOT / "reports/dhera_moe/p0_base_lock.json"
PREREG = ROOT / "reports/dhera_moe/P0_BUDGET_CACHE_PREREGISTRATION.md"
RUN_DIR = ROOT / "reports/runs/dhera_moe/p0_routes"
LAYER_DIR = ROOT / "reports/dhera_moe/p0_route_layers"
RESULT = ROOT / "reports/dhera_moe/p0_route_capture.json"
ORIGINAL_TOPK = torch.topk


def sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if RESULT.exists():
        raise FileExistsError(f"refusing to overwrite {RESULT}")
    if not all(path.is_file() for path in (INPUT_LOCK, INPUT_ARTIFACT, BASE_LOCK, PREREG)):
        raise FileNotFoundError("DHERA locks/preregistration missing")
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    if sha256(INPUT_ARTIFACT) != lock["artifact_sha256"]:
        raise ValueError("DHERA input artifact hash mismatch")
    inputs = load_file(INPUT_ARTIFACT)
    if set(inputs) != set(DOMAINS):
        raise ValueError("domain set mismatch")
    if transformers.__version__ != "4.51.3" or not torch.cuda.is_available():
        raise RuntimeError("pinned transformers and CUDA required")

    model_dir = ROOT / "models/qwen3-30b-a3b-base"
    config = Qwen3MoeConfig.from_pretrained(model_dir, local_files_only=True)
    config._attn_implementation = "sdpa"
    device = torch.device("cuda")
    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats(device)
    peak_rss = process.memory_info().rss
    timer = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    weight_map = checkpoint_weight_map(model_dir)
    hidden = {domain: load_token_embeddings(model_dir, inputs[domain], "cpu", weight_map).contiguous() for domain in DOMAINS}
    position_ids = torch.arange(1024, device=device).unsqueeze(0)
    manifests = {}
    RUN_DIR.mkdir(parents=True, exist_ok=True); LAYER_DIR.mkdir(parents=True, exist_ok=True)

    for layer_index in range(48):
        artifact = RUN_DIR / f"layer_{layer_index:02d}.safetensors"
        report_path = LAYER_DIR / f"layer_{layer_index:02d}.json"
        if artifact.exists() or report_path.exists():
            raise FileExistsError(f"refusing to overwrite layer {layer_index}")
        layer_timer = time.perf_counter()
        layer = load_qwen_decoder_layer(model_dir, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        routes = {}; domain_reports = {}; exact = True
        for domain in DOMAINS:
            next_chunks, ids_chunks = [], []
            for start in range(0, 32, 2):
                batch = hidden[domain][start:start + 2].to(device)
                captured = []

                def intercepted_topk(input, k, *args, **kwargs):
                    output = ORIGINAL_TOPK(input, k, *args, **kwargs)
                    if input.ndim == 2 and input.shape[-1] == 128 and k == 8:
                        captured.append(output.indices.detach())
                    return output

                with torch.inference_mode():
                    position_embeddings = rotary(batch, position_ids)
                    with mock.patch.object(torch, "topk", side_effect=intercepted_topk):
                        output = layer(
                            batch, attention_mask=None, position_ids=position_ids,
                            use_cache=False, output_attentions=False, output_router_logits=False,
                            cache_position=position_ids.squeeze(0), position_embeddings=position_embeddings,
                        )[0]
                if len(captured) != 1:
                    raise RuntimeError(f"expected one official topk call, got {len(captured)}")
                ids_chunks.append(captured[0].to(torch.int16).cpu().contiguous())
                next_chunks.append(output.detach().cpu().contiguous())
                del batch, output, position_embeddings
            ids = torch.cat(ids_chunks)
            hidden[domain] = torch.cat(next_chunks)
            counts = torch.bincount(ids.long().reshape(-1), minlength=128)
            routes[f"{domain}_router_ids"] = ids
            domain_reports[domain] = {"counts": counts.tolist(), "total_invocations": int(counts.sum()), "zero_experts": int((counts == 0).sum())}
            exact &= ids.shape == (32768, 8)
            del ids, ids_chunks, next_chunks
        save_file(routes, artifact, metadata={"kind": "dhera_p0_validation_routes", "layer": str(layer_index), "input_lock_sha256": sha256(INPUT_LOCK)})
        report = {
            "kind": "dhera_p0_validation_route_layer", "layer": layer_index,
            "completed_utc": datetime.now(timezone.utc).isoformat(), "domains": domain_reports,
            "official_topk_captured_exactly_once_per_chunk": exact,
            "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
            "artifact_sha256": sha256(artifact), "elapsed_seconds": time.perf_counter() - layer_timer,
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifests[str(layer_index)] = {"artifact": report["artifact"], "artifact_sha256": report["artifact_sha256"], "report": str(report_path.relative_to(ROOT)).replace("\\", "/"), "report_sha256": sha256(report_path)}
        del layer, rotary, routes
        gc.collect(); torch.cuda.empty_cache(); peak_rss = max(peak_rss, process.memory_info().rss)
        if torch.cuda.max_memory_allocated(device) > 7.5 * 2**30 or peak_rss > 32 * 2**30:
            raise MemoryError("DHERA route capture exceeded resource gate")
        print(json.dumps({"layer": layer_index, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)

    payload = {
        "kind": "dhera_p0_validation_route_capture", "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "input_lock_sha256": sha256(INPUT_LOCK), "base_lock_sha256": sha256(BASE_LOCK),
        "artifacts": manifests, "elapsed_seconds": time.perf_counter() - timer,
        "hardware": {"platform": platform.platform(), "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__, "safetensors": safetensors.__version__, "device": torch.cuda.get_device_name(device), "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_process_rss_bytes": peak_rss},
        "claim_boundary": "Out-of-sample official routes only; cache result and quality not yet computed.",
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "elapsed_seconds": payload["elapsed_seconds"]}, indent=2))
