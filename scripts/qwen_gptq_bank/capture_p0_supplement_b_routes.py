from __future__ import annotations

import gc
import hashlib
import json
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
from moe_lab.rsiv_moe.qwen_stream import (
    checkpoint_weight_map, load_qwen_decoder_layer, load_token_embeddings,
)


ORIGINAL_TOPK = torch.topk
DOMAINS = ("math", "instruction")
ALL_DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, EXPERTS, TOP_K, CONTEXT_TOKENS = 48, 128, 8, 1_024
CHUNK_CONTEXTS = 2
MODEL = ROOT / "models/qwen3-30b-a3b-base"
PREREG = ROOT / "reports/qwen_gptq_bank/P0_SUPPLEMENT_B_PREREGISTRATION.md"
INPUT_LOCK = ROOT / "reports/qwen_gptq_bank/p0_supplement_b_input_lock.json"
INPUT_ARTIFACT = ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_b_input_ids.safetensors"
RUN_DIR = ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_b_routes"
LAYER_DIR = ROOT / "reports/qwen_gptq_bank/p0_supplement_b_route_layers"
RESULT = ROOT / "reports/qwen_gptq_bank/p0_coverage_result_b.json"
REPORT = ROOT / "reports/qwen_gptq_bank/P0_COVERAGE_REPORT_B.md"
ROUTE_SOURCES = (
    ("hera", ROOT / "reports/runs/hera_moe/p0_routes", ALL_DOMAINS),
    ("dhera", ROOT / "reports/runs/dhera_moe/p0_routes", ALL_DOMAINS),
    ("supplement_a", ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_routes", ALL_DOMAINS),
    ("supplement_b", RUN_DIR, DOMAINS),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfficialTopKCapture:
    def __init__(self):
        self.calls = []
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

    def result(self):
        if len(self.calls) != 1:
            raise RuntimeError(f"expected one official top-k call, got {len(self.calls)}")
        return self.calls[0]


if __name__ == "__main__":
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite supplement-B coverage result")
    if transformers.__version__ != "4.51.3" or not torch.cuda.is_available():
        raise RuntimeError("pinned transformers 4.51.3 and CUDA are required")
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    lock_hash = sha256(INPUT_LOCK)
    if sha256(INPUT_ARTIFACT) != lock["artifact_sha256"] or sha256(PREREG) != lock["preregistration_sha256"]:
        raise ValueError("supplement-B lock integrity failure")
    inputs = load_file(INPUT_ARTIFACT)
    if set(inputs) != set(DOMAINS):
        raise ValueError("supplement-B domain set mismatch")

    config = Qwen3MoeConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = "sdpa"
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
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
            raise RuntimeError(f"partial supplement-B checkpoint at layer {layer_index}")
        was_existing = artifact.exists()
        existing = None
        if was_existing:
            with safe_open(artifact, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("input_lock_sha256") != lock_hash:
                    raise ValueError(f"supplement-B checkpoint lock mismatch at layer {layer_index}")
                existing = {key: handle.get_tensor(key) for key in handle.keys()}
        layer = load_qwen_decoder_layer(MODEL, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        routes = {}
        domain_counts = {}
        for domain in DOMAINS:
            state = hidden[domain]
            next_chunks, route_chunks = [], []
            for begin in range(0, state.shape[0], CHUNK_CONTEXTS):
                batch = state[begin : begin + CHUNK_CONTEXTS].to(device)
                with torch.inference_mode():
                    position_embeddings = rotary(batch, position_ids)
                    with OfficialTopKCapture() as capture:
                        output = layer(
                            batch, attention_mask=None, position_ids=position_ids,
                            use_cache=False, output_attentions=False, output_router_logits=False,
                            cache_position=position_ids.squeeze(0), position_embeddings=position_embeddings,
                        )[0]
                route_chunks.append(capture.result())
                next_chunks.append(output.detach().cpu().contiguous())
                del batch, output, position_embeddings
            ids = torch.cat(route_chunks)
            hidden[domain] = torch.cat(next_chunks)
            key = f"{domain}_router_ids"
            routes[key] = ids
            domain_counts[domain] = torch.bincount(ids.long().reshape(-1), minlength=EXPERTS).tolist()
            if existing is not None and not torch.equal(existing[key], ids):
                raise RuntimeError(f"supplement-B resume mismatch at layer {layer_index}, {domain}")
        if not was_existing:
            save_file(routes, artifact, metadata={
                "kind": "qwen_gptq_bank_p0_supplement_b_routes",
                "layer": str(layer_index), "input_lock_sha256": lock_hash,
            })
            payload = {
                "kind": "qwen_gptq_bank_p0_supplement_b_route_layer",
                "layer": layer_index, "completed_utc": datetime.now(timezone.utc).isoformat(),
                "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
                "artifact_sha256": sha256(artifact), "domain_counts": domain_counts,
                "elapsed_seconds": time.perf_counter() - layer_started,
            }
            layer_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            payload = json.loads(layer_report.read_text(encoding="utf-8"))
            if payload["artifact_sha256"] != sha256(artifact):
                raise ValueError(f"supplement-B report mismatch at layer {layer_index}")
        manifests[str(layer_index)] = {
            "artifact_sha256": sha256(artifact), "report_sha256": sha256(layer_report),
        }
        del layer, rotary, routes, existing
        gc.collect(); torch.cuda.empty_cache()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if torch.cuda.max_memory_allocated(device) > int(7.5 * 2**30) or peak_rss > 32 * 2**30:
            raise MemoryError("supplement-B route capture exceeded resource ceiling")
        print(json.dumps({
            "layer": layer_index, "seconds": time.perf_counter() - layer_started,
            "resumed": was_existing,
        }), flush=True)

    layers, deficient = [], []
    for layer_index in range(LAYERS):
        total = torch.zeros(EXPERTS, dtype=torch.int64)
        source_counts = {}
        for source_name, directory, domains in ROUTE_SOURCES:
            with safe_open(directory / f"layer_{layer_index:02d}.safetensors", framework="pt", device="cpu") as handle:
                counts = torch.zeros(EXPERTS, dtype=torch.int64)
                for domain in domains:
                    counts += torch.bincount(
                        handle.get_tensor(f"{domain}_router_ids").long().reshape(-1), minlength=EXPERTS
                    )
            total += counts
            source_counts[source_name] = int(counts.sum())
        low = torch.where(total < 128)[0].tolist()
        layers.append({
            "layer": layer_index, "minimum_rows": int(total.min()),
            "maximum_rows": int(total.max()), "experts_below_128": len(low),
            "counts": total.tolist(), "source_invocations": source_counts,
        })
        deficient.extend(
            {"layer": layer_index, "expert": expert, "rows": int(total[expert])}
            for expert in low
        )
    passed = not deficient
    payload = {
        "kind": "qwen_gptq_bank_p0_coverage_result_b",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "coverage_pass" if passed else "coverage_fail",
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "input_lock_sha256": lock_hash,
            "attempt_a_result_sha256": sha256(ROOT / "reports/qwen_gptq_bank/p0_coverage_result.json"),
        },
        "manifests": manifests,
        "coverage": {
            "required_layer_expert_pairs": LAYERS * EXPERTS, "threshold_rows": 128,
            "minimum_rows": min(row["minimum_rows"] for row in layers),
            "deficient_pairs": deficient, "layers": layers, "all_pairs_pass": passed,
        },
        "resources": {
            "device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_process_rss_bytes": peak_rss, "elapsed_seconds": time.perf_counter() - started,
        },
        "claim_boundary": "Cumulative official route coverage only; no activation or GPTQ claim yet.",
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# Qwen GPTQ Bank — P0 coverage attempt B\n\n"
        f"Uitkomst: **{payload['status']}**.\n\n"
        f"Cumulatief minimum: **{payload['coverage']['minimum_rows']}**; "
        f"paren onder 128: **{len(deficient)}**.\n\n"
        "Dit is uitsluitend de routecoveragegate; activatiecapture en GPTQ blijven afzonderlijke fasen.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"], "minimum_rows": payload["coverage"]["minimum_rows"],
        "deficient_pairs": len(deficient), "elapsed_seconds": payload["resources"]["elapsed_seconds"],
    }, indent=2))
