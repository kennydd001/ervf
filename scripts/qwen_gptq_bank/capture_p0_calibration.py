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
LAYERS = 48
EXPERTS = 128
ROWS = 128
TOP_K = 8
HIDDEN = 2_048
CONTEXT_TOKENS = 1_024
CHUNK_CONTEXTS = 2
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
SOURCES = (
    (
        "hera",
        ROOT / "reports/runs/hera_moe/p0_input_ids.safetensors",
        ROOT / "reports/runs/hera_moe/p0_routes",
        DOMAINS,
    ),
    (
        "dhera",
        ROOT / "reports/runs/dhera_moe/p0_input_ids.safetensors",
        ROOT / "reports/runs/dhera_moe/p0_routes",
        DOMAINS,
    ),
    (
        "supplement_a",
        ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_input_ids.safetensors",
        ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_routes",
        DOMAINS,
    ),
    (
        "supplement_b",
        ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_b_input_ids.safetensors",
        ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_b_routes",
        ("math", "instruction"),
    ),
)
CUDA_LIMIT = int(7.5 * 2**30)
RSS_LIMIT = 32 * 2**30
MODEL = ROOT / "models/qwen3-30b-a3b-base"
PREREG = ROOT / "reports/qwen_gptq_bank/P0_FULL_BANK_PREREGISTRATION.md"
SELECTION_LOCK = ROOT / "reports/qwen_gptq_bank/p0_calibration_selection_lock.json"
SELECTION_ARTIFACT = ROOT / "reports/runs/qwen_gptq_bank/p0_calibration_selection.safetensors"
RUN_DIR = ROOT / "reports/runs/qwen_gptq_bank/p0_calibration"
LAYER_DIR = ROOT / "reports/qwen_gptq_bank/p0_calibration_layers"
RESULT = ROOT / "reports/qwen_gptq_bank/p0_calibration_capture_result.json"
REPORT = ROOT / "reports/qwen_gptq_bank/P0_CALIBRATION_CAPTURE_REPORT.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class OfficialCapture:
    def __init__(self, mlp):
        self.mlp = mlp
        self.inputs: list[torch.Tensor] = []
        self.route_calls: list[torch.Tensor] = []
        self._handle = None
        self._patcher = None

    def __enter__(self):
        def pre(_module, inputs):
            self.inputs.append(inputs[0].detach().cpu().contiguous())

        def intercepted(input, k, *args, **kwargs):
            output = ORIGINAL_TOPK(input, k, *args, **kwargs)
            if input.ndim == 2 and input.shape[-1] == EXPERTS and k == TOP_K:
                self.route_calls.append(output.indices.detach().to(torch.int16).cpu().contiguous())
            return output

        self._handle = self.mlp.register_forward_pre_hook(pre)
        self._patcher = mock.patch.object(torch, "topk", side_effect=intercepted)
        self._patcher.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._handle is not None:
            self._handle.remove()
        if self._patcher is not None:
            self._patcher.stop()

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.inputs) != 1 or len(self.route_calls) != 1:
            raise RuntimeError(
                f"expected one MoE input and route call, got {len(self.inputs)}/{len(self.route_calls)}"
            )
        return self.inputs[0], self.route_calls[0]


if __name__ == "__main__":
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite the calibration capture result")
    if not PREREG.is_file() or not SELECTION_LOCK.is_file():
        raise FileNotFoundError("preregistration or calibration selection lock missing")
    if transformers.__version__ != "4.51.3" or not torch.cuda.is_available():
        raise RuntimeError("pinned transformers 4.51.3 and CUDA are required")
    selection_lock = json.loads(SELECTION_LOCK.read_text(encoding="utf-8"))
    if sha256_file(SELECTION_ARTIFACT) != selection_lock["artifact_sha256"]:
        raise ValueError("calibration selection artifact hash mismatch")
    selections = load_file(SELECTION_ARTIFACT)
    expected_selection_shapes = {
        "source_index": (LAYERS, EXPERTS, ROWS),
        "domain_index": (LAYERS, EXPERTS, ROWS),
        "token_index": (LAYERS, EXPERTS, ROWS),
        "slot_index": (LAYERS, EXPERTS, ROWS),
    }
    if {key: tuple(value.shape) for key, value in selections.items()} != expected_selection_shapes:
        raise ValueError("calibration selection shape mismatch")

    input_sets = {}
    for source_name, input_path, _route_dir, source_domains in SOURCES:
        tensors = load_file(input_path)
        if set(tensors) != set(source_domains):
            raise ValueError(f"domain mismatch for {source_name}")
        input_sets[source_name] = tensors

    selected_source_all = selections["source_index"].long()
    selected_domain_all = selections["domain_index"].long()
    selected_token_all = selections["token_index"].long()
    context_ids = {}
    for source_id, (source_name, _input_path, _route_dir, source_domains) in enumerate(SOURCES):
        for domain in source_domains:
            domain_id = DOMAINS.index(domain)
            mask = (selected_source_all == source_id) & (selected_domain_all == domain_id)
            chosen = torch.unique(
                selected_token_all[mask].div(CONTEXT_TOKENS, rounding_mode="floor"), sorted=True
            )
            if chosen.numel():
                # Preserve the original two-context GEMM batches exactly.
                # BF16 tied router logits can change top-k tie ordering when a
                # context is paired with a different batch neighbour.
                starts = chosen.div(CHUNK_CONTEXTS, rounding_mode="floor") * CHUNK_CONTEXTS
                context_ids[(source_name, domain)] = torch.unique(
                    torch.cat((starts, starts + 1)), sorted=True
                )

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
    hidden = {}
    for key, chosen_contexts in context_ids.items():
        source_name, domain = key
        selected_input_ids = input_sets[source_name][domain].index_select(0, chosen_contexts)
        hidden[key] = load_token_embeddings(
            MODEL, selected_input_ids, "cpu", weight_map
        ).contiguous()
    position_ids = torch.arange(CONTEXT_TOKENS, device=device).unsqueeze(0)
    selection_hash = sha256_file(SELECTION_ARTIFACT)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LAYER_DIR.mkdir(parents=True, exist_ok=True)
    manifests = {}

    for layer_index in range(LAYERS):
        layer_started = time.perf_counter()
        artifact = RUN_DIR / f"layer_{layer_index:02d}.safetensors"
        layer_report = LAYER_DIR / f"layer_{layer_index:02d}.json"
        if artifact.exists() != layer_report.exists():
            raise RuntimeError(f"partial calibration checkpoint at layer {layer_index}")
        was_existing = artifact.exists()
        calibration = torch.empty((EXPERTS, ROWS, HIDDEN), dtype=torch.bfloat16)
        filled = torch.zeros((EXPERTS, ROWS), dtype=torch.bool)
        selected_source = selections["source_index"][layer_index].long()
        selected_domain = selections["domain_index"][layer_index].long()
        selected_token = selections["token_index"][layer_index].long()
        selected_slot = selections["slot_index"][layer_index].long()
        layer = load_qwen_decoder_layer(MODEL, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        route_exact = True

        for source_id, (source_name, _input_path, route_dir, source_domains) in enumerate(SOURCES):
            with safe_open(
                route_dir / f"layer_{layer_index:02d}.safetensors",
                framework="pt",
                device="cpu",
            ) as route_handle:
                expected_by_domain = {
                    domain: route_handle.get_tensor(f"{domain}_router_ids") for domain in source_domains
                }
            for domain in source_domains:
                domain_id = DOMAINS.index(domain)
                if (source_name, domain) not in hidden:
                    continue
                state = hidden[(source_name, domain)]
                contexts = state.shape[0]
                next_chunks = []
                expected_full = expected_by_domain[domain]
                chosen_contexts = context_ids[(source_name, domain)]
                expected_ids = expected_full.reshape(-1, CONTEXT_TOKENS, TOP_K).index_select(
                    0, chosen_contexts
                ).reshape(-1, TOP_K)
                mask = (selected_source == source_id) & (selected_domain == domain_id)
                locations = mask.nonzero(as_tuple=False)
                global_tokens = selected_token[mask]
                slots = selected_slot[mask]
                if locations.shape[0] != global_tokens.shape[0]:
                    raise RuntimeError("selection coordinate cardinality mismatch")
                if global_tokens.numel() and not bool(
                    expected_full[global_tokens, slots].long().eq(locations[:, 0]).all()
                ):
                    raise RuntimeError(
                        f"selection route provenance failed at layer {layer_index}, {source_name}/{domain}"
                    )
                selected_contexts = global_tokens.div(CONTEXT_TOKENS, rounding_mode="floor")
                local_contexts = torch.searchsorted(chosen_contexts, selected_contexts)
                if global_tokens.numel() and not bool(
                    chosen_contexts[local_contexts].eq(selected_contexts).all()
                ):
                    raise RuntimeError("selected context missing from pruned calibration stream")
                tokens = local_contexts * CONTEXT_TOKENS + global_tokens.remainder(CONTEXT_TOKENS)

                for context_begin in range(0, contexts, CHUNK_CONTEXTS):
                    batch = state[context_begin : context_begin + CHUNK_CONTEXTS].to(device)
                    with torch.inference_mode():
                        position_embeddings = rotary(batch, position_ids)
                        with OfficialCapture(layer.mlp) as capture:
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
                    captured_x, captured_ids = capture.result()
                    token_begin = context_begin * CONTEXT_TOKENS
                    token_end = token_begin + captured_x.shape[0] * CONTEXT_TOKENS
                    expected_chunk = expected_ids[token_begin:token_end]
                    route_exact &= torch.equal(captured_ids, expected_chunk)
                    inside = (tokens >= token_begin) & (tokens < token_end)
                    if bool(inside.any()):
                        chosen_locations = locations[inside]
                        local_tokens = tokens[inside] - token_begin
                        flat_x = captured_x.reshape(-1, HIDDEN)
                        calibration[chosen_locations[:, 0], chosen_locations[:, 1]] = flat_x[local_tokens]
                        filled[chosen_locations[:, 0], chosen_locations[:, 1]] = True
                    next_chunks.append(output.detach().cpu().contiguous())
                    del batch, output, position_embeddings, captured_x, captured_ids
                hidden[(source_name, domain)] = torch.cat(next_chunks)
                del state, next_chunks, expected_ids, expected_full, mask, locations
                del global_tokens, selected_contexts, local_contexts, tokens, slots

        if not route_exact or not bool(filled.all()):
            raise RuntimeError(
                f"calibration capture control failed at layer {layer_index}: routes={route_exact}, "
                f"filled={int(filled.sum())}/{filled.numel()}"
            )
        if not bool(torch.isfinite(calibration).all()):
            raise RuntimeError(f"non-finite calibration tensor at layer {layer_index}")
        tensor_hash = sha256_tensor(calibration)
        tensors = {
            "moe_input": calibration.contiguous(),
            "source_index": selected_source.to(torch.int8).contiguous(),
            "domain_index": selected_domain.to(torch.int8).contiguous(),
            "token_index": selected_token.to(torch.int32).contiguous(),
            "slot_index": selected_slot.to(torch.int8).contiguous(),
        }
        if was_existing:
            with safe_open(artifact, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("selection_sha256") != selection_hash:
                    raise ValueError(f"existing calibration metadata mismatch at layer {layer_index}")
                for key, value in tensors.items():
                    if not torch.equal(handle.get_tensor(key), value):
                        raise RuntimeError(f"calibration resume mismatch at layer {layer_index}: {key}")
            payload = json.loads(layer_report.read_text(encoding="utf-8"))
            if payload["artifact_sha256"] != sha256_file(artifact):
                raise ValueError(f"existing calibration report hash mismatch at layer {layer_index}")
        else:
            save_file(tensors, artifact, metadata={
                "kind": "qwen_gptq_bank_p0_calibration",
                "layer": str(layer_index),
                "selection_sha256": selection_hash,
                "moe_input_sha256": tensor_hash,
            })
            payload = {
                "kind": "qwen_gptq_bank_p0_calibration_layer",
                "layer": layer_index,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
                "artifact_sha256": sha256_file(artifact),
                "moe_input_sha256": tensor_hash,
                "shape": list(calibration.shape),
                "dtype": str(calibration.dtype),
                "all_finite": True,
                "all_rows_filled": True,
                "official_routes_exact": route_exact,
                "elapsed_seconds": time.perf_counter() - layer_started,
            }
            layer_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifests[str(layer_index)] = {
            "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
            "artifact_sha256": sha256_file(artifact),
            "report": str(layer_report.relative_to(ROOT)).replace("\\", "/"),
            "report_sha256": sha256_file(layer_report),
            "moe_input_sha256": tensor_hash,
        }
        del layer, rotary, calibration, filled, tensors
        gc.collect()
        torch.cuda.empty_cache()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if torch.cuda.max_memory_allocated(device) > CUDA_LIMIT or peak_rss > RSS_LIMIT:
            raise MemoryError("calibration capture exceeded its resource ceiling")
        print(json.dumps({
            "layer": layer_index,
            "seconds": time.perf_counter() - layer_started,
            "resumed": was_existing,
            "routes_exact": route_exact,
        }), flush=True)

    payload = {
        "kind": "qwen_gptq_bank_p0_calibration_capture_result",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "capture_pass",
        "preregistration_sha256": sha256_file(PREREG),
        "selection_lock_sha256": sha256_file(SELECTION_LOCK),
        "selection_artifact_sha256": selection_hash,
        "layers": manifests,
        "coverage": {
            "layers": LAYERS, "experts_per_layer": EXPERTS,
            "rows_per_expert": ROWS, "total_rows": LAYERS * EXPERTS * ROWS,
            "all_shapes_exact": True, "all_finite": True, "all_official_routes_exact": True,
            "processed_contexts_by_source_domain": {
                f"{source}/{domain}": int(contexts.numel())
                for (source, domain), contexts in context_ids.items()
            },
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
        "claim_boundary": "True routed calibration inputs only; no GPTQ bank or quality/runtime claim yet.",
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# Qwen GPTQ Bank — P0 calibration capture\n\n"
        "Uitkomst: **capture_pass**.\n\n"
        f"Alle {LAYERS * EXPERTS * ROWS:,} vooraf geselecteerde echte routed rijen zijn als originele "
        "BF16 MoE-input vastgelegd. Alle 48 laagvormen zijn `[128, 128, 2048]`, alle waarden zijn "
        "eindig en alle opnieuw onderschepte officiële route-ID's reproduceren de route-artifacts exact.\n\n"
        "Dit opent uitsluitend de GPTQ-equivalentiegate; er is nog geen volledige bankclaim.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "total_rows": payload["coverage"]["total_rows"],
        "elapsed_seconds": payload["resources"]["elapsed_seconds"],
    }, indent=2))
