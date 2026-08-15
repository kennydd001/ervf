from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import time
from unittest import mock
from datetime import datetime, timezone
from pathlib import Path

import psutil
import safetensors
import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_qwen_decoder_layer, load_token_embeddings


ORIGINAL_TOPK = torch.topk


DOMAINS = ("general", "code", "math", "multilingual", "instruction")
THRESHOLD = 128
LAYERS = 48
EXPERTS = 128
TOP_K = 8
CONTEXTS = 32
CONTEXT_TOKENS = 1024
CHUNK_CONTEXTS = 2
PARAMETERS_PER_EXPERT = 4_718_592
NONEXPERT_PARAMETERS = 1_541_093_376
HOT_RATE_BPP = 1.930708991156684
CUDA_LIMIT = int(7.5 * 2**30)
RSS_LIMIT = 32 * 2**30
INPUT_LOCK = ROOT / "reports/hera_moe/p0_input_lock.json"
INPUT_ARTIFACT = ROOT / "reports/runs/hera_moe/p0_input_ids.safetensors"
PREREG = ROOT / "reports/hera_moe/P0_MULTIDOMAIN_TIER_PREREGISTRATION.md"
RUN_DIR = ROOT / "reports/runs/hera_moe/p0_routes"
LAYER_DIR = ROOT / "reports/hera_moe/p0_route_layers"
RESULT = ROOT / "reports/hera_moe/p0_multidomain_tier_result.json"
REPORT = ROOT / "reports/hera_moe/P0_MULTIDOMAIN_TIER_AUDIT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean()), "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)), "p99": float(torch.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


class RouterAuditCapture:
    def __init__(self, block):
        self.block = block
        self.ids = self.weights = self.selected_margins = self.boundary_margin = None
        self.ids_exact = False
        self.logit_error = None
        self._handles = []
        self._actual_topk = []
        self._patcher = None

    def __enter__(self):
        def pre(_module, inputs):
            flat = inputs[0].reshape(-1, inputs[0].shape[-1])
            logits = self.block.gate(flat)
            self._logits = logits.detach()

        def intercepted_topk(input, k, *args, **kwargs):
            output = ORIGINAL_TOPK(input, k, *args, **kwargs)
            if input.ndim == 2 and input.shape[-1] == EXPERTS and k == TOP_K:
                self._actual_topk.append((output.values.detach(), output.indices.detach()))
            return output

        def post(_module, _inputs, output):
            official_logits = output[1]
            if len(self._actual_topk) != 1:
                raise RuntimeError(f"expected one official router topk call, got {len(self._actual_topk)}")
            raw_top8, ids = self._actual_topk[0]
            probs = torch.softmax(official_logits, dim=1, dtype=torch.float)
            top9_values, _top9_ids = ORIGINAL_TOPK(probs, TOP_K + 1, dim=-1)
            weights = raw_top8
            if self.block.norm_topk_prob:
                weights = weights / weights.sum(dim=-1, keepdim=True)
            self.logit_error = float((self._logits.float() - official_logits.float()).abs().max())
            # IDs are the exact return value of the one topk call executed by
            # the official block.  Do not make this control depend on a second
            # softmax evaluation: tied BF16 logits can make that redundant
            # value comparison non-bit-exact even though the intercepted IDs
            # are authoritative.
            self.ids_exact = self.logit_error == 0.0 and len(self._actual_topk) == 1
            ninth = top9_values[:, TOP_K:TOP_K + 1]
            self.ids = ids.to(torch.int16).cpu().contiguous()
            self.weights = weights.float().cpu().contiguous()
            self.selected_margins = (raw_top8 - ninth).float().cpu().contiguous()
            self.boundary_margin = (top9_values[:, TOP_K - 1] - top9_values[:, TOP_K]).float().cpu().contiguous()

        self._handles = [self.block.register_forward_pre_hook(pre), self.block.register_forward_hook(post)]
        self._patcher = mock.patch.object(torch, "topk", side_effect=intercepted_topk)
        self._patcher.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in reversed(self._handles):
            handle.remove()
        if self._patcher is not None:
            self._patcher.stop()


def domain_metrics(ids: torch.Tensor, weights: torch.Tensor, margins: torch.Tensor, boundary: torch.Tensor) -> dict:
    flat_ids = ids.long().reshape(-1)
    flat_weights = weights.reshape(-1).double()
    flat_margins = margins.reshape(-1).double()
    counts = torch.bincount(flat_ids, minlength=EXPERTS)
    mass = torch.zeros(EXPERTS, dtype=torch.float64).scatter_add_(0, flat_ids, flat_weights)
    squared_mass = torch.zeros(EXPERTS, dtype=torch.float64).scatter_add_(0, flat_ids, flat_weights.square())
    margin_sum = torch.zeros(EXPERTS, dtype=torch.float64).scatter_add_(0, flat_ids, flat_margins)
    margin_min = []
    for expert in range(EXPERTS):
        selected = flat_margins[flat_ids == expert]
        margin_min.append(None if selected.numel() == 0 else float(selected.min()))
    return {
        "counts": counts.tolist(), "router_weight_sum": mass.tolist(),
        "router_weight_squared_sum": squared_mass.tolist(),
        "selected_margin_sum": margin_sum.tolist(), "selected_margin_minimum": margin_min,
        "boundary_margin": quantiles(boundary), "hot_experts": int((counts >= THRESHOLD).sum()),
        "zero_experts": int((counts == 0).sum()), "total_invocations": int(counts.sum()),
    }


if __name__ == "__main__":
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite HERA P0 result")
    if not PREREG.is_file() or not INPUT_LOCK.is_file():
        raise FileNotFoundError("HERA preregistration/input lock missing")
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    if sha256(INPUT_ARTIFACT) != lock["artifact_sha256"]:
        raise ValueError("HERA input artifact hash mismatch")
    input_tensors = load_file(INPUT_ARTIFACT)
    if set(input_tensors) != set(DOMAINS):
        raise ValueError("HERA domain set differs from preregistration")
    if transformers.__version__ != "4.51.3" or not torch.cuda.is_available():
        raise RuntimeError("pinned transformers and CUDA are required")

    model_dir = ROOT / "models/qwen3-30b-a3b-base"
    config = Qwen3MoeConfig.from_pretrained(model_dir, local_files_only=True)
    config._attn_implementation = "sdpa"
    if (config.num_hidden_layers, config.num_experts, config.num_experts_per_tok, config.norm_topk_prob) != (48, 128, 8, True):
        raise RuntimeError("unexpected Qwen MoE configuration")
    device = torch.device("cuda")
    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats(device)
    peak_rss = process.memory_info().rss
    timer = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    weight_map = checkpoint_weight_map(model_dir)
    hidden = {
        domain: load_token_embeddings(model_dir, input_tensors[domain], "cpu", weight_map).contiguous()
        for domain in DOMAINS
    }
    position_ids = torch.arange(CONTEXT_TOKENS, device=device).unsqueeze(0)
    hot_masks = {domain: torch.zeros((LAYERS, EXPERTS), dtype=torch.bool) for domain in DOMAINS}
    manifests = {}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LAYER_DIR.mkdir(parents=True, exist_ok=True)

    for layer_index in range(LAYERS):
        artifact = RUN_DIR / f"layer_{layer_index:02d}.safetensors"
        layer_report = LAYER_DIR / f"layer_{layer_index:02d}.json"
        if artifact.exists() or layer_report.exists():
            raise FileExistsError(f"refusing to overwrite HERA layer {layer_index}")
        layer_timer = time.perf_counter()
        layer = load_qwen_decoder_layer(model_dir, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        routes_to_save = {}
        layer_payload = {}
        exact = True
        maximum_logit_error = 0.0
        for domain in DOMAINS:
            next_chunks, ids_chunks, weight_chunks, margin_chunks, boundary_chunks = [], [], [], [], []
            for start in range(0, CONTEXTS, CHUNK_CONTEXTS):
                batch = hidden[domain][start:start + CHUNK_CONTEXTS].to(device)
                with torch.inference_mode():
                    position_embeddings = rotary(batch, position_ids)
                    with RouterAuditCapture(layer.mlp) as capture:
                        output = layer(
                            batch, attention_mask=None, position_ids=position_ids,
                            use_cache=False, output_attentions=False, output_router_logits=False,
                            cache_position=position_ids.squeeze(0), position_embeddings=position_embeddings,
                        )[0]
                exact &= capture.ids_exact
                maximum_logit_error = max(maximum_logit_error, capture.logit_error)
                ids_chunks.append(capture.ids); weight_chunks.append(capture.weights)
                margin_chunks.append(capture.selected_margins); boundary_chunks.append(capture.boundary_margin)
                next_chunks.append(output.detach().cpu().contiguous())
                del batch, output, position_embeddings
            ids = torch.cat(ids_chunks); weights = torch.cat(weight_chunks)
            margins = torch.cat(margin_chunks); boundary = torch.cat(boundary_chunks)
            hidden[domain] = torch.cat(next_chunks)
            metrics = domain_metrics(ids, weights, margins, boundary)
            hot_masks[domain][layer_index] = torch.tensor(metrics["counts"]) >= THRESHOLD
            routes_to_save[f"{domain}_router_ids"] = ids
            layer_payload[domain] = metrics
            del ids, weights, margins, boundary, ids_chunks, weight_chunks, margin_chunks, boundary_chunks, next_chunks
        save_file(routes_to_save, artifact, metadata={
            "kind": "hera_moe_p0_routes", "layer": str(layer_index),
            "input_lock_sha256": sha256(INPUT_LOCK),
        })
        payload = {
            "kind": "hera_moe_p0_route_layer", "layer": layer_index,
            "completed_utc": datetime.now(timezone.utc).isoformat(), "domains": layer_payload,
            "route_ids_exact": exact, "router_logit_maximum_absolute_error": maximum_logit_error,
            "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
            "artifact_sha256": sha256(artifact), "elapsed_seconds": time.perf_counter() - layer_timer,
        }
        layer_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifests[str(layer_index)] = {
            "artifact": payload["artifact"], "artifact_sha256": payload["artifact_sha256"],
            "report": str(layer_report.relative_to(ROOT)).replace("\\", "/"),
            "report_sha256": sha256(layer_report),
        }
        del layer, rotary, routes_to_save
        gc.collect(); torch.cuda.empty_cache()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if torch.cuda.max_memory_allocated(device) > CUDA_LIMIT or peak_rss > RSS_LIMIT:
            raise MemoryError("HERA P0 exceeded preregistered resources")
        print(json.dumps({
            "layer": layer_index, "hot_by_domain": {d: int(hot_masks[d][layer_index].sum()) for d in DOMAINS},
            "elapsed_seconds": payload["elapsed_seconds"],
        }), flush=True)

    hot_union = torch.stack([hot_masks[d] for d in DOMAINS]).any(dim=0)
    union_growth = []
    cumulative = torch.zeros_like(hot_union)
    for domain in DOMAINS:
        before = int(cumulative.sum())
        cumulative |= hot_masks[domain]
        union_growth.append({"domain": domain, "new_hot_experts": int(cumulative.sum()) - before, "cumulative_hot_experts": int(cumulative.sum())})
    pairwise_jaccard = {}
    for index, left in enumerate(DOMAINS):
        for right in DOMAINS[index + 1:]:
            intersection = int((hot_masks[left] & hot_masks[right]).sum())
            union = int((hot_masks[left] | hot_masks[right]).sum())
            pairwise_jaccard[f"{left}__{right}"] = intersection / union

    cold_calls = {domain: torch.zeros(CONTEXTS * CONTEXT_TOKENS, dtype=torch.int16) for domain in DOMAINS}
    maximum_layer_cold_fraction = {domain: 0.0 for domain in DOMAINS}
    for layer_index in range(LAYERS):
        routes = load_file(RUN_DIR / f"layer_{layer_index:02d}.safetensors")
        cold_for_layer = ~hot_union[layer_index]
        for domain in DOMAINS:
            calls = cold_for_layer[routes[f"{domain}_router_ids"].long()].sum(dim=1).to(torch.int16)
            cold_calls[domain] += calls
            maximum_layer_cold_fraction[domain] = max(
                maximum_layer_cold_fraction[domain], float(calls.sum()) / (calls.numel() * TOP_K)
            )

    hot_count = int(hot_union.sum())
    cold_count = LAYERS * EXPERTS - hot_count
    hot_gib = hot_count * PARAMETERS_PER_EXPERT * HOT_RATE_BPP / 8 / 2**30
    trunk_gib = NONEXPERT_PARAMETERS * 4 / 8 / 2**30
    cold_gib = cold_count * PARAMETERS_PER_EXPERT * 16 / 8 / 2**30
    general_matches = True
    general_count_l1 = 0
    general_experts_different = 0
    for layer_index in range(LAYERS):
        old = json.loads((ROOT / f"reports/e2gq_moe/p0_capture_layers/layer_{layer_index:02d}.json").read_text(encoding="utf-8"))
        new = json.loads((LAYER_DIR / f"layer_{layer_index:02d}.json").read_text(encoding="utf-8"))
        old_counts = old["router_counts"]
        new_counts = new["domains"]["general"]["counts"]
        general_matches &= old_counts == new_counts
        general_count_l1 += sum(abs(a - b) for a, b in zip(old_counts, new_counts))
        general_experts_different += sum(a != b for a, b in zip(old_counts, new_counts))

    controls = {
        "all_48_layers_present": len(manifests) == 48,
        "all_route_ids_exact": all(json.loads((LAYER_DIR / f"layer_{i:02d}.json").read_text(encoding="utf-8"))["route_ids_exact"] for i in range(LAYERS)),
        "all_router_logits_exact": all(json.loads((LAYER_DIR / f"layer_{i:02d}.json").read_text(encoding="utf-8"))["router_logit_maximum_absolute_error"] == 0.0 for i in range(LAYERS)),
        "resource_limits_pass": torch.cuda.max_memory_allocated(device) <= CUDA_LIMIT and peak_rss <= RSS_LIMIT,
    }
    memory_gate = hot_gib + trunk_gib <= 5.75 and cold_gib <= 24.0
    verdict = "tier_positive" if all(controls.values()) and memory_gate else "static_tier_negative"
    result = {
        "kind": "hera_moe_p0_multidomain_tier_audit", "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "verdict": verdict,
        "p1_authorized": verdict == "tier_positive", "input_lock_sha256": sha256(INPUT_LOCK),
        "threshold_rows": THRESHOLD, "domains": list(DOMAINS), "controls": controls,
        "hot_experts_by_domain": {domain: int(hot_masks[domain].sum()) for domain in DOMAINS},
        "hot_union_experts": hot_count, "cold_experts": cold_count,
        "union_growth": union_growth, "pairwise_hot_jaccard": pairwise_jaccard,
        "general_e2gq_reproduction_diagnostic": {
            "exact_counts_match": general_matches,
            "count_l1_across_all_layers": general_count_l1,
            "layer_expert_counts_different": general_experts_different,
            "interpretation": "Historical E2GQ routes were reconstructed by a second topk over tied BF16 logits; HERA captures the actual official topk call.",
        },
        "cold_calls_per_token": {domain: quantiles(cold_calls[domain]) for domain in DOMAINS},
        "maximum_layer_cold_invocation_fraction": maximum_layer_cold_fraction,
        "memory_projection": {
            "hot_rate_bpp_assumption": HOT_RATE_BPP, "hot_entropy_gib": hot_gib,
            "nonexpert_int4_gib": trunk_gib, "resident_weight_gib": hot_gib + trunk_gib,
            "cold_bf16_host_gib": cold_gib, "hot_plus_trunk_gate_gib": 5.75,
            "cold_host_gate_gib": 24.0, "memory_gate_pass": memory_gate,
        },
        "artifacts": manifests, "elapsed_seconds": time.perf_counter() - timer,
        "hardware": {
            "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
            "transformers": transformers.__version__, "safetensors": safetensors.__version__,
            "device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_process_rss_bytes": peak_rss,
        },
        "claim_boundary": "P0 routing/tier result only; hot rate remains an extrapolation and no GPTQ quality, actual pack, cold transfer or runtime was measured.",
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = [
        f"| {domain} | {result['hot_experts_by_domain'][domain]} | {result['cold_calls_per_token'][domain]['mean']:.3f} | {result['cold_calls_per_token'][domain]['p95']:.0f} | {result['cold_calls_per_token'][domain]['p99']:.0f} |"
        for domain in DOMAINS
    ]
    lines = [
        "# HERA-MoE P0 — multidomain-tieraudit", "",
        f"Uitkomst: **{verdict}**. De vaste multidomainunion bevat **{hot_count:,}** hot en **{cold_count:,}** cold laag-expertparen.", "",
        f"De geprojecteerde resident weights zijn **{hot_gib + trunk_gib:.3f} GiB** tegenover de vooraf vastgelegde 5,75-GiB-gate; cold BF16 is **{cold_gib:.3f} GiB**.", "",
        "| Domein | Hot per domein | Gem. cold calls/token | p95 | p99 |", "|---|---:|---:|---:|---:|", *table, "",
        f"Alle officiële routecalls en logits zijn exact onderschept: `{controls['all_route_ids_exact'] and controls['all_router_logits_exact']}`. De historische E2GQ-counts zijn niet exact reproduceerbaar door tied BF16-topk (`L1={general_count_l1:,}`); dit is diagnostiek en geen gate-input.", "",
        "Dit is uitsluitend een router/tierbesluit. Er is geen GPTQ, kwaliteitsmeting, werkelijk entropybestand, cold-transferbenchmark of tokens/s-resultaat geopend.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "hot_union_experts": hot_count, "cold_experts": cold_count, "resident_weight_gib": hot_gib + trunk_gib, "cold_bf16_host_gib": cold_gib}, indent=2))
