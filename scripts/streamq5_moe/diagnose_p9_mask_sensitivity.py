from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_qwen_decoder_layer
from scripts.streamq5_moe.run_p0c_model_quality import quantize_experts_q5_, quantize_trunk_, selected_embeddings
from scripts.streamq5_moe.run_p9b_structured_wanda_pruning import EVAL, MODEL, apply_structured_pruning_, forward_chunks

OUT = ROOT / "reports/streamq5_moe/p9_mask_sensitivity_diagnostic.json"
P9B = ROOT / "reports/runs/streamq5_moe/p9b_structured_wanda_keep.safetensors"
P9E = ROOT / "reports/runs/streamq5_moe/p9e1_group_balanced_keep.safetensors"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    a = left.float().cpu().numpy()
    b = right.float().cpu().numpy()
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {
        "elements": int(a.size),
        "different_bf16": int(np.count_nonzero(left.cpu().view(torch.uint16).numpy() != right.cpu().view(torch.uint16).numpy())),
        "max_abs": float(np.abs(delta).max(initial=0.0)),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(b.astype(np.float64)), 1e-30)),
    }


@torch.inference_mode()
def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    ids = load_file(EVAL)["validation_general"][:2].long().contiguous()
    masks_b = load_file(P9B)["layer_00"].long()
    masks_e = load_file(P9E)["layer_00"].long()
    device = torch.device("cuda")
    config = Qwen3MoeConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = "sdpa"
    weight_map = checkpoint_weight_map(MODEL)
    hidden = selected_embeddings(MODEL, ids, device, weight_map, 8)

    outputs, weight_hashes, router_ids = {}, {}, {}
    for name, masks in (("p9b", masks_b), ("p9e", masks_e)):
        layer = load_qwen_decoder_layer(MODEL, config, 0, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        with torch.no_grad():
            router = layer.mlp.gate(hidden.to(device))
            router_ids[name] = torch.topk(router, layer.mlp.top_k, dim=-1).indices.cpu()
            apply_structured_pruning_(layer, masks)
            quantize_experts_q5_(layer)
            quantize_trunk_(layer, 8)
        weight_hashes[name] = {
            "expert0_gate": digest_tensor(layer.mlp.experts[0].gate_proj.weight),
            "expert0_down": digest_tensor(layer.mlp.experts[0].down_proj.weight),
        }
        outputs[name] = forward_chunks(layer, rotary, hidden, device)
        del layer, rotary
        torch.cuda.empty_cache()

    routed = torch.unique(router_ids["p9b"]).tolist()
    payload = {
        "kind": "p9_mask_sensitivity_diagnostic",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"evaluation_sha256": sha256(EVAL), "p9b_masks_sha256": sha256(P9B), "p9e_masks_sha256": sha256(P9E), "evaluator_sha256": sha256(Path(__file__))},
        "mask_element_differences_layer0": int((masks_b != masks_e).sum()),
        "routed_experts": routed,
        "routed_experts_with_different_sets": int(sum(set(masks_b[i].tolist()) != set(masks_e[i].tolist()) for i in routed)),
        "router_ids_equal": bool(torch.equal(router_ids["p9b"], router_ids["p9e"])),
        "weight_hashes": weight_hashes,
        "outputs": {"p9b_sha256": digest_tensor(outputs["p9b"]), "p9e_sha256": digest_tensor(outputs["p9e"]), "comparison": compare(outputs["p9b"], outputs["p9e"])},
        "interpretation": "Different masks must change at least routed expert weights and normally outputs; equality would expose an evaluator alias/path bug.",
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
