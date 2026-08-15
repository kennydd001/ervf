from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path, chunk_size: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def layer_prefix(layer: int) -> str:
    if layer < 0:
        raise ValueError("layer must be non-negative")
    return f"model.layers.{layer}."


def expected_moe_layout(config: dict[str, Any]) -> dict[str, Any]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    experts = int(config["n_routed_experts"])
    shared = int(config.get("n_shared_experts") or 0)
    top_k = int(config["num_experts_per_tok"])
    layers = int(config["num_hidden_layers"])
    dense_layers = int(config["first_k_dense_replace"])
    bytes_per_bf16 = 2

    per_expert_parameters = 3 * hidden * intermediate
    routed_parameters_per_layer = experts * per_expert_parameters
    shared_parameters_per_layer = 3 * hidden * (shared * intermediate)
    gate_parameters_per_layer = experts * hidden
    active_routed_parameters_per_token_per_layer = top_k * per_expert_parameters

    return {
        "hidden_size": hidden,
        "moe_intermediate_size": intermediate,
        "routed_experts": experts,
        "selected_experts_per_token": top_k,
        "shared_experts": shared,
        "moe_layers": layers - dense_layers,
        "per_expert_parameters": per_expert_parameters,
        "per_expert_bf16_mib": round(per_expert_parameters * bytes_per_bf16 / 2**20, 3),
        "routed_parameters_per_layer": routed_parameters_per_layer,
        "routed_bf16_gib_per_layer": round(
            routed_parameters_per_layer * bytes_per_bf16 / 2**30, 3
        ),
        "shared_parameters_per_layer": shared_parameters_per_layer,
        "shared_bf16_mib_per_layer": round(
            shared_parameters_per_layer * bytes_per_bf16 / 2**20, 3
        ),
        "gate_parameters_per_layer": gate_parameters_per_layer,
        "active_routed_parameters_per_token_per_layer": (
            active_routed_parameters_per_token_per_layer
        ),
        "active_routed_bf16_mib_per_token_per_layer": round(
            active_routed_parameters_per_token_per_layer * bytes_per_bf16 / 2**20, 3
        ),
        "expected_expert_shapes": {
            "gate_proj.weight": [intermediate, hidden],
            "up_proj.weight": [intermediate, hidden],
            "down_proj.weight": [hidden, intermediate],
        },
    }


def shard_inventory(model_dir: Path) -> list[dict[str, Any]]:
    index = load_json(model_dir / "model.safetensors.index.json")
    shard_names = sorted(set(index["weight_map"].values()))
    inventory = []
    for name in shard_names:
        path = model_dir / name
        inventory.append(
            {
                "name": name,
                "expected": True,
                "present": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256(path) if path.is_file() else None,
            }
        )
    return inventory


def layer_shards(model_dir: Path, layer: int) -> list[str]:
    index = load_json(model_dir / "model.safetensors.index.json")
    prefix = layer_prefix(layer)
    return sorted(
        {
            shard
            for name, shard in index["weight_map"].items()
            if name.startswith(prefix)
        }
    )


def inspect_layer_headers(model_dir: Path, layer: int) -> dict[str, Any]:
    index = load_json(model_dir / "model.safetensors.index.json")
    prefix = layer_prefix(layer)
    names = sorted(name for name in index["weight_map"] if name.startswith(prefix))
    missing_shards = [
        shard for shard in layer_shards(model_dir, layer) if not (model_dir / shard).is_file()
    ]
    if missing_shards:
        return {
            "layer": layer,
            "status": "waiting_for_shards",
            "required_shards": layer_shards(model_dir, layer),
            "missing_shards": missing_shards,
            "tensor_count_from_index": len(names),
        }

    tensors: list[dict[str, Any]] = []
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(index["weight_map"][name], []).append(name)
    for shard, shard_names in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in shard_names:
                tensor_slice = handle.get_slice(name)
                tensors.append(
                    {
                        "name": name,
                        "shape": list(tensor_slice.get_shape()),
                        "shard": shard,
                    }
                )
    return {
        "layer": layer,
        "status": "complete",
        "required_shards": sorted(by_shard),
        "tensor_count_from_index": len(names),
        "tensors": tensors,
    }
