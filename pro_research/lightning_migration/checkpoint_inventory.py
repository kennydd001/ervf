from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LAYER_RE = re.compile(r"(?:^|\.)(?:layers?|blocks?)\.(\d+)(?:\.|$)")
MTP_RE = re.compile(
    r"(?:^|[._/\-])(?:mtp|multi[_\-]?token|nextn|"
    r"prediction[_\-]?head|speculator)(?:$|[._/\-])", re.I,
)
LATENT_RE = re.compile(
    r"(?:latent[_\-]?moe|latentmoe|latent[_\-]?expert|"
    r"latent[_\-]?proj|expert[_\-]?latent)", re.I,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def tensor_family(name: str) -> str:
    low = name.lower()
    if MTP_RE.search(name): return "mtp"
    if LATENT_RE.search(name): return "latentmoe"
    if "mamba" in low or "ssm" in low or "conv1d" in low: return "mamba"
    if any(x in low for x in ("q_proj", "k_proj", "v_proj", "o_proj")): return "attention"
    if "router" in low or "gate" in low: return "router"
    if "expert" in low or ".mlp." in low or ".moe." in low: return "moe"
    if "embed" in low: return "embedding"
    if "lm_head" in low or "output_layer" in low: return "lm_head"
    if "norm" in low: return "norm"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-safetensors", action="store_true",
                        help="Open shard headers to collect shape/dtype without loading data.")
    parser.add_argument("--hash-shards", action="store_true",
                        help="Expensive: SHA256 every weight shard.")
    args = parser.parse_args()

    root = args.model_dir.resolve()
    config = load_json(root / "config.json") or {}
    index = load_json(root / "model.safetensors.index.json") or {}
    weight_map = index.get("weight_map") or {}
    names = sorted(str(x) for x in weight_map)

    shard_names = sorted(set(str(x) for x in weight_map.values()))
    shards = []
    for name in shard_names:
        path = root / name
        shards.append({
            "name": name,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256(path) if path.exists() and args.hash_shards else None,
        })

    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    safetensor_error = None
    if args.inspect_safetensors:
        try:
            from safetensors import safe_open
            by_shard: dict[str, list[str]] = defaultdict(list)
            for tensor, shard in weight_map.items():
                by_shard[str(shard)].append(str(tensor))
            for shard, tensor_names in by_shard.items():
                path = root / shard
                with safe_open(str(path), framework="np", device="cpu") as handle:
                    for tensor in tensor_names:
                        sl = handle.get_slice(tensor)
                        shapes[tensor] = [int(x) for x in sl.get_shape()]
                        dtypes[tensor] = str(sl.get_dtype())
        except Exception as exc:
            safetensor_error = f"{type(exc).__name__}: {exc}"

    family_counts = Counter(tensor_family(x) for x in names)
    layer_tensors: dict[str, int] = Counter()
    for name in names:
        match = LAYER_RE.search(name)
        if match:
            layer_tensors[match.group(1)] += 1

    small_hashes = {}
    for name in (
        "config.json", "generation_config.json", "tokenizer_config.json",
        "model.safetensors.index.json", "configuration_nemotron_h.py",
        "modeling_nemotron_h.py", "README.md", "ACQUISITION_PROVENANCE.json",
    ):
        path = root / name
        small_hashes[name] = sha256(path) if path.exists() else None

    output = {
        "kind": "s100_checkpoint_inventory",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_dir": str(root),
        "config": config,
        "metadata_sha256": small_hashes,
        "tensor_count": len(names),
        "tensor_names": names,
        "tensor_shapes": shapes,
        "tensor_dtypes": dtypes,
        "family_counts": dict(family_counts),
        "layer_tensor_counts": dict(sorted(layer_tensors.items())),
        "mtp_tensor_hits": [x for x in names if MTP_RE.search(x)],
        "latentmoe_tensor_hits": [x for x in names if LATENT_RE.search(x)],
        "shards": shards,
        "total_shard_bytes": sum(int(x["bytes"] or 0) for x in shards),
        "safetensors_header_error": safetensor_error,
        "complete_weight_headers": bool(
            args.inspect_safetensors and not safetensor_error and len(shapes) == len(names)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "tensor_count": output["tensor_count"],
        "family_counts": output["family_counts"],
        "mtp_hits": len(output["mtp_tensor_hits"]),
        "latentmoe_hits": len(output["latentmoe_tensor_hits"]),
        "shard_bytes": output["total_shard_bytes"],
        "header_complete": output["complete_weight_headers"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
