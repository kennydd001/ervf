from __future__ import annotations

import argparse
from pathlib import Path

from moe_lab.deepseek_v2 import (
    expected_moe_layout,
    inspect_layer_headers,
    load_json,
    shard_inventory,
    sha256,
)
from moe_lab.reporting import ROOT, envelope, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 13, 26])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    config_path = model_dir / "config.json"
    code_files = [
        model_dir / "configuration_deepseek.py",
        model_dir / "modeling_deepseek.py",
        model_dir / "tokenization_deepseek_fast.py",
    ]
    payload = {
        "model_dir": str(model_dir.resolve()),
        "config_sha256": sha256(config_path),
        "official_code_sha256": {
            path.name: sha256(path) for path in code_files if path.is_file()
        },
        "expected_moe_layout": expected_moe_layout(load_json(config_path)),
        "shards": shard_inventory(model_dir),
        "layers": [inspect_layer_headers(model_dir, layer) for layer in args.layers],
    }
    path = write_json("checkpoint_layout.json", envelope("checkpoint_layout", payload))
    print(path)
    for layer in payload["layers"]:
        print(f"layer {layer['layer']}: {layer['status']}")

