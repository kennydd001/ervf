from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(item, name))
    elif isinstance(value, list):
        out[prefix] = value
    else:
        out[prefix] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nano", type=Path, required=True)
    parser.add_argument("--lightning", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nano = load(args.nano)
    lightning = load(args.lightning)
    n_names = set(nano.get("tensor_names") or [])
    l_names = set(lightning.get("tensor_names") or [])
    common = n_names & l_names

    n_shapes = nano.get("tensor_shapes") or {}
    l_shapes = lightning.get("tensor_shapes") or {}
    shape_changes = []
    for name in sorted(common):
        if name in n_shapes and name in l_shapes and n_shapes[name] != l_shapes[name]:
            shape_changes.append({"tensor": name, "nano": n_shapes[name], "lightning": l_shapes[name]})

    nc = flatten(nano.get("config") or {})
    lc = flatten(lightning.get("config") or {})
    config_changes = []
    for key in sorted(set(nc) | set(lc)):
        if nc.get(key) != lc.get(key):
            config_changes.append({"key": key, "nano": nc.get(key), "lightning": lc.get(key)})

    result = {
        "kind": "s100_nano_lightning_checkpoint_diff",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "nano_inventory": str(args.nano),
        "lightning_inventory": str(args.lightning),
        "nano_tensor_count": len(n_names),
        "lightning_tensor_count": len(l_names),
        "common_tensor_count": len(common),
        "only_nano_count": len(n_names - l_names),
        "only_lightning_count": len(l_names - n_names),
        "only_nano": sorted(n_names - l_names),
        "only_lightning": sorted(l_names - n_names),
        "shape_change_count": len(shape_changes),
        "shape_changes": shape_changes,
        "config_change_count": len(config_changes),
        "config_changes": config_changes,
        "family_counts": {"nano": nano.get("family_counts"), "lightning": lightning.get("family_counts")},
        "mtp": {"nano_hits": nano.get("mtp_tensor_hits"), "lightning_hits": lightning.get("mtp_tensor_hits")},
        "latentmoe": {"nano_hits": nano.get("latentmoe_tensor_hits"), "lightning_hits": lightning.get("latentmoe_tensor_hits")},
        "shard_bytes": {"nano": nano.get("total_shard_bytes"), "lightning": lightning.get("total_shard_bytes")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "common": len(common), "only_nano": len(n_names - l_names),
        "only_lightning": len(l_names - n_names),
        "shape_changes": len(shape_changes), "config_changes": len(config_changes),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
