from __future__ import annotations

import json

from safetensors import safe_open

from moe_lab.deepseek_v2 import load_json, sha256
from moe_lab.reporting import ROOT, envelope, write_json


if __name__ == "__main__":
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    model_report = load_json(ROOT / "reports" / "baseline" / "model.json")
    expected_files = {
        item["path"]: item["size_bytes"]
        for item in model_report["payload"]["files"]
        if item["path"].endswith(".safetensors")
    }
    index = load_json(model_dir / "model.safetensors.index.json")
    shard_names = sorted(set(index["weight_map"].values()))
    results = []
    for shard_name in shard_names:
        path = model_dir / shard_name
        expected_size = expected_files.get(shard_name)
        item = {
            "name": shard_name,
            "present": path.is_file(),
            "expected_size_bytes": expected_size,
            "actual_size_bytes": path.stat().st_size if path.is_file() else None,
            "size_matches": path.is_file()
            and expected_size is not None
            and path.stat().st_size == expected_size,
            "header_valid": False,
            "tensor_count": None,
            "sha256": None,
        }
        if path.is_file():
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
            item["header_valid"] = bool(keys)
            item["tensor_count"] = len(keys)
            item["sha256"] = sha256(path)
        results.append(item)

    all_complete = bool(results) and all(
        item["present"] and item["size_matches"] and item["header_valid"]
        for item in results
    )
    actual_tensor_bytes = int(index["metadata"]["total_size"])
    payload = {
        "status": "complete" if all_complete else "incomplete",
        "model_revision": model_report["payload"]["revision"],
        "index_tensor_bytes": actual_tensor_bytes,
        "index_tensor_gib": round(actual_tensor_bytes / 2**30, 3),
        "shards": results,
    }
    path = write_json("checkpoint_verification.json", envelope("checkpoint_verification", payload))
    print(path)
    print(json.dumps({"status": payload["status"], "shards": results}, indent=2))
    if not all_complete:
        raise SystemExit(2)

