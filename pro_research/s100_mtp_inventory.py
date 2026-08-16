"""S100-MTP phase 0: GPU-free inventory of explicitly named MTP structures."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import REPO, require_model_dir, sha256_file, utc_now

sys.path.insert(0, str(REPO / "src"))
from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

RESULT_DIR = REPO / "pro_research" / "results" / "s100_mtp_inventory"
OUT = RESULT_DIR / "PRO_S100_MTP_INVENTORY.json"
PREREG = REPO / "pro_research" / "S100_MTP_FEASIBILITY_PREREGISTRATION.md"

TERMS = ("mtp", "nextn", "next_n", "next_token", "multi_token", "multitoken", "speculative")
TOKEN_RE = re.compile(r"(?:^|[._/\-])(" + "|".join(re.escape(x) for x in TERMS) + r")(?:$|[._/\-])", re.I)


def _matches(text: str) -> list[str]:
    low = text.lower()
    return sorted({t for t in TERMS if t in low}) if TOKEN_RE.search(low) or any(t in low for t in TERMS) else []


def _walk_config(value: Any, path: str = ""):
    if isinstance(value, dict):
        for k, v in value.items():
            p = f"{path}.{k}" if path else str(k)
            km = _matches(str(k))
            if km:
                yield {"path": p, "match_terms": km, "value": v if not isinstance(v, (dict, list)) else None,
                       "value_type": type(v).__name__}
            yield from _walk_config(v, p)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _walk_config(v, f"{path}[{i}]")
    elif isinstance(value, str):
        vm = _matches(value)
        if vm:
            yield {"path": path, "match_terms": vm, "value": value, "value_type": "str"}


def _name_group(name: str) -> str:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if _matches(p):
            return ".".join(parts[:min(len(parts), i + 3)])
    return ".".join(parts[:3])


def _mtp_indices(name: str) -> list[int]:
    parts = re.split(r"[./]", name)
    out = []
    for i, p in enumerate(parts):
        if _matches(p):
            for q in parts[i + 1:i + 4]:
                if q.isdigit():
                    out.append(int(q))
                    break
    return out


def _write(payload):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def main() -> int:
    model = require_model_dir()
    idx = ShardIndex(model)
    config_matches = list(_walk_config(idx.config))

    tensor_matches = []
    dtype_bytes = Counter()
    group_bytes = Counter()
    shape_groups = Counter()
    shards = Counter()
    indices = set()
    invalid_ranges = []
    suffix_groups = Counter()

    for name, e in sorted(idx.entries.items()):
        terms = _matches(name)
        if not terms:
            continue
        if e.end <= e.start:
            invalid_ranges.append(name)
        nbytes = int(e.nbytes)
        group = _name_group(name)
        suffix = ".".join(name.split(".")[-3:])
        for x in _mtp_indices(name):
            indices.add(x)
        dtype_bytes[e.dtype] += nbytes
        group_bytes[group] += nbytes
        shape_groups[f"{e.dtype}:{list(e.shape)}"] += 1
        suffix_groups[suffix] += 1
        shards[e.shard] += nbytes
        tensor_matches.append({
            "name": name,
            "match_terms": terms,
            "dtype": e.dtype,
            "shape": list(e.shape),
            "nbytes": nbytes,
            "shard": e.shard,
            "start": int(e.start),
            "end": int(e.end),
            "name_group": group,
            "suffix3": suffix,
        })

    total = sum(x["nbytes"] for x in tensor_matches)
    named = len(tensor_matches) > 0
    nontrivial_groups = len(shape_groups) >= 2 or len(suffix_groups) >= 2
    structured = bool(
        named and total > 0 and not invalid_ranges
        and (bool(config_matches) or nontrivial_groups)
    )

    index_path = model / "model.safetensors.index.json"
    config_path = model / "config.json"
    payload = {
        "kind": "pro_s100_mtp_inventory",
        "status": "structured_mtp_candidate" if structured else ("named_mtp_present" if named else "no_named_mtp_tensors"),
        "created_utc": utc_now(),
        "claim_boundary": "GPU-free checkpoint metadata inventory only; no MTP forward/speculative speed claim",
        "preregistration": str(PREREG.relative_to(REPO)),
        "model_dir": str(model),
        "metadata_hashes": {
            "config_json_sha256": sha256_file(config_path),
            "safetensors_index_sha256": sha256_file(index_path),
        },
        "model_summary": {
            "entry_count": len(idx.entries),
            "shard_count": len({e.shard for e in idx.entries.values()}),
            "config_architectures": idx.config.get("architectures"),
            "model_type": idx.config.get("model_type"),
        },
        "match_vocabulary": list(TERMS),
        "config_matches": config_matches,
        "tensor_match_count": len(tensor_matches),
        "tensor_total_bytes": total,
        "tensor_total_gib": total / (1024 ** 3),
        "dtype_bytes": dict(dtype_bytes),
        "group_bytes": dict(group_bytes),
        "shape_group_counts": dict(shape_groups),
        "suffix_group_counts": dict(suffix_groups),
        "shard_bytes": dict(shards),
        "name_derived_indices": sorted(indices),
        "invalid_ranges": invalid_ranges,
        "tensors": tensor_matches,
        "gates": {
            "named_mtp_present": named,
            "total_bytes_gt_zero": total > 0,
            "valid_metadata_ranges": not invalid_ranges,
            "config_or_nontrivial_tensor_structure": bool(config_matches) or nontrivial_groups,
            "structured_mtp_candidate": structured,
        },
    }
    _write(payload)
    print(json.dumps({
        "status": payload["status"],
        "output": str(OUT),
        "tensor_match_count": len(tensor_matches),
        "tensor_total_gib": payload["tensor_total_gib"],
        "config_match_count": len(config_matches),
        "name_derived_indices": sorted(indices),
        "gates": payload["gates"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
