"""S100-MTP phase 0: GPU-free inventory + official NemotronV3 name fingerprint."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from common import REPO, require_model_dir, sha256_file, utc_now

sys.path.insert(0, str(REPO / "src"))
from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

RESULT_DIR = REPO / "pro_research" / "results" / "s100_mtp_inventory"
OUT = RESULT_DIR / "PRO_S100_MTP_INVENTORY.json"
PREREG = REPO / "pro_research" / "S100_MTP_FEASIBILITY_PREREGISTRATION.md"
ADDENDUM = REPO / "pro_research" / "S100_MTP_NEMOTRONV3_MAPPING_ADDENDUM.md"

TERMS = ("mtp", "nextn", "next_n", "next_token", "multi_token", "multitoken", "speculative")
TOKEN_RE = re.compile(r"(?:^|[._/\-])(" + "|".join(re.escape(x) for x in TERMS) + r")(?:$|[._/\-])", re.I)
OFFICIAL_LAYER_RE = re.compile(r"(?:^|\.)mtp\.layers\.(\d+)\.")
OFFICIAL_CONFIG_FIELDS = ("num_nextn_predict_layers", "mtp_hybrid_override_pattern", "mtp_layers_block_type")
FUSION_MARKERS = ("enorm", "hnorm", "eh_proj", "final_layernorm")


def _matches(text: str) -> list[str]:
    low = text.lower()
    return sorted({t for t in TERMS if t in low}) if TOKEN_RE.search(low) or any(t in low for t in TERMS) else []


def _walk_config(value: Any, path: str = ""):
    if isinstance(value, dict):
        for k, v in value.items():
            p = f"{path}.{k}" if path else str(k)
            km = _matches(str(k))
            if km:
                yield {"path": p, "match_terms": km, "value": v if not isinstance(v, (dict, list)) else v,
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
    official_config = {k: idx.config.get(k) for k in OFFICIAL_CONFIG_FIELDS if k in idx.config}

    tensor_matches = []
    dtype_bytes = Counter()
    group_bytes = Counter()
    shape_groups = Counter()
    shards = Counter()
    generic_indices = set()
    official_layer_indices = set()
    invalid_ranges = []
    suffix_groups = Counter()
    fusion_counts = Counter()
    fusion_bytes = Counter()

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
            generic_indices.add(x)
        m = OFFICIAL_LAYER_RE.search(name)
        if m:
            official_layer_indices.add(int(m.group(1)))
        for marker in FUSION_MARKERS:
            if f".{marker}." in f".{name}.":
                fusion_counts[marker] += 1
                fusion_bytes[marker] += nbytes
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
            "official_mtp_layer_index": int(m.group(1)) if m else None,
        })

    total = sum(x["nbytes"] for x in tensor_matches)
    named = len(tensor_matches) > 0
    nontrivial_groups = len(shape_groups) >= 2 or len(suffix_groups) >= 2
    structured = bool(named and total > 0 and not invalid_ranges and (bool(config_matches) or nontrivial_groups))

    observed_layers = sorted(official_layer_indices)
    contiguous = bool(observed_layers) and observed_layers == list(range(observed_layers[-1] + 1))
    pattern = official_config.get("mtp_hybrid_override_pattern")
    block_types = official_config.get("mtp_layers_block_type")
    if isinstance(pattern, str) and pattern:
        pattern_length = len(pattern)
    elif isinstance(block_types, list) and block_types:
        pattern_length = len(block_types)
    else:
        pattern_length = None
    logical_depths = official_config.get("num_nextn_predict_layers")
    naive_nonrepeated_sublayers = None
    if isinstance(logical_depths, int) and logical_depths > 0 and pattern_length:
        naive_nonrepeated_sublayers = logical_depths * pattern_length
    marker_present = sum(fusion_counts.values()) > 0
    official_alignment = bool(
        observed_layers and contiguous and bool(official_config) and marker_present
    )

    if official_alignment:
        status = "official_nemotron_v3_name_alignment"
    elif structured:
        status = "structured_mtp_candidate"
    elif named:
        status = "named_mtp_present"
    else:
        status = "no_named_mtp_tensors"

    index_path = model / "model.safetensors.index.json"
    config_path = model / "config.json"
    payload = {
        "kind": "pro_s100_mtp_inventory",
        "runner_revision": "v2_official_nemotron_v3_fingerprint",
        "status": status,
        "created_utc": utc_now(),
        "claim_boundary": "GPU-free checkpoint metadata/name inventory only; no MTP forward/speculative speed claim",
        "preregistration": str(PREREG.relative_to(REPO)),
        "mapping_addendum": str(ADDENDUM.relative_to(REPO)),
        "official_reference": {
            "repository": "NVIDIA-NeMo/Automodel",
            "commit": "5001dd45f051fe137f8bc284f53577f5e0da2fdb",
            "nemotron_v3_mtp_path": "nemo_automodel/components/models/nemotron_v3/mtp.py",
            "common_mtp_path": "nemo_automodel/components/models/common/mtp/mtp.py",
            "expected_flat_prefix": "mtp.layers.{global_idx}.*",
        },
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
        "official_mtp_config": official_config,
        "official_pattern_length": pattern_length,
        "official_logical_depths_configured": logical_depths,
        "naive_nonrepeated_expected_sublayers": naive_nonrepeated_sublayers,
        "note_on_expected_sublayers": "diagnostic only: official implementation can reuse one physical depth via use_repeated_layer, so this is not a gate",
        "tensor_match_count": len(tensor_matches),
        "tensor_total_bytes": total,
        "tensor_total_gib": total / (1024 ** 3),
        "dtype_bytes": dict(dtype_bytes),
        "group_bytes": dict(group_bytes),
        "shape_group_counts": dict(shape_groups),
        "suffix_group_counts": dict(suffix_groups),
        "shard_bytes": dict(shards),
        "generic_name_derived_indices": sorted(generic_indices),
        "official_mtp_layer_indices": observed_layers,
        "official_mtp_layer_indices_contiguous_from_zero": contiguous,
        "official_fusion_marker_counts": dict(fusion_counts),
        "official_fusion_marker_bytes": dict(fusion_bytes),
        "invalid_ranges": invalid_ranges,
        "tensors": tensor_matches,
        "gates": {
            "named_mtp_present": named,
            "total_bytes_gt_zero": total > 0,
            "valid_metadata_ranges": not invalid_ranges,
            "config_or_nontrivial_tensor_structure": bool(config_matches) or nontrivial_groups,
            "structured_mtp_candidate": structured,
            "official_flat_mtp_layers_present": bool(observed_layers),
            "official_layer_indices_contiguous_from_zero": contiguous,
            "official_mtp_config_field_present": bool(official_config),
            "official_fusion_or_final_norm_marker_present": marker_present,
            "official_nemotron_v3_name_alignment": official_alignment,
        },
    }
    _write(payload)
    print(json.dumps({
        "status": status,
        "output": str(OUT),
        "tensor_match_count": len(tensor_matches),
        "tensor_total_gib": payload["tensor_total_gib"],
        "official_mtp_config": official_config,
        "official_mtp_layer_indices": observed_layers,
        "official_fusion_marker_counts": dict(fusion_counts),
        "gates": payload["gates"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
