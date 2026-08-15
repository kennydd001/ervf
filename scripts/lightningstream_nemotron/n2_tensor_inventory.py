"""N2 steps 4-9: tensor inventory, partition, layout adjudication, contiguity.

Reads only safetensors headers and the official index.  No tensor payload is
decoded here and no BF16 model is materialized.

The central job is to confirm or falsify the N0R layout hypothesis against real
tensor entries.  A hypothesis that reproduces byte totals is not a layout proof;
this compares dtypes, shapes and per-field byte counts directly.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"

SHARDS = [f"model-0000{i}-of-00005.safetensors" for i in range(1, 6)]

# Frozen config facts from N0R.
HIDDEN = 2688
MOE_INTERMEDIATE = 1856
SHARED_INTERMEDIATE = 3712
GROUP_SIZE = 16
N_ROUTED = 128
HYBRID = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"

# Frozen N1 header-inventory values (protected, read-only).
N1 = {
    "tensor_count": 24_147,
    "tensor_bytes": 19_339_781_632,
    "routed_experts": 16_523_376_640,
    "shared_experts": 258_177_392,
    "trunk_other": 2_558_227_600,
    "routed_records": 2_944,
    "routed_record_bytes": 5_612_560,
    "moe_layers": 23,
    "shard1_header_bytes": 429_488,
    # N1 hashed the 8-byte length prefix together with the header body.
    "shard1_header_sha256": "f9b2428248cfb2b8d36dbd879882f72e1a9ed417d4a734c65d80c7192c5a1a78",
    "shard1_header_sha256_convention": "sha256(u64_length_prefix + header_body)",
    "shard1_tensors": 3_454,
    "shard1_payload_extent": 3_998_409_368,
}

DTYPE_BYTES = {
    "F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2,
    "I16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1, "BOOL": 1,
}

ROUTED_RE = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.(.+)$")
SHARED_RE = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.shared_experts\.(up_proj|down_proj)\.(.+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_header(path: Path) -> tuple[dict, int, str, str]:
    """Return the parsed header plus both header-hash conventions.

    N1 hashed the 8-byte little-endian length prefix together with the header
    body; hashing the body alone gives a different digest.  Both are recorded so
    the N1 comparison is like-for-like and the ambiguity cannot resurface.
    """
    with path.open("rb") as handle:
        prefix = handle.read(8)
        (header_len,) = struct.unpack("<Q", prefix)
        raw = handle.read(header_len)
    return (
        json.loads(raw.decode("utf-8")),
        header_len,
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(prefix + raw).hexdigest(),
    )


def nbytes(entry: dict) -> int:
    start, end = entry["data_offsets"]
    return end - start


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [s for s in SHARDS if not (MODEL_DIR / s).is_file()]
    if missing:
        print(f"missing shards: {missing}")
        return 3

    # ---------------------------------------------------------------- headers
    tensors: dict[str, dict] = {}
    shard_rows = []
    for name in SHARDS:
        path = MODEL_DIR / name
        header, header_len, header_sha_body, header_sha_prefixed = read_header(path)
        entries = {k: v for k, v in header.items() if k != "__metadata__"}
        payload_extent = max(nb["data_offsets"][1] for nb in entries.values())
        shard_rows.append({
            "shard": name,
            "file_bytes": path.stat().st_size,
            "header_bytes": header_len,
            "header_sha256_body_only": header_sha_body,
            "header_sha256_with_length_prefix": header_sha_prefixed,
            "tensors": len(entries),
            "payload_extent_bytes": payload_extent,
            "metadata": header.get("__metadata__"),
        })
        for key, value in entries.items():
            if key in tensors:
                raise ValueError(f"duplicate tensor name across shards: {key}")
            tensors[key] = {**value, "shard": name}

    total_bytes = sum(nbytes(v) for v in tensors.values())

    # -------------------------------------------------------- dtype breakdown
    dtype_bytes: dict[str, int] = defaultdict(int)
    dtype_counts: dict[str, int] = defaultdict(int)
    for value in tensors.values():
        dtype_bytes[value["dtype"]] += nbytes(value)
        dtype_counts[value["dtype"]] += 1

    # ------------------------------------------------------------- partition
    routed_bytes = shared_bytes = 0
    routed_by_expert: dict[tuple[int, int], dict] = defaultdict(dict)
    shared_by_layer: dict[int, dict] = defaultdict(dict)

    for key, value in tensors.items():
        routed = ROUTED_RE.match(key)
        if routed:
            layer, expert, matrix, field = int(routed[1]), int(routed[2]), routed[3], routed[4]
            routed_bytes += nbytes(value)
            routed_by_expert[(layer, expert)][f"{matrix}.{field}"] = value
            continue
        shared = SHARED_RE.match(key)
        if shared:
            layer, matrix, field = int(shared[1]), shared[2], shared[3]
            shared_bytes += nbytes(value)
            shared_by_layer[layer][f"{matrix}.{field}"] = value

    trunk_bytes = total_bytes - routed_bytes - shared_bytes

    # -------------------------------------------------- layout adjudication
    expected_fields = {
        "up_proj.weight", "up_proj.weight_scale", "up_proj.weight_scale_2", "up_proj.input_scale",
        "down_proj.weight", "down_proj.weight_scale", "down_proj.weight_scale_2", "down_proj.input_scale",
    }

    def adjudicate(group: dict, intermediate: int) -> dict:
        """Field-by-field test of the N0R hypothesis for one expert."""
        checks: dict[str, bool] = {}
        checks["field_set_exact"] = set(group) == expected_fields
        if not checks["field_set_exact"]:
            return {"checks": checks, "pass": False,
                    "observed_fields": sorted(group), "missing": sorted(expected_fields - set(group)),
                    "extra": sorted(set(group) - expected_fields)}

        up_w, down_w = group["up_proj.weight"], group["down_proj.weight"]
        up_s, down_s = group["up_proj.weight_scale"], group["down_proj.weight_scale"]

        checks["up_weight_dtype_u8"] = up_w["dtype"] == "U8"
        checks["down_weight_dtype_u8"] = down_w["dtype"] == "U8"
        checks["up_scale_dtype_f8e4m3"] = up_s["dtype"] == "F8_E4M3"
        checks["down_scale_dtype_f8e4m3"] = down_s["dtype"] == "F8_E4M3"
        checks["globals_dtype_f32"] = all(
            group[f]["dtype"] == "F32" for f in
            ("up_proj.weight_scale_2", "up_proj.input_scale",
             "down_proj.weight_scale_2", "down_proj.input_scale")
        )

        # up_proj maps hidden -> intermediate, so rows=intermediate, packed cols=hidden/2
        checks["up_weight_shape_packed_half"] = up_w["shape"] == [intermediate, HIDDEN // 2]
        checks["up_scale_shape_group16"] = up_s["shape"] == [intermediate, HIDDEN // GROUP_SIZE]
        # down_proj maps intermediate -> hidden
        checks["down_weight_shape_packed_half"] = down_w["shape"] == [HIDDEN, intermediate // 2]
        checks["down_scale_shape_group16"] = down_s["shape"] == [HIDDEN, intermediate // GROUP_SIZE]

        n_weights = intermediate * HIDDEN
        checks["up_code_bytes_half_n"] = nbytes(up_w) == n_weights // 2
        checks["down_code_bytes_half_n"] = nbytes(down_w) == n_weights // 2
        checks["up_scale_bytes_n_over_group"] = nbytes(up_s) == n_weights // GROUP_SIZE
        checks["down_scale_bytes_n_over_group"] = nbytes(down_s) == n_weights // GROUP_SIZE
        checks["global_bytes_four_bytes_each"] = all(
            nbytes(group[f]) == 4 for f in
            ("up_proj.weight_scale_2", "up_proj.input_scale",
             "down_proj.weight_scale_2", "down_proj.input_scale")
        )

        record = sum(nbytes(v) for v in group.values())
        derived = 2 * (n_weights // 2) + 2 * (n_weights // GROUP_SIZE) + 4 * 4
        checks["record_bytes_match_derivation"] = record == derived

        return {"checks": checks, "pass": all(checks.values()), "record_bytes": record}

    routed_adj = {}
    routed_record_sizes: dict[int, int] = defaultdict(int)
    routed_fail = []
    for (layer, expert), group in routed_by_expert.items():
        verdict = adjudicate(group, MOE_INTERMEDIATE)
        routed_record_sizes[verdict.get("record_bytes", -1)] += 1
        if not verdict["pass"]:
            routed_fail.append({"layer": layer, "expert": expert, **verdict})
    routed_adj = {
        "experts_examined": len(routed_by_expert),
        "all_pass": not routed_fail,
        "failures": routed_fail[:20],
        "failure_count": len(routed_fail),
        "record_size_distribution": dict(routed_record_sizes),
    }

    shared_fail = []
    shared_record_sizes: dict[int, int] = defaultdict(int)
    for layer, group in shared_by_layer.items():
        verdict = adjudicate(group, SHARED_INTERMEDIATE)
        shared_record_sizes[verdict.get("record_bytes", -1)] += 1
        if not verdict["pass"]:
            shared_fail.append({"layer": layer, **verdict})
    shared_adj = {
        "layers_examined": len(shared_by_layer),
        "all_pass": not shared_fail,
        "failures": shared_fail[:20],
        "failure_count": len(shared_fail),
        "record_size_distribution": dict(shared_record_sizes),
    }

    # ------------------------------------------------------------ contiguity
    # How many contiguous byte ranges does fetching one routed expert require?
    # safetensors groups tensors by dtype, so an expert's eight tensors are not
    # one run -- but its two U8 code tensors, its two FP8 scale tensors and its
    # four FP32 scalars may each be adjacent.  The minimal range count is what
    # the H3 transport design actually depends on.
    def merge_runs(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            if merged and start == merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    range_counts: dict[int, int] = defaultdict(int)
    per_class_runs: dict[str, dict[int, int]] = {
        "codes_u8": defaultdict(int),
        "scales_f8": defaultdict(int),
        "globals_f32": defaultdict(int),
    }
    contiguity = {"single_shard": 0, "split_across_shards": 0,
                  "split_examples": [], "examples": []}

    for (layer, expert), group in sorted(routed_by_expert.items()):
        shards_used = {v["shard"] for v in group.values()}
        if len(shards_used) > 1:
            contiguity["split_across_shards"] += 1
            if len(contiguity["split_examples"]) < 8:
                contiguity["split_examples"].append({
                    "layer": layer, "expert": expert,
                    "shards": sorted(shards_used),
                    "tensors_per_shard": {
                        s: sorted(k for k, v in group.items() if v["shard"] == s)
                        for s in sorted(shards_used)
                    },
                })
            continue
        contiguity["single_shard"] += 1

        classes = {
            "codes_u8": [v for k, v in group.items() if k.endswith(".weight")],
            "scales_f8": [v for k, v in group.items() if k.endswith(".weight_scale")],
            "globals_f32": [v for k, v in group.items()
                            if k.endswith(".weight_scale_2") or k.endswith(".input_scale")],
        }
        total_ranges = 0
        detail = {}
        for label, members in classes.items():
            runs = merge_runs([tuple(v["data_offsets"]) for v in members])
            per_class_runs[label][len(runs)] += 1
            total_ranges += len(runs)
            detail[label] = {"runs": len(runs),
                             "bytes": sum(e - s for s, e in runs)}
        range_counts[total_ranges] += 1

        if len(contiguity["examples"]) < 3:
            contiguity["examples"].append({"layer": layer, "expert": expert,
                                           "total_ranges": total_ranges, **detail})

    contiguity["ranges_per_expert_distribution"] = dict(sorted(range_counts.items()))
    contiguity["runs_per_class_distribution"] = {
        label: dict(sorted(counts.items())) for label, counts in per_class_runs.items()
    }

    # --------------------------------------------------------------- excluded
    bf16_tensors = {k: v for k, v in tensors.items() if v["dtype"] == "BF16"}
    excluded_summary = {
        "bf16_tensor_count": len(bf16_tensors),
        "bf16_bytes": sum(nbytes(v) for v in bf16_tensors.values()),
        "lm_head_bytes": nbytes(tensors["lm_head.weight"]) if "lm_head.weight" in tensors else None,
        "lm_head_dtype": tensors.get("lm_head.weight", {}).get("dtype"),
        "embeddings_bytes": nbytes(tensors["backbone.embeddings.weight"]) if "backbone.embeddings.weight" in tensors else None,
        "embeddings_dtype": tensors.get("backbone.embeddings.weight", {}).get("dtype"),
    }

    # ---------------------------------------------------------------- index
    index_path = MODEL_DIR / "model.safetensors.index.json"
    index_check = {"present": index_path.is_file()}
    if index_check["present"]:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        index_check["index_keys"] = len(weight_map)
        index_check["keys_match_headers"] = set(weight_map) == set(tensors)
        index_check["shard_assignment_matches"] = all(
            weight_map.get(k) == v["shard"] for k, v in tensors.items()
        )
        index_check["declared_total_size"] = index.get("metadata", {}).get("total_size")

    # ------------------------------------------------------------ layer roles
    roles = {"M": [], "E": [], "*": []}
    for idx, char in enumerate(HYBRID):
        roles[char].append(idx)
    moe_layers_from_tensors = sorted({layer for layer, _ in routed_by_expert})

    layout_verdict = (
        "confirmed" if (routed_adj["all_pass"] and shared_adj["all_pass"]) else "falsified"
    )

    # --------------------------------------------------------- gates (frozen)
    # Exactly the preregistration §5 items that this runner is responsible for.
    # Gate 5 requires an explicit verdict, not a favourable one: a falsified
    # layout is a valid scientific result and still passes the gate.
    gates = {
        "g2_five_headers_parse_and_tensor_count_24147": (
            len(shard_rows) == 5 and len(tensors) == N1["tensor_count"]
        ),
        "g3_tensor_bytes_match_n1": total_bytes == N1["tensor_bytes"],
        "g4_routed_shared_trunk_buckets_match_n1": (
            routed_bytes == N1["routed_experts"]
            and shared_bytes == N1["shared_experts"]
            and trunk_bytes == N1["trunk_other"]
        ),
        "g5_layout_hypothesis_explicitly_adjudicated": layout_verdict in {"confirmed", "falsified"},
        "g7_no_payload_decode_no_bf16_materialization_no_gpu": True,
    }

    # ------------------------------------------------- additional observations
    # Performed and reported per preregistration §3, but not pass/fail gates.
    # They must never be able to change the phase verdict.
    additional_checks = {
        "routed_records_2944": len(routed_by_expert) == N1["routed_records"],
        "routed_record_bytes_uniform_5612560": (
            list(routed_record_sizes) == [N1["routed_record_bytes"]]
        ),
        "moe_layers_match_hybrid_pattern": moe_layers_from_tensors == roles["E"],
        # N1's convention: SHA-256 over the 8-byte length prefix + header body.
        "shard1_header_matches_n1": (
            shard_rows[0]["header_bytes"] == N1["shard1_header_bytes"]
            and shard_rows[0]["header_sha256_with_length_prefix"] == N1["shard1_header_sha256"]
            and shard_rows[0]["tensors"] == N1["shard1_tensors"]
        ),
        "shard1_payload_extent_matches_n1": (
            shard_rows[0]["payload_extent_bytes"] == N1["shard1_payload_extent"]
        ),
        "routed_layout_adjudication_passed": routed_adj["all_pass"],
        "shared_layout_adjudication_passed": shared_adj["all_pass"],
        "index_keys_match_headers": index_check.get("keys_match_headers", False),
        "index_shard_assignment_matches": index_check.get("shard_assignment_matches", False),
        "every_routed_expert_in_single_shard": contiguity["split_across_shards"] == 0,
    }

    result = {
        "kind": "lightningstream_nemotron_n2_tensor_inventory",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS",
        "completed_utc": utc_now(),
        "runner_sha256": sha256_path(Path(__file__)),
        "payload_decoded": False,
        "bf16_model_materialized": False,
        "gpu_used": False,
        "shards": shard_rows,
        "totals": {
            "tensor_count": len(tensors),
            "tensor_bytes": total_bytes,
            "routed_expert_bytes": routed_bytes,
            "shared_expert_bytes": shared_bytes,
            "trunk_other_bytes": trunk_bytes,
        },
        "n1_frozen": N1,
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "layer_roles": {"mamba": roles["M"], "moe": roles["E"], "attention": roles["*"]},
        "moe_layers_from_tensors": moe_layers_from_tensors,
        "routed_layout_adjudication": routed_adj,
        "shared_layout_adjudication": shared_adj,
        "layout_hypothesis_verdict": layout_verdict,
        "contiguity": contiguity,
        "excluded_and_bf16": excluded_summary,
        "index_check": index_check,
        "gates": gates,
        "gates_all_pass": all(gates.values()),
        "additional_checks": additional_checks,
        "additional_checks_all_pass": all(additional_checks.values()),
        "claim_boundary": (
            "Header/index/layout evidence only. No decode of payload here, no "
            "quality, latency or throughput claim, no full-model statement."
        ),
    }

    out = OUT_DIR / "n2_tensor_inventory.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"tensors            : {len(tensors):,}  (N1 {N1['tensor_count']:,})")
    print(f"tensor bytes       : {total_bytes:,}  (N1 {N1['tensor_bytes']:,})")
    print(f"routed bytes       : {routed_bytes:,}  (N1 {N1['routed_experts']:,})")
    print(f"shared bytes       : {shared_bytes:,}  (N1 {N1['shared_experts']:,})")
    print(f"trunk bytes        : {trunk_bytes:,}  (N1 {N1['trunk_other']:,})")
    print(f"routed experts     : {len(routed_by_expert):,}")
    print(f"record sizes       : {dict(routed_record_sizes)}")
    print(f"layout verdict     : {layout_verdict}")
    print(f"single-shard expts : {contiguity['single_shard']}  "
          f"split across shards: {contiguity['split_across_shards']}")
    print(f"ranges per expert  : {contiguity['ranges_per_expert_distribution']}")
    print(f"runs per class     : {contiguity['runs_per_class_distribution']}")
    print(f"BF16 bytes         : {excluded_summary['bf16_bytes']:,} in {excluded_summary['bf16_tensor_count']} tensors")
    print()
    print("frozen gates (preregistration §5):")
    for key, value in gates.items():
        print(f"  {'OK  ' if value else 'FAIL'} {key}")
    print("additional observations (not pass/fail):")
    for key, value in additional_checks.items():
        print(f"  {'OK  ' if value else 'note'} {key}")
    print(f"\ngates all pass     : {result['gates_all_pass']}")
    print(f"written            : {out}")
    return 0 if result["gates_all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
