#!/usr/bin/env python3
"""Read-only independent audit of the completed T0R12D2R3 diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization"
RAW = RUN_DIR / "t0r12d2_raw.safetensors"
RESULT = RUN_DIR / "t0r12d2_result.json"
REPORTS = ROOT / "reports/streamq5_moe"
RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_t0r12d2r3_cloned_serialization.py"
VERIFIER = ROOT / "scripts/streamq5_moe/verify_port80b_t0r12d2r3_cloned_serialization.py"
VERIFIER_LOCK = REPORTS / "port80b_t0r12d2r3_verifier_lock.json"
PREREG = REPORTS / "PORT80B_T0R12D2R3_CLONED_SERIALIZATION_REPAIR_2026-08-13.md"
D2_SOURCE = ROOT / "scripts/streamq5_moe/run_port80b_t0r12d2_full_stage_diagnostic.py"
BASE = ROOT / "scripts/streamq5_moe/run_port80b_t0r12_official_cpu_reference_only.py"
ORIGINAL_FAILURE = (
    ROOT
    / "reports/runs/streamq5_moe/port80b_t0r12_official_cpu_reference_only/t0r12_capture_1_failure.json"
)
SERIALIZATION_FAILURE = REPORTS / "port80b_t0r12d2r2_shared_storage_serialization_failure.json"

STAGES = {
    "input_norm": (torch.bfloat16, 2048, True),
    "gdn": (torch.bfloat16, 2048, True),
    "post_norm": (torch.bfloat16, 2048, True),
    "official_router_logits": (torch.bfloat16, 512, False),
    "official_router_weights": (torch.bfloat16, 10, False),
    "official_router_ids": (torch.int64, 10, False),
    "experts": (torch.bfloat16, 2048, False),
    "shared": (torch.bfloat16, 2048, False),
    "shared_gate": (torch.bfloat16, 1, False),
    "layer_output": (torch.bfloat16, 2048, True),
}
ROUTES = {
    "diagnostic_router_logits": (torch.bfloat16, 512),
    "diagnostic_router_weights": (torch.bfloat16, 10),
    "diagnostic_router_ids": (torch.int64, 10),
    "router_logits_fp32": (torch.float32, 512),
    "router_probs_fp32": (torch.float32, 512),
    "router_top10_ids_recomputed": (torch.int64, 10),
    "router_weights_precast_fp32": (torch.float32, 10),
    "router_weights_recomputed_bf16": (torch.bfloat16, 10),
    "router_top10_top11_margin_fp32": (torch.float32, None),
    "router_boundary_tie_mask": (torch.bool, 512),
    "router_selected_boundary_mask": (torch.bool, 10),
    "router_top11_ids": (torch.int64, 11),
    "router_top11_native_bf16_logits": (torch.bfloat16, 11),
}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.contiguous().view(torch.uint8).numpy().tobytes()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(value)).hexdigest()


def ordered_bf16(value: torch.Tensor) -> torch.Tensor:
    bits = value.contiguous().view(torch.uint16).to(torch.int32)
    return torch.where((bits & 0x8000) != 0, 0x8000 - (bits & 0x7FFF), 0x8000 + bits)


def compare(reference: torch.Tensor, observed: torch.Tensor) -> dict:
    different = int((reference != observed).sum())
    item = {
        "dtype": str(reference.dtype),
        "shape": list(reference.shape),
        "different_elements": different,
        "exact_equal": bool(torch.equal(reference, observed)),
        "reference_sha256": tensor_sha(reference),
        "observed_sha256": tensor_sha(observed),
    }
    if reference.dtype == torch.bfloat16:
        delta = reference.float() - observed.float()
        item.update(
            max_bf16_ulp=int((ordered_bf16(reference) - ordered_bf16(observed)).abs().max()),
            max_abs=float(delta.abs().max()),
            rel_l2=float(
                torch.linalg.vector_norm(delta)
                / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
            ),
        )
    return item


def prefixes() -> list[tuple[str, int]]:
    values = [(f"p{prompt}_whole", 16) for prompt in range(4)]
    values += [(f"p{prompt}_n{length}", length) for prompt in range(4) for length in range(1, 17)]
    values += [("p1_whole_repeat", 16), ("p1_n3_repeat", 3)]
    return values


def expected_schema() -> dict[str, tuple[torch.dtype, list[int]]]:
    schema = {"token_ids": (torch.int64, [4, 16]), "embedding": (torch.bfloat16, [4, 16, 2048])}
    for prefix, length in prefixes():
        for name, (dtype, width, batched) in STAGES.items():
            schema[f"{prefix}_{name}"] = (dtype, [1, length, width] if batched else [length, width])
        for name, (dtype, width) in ROUTES.items():
            schema[f"{prefix}_{name}"] = (dtype, [length] if width is None else [length, width])
        schema[f"{prefix}_cache_conv"] = (torch.bfloat16, [1, 8192, 4])
        schema[f"{prefix}_cache_recurrent"] = (torch.float32, [1, 32, 128, 128])
    return schema


def final_position(value: torch.Tensor, stage: str, length: int) -> torch.Tensor:
    if STAGES[stage][2]:
        return value[:, length - 1 : length]
    return value[length - 1 : length]


def audit() -> dict:
    result = json.loads(RESULT.read_text())
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(RAW, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)

    schema = expected_schema()
    manifest = {
        key: {
            "semantic_key": key,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "bytes": value.numel() * value.element_size(),
            "sha256": tensor_sha(value),
        }
        for key, value in sorted(tensors.items())
    }
    schema_ok = set(tensors) == set(schema) and all(
        tensors[key].dtype == dtype and list(tensors[key].shape) == shape
        for key, (dtype, shape) in schema.items()
    )

    rows = []
    stage_summary = {
        stage: {
            "comparisons": 64,
            "divergent_comparisons": 0,
            "different_elements": 0,
            "max_bf16_ulp": 0 if dtype == torch.bfloat16 else None,
            "max_abs": 0.0 if dtype == torch.bfloat16 else None,
            "max_rel_l2": 0.0 if dtype == torch.bfloat16 else None,
            "divergent_lengths_by_prompt": {str(prompt): [] for prompt in range(4)},
        }
        for stage, (dtype, _width, _batched) in STAGES.items()
    }
    for prompt in range(4):
        for length in range(1, 17):
            stage_metrics = {}
            for stage in STAGES:
                metric = compare(
                    final_position(tensors[f"p{prompt}_whole_{stage}"], stage, length),
                    final_position(tensors[f"p{prompt}_n{length}_{stage}"], stage, length),
                )
                stage_metrics[stage] = metric
                summary = stage_summary[stage]
                summary["different_elements"] += metric["different_elements"]
                if not metric["exact_equal"]:
                    summary["divergent_comparisons"] += 1
                    summary["divergent_lengths_by_prompt"][str(prompt)].append(length)
                if "max_bf16_ulp" in metric:
                    summary["max_bf16_ulp"] = max(summary["max_bf16_ulp"], metric["max_bf16_ulp"])
                    summary["max_abs"] = max(summary["max_abs"], metric["max_abs"])
                    summary["max_rel_l2"] = max(summary["max_rel_l2"], metric["rel_l2"])
            rows.append({"prompt": prompt, "length": length, "stages": stage_metrics})

    repeat_pairs = (("p1_whole", "p1_whole_repeat"), ("p1_n3", "p1_n3_repeat"))
    repeat_metrics = {}
    for reference, observed in repeat_pairs:
        repeat_metrics[observed] = {
            "stages": {
                stage: compare(tensors[f"{reference}_{stage}"], tensors[f"{observed}_{stage}"])
                for stage in STAGES
            },
            "cache_conv": compare(tensors[f"{reference}_cache_conv"], tensors[f"{observed}_cache_conv"]),
            "cache_recurrent": compare(
                tensors[f"{reference}_cache_recurrent"], tensors[f"{observed}_cache_recurrent"]
            ),
        }
    repeat_all_exact = all(
        metric["exact_equal"]
        for pair in repeat_metrics.values()
        for metric in list(pair["stages"].values()) + [pair["cache_conv"], pair["cache_recurrent"]]
    )

    cache16 = {
        str(prompt): {
            "conv": compare(tensors[f"p{prompt}_whole_cache_conv"], tensors[f"p{prompt}_n16_cache_conv"]),
            "recurrent": compare(
                tensors[f"p{prompt}_whole_cache_recurrent"], tensors[f"p{prompt}_n16_cache_recurrent"]
            ),
        }
        for prompt in range(4)
    }
    cache16_all_exact = all(item["exact_equal"] for pair in cache16.values() for item in pair.values())

    direct_tuple_ok = True
    router_recompute_ok = True
    tie_evidence_ok = True
    router_token_rows = 0
    boundary_tie_rows = 0
    selected_boundary_tie_rows = 0
    zero_margin_rows = 0
    for prefix, _length in prefixes():
        logits = tensors[f"{prefix}_official_router_logits"]
        weights = tensors[f"{prefix}_official_router_weights"]
        ids = tensors[f"{prefix}_official_router_ids"]
        direct_tuple_ok &= (
            torch.equal(logits, tensors[f"{prefix}_diagnostic_router_logits"])
            and torch.equal(weights, tensors[f"{prefix}_diagnostic_router_weights"])
            and torch.equal(ids, tensors[f"{prefix}_diagnostic_router_ids"])
        )
        probs = torch.softmax(logits.float(), dim=-1)
        values11, ids11 = torch.topk(probs, 11, dim=-1)
        top_weights = values11[:, :10]
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        ids10 = ids11[:, :10]
        boundary = probs == values11[:, 9:10]
        selected_boundary = torch.gather(boundary, 1, ids)
        margin = values11[:, 9] - values11[:, 10]
        router_recompute_ok &= (
            torch.equal(logits.float(), tensors[f"{prefix}_router_logits_fp32"])
            and torch.equal(probs, tensors[f"{prefix}_router_probs_fp32"])
            and torch.equal(ids10, tensors[f"{prefix}_router_top10_ids_recomputed"])
            and torch.equal(top_weights, tensors[f"{prefix}_router_weights_precast_fp32"])
            and torch.equal(top_weights.to(torch.bfloat16), tensors[f"{prefix}_router_weights_recomputed_bf16"])
            and torch.equal(weights, top_weights.to(torch.bfloat16))
            and torch.equal(ids, ids10)
        )
        tie_evidence_ok &= (
            torch.equal(boundary, tensors[f"{prefix}_router_boundary_tie_mask"])
            and torch.equal(selected_boundary, tensors[f"{prefix}_router_selected_boundary_mask"])
            and torch.equal(ids11, tensors[f"{prefix}_router_top11_ids"])
            and torch.equal(torch.gather(logits, 1, ids11), tensors[f"{prefix}_router_top11_native_bf16_logits"])
            and torch.equal(margin, tensors[f"{prefix}_router_top10_top11_margin_fp32"])
        )
        router_token_rows += logits.shape[0]
        boundary_tie_rows += int((boundary.sum(dim=1) > 1).sum())
        selected_boundary_tie_rows += int((selected_boundary.sum(dim=1) > 1).sum())
        zero_margin_rows += int((margin == 0).sum())

    first = {
        str(prompt): {
            stage: next(
                (
                    row["length"]
                    for row in rows
                    if row["prompt"] == prompt and not row["stages"][stage]["exact_equal"]
                ),
                None,
            )
            for stage in STAGES
        }
        for prompt in range(4)
    }
    interpretation = {
        "same_length_nondeterminism_observed": not repeat_all_exact,
        "first_divergent_length_by_prompt_stage": first,
        "whole_prefix16_cache": cache16,
        "whole_prefix16_cache_divergence": not cache16_all_exact,
    }
    provenance = {
        "runner": result.get("runner_sha256") == file_sha(RUNNER),
        "verifier": result.get("verifier_sha256") == file_sha(VERIFIER),
        "verifier_lock": result.get("verifier_lock_sha256") == file_sha(VERIFIER_LOCK),
        "prereg": result.get("prereg_sha256") == file_sha(PREREG),
        "d2_source": result.get("d2_source_sha256") == file_sha(D2_SOURCE),
        "base": result.get("base_sha256") == file_sha(BASE),
        "original_failure": result.get("failure_sha256") == file_sha(ORIGINAL_FAILURE),
        "serialization_failure": result.get("serialization_failure_sha256")
        == file_sha(SERIALIZATION_FAILURE),
    }
    input_file_hashes = {
        key: Path(key).is_file() and file_sha(Path(key)) == value
        for key, value in result.get("inputs", {}).items()
        if isinstance(value, str)
    }
    source_hashes_well_formed = bool(result.get("source_tensor_sha256")) and all(
        isinstance(value, str) and len(value) == 64 for value in result.get("source_tensor_sha256", {}).values()
    )
    checks = {
        "raw_file_sha256": file_sha(RAW) == result.get("raw_sha256"),
        "schema_exact": schema_ok,
        "manifest_exact": manifest == result.get("raw_manifest"),
        "all_tensors_finite": all(bool(torch.isfinite(value.float()).all()) for value in tensors.values()),
        "stored_stage_metrics_recomputed": rows == result.get("stage_metrics"),
        "stored_repeat_metrics_recomputed": repeat_metrics == result.get("repeat_metrics"),
        "stored_interpretation_recomputed": interpretation == result.get("interpretation_classes"),
        "direct_official_tuple_recomputed": bool(direct_tuple_ok),
        "router_arithmetic_recomputed": bool(router_recompute_ok),
        "router_tie_evidence_recomputed": bool(tie_evidence_ok),
        "provenance_all_bound": all(provenance.values()),
        "input_file_hashes_bound": bool(input_file_hashes) and all(input_file_hashes.values()),
        "source_tensor_hashes_present_and_well_formed": source_hashes_well_formed,
        "resource_gates": result["resources"]["windows_peak_working_set_bytes"] <= 12 * 2**30
        and result["resources"]["minimum_available_ram_bytes"] >= 2 * 2**30,
        "diagnostic_only_boundary": result.get("status") == "diagnostic_only_not_pass"
        and result.get("cuda_initialized") is False
        and result.get("serialization_repair_only") is True,
    }
    return {
        "kind": "port80b_t0r12d2r3_independent_artifact_audit",
        "valid_diagnostic": all(checks.values()),
        "scientific_pass": False,
        "checks": checks,
        "provenance_checks": provenance,
        "input_file_hash_checks": input_file_hashes,
        "artifact": {
            "raw_sha256": file_sha(RAW),
            "result_sha256": file_sha(RESULT),
            "tensor_count": len(tensors),
            "raw_bytes": RAW.stat().st_size,
            "result_bytes": RESULT.stat().st_size,
            "source_tensor_identity_count": len(result.get("source_tensor_sha256", {})),
        },
        "stage_divergence": stage_summary,
        "repeat_determinism": {
            "pairs": 2,
            "all_stages_and_caches_exact": repeat_all_exact,
        },
        "cache": {
            "whole_vs_prefix16_pairs": 8,
            "all_exact": cache16_all_exact,
        },
        "router": {
            "captures": len(prefixes()),
            "token_rows": router_token_rows,
            "whole_prefix_final_position_comparisons": 64,
            "logits_weights_ids_exact_comparisons": {
                stage: 64 - stage_summary[stage]["divergent_comparisons"]
                for stage in ("official_router_logits", "official_router_weights", "official_router_ids")
            },
            "boundary_tie_rows": boundary_tie_rows,
            "selected_multiple_boundary_rows": selected_boundary_tie_rows,
            "zero_top10_top11_margin_rows": zero_margin_rows,
        },
        "claim_boundary": (
            "Valid four-prompt, layer-0, CPU-BF16 localization diagnostic only; "
            "not a model-quality, Q5, bank, GPU, performance, or deployment pass."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    rendered = json.dumps(result, indent=None if args.compact else 2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if result["valid_diagnostic"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
