from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import get_safetensors_metadata


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
PREREG = REPORTS / "N4A_SYNTHETIC_80B_SHAPE_CAPACITY_PREREGISTRATION.md"
P7C = REPORTS / "p7c_ervf_end_to_end_test.json"
OUTPUT = REPORTS / "n4a_synthetic_80b_shape_capacity.json"

MODEL_ID = "Qwen/Qwen3-Coder-Next"
REVISION = "a19358a7659bd1f564300250ee189120c49a562f"
CONFIG_URL = f"https://huggingface.co/{MODEL_ID}/blob/{REVISION}/config.json"
MODEL_CODE_URL = (
    "https://github.com/huggingface/transformers/blob/main/"
    "src/transformers/models/qwen3_next/modular_qwen3_next.py"
)

LAYERS = 48
EXPERTS = 512
TOP_K = 10
HIDDEN = 2048
MOE_INTERMEDIATE = 512
VOCAB = 151_936
FULL_ATTN_INTERVAL = 4
FULL_ATTN_LAYERS = LAYERS // FULL_ATTN_INTERVAL
LINEAR_ATTN_LAYERS = LAYERS - FULL_ATTN_LAYERS
ATTN_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 256
LINEAR_K_HEADS = 16
LINEAR_V_HEADS = 32
LINEAR_K_DIM = 128
LINEAR_V_DIM = 128
CONV_KERNEL = 4

GROUP = 128
Q5_HEADER = 64
Q5_ALIGNMENT = 4096
HOST_LIMIT = 58 * 2**30
HOST_PROCESS_RESERVE = 1 * 2**30
DEVICE_SCRATCH_RESERVE = 256 * 2**20
PINNED_WINDOWS = 8
QWEN30_ROUTED_ACTIVE = 48 * 8 * 3 * 2048 * 768


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def q8_bytes(shape: tuple[int, ...]) -> int:
    """STREAMQ5 Q8 matrices; non-matrix tensors stay BF16."""
    weights = numel(shape)
    if len(shape) == 2 and shape[-1] % GROUP == 0:
        return weights + weights // GROUP * 2
    return weights * 2


def aligned(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def q5_record_bytes(shape: tuple[int, int]) -> dict[str, int]:
    weights = numel(shape)
    if weights % 8 or weights % GROUP:
        raise ValueError(f"unsupported Q5 shape {shape}")
    codes = weights * 5 // 8
    scales = weights // GROUP * 2
    payload_with_header = Q5_HEADER + codes + scales
    return {
        "weights": weights,
        "code_bytes": codes,
        "scale_bytes": scales,
        "header_bytes": Q5_HEADER,
        "payload_with_header_bytes": payload_with_header,
        "padding_bytes": aligned(payload_with_header, Q5_ALIGNMENT) - payload_with_header,
        "record_bytes": aligned(payload_with_header, Q5_ALIGNMENT),
    }


def expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (VOCAB, HIDDEN),
        "model.norm.weight": (HIDDEN,),
        "lm_head.weight": (VOCAB, HIDDEN),
    }
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}"
        shapes[f"{prefix}.input_layernorm.weight"] = (HIDDEN,)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (HIDDEN,)
        shapes[f"{prefix}.mlp.gate.weight"] = (EXPERTS, HIDDEN)
        shapes[f"{prefix}.mlp.shared_expert_gate.weight"] = (1, HIDDEN)
        for expert in range(EXPERTS):
            base = f"{prefix}.mlp.experts.{expert}"
            shapes[f"{base}.gate_proj.weight"] = (MOE_INTERMEDIATE, HIDDEN)
            shapes[f"{base}.up_proj.weight"] = (MOE_INTERMEDIATE, HIDDEN)
            shapes[f"{base}.down_proj.weight"] = (HIDDEN, MOE_INTERMEDIATE)
        shared = f"{prefix}.mlp.shared_expert"
        shapes[f"{shared}.gate_proj.weight"] = (MOE_INTERMEDIATE, HIDDEN)
        shapes[f"{shared}.up_proj.weight"] = (MOE_INTERMEDIATE, HIDDEN)
        shapes[f"{shared}.down_proj.weight"] = (HIDDEN, MOE_INTERMEDIATE)

        if (layer + 1) % FULL_ATTN_INTERVAL == 0:
            attn = f"{prefix}.self_attn"
            shapes[f"{attn}.q_proj.weight"] = (ATTN_HEADS * HEAD_DIM * 2, HIDDEN)
            shapes[f"{attn}.k_proj.weight"] = (KV_HEADS * HEAD_DIM, HIDDEN)
            shapes[f"{attn}.v_proj.weight"] = (KV_HEADS * HEAD_DIM, HIDDEN)
            shapes[f"{attn}.o_proj.weight"] = (HIDDEN, ATTN_HEADS * HEAD_DIM)
            shapes[f"{attn}.q_norm.weight"] = (HEAD_DIM,)
            shapes[f"{attn}.k_norm.weight"] = (HEAD_DIM,)
        else:
            linear = f"{prefix}.linear_attn"
            key_dim = LINEAR_K_HEADS * LINEAR_K_DIM
            value_dim = LINEAR_V_HEADS * LINEAR_V_DIM
            conv_dim = key_dim * 2 + value_dim
            shapes[f"{linear}.in_proj_qkvz.weight"] = (key_dim * 2 + value_dim * 2, HIDDEN)
            shapes[f"{linear}.in_proj_ba.weight"] = (LINEAR_V_HEADS * 2, HIDDEN)
            shapes[f"{linear}.conv1d.weight"] = (conv_dim, 1, CONV_KERNEL)
            shapes[f"{linear}.A_log"] = (LINEAR_V_HEADS,)
            shapes[f"{linear}.dt_bias"] = (LINEAR_V_HEADS,)
            shapes[f"{linear}.norm.weight"] = (LINEAR_V_DIM,)
            shapes[f"{linear}.out_proj.weight"] = (HIDDEN, value_dim)
    return shapes


def collect_official_shapes() -> tuple[object, dict[str, tuple[int, ...]]]:
    repo = get_safetensors_metadata(MODEL_ID, revision=REVISION)
    tensors = {}
    for file_metadata in repo.files_metadata.values():
        for key, info in file_metadata.tensors.items():
            if key in tensors:
                raise ValueError(f"duplicate official tensor key: {key}")
            tensors[key] = tuple(info.shape)
    return repo, tensors


def classify(key: str) -> str:
    if ".mlp.experts." in key:
        return "routed_experts"
    if ".mlp.shared_expert." in key:
        return "shared_experts"
    if key == "model.embed_tokens.weight":
        return "embedding"
    if key == "lm_head.weight":
        return "lm_head"
    return "dense_core"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not PREREG.is_file() or not P7C.is_file():
        raise FileNotFoundError("N4A preregistration and P7C hardware evidence are required")

    expected = expected_shapes()
    repo, official = collect_official_shapes()
    missing = sorted(set(expected) - set(official))
    unexpected = sorted(set(official) - set(expected))
    mismatched = [
        {"key": key, "expected": expected[key], "official": official[key]}
        for key in sorted(set(expected) & set(official))
        if expected[key] != official[key]
    ]

    groups: dict[str, dict[str, int]] = {}
    for key, shape in official.items():
        group = classify(key)
        row = groups.setdefault(group, {"tensors": 0, "parameters": 0, "streamq8_bytes": 0})
        row["tensors"] += 1
        row["parameters"] += numel(shape)
        row["streamq8_bytes"] += q8_bytes(shape)

    gate_record = q5_record_bytes((MOE_INTERMEDIATE, HIDDEN))
    up_record = q5_record_bytes((MOE_INTERMEDIATE, HIDDEN))
    down_record = q5_record_bytes((HIDDEN, MOE_INTERMEDIATE))
    expert_record_bytes = gate_record["record_bytes"] + up_record["record_bytes"] + down_record["record_bytes"]
    routed_bank_bytes = LAYERS * EXPERTS * expert_record_bytes
    shared_bank_bytes = LAYERS * expert_record_bytes
    full_q5_bank_bytes = routed_bank_bytes + shared_bank_bytes

    routed_active = LAYERS * TOP_K * 3 * HIDDEN * MOE_INTERMEDIATE
    shared_active = LAYERS * 3 * HIDDEN * MOE_INTERMEDIATE
    active_total = routed_active + shared_active
    all_cold_routed_bytes = LAYERS * TOP_K * expert_record_bytes

    p7c = json.loads(P7C.read_text(encoding="utf-8"))
    physical = p7c["physical"]
    device_total = int(physical["total_vram_bytes"])
    cuda_context_reserve = device_total - int(physical["free_before_bytes"])
    device_q8_shell = groups["dense_core"]["streamq8_bytes"] + groups["lm_head"]["streamq8_bytes"]
    host_embedding = groups["embedding"]["streamq8_bytes"]
    pinned_staging = PINNED_WINDOWS * expert_record_bytes
    host_accounted = full_q5_bank_bytes + host_embedding + pinned_staging
    host_with_reserve = host_accounted + HOST_PROCESS_RESERVE

    recurrent_state = LINEAR_ATTN_LAYERS * LINEAR_V_HEADS * LINEAR_K_DIM * LINEAR_V_DIM * 4
    conv_state = LINEAR_ATTN_LAYERS * (LINEAR_K_HEADS * LINEAR_K_DIM * 2 + LINEAR_V_HEADS * LINEAR_V_DIM) * CONV_KERNEL * 2
    contexts = {}
    for context in (4096, 32768):
        kv = FULL_ATTN_LAYERS * context * KV_HEADS * HEAD_DIM * 2 * 2
        state = kv + recurrent_state + conv_state
        available_cache = (
            device_total
            - cuda_context_reserve
            - device_q8_shell
            - shared_bank_bytes
            - state
            - DEVICE_SCRATCH_RESERVE
        )
        slots = available_cache // expert_record_bytes
        contexts[str(context)] = {
            "full_attention_kv_bytes": kv,
            "delta_recurrent_fp32_bytes": recurrent_state,
            "delta_conv_bf16_bytes": conv_state,
            "total_state_bytes": state,
            "available_routed_cache_bytes": available_cache,
            "routed_expert_slots": slots,
            "mean_slots_per_layer": slots / LAYERS,
            "active_sets_of_480_records": slots / (LAYERS * TOP_K),
        }

    official_parameter_count = sum(numel(shape) for shape in official.values())
    gates = {
        "official_keyset_exact": not missing and not unexpected,
        "official_shapes_exact": not mismatched,
        "official_tensor_count_74391": len(official) == 74_391,
        "official_parameter_count_exact": official_parameter_count == 79_674_391_296,
        "official_metadata_total_size_exact": int(repo.metadata["total_size"]) == official_parameter_count * 2,
        "routed_tensor_count_exact": groups["routed_experts"]["tensors"] == LAYERS * EXPERTS * 3,
        "shared_tensor_count_exact": groups["shared_experts"]["tensors"] == LAYERS * 3,
        "full_q5_bank_le_50_gib": full_q5_bank_bytes <= 50 * 2**30,
        "device_q8_shell_le_2_gib": device_q8_shell <= 2 * 2**30,
        "host_with_1gib_reserve_le_58_gib": host_with_reserve <= HOST_LIMIT,
        "4k_cache_at_least_32_per_layer": contexts["4096"]["routed_expert_slots"] >= LAYERS * 32,
        "32k_cache_at_least_32_per_layer": contexts["32768"]["routed_expert_slots"] >= LAYERS * 32,
        "active_expert_mass_lt_qwen30_routed": active_total < QWEN30_ROUTED_ACTIVE,
    }

    result = {
        "kind": "streamq5_moe_n4a_synthetic_80b_shape_capacity",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "cpu_shape_capacity_pass_physical_performance_pending" if all(gates.values()) else "cpu_shape_capacity_fail",
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "script_sha256": sha256(Path(__file__)),
            "p7c_sha256": sha256(P7C),
            "model_id": MODEL_ID,
            "revision": REVISION,
            "official_config_url": CONFIG_URL,
            "official_reference_code_url": MODEL_CODE_URL,
            "weight_payload_downloaded": False,
            "metadata_only": True,
        },
        "architecture": {
            "layers": LAYERS,
            "linear_attention_layers": LINEAR_ATTN_LAYERS,
            "full_attention_layers": FULL_ATTN_LAYERS,
            "routed_experts_per_layer": EXPERTS,
            "top_k": TOP_K,
            "hidden_size": HIDDEN,
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "shared_experts_per_layer": 1,
            "shared_expert_intermediate_size": MOE_INTERMEDIATE,
            "vocab_size": VOCAB,
            "official_tensor_count": len(official),
            "official_parameter_count": official_parameter_count,
            "official_bf16_checkpoint_bytes": int(repo.metadata["total_size"]),
        },
        "shape_verification": {
            "generated_tensor_count": len(expected),
            "missing": missing[:20],
            "unexpected": unexpected[:20],
            "mismatched": mismatched[:20],
            "truncated_failure_lists": len(missing) > 20 or len(unexpected) > 20 or len(mismatched) > 20,
            "groups": groups,
        },
        "q5_record_contract": {
            "group_size": GROUP,
            "header_bytes": Q5_HEADER,
            "alignment_bytes": Q5_ALIGNMENT,
            "gate": gate_record,
            "up": up_record,
            "down": down_record,
            "expert_record_bytes": expert_record_bytes,
        },
        "expert_accounting": {
            "routed_parameters": groups["routed_experts"]["parameters"],
            "shared_parameters": groups["shared_experts"]["parameters"],
            "routed_bank_bytes": routed_bank_bytes,
            "shared_bank_bytes": shared_bank_bytes,
            "full_q5_bank_bytes": full_q5_bank_bytes,
            "routed_active_parameters_per_token": routed_active,
            "shared_active_parameters_per_token": shared_active,
            "total_active_expert_parameters_per_token": active_total,
            "qwen30_routed_active_reference": QWEN30_ROUTED_ACTIVE,
            "active_mass_ratio_vs_qwen30_routed": active_total / QWEN30_ROUTED_ACTIVE,
            "all_cold_routed_record_bytes_per_token": all_cold_routed_bytes,
            "all_cold_h2d_ms_at_local_26_16_decimal_gbps": all_cold_routed_bytes / 26.16e9 * 1000,
        },
        "host_budget": {
            "limit_bytes": HOST_LIMIT,
            "full_q5_bank_bytes": full_q5_bank_bytes,
            "persistent_q8_embedding_bytes": host_embedding,
            "eight_pinned_staging_windows_bytes": pinned_staging,
            "accounted_bytes": host_accounted,
            "explicit_process_reserve_bytes": HOST_PROCESS_RESERVE,
            "accounted_plus_reserve_bytes": host_with_reserve,
            "headroom_bytes": HOST_LIMIT - host_with_reserve,
        },
        "device_budget": {
            "measured_total_vram_bytes": device_total,
            "measured_cuda_context_reserve_bytes": cuda_context_reserve,
            "q8_dense_core_bytes": groups["dense_core"]["streamq8_bytes"],
            "q8_lm_head_bytes": groups["lm_head"]["streamq8_bytes"],
            "q8_device_shell_bytes": device_q8_shell,
            "resident_shared_q5_bank_bytes": shared_bank_bytes,
            "scratch_staging_reserve_bytes": DEVICE_SCRATCH_RESERVE,
            "contexts": contexts,
        },
        "gates": gates,
        "overall_pass": all(gates.values()),
        "next_stage_authorized": "separately_preregistered_gpu_shape_benchmark_only" if all(gates.values()) else False,
        "claim_boundary": (
            "CPU-only exact official tensor-shape and analytical memory-capacity gate. "
            "No GPU kernel, routing, quality, checkpoint payload, latency, throughput, prefill, "
            "32K attention-time, or end-to-end 80B claim."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "architecture": result["architecture"],
        "expert_accounting": result["expert_accounting"],
        "host_budget": result["host_budget"],
        "device_budget": result["device_budget"],
        "gates": gates,
    }, indent=2))


if __name__ == "__main__":
    main()
