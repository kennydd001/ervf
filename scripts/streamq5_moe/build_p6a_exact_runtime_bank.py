from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from moe_lab.reporting import ROOT


MODEL = ROOT / "models/qwen3-30b-a3b-base"
R = ROOT / "reports/streamq5_moe"
PREREG = R / "P6A_END_TO_END_DECODE_PREREGISTRATION.md"
OUT_DIR = ROOT / "reports/runs/streamq5_moe/p6a_exact_runtime_bank"
RESULT = R / "p6a_exact_runtime_bank_result.json"
GROUP = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_specs():
    specs = []
    for layer in range(48):
        prefix = f"model.layers.{layer}"
        specs.extend(
            [
                (layer, "q", f"{prefix}.self_attn.q_proj.weight", 4096, 2048, "device"),
                (layer, "k", f"{prefix}.self_attn.k_proj.weight", 512, 2048, "device"),
                (layer, "v", f"{prefix}.self_attn.v_proj.weight", 512, 2048, "device"),
                (layer, "o", f"{prefix}.self_attn.o_proj.weight", 2048, 4096, "device"),
                (layer, "router", f"{prefix}.mlp.gate.weight", 128, 2048, "device"),
            ]
        )
    specs.append((48, "head", "lm_head.weight", 151936, 2048, "device"))
    specs.append((49, "embed", "model.embed_tokens.weight", 151936, 2048, "host"))
    return specs


def norm_specs():
    specs = []
    for layer in range(48):
        prefix = f"model.layers.{layer}"
        specs.extend(
            [
                (layer, "input", f"{prefix}.input_layernorm.weight", 2048),
                (layer, "post", f"{prefix}.post_attention_layernorm.weight", 2048),
                (layer, "q_norm", f"{prefix}.self_attn.q_norm.weight", 128),
                (layer, "k_norm", f"{prefix}.self_attn.k_norm.weight", 128),
            ]
        )
    specs.append((48, "final", "model.norm.weight", 2048))
    return specs


@torch.no_grad()
def quantize_exact_p0c(value: torch.Tensor):
    rows, cols = value.shape
    codes = np.empty((rows, cols), dtype=np.int8)
    scale_bits = np.empty((rows, cols // GROUP), dtype="<u2")
    for begin in range(0, rows, 512):
        end = min(rows, begin + 512)
        work = value[begin:end].float().reshape(end - begin, cols // GROUP, GROUP)
        maximum = work.abs().amax(dim=-1)
        scale_fp32 = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
        quantized = torch.round(work / scale_fp32.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        stored_scale = scale_fp32.to(torch.bfloat16)
        codes[begin:end] = quantized.reshape(end - begin, cols).numpy()
        scale_bits[begin:end] = stored_scale.view(torch.int16).numpy().astype("<u2", copy=False)
    return codes, scale_bits


def main():
    if OUT_DIR.exists() or RESULT.exists():
        raise FileExistsError("refusing to overwrite immutable P6A bank")
    OUT_DIR.mkdir(parents=True)
    index_path = MODEL / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    specs = matrix_specs()
    by_file: dict[str, list[tuple]] = {}
    for spec in specs:
        by_file.setdefault(weight_map[spec[2]], []).append(spec)
    records = []
    total_weights = total_codes = total_scales = total_bytes = device_bytes = 0
    started = time.perf_counter()
    for shard, shard_specs in by_file.items():
        with safe_open(MODEL / shard, framework="pt", device="cpu") as handle:
            for layer, name, key, rows, cols, residency in shard_specs:
                value = handle.get_tensor(key)
                if tuple(value.shape) != (rows, cols):
                    raise ValueError(f"shape mismatch for {key}: {tuple(value.shape)}")
                codes, scale_bits = quantize_exact_p0c(value)
                filename = "lm_head.q8bin" if name == "head" else "embed.q8bin" if name == "embed" else f"layer_{layer:02d}_{name}.q8bin"
                path = OUT_DIR / filename
                with path.open("xb") as output:
                    output.write(codes.tobytes(order="C"))
                    output.write(scale_bits.tobytes(order="C"))
                code_bytes, scale_bytes = codes.nbytes, scale_bits.nbytes
                record = {
                    "layer": layer,
                    "name": name,
                    "source_key": key,
                    "source_shard": shard,
                    "rows": rows,
                    "cols": cols,
                    "groups": rows * (cols // GROUP),
                    "weights": rows * cols,
                    "code_bytes": code_bytes,
                    "scale_bytes": scale_bytes,
                    "bytes": path.stat().st_size,
                    "residency": residency,
                    "artifact": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "artifact_sha256": sha256(path),
                }
                records.append(record)
                total_weights += rows * cols
                total_codes += codes.size
                total_scales += scale_bits.size
                total_bytes += record["bytes"]
                if residency == "device":
                    device_bytes += record["bytes"]
                print(json.dumps({"record": len(records), "layer": layer, "name": name, "bytes": record["bytes"]}), flush=True)

    norm_path = OUT_DIR / "norms.bf16bin"
    norm_records = []
    norm_cursor = 0
    norm_by_file: dict[str, list[tuple]] = {}
    for spec in norm_specs():
        norm_by_file.setdefault(weight_map[spec[2]], []).append(spec)
    with norm_path.open("xb") as output:
        for shard, shard_specs in norm_by_file.items():
            with safe_open(MODEL / shard, framework="pt", device="cpu") as handle:
                for layer, name, key, elements in shard_specs:
                    value = handle.get_tensor(key).to(torch.bfloat16).contiguous()
                    if value.numel() != elements:
                        raise ValueError(f"norm shape mismatch for {key}")
                    raw = value.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes(order="C")
                    output.write(raw)
                    norm_records.append(
                        {
                            "layer": layer,
                            "name": name,
                            "source_key": key,
                            "source_shard": shard,
                            "elements": elements,
                            "offset": norm_cursor,
                            "bytes": len(raw),
                            "payload_sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    )
                    norm_cursor += len(raw)

    result = {
        "kind": "streamq5_moe_p6a_exact_runtime_bank",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "p6a_bank_built_pending_independent_verification",
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "model_index_sha256": sha256(index_path),
        },
        "quantization": {
            "bits": 8,
            "group_size": GROUP,
            "qmax": 127,
            "code_selection_scale": "FP32 maxabs/127",
            "code_rounding": "torch_round_nearest_even",
            "persisted_scale": "BF16 bits little-endian",
            "physical_weight": "BF16(code * float(BF16 scale))",
        },
        "aggregate": {
            "records": len(records),
            "device_records": sum(r["residency"] == "device" for r in records),
            "host_records": sum(r["residency"] == "host" for r in records),
            "weights": total_weights,
            "codes": total_codes,
            "scales": total_scales,
            "bytes": total_bytes,
            "device_bytes": device_bytes,
            "host_embedding_bytes": total_bytes - device_bytes,
            "gib": total_bytes / 2**30,
        },
        "records": records,
        "norm_bank": {
            "artifact": str(norm_path.relative_to(ROOT)).replace("\\", "/"),
            "artifact_sha256": sha256(norm_path),
            "bytes": norm_path.stat().st_size,
            "records": norm_records,
        },
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": "Exact P0C-semantic physical Q8/embedding/norm bank only; independent verification and end-to-end runtime remain unopened.",
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"], "norm_bytes": result["norm_bank"]["bytes"], "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
