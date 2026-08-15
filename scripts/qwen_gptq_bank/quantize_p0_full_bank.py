from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from moe_lab.qwen_gptq_bank import (
    codes_from_quantized,
    fastgrid_pure_gptq_projection,
    pack_2bit_codes,
    unpack_2bit_codes,
)
from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_checkpoint_tensors


MODEL = ROOT / "models/qwen3-30b-a3b-base"
GSQ_ROOT = ROOT / "third_party/GSQ"
PREREG = ROOT / "reports/qwen_gptq_bank/P0_FULL_BANK_PREREGISTRATION.md"
EQUIVALENCE = ROOT / "reports/qwen_gptq_bank/p0_fastgrid_equivalence_result_h.json"
CALIBRATION_DIR = ROOT / "reports/runs/qwen_gptq_bank/p0_calibration"
RUN_DIR = ROOT / "reports/runs/qwen_gptq_bank/p0_bank"
LAYER_DIR = ROOT / "reports/qwen_gptq_bank/p0_bank_layers"
LAYERS, EXPERTS, ROWS = 48, 128, 128
HIDDEN, INTERMEDIATE, GROUP = 2_048, 768, 128
EXPECTED_CODES_PER_LAYER = EXPERTS * 3 * INTERMEDIATE * HIDDEN
EXPECTED_PACKED_BYTES_PER_LAYER = EXPECTED_CODES_PER_LAYER // 4
EXPECTED_SCALE_ELEMENTS_PER_LAYER = EXPERTS * (
    2 * INTERMEDIATE * (HIDDEN // GROUP) + HIDDEN * (INTERMEDIATE // GROUP)
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, action="append", choices=range(LAYERS))
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_layer_weights(layer: int, weight_map: dict[str, str]):
    identities = {}
    names = []
    for expert in range(EXPERTS):
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        identities[expert] = {
            kind: f"{base}.{kind}_proj.weight" for kind in ("gate", "up", "down")
        }
        names.extend(identities[expert].values())
    loaded = load_checkpoint_tensors(MODEL, names, weight_map)
    result = {
        kind: torch.stack([loaded[identities[expert][kind]] for expert in range(EXPERTS)]).contiguous()
        for kind in ("gate", "up", "down")
    }
    del loaded
    expected = {
        "gate": (EXPERTS, INTERMEDIATE, HIDDEN),
        "up": (EXPERTS, INTERMEDIATE, HIDDEN),
        "down": (EXPERTS, HIDDEN, INTERMEDIATE),
    }
    if {kind: tuple(value.shape) for kind, value in result.items()} != expected:
        raise ValueError(f"unexpected source weight shape at layer {layer}")
    if any(value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()) for value in result.values()):
        raise ValueError(f"invalid source weights at layer {layer}")
    return result


def original_down_inputs(x, gate, up):
    parts = []
    for index in range(x.shape[0]):
        parts.append(F.silu(F.linear(x[index], gate[index])) * F.linear(x[index], up[index]))
    return torch.stack(parts).contiguous()


def histogram(codes: torch.Tensor) -> list[int]:
    return torch.bincount((codes.long() + 2).reshape(-1).cpu(), minlength=4).tolist()


def verify_orphan_partial(layer: int):
    partial = RUN_DIR / f"layer_{layer:02d}.safetensors.inprogress"
    if partial.exists():
        failed = RUN_DIR / "failed_attempts"
        failed.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        partial.replace(failed / f"layer_{layer:02d}_{stamp}.safetensors.inprogress")


def quantize_layer(layer: int, batch_size: int, device, weight_map):
    artifact = RUN_DIR / f"layer_{layer:02d}.safetensors"
    report = LAYER_DIR / f"layer_{layer:02d}.json"
    if artifact.exists() or report.exists():
        if not artifact.exists() or not report.exists():
            raise RuntimeError(f"partial completed bank checkpoint at layer {layer}")
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload["artifact_sha256"] != sha256_file(artifact):
            raise ValueError(f"existing bank checkpoint hash mismatch at layer {layer}")
        print(json.dumps({"layer": layer, "status": "verified_skip"}), flush=True)
        return
    verify_orphan_partial(layer)
    layer_started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()
    rss_peak = process.memory_info().rss
    calibration_path = CALIBRATION_DIR / f"layer_{layer:02d}.safetensors"
    with safe_open(calibration_path, framework="pt", device="cpu") as handle:
        calibration = handle.get_tensor("moe_input")
        calibration_metadata = handle.metadata() or {}
    if tuple(calibration.shape) != (EXPERTS, ROWS, HIDDEN) or calibration.dtype != torch.bfloat16:
        raise ValueError(f"invalid calibration at layer {layer}")
    original = load_layer_weights(layer, weight_map)
    source_hashes = {kind: sha256_tensor(value) for kind, value in original.items()}

    tensors = {
        "gate_codes_packed": torch.empty((EXPERTS, INTERMEDIATE, HIDDEN // 4), dtype=torch.uint8),
        "gate_scales": torch.empty((EXPERTS, INTERMEDIATE, HIDDEN // GROUP), dtype=torch.bfloat16),
        "up_codes_packed": torch.empty((EXPERTS, INTERMEDIATE, HIDDEN // 4), dtype=torch.uint8),
        "up_scales": torch.empty((EXPERTS, INTERMEDIATE, HIDDEN // GROUP), dtype=torch.bfloat16),
        "down_codes_packed": torch.empty((EXPERTS, HIDDEN, INTERMEDIATE // 4), dtype=torch.uint8),
        "down_scales": torch.empty((EXPERTS, HIDDEN, INTERMEDIATE // GROUP), dtype=torch.bfloat16),
    }
    histograms = {kind: torch.zeros(4, dtype=torch.int64) for kind in ("gate", "up", "down")}
    batch_timings = []

    for begin in range(0, EXPERTS, batch_size):
        end = min(begin + batch_size, EXPERTS)
        batch_started = time.perf_counter()
        x = calibration[begin:end].to(device).contiguous()
        gate = original["gate"][begin:end].to(device).contiguous()
        up = original["up"][begin:end].to(device).contiguous()
        down_weight = original["down"][begin:end].to(device).contiguous()
        z = original_down_inputs(x, gate, up)
        quantized = {kind: [] for kind in ("gate", "up", "down")}
        for index in range(end - begin):
            quantized["gate"].append(fastgrid_pure_gptq_projection(gate[index], x[index], GSQ_ROOT))
            quantized["up"].append(fastgrid_pure_gptq_projection(up[index], x[index], GSQ_ROOT))
            quantized["down"].append(fastgrid_pure_gptq_projection(down_weight[index], z[index], GSQ_ROOT))
        batch_values = {
            kind: (
                torch.stack([item.weight for item in quantized[kind]]),
                torch.stack([item.scales for item in quantized[kind]]),
            )
            for kind in quantized
        }

        for kind, (quantized_weights, scales) in batch_values.items():
            code_values = codes_from_quantized(quantized_weights, scales)
            packed = pack_2bit_codes(code_values).cpu()
            if not torch.equal(unpack_2bit_codes(packed), code_values.cpu()):
                raise RuntimeError(f"packed round trip failed at layer {layer}, batch {begin}, {kind}")
            tensors[f"{kind}_codes_packed"][begin:end] = packed
            tensors[f"{kind}_scales"][begin:end] = scales.cpu()
            histograms[kind] += torch.tensor(histogram(code_values), dtype=torch.int64)
        if any(
            not bool(torch.isfinite(scales).all() and (scales != 0).all())
            for _weights, scales in batch_values.values()
        ):
            raise RuntimeError(f"invalid scale at layer {layer}, batch {begin}")
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - batch_started
        batch_timings.append({"begin": begin, "end": end, "seconds": seconds})
        rss_peak = max(rss_peak, process.memory_info().rss)
        print(json.dumps({
            "layer": layer, "experts": [begin, end - 1], "seconds": seconds,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }), flush=True)
        del x, gate, up, down_weight, z, quantized, batch_values
        del quantized_weights, scales, code_values, packed
        gc.collect(); torch.cuda.empty_cache()

    packed_bytes = sum(value.numel() for key, value in tensors.items() if key.endswith("_packed"))
    scale_elements = sum(value.numel() for key, value in tensors.items() if key.endswith("_scales"))
    if packed_bytes != EXPECTED_PACKED_BYTES_PER_LAYER or scale_elements != EXPECTED_SCALE_ELEMENTS_PER_LAYER:
        raise RuntimeError(f"physical tensor count mismatch at layer {layer}")
    if sum(sum(values.tolist()) for values in histograms.values()) != EXPECTED_CODES_PER_LAYER:
        raise RuntimeError(f"code histogram count mismatch at layer {layer}")
    if torch.cuda.max_memory_allocated(device) > int(7.5 * 2**30) or rss_peak > 40 * 2**30:
        raise MemoryError(f"bank producer exceeded resource ceiling at layer {layer}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LAYER_DIR.mkdir(parents=True, exist_ok=True)
    partial = RUN_DIR / f"layer_{layer:02d}.safetensors.inprogress"
    save_file(tensors, partial, metadata={
        "kind": "qwen_gptq_bank_p0_layer",
        "layer": str(layer), "experts": str(EXPERTS), "rows_per_expert": str(ROWS),
        "group_size": str(GROUP), "code_mapping": "stored_unsigned=code+2; four little-order 2bit codes per byte",
        "calibration_sha256": sha256_file(calibration_path),
        "equivalence_result_sha256": sha256_file(EQUIVALENCE),
    })
    os.replace(partial, artifact)
    payload = {
        "kind": "qwen_gptq_bank_p0_layer_result",
        "layer": layer, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256_file(artifact), "artifact_bytes": artifact.stat().st_size,
        "calibration": str(calibration_path.relative_to(ROOT)).replace("\\", "/"),
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_moe_input_sha256": calibration_metadata.get("moe_input_sha256"),
        "source_weight_sha256": source_hashes,
        "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
        "tensor_dtypes": {key: str(value.dtype) for key, value in tensors.items()},
        "codes": EXPECTED_CODES_PER_LAYER, "packed_code_bytes": packed_bytes,
        "scale_elements": scale_elements, "scale_bytes": scale_elements * 2,
        "histograms": {
            kind: {str(code - 2): int(value) for code, value in enumerate(values.tolist())}
            for kind, values in histograms.items()
        },
        "controls": {
            "all_codes_in_alphabet": True, "all_scales_finite_nonzero": True,
            "all_packed_roundtrips_exact": True, "all_128_experts_present": True,
        },
        "batch_size": batch_size, "batch_timings": batch_timings,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_process_rss_bytes": rss_peak, "elapsed_seconds": time.perf_counter() - layer_started,
    }
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "layer": layer, "status": "complete", "artifact_bytes": payload["artifact_bytes"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }), flush=True)


if __name__ == "__main__":
    args = parse_args()
    if args.batch_size < 1 or args.batch_size > 32:
        raise ValueError("batch size must be in [1, 32]")
    equivalence = json.loads(EQUIVALENCE.read_text(encoding="utf-8"))
    if equivalence["status"] != "equivalence_pass":
        raise RuntimeError("batched GPTQ equivalence gate did not pass")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    selected_layers = sorted(set(args.layer if args.layer is not None else range(LAYERS)))
    weight_map = checkpoint_weight_map(MODEL)
    device = torch.device("cuda")
    for selected_layer in selected_layers:
        quantize_layer(selected_layer, args.batch_size, device, weight_map)
