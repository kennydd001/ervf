from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import struct
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_checkpoint_tensors


MODEL = ROOT / "models/qwen3-30b-a3b-base"
PREREG = ROOT / "reports/streamq5_moe/P1B_PHYSICAL_BANK_PREREGISTRATION.md"
P1A_VERIFY = ROOT / "reports/streamq5_moe/p1a_cache_verification.json"
RUN_DIR = ROOT / "reports/runs/streamq5_moe/p1b_q5_bank"
LAYER_DIR = ROOT / "reports/streamq5_moe/p1b_bank_layers"
RESULT = ROOT / "reports/streamq5_moe/p1b_physical_bank_result.json"
REPORT = ROOT / "reports/streamq5_moe/P1B_PHYSICAL_BANK_REPORT.md"
LAYERS, EXPERTS, GROUP, BATCH = 48, 128, 128, 8
MATRICES = (("gate", 768, 2048, 0), ("up", 768, 2048, 1), ("down", 2048, 768, 2))
HEADER_FORMAT = "<4sHHHBBIIH2xIII28s"
HEADER_BYTES, RECORD_BYTES = 64, 1_011_712
EXPERT_BYTES, LAYER_BYTES, BANK_BYTES = 3_035_136, 388_497_408, 18_647_875_584


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def layer_weights(layer: int, weight_map: dict[str, str]):
    identities = {}
    names = []
    for expert in range(EXPERTS):
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        identities[expert] = {kind: f"{base}.{kind}_proj.weight" for kind, *_rest in MATRICES}
        names.extend(identities[expert].values())
    loaded = load_checkpoint_tensors(MODEL, names, weight_map)
    result = {kind: torch.stack([loaded[identities[expert][kind]] for expert in range(EXPERTS)]).contiguous() for kind, *_rest in MATRICES}
    del loaded
    return result


@torch.no_grad()
def quantize_batch(value: torch.Tensor, device: torch.device):
    original_shape = value.shape
    work = value.to(device).float().reshape(original_shape[0], original_shape[1], original_shape[2] // GROUP, GROUP)
    maximum = work.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 15, torch.ones_like(maximum))
    codes = torch.round(work / scale).clamp(-15, 15).to(torch.int8)
    return codes.reshape(original_shape).cpu().contiguous(), scale.squeeze(-1).to(torch.bfloat16).cpu().contiguous()


def pack_q5(codes: torch.Tensor) -> list[bytes]:
    values = (codes.numpy().astype(np.int16) + 15).astype(np.uint64)
    grouped = values.reshape(values.shape[0], values.shape[1], -1, 8)
    shifts = (np.arange(8, dtype=np.uint64) * 5).reshape(1, 1, 1, 8)
    words = np.bitwise_or.reduce(grouped << shifts, axis=-1)
    byte_shifts = (np.arange(5, dtype=np.uint64) * 8).reshape(1, 1, 1, 5)
    packed = ((words[..., None] >> byte_shifts) & 0xFF).astype(np.uint8).reshape(values.shape[0], values.shape[1], -1)
    return [packed[index].tobytes(order="C") for index in range(packed.shape[0])]


def header(layer, expert, projection, rows, columns, code_bytes, scale_bytes, crc):
    value = struct.pack(HEADER_FORMAT, b"SQ5M", 1, layer, expert, projection, 5, rows, columns, GROUP, code_bytes, scale_bytes, crc, b"\x00" * 28)
    if len(value) != HEADER_BYTES:
        raise RuntimeError("header size contract failed")
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, action="append", choices=range(LAYERS))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite completed P1B result")
    p1a = json.loads(P1A_VERIFY.read_text(encoding="utf-8"))
    if p1a.get("status") != "p1a_cache_verification_pass":
        raise RuntimeError("independent P1A pass required")
    device = torch.device("cuda")
    weight_map = checkpoint_weight_map(MODEL)
    process = psutil.Process(); torch.cuda.reset_peak_memory_stats(device)
    peak_rss = process.memory_info().rss; started = time.perf_counter()
    RUN_DIR.mkdir(parents=True, exist_ok=True); LAYER_DIR.mkdir(parents=True, exist_ok=True)
    selected = args.layer if args.layer else list(range(LAYERS))

    for layer in selected:
        artifact = RUN_DIR / f"layer_{layer:02d}.q5bin"
        layer_report = LAYER_DIR / f"layer_{layer:02d}.json"
        if artifact.exists() or layer_report.exists():
            if not artifact.exists() or not layer_report.exists() or artifact.stat().st_size != LAYER_BYTES:
                raise RuntimeError(f"partial or invalid completed layer {layer}")
            saved = json.loads(layer_report.read_text(encoding="utf-8"))
            if saved["artifact_sha256"] != sha256(artifact):
                raise ValueError(f"existing layer hash mismatch {layer}")
            print(json.dumps({"layer": layer, "status": "verified_skip"}), flush=True)
            continue
        layer_started = time.perf_counter(); values = layer_weights(layer, weight_map)
        source_hashes = {kind: tensor_sha(value) for kind, value in values.items()}
        partial = artifact.with_suffix(".q5bin.inprogress")
        if partial.exists():
            failed = RUN_DIR / "failed_attempts"; failed.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            os.replace(partial, failed / f"layer_{layer:02d}_{stamp}.q5bin.inprogress")
        records = 0; codes_total = scales_total = 0; crc_xor = 0
        with partial.open("wb", buffering=8 * 2**20) as handle_out:
            for begin in range(0, EXPERTS, BATCH):
                end = min(begin + BATCH, EXPERTS)
                encoded = {}
                for kind, rows, columns, projection in MATRICES:
                    codes, scales = quantize_batch(values[kind][begin:end], device)
                    packed_parts = pack_q5(codes)
                    scale_parts = [scales[index].view(torch.uint16).numpy().tobytes(order="C") for index in range(end - begin)]
                    encoded[kind] = (packed_parts, scale_parts, rows, columns, projection)
                    del codes, scales
                for offset, expert in enumerate(range(begin, end)):
                    for kind, _rows, _columns, _projection in MATRICES:
                        packed_parts, scale_parts, rows, columns, projection = encoded[kind]
                        code_payload, scale_payload = packed_parts[offset], scale_parts[offset]
                        payload_crc = zlib.crc32(code_payload); payload_crc = zlib.crc32(scale_payload, payload_crc) & 0xFFFFFFFF
                        record_header = header(layer, expert, projection, rows, columns, len(code_payload), len(scale_payload), payload_crc)
                        padding = RECORD_BYTES - HEADER_BYTES - len(code_payload) - len(scale_payload)
                        if len(code_payload) != 983_040 or len(scale_payload) != 24_576 or padding != 4_032:
                            raise RuntimeError("record byte contract failed")
                        handle_out.write(record_header); handle_out.write(code_payload); handle_out.write(scale_payload); handle_out.write(b"\x00" * padding)
                        records += 1; codes_total += rows * columns; scales_total += rows * (columns // GROUP); crc_xor ^= payload_crc
                del encoded
                gc.collect(); torch.cuda.empty_cache()
        os.replace(partial, artifact)
        if artifact.stat().st_size != LAYER_BYTES or records != EXPERTS * 3:
            raise RuntimeError(f"layer physical size/count failed {layer}")
        payload = {
            "kind": "streamq5_moe_p1b_layer", "layer": layer, "completed_utc": datetime.now(timezone.utc).isoformat(),
            "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"), "artifact_sha256": sha256(artifact), "artifact_bytes": artifact.stat().st_size,
            "records": records, "codes": codes_total, "scale_elements": scales_total, "crc32_xor": crc_xor,
            "source_weight_sha256": source_hashes, "seconds": time.perf_counter() - layer_started,
        }
        layer_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        del values; gc.collect(); torch.cuda.empty_cache()
        peak_rss = max(peak_rss, process.memory_info().rss)
        print(json.dumps({"layer": layer, "status": "complete", "seconds": payload["seconds"], "bytes": payload["artifact_bytes"]}), flush=True)

    if selected != list(range(LAYERS)):
        raise SystemExit(0)
    manifests = {}
    for layer in range(LAYERS):
        artifact = RUN_DIR / f"layer_{layer:02d}.q5bin"; layer_report = LAYER_DIR / f"layer_{layer:02d}.json"
        if not artifact.is_file() or artifact.stat().st_size != LAYER_BYTES:
            raise RuntimeError(f"missing complete layer {layer}")
        manifests[str(layer)] = {"artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"), "artifact_sha256": sha256(artifact), "report_sha256": sha256(layer_report), "bytes": artifact.stat().st_size}
    total_bytes = sum(row["bytes"] for row in manifests.values())
    result = {
        "kind": "streamq5_moe_p1b_physical_bank", "status": "physical_bank_built_pending_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"preregistration_sha256": sha256(PREREG), "p1a_verification_sha256": sha256(P1A_VERIFY), "model_index_sha256": sha256(MODEL / "model.safetensors.index.json")},
        "format": {"header_bytes": HEADER_BYTES, "record_alignment": 4096, "matrix_record_bytes": RECORD_BYTES, "expert_bytes": EXPERT_BYTES, "layer_bytes": LAYER_BYTES, "code_mapping": "unsigned=code+15; eight little-order 5-bit codes in five bytes", "scales": "raw_bfloat16_group128"},
        "bank": {"layers": LAYERS, "experts": LAYERS * EXPERTS, "records": LAYERS * EXPERTS * 3, "codes": 28_991_029_248, "scale_elements": 226_492_416, "bytes": total_bytes, "gib": total_bytes / 2**30},
        "manifests": manifests, "runtime": {"seconds": time.perf_counter() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_rss_bytes": peak_rss},
        "claim_boundary": "Physical Q5 full bank built; independent decoding is still required and no kernel or wall-clock claim is made.",
    }
    if total_bytes != BANK_BYTES or result["runtime"]["peak_cuda_allocated_bytes"] > int(7.5 * 2**30) or peak_rss > 32 * 2**30:
        raise RuntimeError("full-bank gate failed")
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(f"# STREAMQ5-MoE P1B - fysieke bank\n\nUitkomst: **{result['status']}**. {total_bytes:,} bytes ({total_bytes / 2**30:.6f} GiB), 18.432 records.\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "bank": result["bank"], "runtime": result["runtime"]}, indent=2), flush=True)
