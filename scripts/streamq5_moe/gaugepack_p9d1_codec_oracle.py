from __future__ import annotations

import argparse
import gc
import hashlib
import json
import mmap
import os
import struct
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH / "src"))

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_checkpoint_tensors


MODEL = ROOT / "models/qwen3-30b-a3b-base"
MASKS = ROOT / "reports/runs/streamq5_moe/p9b_structured_wanda_keep.safetensors"
P1D_BANK = ROOT / "reports/runs/streamq5_moe/p1d_q5_bank/layer_00.q5bin"
P1D_RESULT = ROOT / "reports/streamq5_moe/p1d_physical_bank_result.json"
P1D_VERIFY = ROOT / "reports/streamq5_moe/p1d_physical_bank_verification.json"
P9B_VALIDATION = ROOT / "reports/streamq5_moe/p9b_structured_wanda_validation.json"
P9B_TEST = ROOT / "reports/streamq5_moe/p9b_structured_wanda_test.json"
P9C_VALIDATION = ROOT / "reports/streamq5_moe/p9c_compact_regroup_q5_validation.json"
PREREG = ROOT / "reports/streamq5_moe/GAUGEPACK_P9D1_PREREGISTRATION.md"
ARTIFACT = ROOT / "reports/runs/streamq5_moe/gaugepack_p9d1_layer00_e000_015.gpk"
RESULT = ROOT / "reports/streamq5_moe/gaugepack_p9d1_result.json"
REPORT = ROOT / "reports/streamq5_moe/GAUGEPACK_P9D1_REPORT_2026-08-12.md"

LAYER, FIRST_EXPERT, EXPERT_COUNT = 0, 0, 16
WIDTH, KEEP, GROUP = 768, 384, 128
MATRICES = (("gate", 768, 2048, 0), ("up", 768, 2048, 1), ("down", 2048, 768, 2))

P1D_HEADER_FORMAT = "<4sHHHBBIIH2xIII28s"
P1D_HEADER_BYTES, P1D_RECORD_BYTES = 64, 1_011_712
P1D_CODE_BYTES, P1D_SCALE_BYTES = 983_040, 24_576
P1D_EXPERT_BYTES = 3_035_136

GLOBAL_HEADER_FORMAT = "<8s7HI38s"
EXPERT_HEADER_FORMAT = "<4sHH6HIII32s"
PROJECTION_HEADER_FORMAT = "<4sB3xIIIIIII28s"
GLOBAL_HEADER_BYTES = struct.calcsize(GLOBAL_HEADER_FORMAT)
EXPERT_HEADER_BYTES = struct.calcsize(EXPERT_HEADER_FORMAT)
PROJECTION_HEADER_BYTES = struct.calcsize(PROJECTION_HEADER_FORMAT)
MASK_BYTES = KEEP * 2
COMPACT_CODE_COUNT = KEEP * 2048
COMPACT_CODE_BYTES = COMPACT_CODE_COUNT * 5 // 8
GATE_UP_SCALE_COUNT = KEEP * (2048 // GROUP)
DOWN_SCALE_COUNT = 2048 * (WIDTH // GROUP)
EXPERT_STRIDE = (
    EXPERT_HEADER_BYTES
    + MASK_BYTES
    + 3 * PROJECTION_HEADER_BYTES
    + 3 * COMPACT_CODE_BYTES
    + 2 * GATE_UP_SCALE_COUNT * 2
    + DOWN_SCALE_COUNT * 2
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crc32_parts(*parts: bytes) -> int:
    result = 0
    for part in parts:
        result = zlib.crc32(part, result)
    return result & 0xFFFFFFFF


def pack_q5(codes: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(codes, dtype=np.int8).reshape(-1)
    if flat.size % 8 or int(flat.min()) < -15 or int(flat.max()) > 15:
        raise ValueError("invalid Q5 code stream")
    values = (flat.astype(np.int16) + 15).astype(np.uint64).reshape(-1, 8)
    shifts = np.arange(8, dtype=np.uint64) * 5
    words = np.bitwise_or.reduce(values << shifts, axis=1)
    byte_shifts = np.arange(5, dtype=np.uint64) * 8
    return ((words[:, None] >> byte_shifts) & 0xFF).astype(np.uint8).tobytes(order="C")


def unpack_q5(payload: bytes | memoryview, code_count: int) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 5).astype(np.uint64)
    words = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16) | (packed[:, 3] << 24) | (packed[:, 4] << 32)
    decoded = np.empty((words.size, 8), dtype=np.int8)
    for slot in range(8):
        decoded[:, slot] = ((words >> (slot * 5)) & 31).astype(np.int8) - 15
    return decoded.reshape(code_count)


def decode_p1d_record(mapped: mmap.mmap, expert: int, projection: int, rows: int, columns: int) -> tuple[np.ndarray, np.ndarray]:
    offset = expert * P1D_EXPERT_BYTES + projection * P1D_RECORD_BYTES
    fields = struct.unpack_from(P1D_HEADER_FORMAT, mapped, offset)
    magic, version, layer, got_expert, got_projection, bits, got_rows, got_columns, group, code_bytes, scale_bytes, stored_crc, reserved = fields
    if not (
        magic == b"SQ5M" and version == 1 and layer == LAYER and got_expert == expert
        and got_projection == projection and bits == 5 and got_rows == rows
        and got_columns == columns and group == GROUP and code_bytes == P1D_CODE_BYTES
        and scale_bytes == P1D_SCALE_BYTES and reserved == b"\x00" * 28
    ):
        raise ValueError(f"invalid P1D header expert={expert} projection={projection}")
    code_begin = offset + P1D_HEADER_BYTES
    scale_begin = code_begin + code_bytes
    end = scale_begin + scale_bytes
    code_payload = memoryview(mapped)[code_begin:scale_begin]
    scale_payload = memoryview(mapped)[scale_begin:end]
    if crc32_parts(code_payload, scale_payload) != stored_crc:
        raise ValueError(f"P1D CRC mismatch expert={expert} projection={projection}")
    codes = unpack_q5(code_payload, rows * columns).reshape(rows, columns).copy()
    scales = np.frombuffer(scale_payload, dtype="<u2").reshape(rows, columns // GROUP).copy()
    del code_payload, scale_payload
    return codes, scales


@torch.no_grad()
def p9b_reference(source: torch.Tensor, keep: np.ndarray, kind: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = source.shape
    work = source.to(device).float().reshape(rows, columns // GROUP, GROUP)
    selected = torch.from_numpy(np.asarray(keep, dtype=np.int64)).to(device)
    if kind in ("gate", "up"):
        row_mask = torch.zeros(rows, dtype=torch.bool)
        row_mask[selected] = True
        work = work * row_mask.view(rows, 1, 1)
    else:
        column_mask = torch.zeros(columns, dtype=torch.bool)
        column_mask[selected] = True
        work = work * column_mask.view(1, columns // GROUP, GROUP)
    maximum = work.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 15, torch.ones_like(maximum))
    codes = torch.round(work / scale).clamp(-15, 15).to(torch.int8)
    scale_bits = scale.squeeze(-1).to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy().copy()
    code_values = codes.reshape(rows, columns).cpu().contiguous().numpy().copy()
    del work, maximum, scale, codes
    return code_values, scale_bits


def bf16_decode_bits(codes: np.ndarray, scale_bits: np.ndarray) -> np.ndarray:
    rows, columns = codes.shape
    code_tensor = torch.from_numpy(np.ascontiguousarray(codes)).float().reshape(rows, columns // GROUP, GROUP)
    scale_tensor = torch.from_numpy(np.ascontiguousarray(scale_bits)).view(torch.bfloat16).float().unsqueeze(-1)
    decoded = (code_tensor * scale_tensor).reshape(rows, columns).to(torch.bfloat16)
    result = decoded.contiguous().view(torch.uint16).numpy().copy()
    del code_tensor, scale_tensor, decoded
    return result


def source_names(weight_map: dict[str, str]) -> tuple[dict[int, dict[str, str]], list[str]]:
    identities: dict[int, dict[str, str]] = {}
    names: list[str] = []
    for expert in range(FIRST_EXPERT, FIRST_EXPERT + EXPERT_COUNT):
        base = f"model.layers.{LAYER}.mlp.experts.{expert}"
        identities[expert] = {kind: f"{base}.{kind}_proj.weight" for kind, *_ in MATRICES}
        names.extend(identities[expert].values())
    missing = [name for name in names if name not in weight_map]
    if missing:
        raise KeyError(missing[:3])
    return identities, names


def projection_chunk(projection: int, rows: int, columns: int, codes: np.ndarray, scales: np.ndarray) -> bytes:
    code_payload = pack_q5(codes)
    scale_payload = np.ascontiguousarray(scales, dtype="<u2").tobytes(order="C")
    header = struct.pack(
        PROJECTION_HEADER_FORMAT,
        b"GPRJ", projection, rows, columns, int(codes.size), int(scales.size),
        len(code_payload), len(scale_payload), crc32_parts(code_payload, scale_payload), b"\x00" * 28,
    )
    return header + code_payload + scale_payload


def load_projection(body: memoryview, cursor: int, expected_projection: int) -> tuple[np.ndarray, np.ndarray, int, tuple[int, int]]:
    fields = struct.unpack_from(PROJECTION_HEADER_FORMAT, body, cursor)
    magic, projection, rows, columns, code_count, scale_count, code_bytes, scale_bytes, stored_crc, reserved = fields
    if magic != b"GPRJ" or projection != expected_projection or reserved != b"\x00" * 28:
        raise ValueError("invalid GaugePack projection header")
    cursor += PROJECTION_HEADER_BYTES
    code_payload = body[cursor:cursor + code_bytes]
    cursor += code_bytes
    scale_payload = body[cursor:cursor + scale_bytes]
    cursor += scale_bytes
    if len(code_payload) != code_bytes or len(scale_payload) != scale_bytes or crc32_parts(code_payload, scale_payload) != stored_crc:
        raise ValueError("invalid GaugePack projection payload")
    codes = unpack_q5(code_payload, code_count).copy()
    scales = np.frombuffer(scale_payload, dtype="<u2", count=scale_count).copy()
    return codes, scales, cursor, (rows, columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reconstruction requested but unavailable")
    device = torch.device(args.device)
    if ARTIFACT.exists() or RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite completed GaugePack P9D-1 output")
    started = time.perf_counter()
    required = (MODEL / "model.safetensors.index.json", MASKS, P1D_BANK, P1D_RESULT, P1D_VERIFY, P9B_VALIDATION, P9B_TEST, P9C_VALIDATION, PREREG)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    p1d_verification = json.loads(P1D_VERIFY.read_text(encoding="utf-8"))
    p9b_validation = json.loads(P9B_VALIDATION.read_text(encoding="utf-8"))
    p9b_test = json.loads(P9B_TEST.read_text(encoding="utf-8"))
    p9c_validation = json.loads(P9C_VALIDATION.read_text(encoding="utf-8"))
    if p1d_verification.get("status") != "p1d_physical_bank_verification_pass":
        raise RuntimeError("independently verified P1D bank required")
    if p9b_validation.get("status") != "validation_pass_test_authorized" or not p9b_test.get("overall_pass"):
        raise RuntimeError("P9B validation and test quality pass required")
    if p9c_validation.get("status") != "validation_closed":
        raise RuntimeError("expected closed P9C validation")

    mask_tensors = load_file(MASKS)
    layer_masks = mask_tensors[f"layer_{LAYER:02d}"][FIRST_EXPERT:FIRST_EXPERT + EXPERT_COUNT].long().numpy().copy()
    mask_failures = 0
    for keep in layer_masks:
        mask_failures += int(keep.shape != (KEEP,))
        mask_failures += int(int(keep.min()) < 0 or int(keep.max()) >= WIDTH)
        mask_failures += int(np.unique(keep).size != KEEP)
        mask_failures += int(not np.all(keep[1:] > keep[:-1]))

    weight_map = checkpoint_weight_map(MODEL)
    identities, names = source_names(weight_map)
    sources = load_checkpoint_tensors(MODEL, names, weight_map)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    partial = ARTIFACT.with_suffix(".gpk.inprogress")
    if partial.exists():
        partial.unlink()

    producer_reference: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    diagnostics = {
        "gate_up_p1d_survivor_code_mismatches": 0,
        "gate_up_p1d_survivor_scale_bit_mismatches": 0,
        "down_p1d_vs_p9b_survivor_code_mismatches": 0,
        "down_p1d_vs_p9b_group_scale_bit_mismatches": 0,
        "down_groups_compared": 0,
    }
    group_counts_all: list[list[int]] = []
    global_header = struct.pack(
        GLOBAL_HEADER_FORMAT, b"GAUGEP1\x00", 1, LAYER, FIRST_EXPERT, EXPERT_COUNT,
        WIDTH, KEEP, GROUP, EXPERT_STRIDE, b"\x00" * 38,
    )
    with P1D_BANK.open("rb") as p1d_handle, partial.open("wb") as output:
        mapped_p1d = mmap.mmap(p1d_handle.fileno(), 0, access=mmap.ACCESS_READ)
        output.write(global_header)
        try:
            for local_index, expert in enumerate(range(FIRST_EXPERT, FIRST_EXPERT + EXPERT_COUNT)):
                keep = layer_masks[local_index].astype(np.int64, copy=False)
                group_counts = np.bincount(keep // GROUP, minlength=WIDTH // GROUP).astype(np.uint16)
                group_counts_all.append(group_counts.astype(int).tolist())
                projection_chunks: list[bytes] = []
                for kind, rows, columns, projection in MATRICES:
                    source = sources[identities[expert][kind]]
                    reference_codes, reference_scales = p9b_reference(source, keep, kind, device)
                    producer_reference[(expert, kind)] = (reference_codes, reference_scales)
                    p1d_codes, p1d_scales = decode_p1d_record(mapped_p1d, expert, projection, rows, columns)
                    if kind in ("gate", "up"):
                        compact_codes = p1d_codes[keep, :].copy()
                        compact_scales = p1d_scales[keep, :].copy()
                        diagnostics["gate_up_p1d_survivor_code_mismatches"] += int(np.count_nonzero(compact_codes != reference_codes[keep, :]))
                        diagnostics["gate_up_p1d_survivor_scale_bit_mismatches"] += int(np.count_nonzero(compact_scales != reference_scales[keep, :]))
                    else:
                        compact_codes = reference_codes[:, keep].copy()
                        compact_scales = reference_scales.copy()
                        diagnostics["down_p1d_vs_p9b_survivor_code_mismatches"] += int(np.count_nonzero(p1d_codes[:, keep] != compact_codes))
                        diagnostics["down_p1d_vs_p9b_group_scale_bit_mismatches"] += int(np.count_nonzero(p1d_scales != compact_scales))
                        diagnostics["down_groups_compared"] += int(compact_scales.size)
                    projection_chunks.append(projection_chunk(projection, rows, columns, compact_codes, compact_scales))
                    del p1d_codes, p1d_scales, compact_codes, compact_scales
                keep_bytes = np.ascontiguousarray(keep, dtype="<u2").tobytes(order="C")
                body = keep_bytes + b"".join(projection_chunks)
                expert_header = struct.pack(
                    EXPERT_HEADER_FORMAT, b"GEXP", LAYER, expert, *group_counts.tolist(), KEEP,
                    zlib.crc32(keep_bytes) & 0xFFFFFFFF, zlib.crc32(body) & 0xFFFFFFFF, b"\x00" * 32,
                )
                if len(expert_header) + len(body) != EXPERT_STRIDE:
                    raise RuntimeError("GaugePack expert stride mismatch")
                output.write(expert_header)
                output.write(body)
                print(json.dumps({"phase": "encode", "layer": LAYER, "expert": expert}), flush=True)
        finally:
            mapped_p1d.close()
    os.replace(partial, ARTIFACT)

    expected_bytes = GLOBAL_HEADER_BYTES + EXPERT_COUNT * EXPERT_STRIDE
    counters = {
        "experts": 0,
        "mask_index_mismatches": 0,
        "group_count_mismatches": 0,
        "survivor_code_mismatches": 0,
        "stored_scale_bit_mismatches": 0,
        "dense_code_mismatches": 0,
        "dense_scale_bit_mismatches": 0,
        "dense_bf16_decode_bit_mismatches": 0,
        "dense_elements_verified": 0,
        "crc_or_header_failures": 0,
    }
    with ARTIFACT.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            fields = struct.unpack_from(GLOBAL_HEADER_FORMAT, mapped, 0)
            expected_global = (b"GAUGEP1\x00", 1, LAYER, FIRST_EXPERT, EXPERT_COUNT, WIDTH, KEEP, GROUP, EXPERT_STRIDE, b"\x00" * 38)
            counters["crc_or_header_failures"] += int(fields != expected_global)
            for local_index, expert in enumerate(range(FIRST_EXPERT, FIRST_EXPERT + EXPERT_COUNT)):
                expert_begin = GLOBAL_HEADER_BYTES + local_index * EXPERT_STRIDE
                expert_end = expert_begin + EXPERT_STRIDE
                header = struct.unpack_from(EXPERT_HEADER_FORMAT, mapped, expert_begin)
                magic, got_layer, got_expert, *tail = header
                counts = np.asarray(tail[:6], dtype=np.uint16)
                got_keep_count, mask_crc, body_crc, reserved = tail[6:]
                body = memoryview(mapped)[expert_begin + EXPERT_HEADER_BYTES:expert_end]
                if magic != b"GEXP" or got_layer != LAYER or got_expert != expert or got_keep_count != KEEP or reserved != b"\x00" * 32:
                    counters["crc_or_header_failures"] += 1
                if zlib.crc32(body) & 0xFFFFFFFF != body_crc:
                    counters["crc_or_header_failures"] += 1
                keep_payload = body[:MASK_BYTES]
                if zlib.crc32(keep_payload) & 0xFFFFFFFF != mask_crc:
                    counters["crc_or_header_failures"] += 1
                decoded_keep = np.frombuffer(keep_payload, dtype="<u2").astype(np.int64).copy()
                original_keep = layer_masks[local_index].astype(np.int64, copy=False)
                counters["mask_index_mismatches"] += int(np.count_nonzero(decoded_keep != original_keep))
                expected_counts = np.bincount(decoded_keep // GROUP, minlength=WIDTH // GROUP).astype(np.uint16)
                counters["group_count_mismatches"] += int(np.count_nonzero(counts != expected_counts))
                cursor = MASK_BYTES
                for kind, rows, columns, projection in MATRICES:
                    compact_codes, compact_scales, cursor, shape = load_projection(body, cursor, projection)
                    if shape != (rows, columns):
                        counters["crc_or_header_failures"] += 1
                    reference_codes, reference_scales = producer_reference[(expert, kind)]
                    dense_codes = np.zeros((rows, columns), dtype=np.int8)
                    if kind in ("gate", "up"):
                        compact_codes = compact_codes.reshape(KEEP, columns)
                        compact_scales = compact_scales.reshape(KEEP, columns // GROUP)
                        dense_codes[decoded_keep, :] = compact_codes
                        one_bits = torch.ones((), dtype=torch.bfloat16).view(torch.uint16).item()
                        dense_scales = np.full((rows, columns // GROUP), one_bits, dtype=np.uint16)
                        dense_scales[decoded_keep, :] = compact_scales
                        counters["survivor_code_mismatches"] += int(np.count_nonzero(compact_codes != reference_codes[original_keep, :]))
                        counters["stored_scale_bit_mismatches"] += int(np.count_nonzero(compact_scales != reference_scales[original_keep, :]))
                    else:
                        compact_codes = compact_codes.reshape(rows, KEEP)
                        compact_scales = compact_scales.reshape(rows, columns // GROUP)
                        dense_codes[:, decoded_keep] = compact_codes
                        dense_scales = compact_scales
                        counters["survivor_code_mismatches"] += int(np.count_nonzero(compact_codes != reference_codes[:, original_keep]))
                        counters["stored_scale_bit_mismatches"] += int(np.count_nonzero(compact_scales != reference_scales))
                    counters["dense_code_mismatches"] += int(np.count_nonzero(dense_codes != reference_codes))
                    counters["dense_scale_bit_mismatches"] += int(np.count_nonzero(dense_scales != reference_scales))
                    decoded_bits = bf16_decode_bits(dense_codes, dense_scales)
                    reference_bits = bf16_decode_bits(reference_codes, reference_scales)
                    counters["dense_bf16_decode_bit_mismatches"] += int(np.count_nonzero(decoded_bits != reference_bits))
                    counters["dense_elements_verified"] += int(decoded_bits.size)
                    del compact_codes, compact_scales, dense_codes, dense_scales, decoded_bits, reference_bits
                if cursor != len(body):
                    counters["crc_or_header_failures"] += 1
                counters["experts"] += 1
                del body, keep_payload
                print(json.dumps({"phase": "oracle", "layer": LAYER, "expert": expert}), flush=True)
        finally:
            mapped.close()

    full_reference_bytes = EXPERT_COUNT * P1D_EXPERT_BYTES
    byte_ratio = ARTIFACT.stat().st_size / full_reference_bytes
    gates = {
        "mask_cardinality_unique_range": mask_failures == 0,
        "mask_and_group_identity_exact": counters["mask_index_mismatches"] == 0 and counters["group_count_mismatches"] == 0,
        "zero_survivor_code_mismatches": counters["survivor_code_mismatches"] == 0,
        "zero_stored_scale_bit_mismatches": counters["stored_scale_bit_mismatches"] == 0,
        "zero_dense_p9b_code_or_scale_mismatches": counters["dense_code_mismatches"] == 0 and counters["dense_scale_bit_mismatches"] == 0,
        "zero_dense_bf16_decode_bit_mismatches": counters["dense_bf16_decode_bit_mismatches"] == 0,
        "gate_up_literal_p1d_bytes_preserved": diagnostics["gate_up_p1d_survivor_code_mismatches"] == 0 and diagnostics["gate_up_p1d_survivor_scale_bit_mismatches"] == 0,
        "headers_crcs_and_size_valid": counters["crc_or_header_failures"] == 0 and ARTIFACT.stat().st_size == expected_bytes,
        "sixteen_experts_complete": counters["experts"] == EXPERT_COUNT,
        "physical_byte_ratio_le_0_51": byte_ratio <= 0.51,
        "cuda_reference_reconstruction": args.device == "cuda",
    }
    passed = all(gates.values())
    result = {
        "kind": "gaugepack_p9d1_codec_oracle",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "gaugepack_p9d1_pass" if passed else "gaugepack_p9d1_fail",
        "scope": {"layer": LAYER, "experts": [FIRST_EXPERT, FIRST_EXPERT + EXPERT_COUNT - 1], "expert_count": EXPERT_COUNT},
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "producer_oracle_sha256": sha256(Path(__file__)),
            "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
            "p9b_masks_sha256": sha256(MASKS),
            "p1d_layer00_sha256": sha256(P1D_BANK),
            "p1d_result_sha256": sha256(P1D_RESULT),
            "p1d_verification_sha256": sha256(P1D_VERIFY),
            "p9b_validation_sha256": sha256(P9B_VALIDATION),
            "p9b_test_sha256": sha256(P9B_TEST),
            "p9c_validation_sha256": sha256(P9C_VALIDATION),
        },
        "audit": {
            "available_mask": "48 layers x 128 experts x 384 sorted int16 survivor indices",
            "available_quantized_bank": "P1D full unpruned group-128 Q5 bank, 48 x 128 experts x 3 matrices",
            "historical_p9b_quantized_bank_persisted": False,
            "gate_up_reuse": "survivor codes and scale bits copied literally from P1D after independent P9B recomputation matched",
            "down_reuse": "P9B dense-zero Q5 codes/scales reconstructed from source BF16 plus frozen mask; original P1D down is only a diagnostic comparator",
        },
        "format": {
            "global_header_bytes": GLOBAL_HEADER_BYTES,
            "expert_header_bytes": EXPERT_HEADER_BYTES,
            "projection_header_bytes": PROJECTION_HEADER_BYTES,
            "mask_bytes_per_expert": MASK_BYTES,
            "expert_stride_bytes": EXPERT_STRIDE,
            "code_mapping": "unsigned=code+15; eight little-order 5-bit codes in five bytes",
            "scales": "raw little-endian bfloat16 bits",
            "down_group_topology": "six original groups of 128 per output row retained",
        },
        "artifact": {
            "path": str(ARTIFACT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(ARTIFACT),
            "bytes": ARTIFACT.stat().st_size,
            "full_p1d_reference_bytes": full_reference_bytes,
            "physical_byte_ratio": byte_ratio,
            "projected_full_bank_gib_at_same_ratio": 17.3671875 * byte_ratio,
        },
        "group_survivor_counts": {
            "per_expert": group_counts_all,
            "minimum": int(np.min(group_counts_all)),
            "maximum": int(np.max(group_counts_all)),
            "mean": float(np.mean(group_counts_all)),
        },
        "diagnostics": diagnostics,
        "oracle_counters": counters,
        "gates": gates,
        "runtime": {"seconds": time.perf_counter() - started, "device": args.device},
        "claim_boundary": "Exact codec/oracle pass for layer 0 experts 0-15 only; no kernel timing, full-bank, full-depth quality, runtime throughput, 80B, or novelty claim.",
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# GaugePack P9D-1 — codec/oracle\n\n"
        f"Uitkomst: **{result['status']}**. De codec bewaart P9B voor laag 0, experts 0–15 met "
        f"{counters['dense_bf16_decode_bit_mismatches']:,} BF16-decodemismatches over "
        f"{counters['dense_elements_verified']:,} gereconstrueerde matrixelementen.\n\n"
        f"Het bestand is {ARTIFACT.stat().st_size:,} bytes; ratio tegenover dezelfde volledige P1D-records: "
        f"**{byte_ratio:.6f}**. Lineaire full-bankprojectie: {result['artifact']['projected_full_bank_gib_at_same_ratio']:.3f} GiB.\n\n"
        "Gate/up-survivorcodes en raw BF16-schalen zijn letterlijk uit P1D overgenomen en onafhankelijk "
        "tegen de P9B-reconstructie gecontroleerd. Down is uit bron-BF16 plus het frozen P9B-masker "
        "gereconstrueerd, omdat P9B's nulmasker groepsmaxima kan wijzigen en er historisch geen P9B-codebank is opgeslagen.\n\n"
        f"Bij down verschilden {diagnostics['down_p1d_vs_p9b_group_scale_bit_mismatches']:,} van "
        f"{diagnostics['down_groups_compared']:,} groepsschalen van de ongesnoeide P1D-bank; dit bevestigt dat "
        "blind P1D-downbytes kopiëren semantisch fout zou zijn.\n\n"
        "Claimgrens: één laag en zestien experts; nog geen kernel-, full-bank-, kwaliteits- of throughputbewijs.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "artifact": result["artifact"], "diagnostics": diagnostics, "counters": counters, "gates": gates, "runtime": result["runtime"]}, indent=2), flush=True)
    del sources, producer_reference
    gc.collect()


if __name__ == "__main__":
    main()
