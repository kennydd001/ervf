from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from moe_lab.reporting import ROOT


PREREG = ROOT / "reports/coretail_moe/P0_FULL_BANK_EXECUTION_PREREGISTRATION.md"
SOURCE_VERIFICATION = ROOT / "reports/qwen_gptq_bank/p0_full_bank_verification.json"
SOURCE_DIR = ROOT / "reports/runs/qwen_gptq_bank/p0_bank"
OUT_DIR = ROOT / "reports/runs/coretail_moe/p0_full_bank"
CORE_FILE = OUT_DIR / "full_bank.core.bin"
TAIL_FILE = OUT_DIR / "full_bank.tail.bin"
CORE_PART = OUT_DIR / "full_bank.core.bin.partial"
TAIL_PART = OUT_DIR / "full_bank.tail.bin.partial"
CHECKPOINT = ROOT / "reports/coretail_moe/p0_full_bank_format_checkpoint.json"
RESULT = ROOT / "reports/coretail_moe/p0_full_bank_format_result.json"
REPORT = ROOT / "reports/coretail_moe/P0_FULL_BANK_FORMAT_REPORT.md"

LAYERS = 48
EXPERTS = 128
GROUP = 128
EXPECTED_WEIGHTS = 28_991_029_248
EXPECTED_SCALES = 226_492_416
ALIGNMENT = 4096
TAIL_BLOCK_ROWS = 64
CORE_HEADER = struct.Struct("<8sHHB3xIIHHQQQQI")
TAIL_HEADER = struct.Struct("<8sHHB3xIIHHQIIQQI")
TAIL_INDEX = struct.Struct("<QIIIIB7x")
MATRIX_ID = {"gate": 0, "up": 1, "down": 2}
MATRIX_NAMES = ("gate", "up", "down")

INT4_TRUNK_GIB = 0.7176275253295898
BF16_KV_4K_GIB = 0.375
RUNTIME_RESERVE_GIB = 0.75
REPORTED_VRAM_GIB = 7.9599609375


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def pack_bits(values: np.ndarray) -> bytes:
    return np.packbits(values.astype(np.uint8, copy=False), bitorder="little").tobytes()


def unpack_codes(packed: np.ndarray) -> np.ndarray:
    values = np.stack(tuple((packed >> shift) & 3 for shift in (0, 2, 4, 6)), axis=-1)
    return (values.reshape(*packed.shape[:-1], -1).astype(np.int8) - 2)


def core_record(layer: int, expert: int, name: str, codes: np.ndarray, scale_bytes: bytes):
    rows, cols = codes.shape
    bitmap = np.packbits(codes != 0, axis=1, bitorder="little").tobytes()
    sign_parts: list[bytes] = []
    offsets = np.empty(rows + 1, dtype="<u4")
    offsets[0] = 0
    cursor = 0
    for row_id, row in enumerate(codes):
        packed = pack_bits(row[row != 0] > 0)
        sign_parts.append(packed)
        cursor += len(packed)
        offsets[row_id + 1] = cursor
    sign = b"".join(sign_parts)
    offset_bytes = offsets.tobytes()
    payload = scale_bytes + bitmap + offset_bytes + sign
    crc = binascii.crc32(payload) & 0xFFFFFFFF
    header = CORE_HEADER.pack(
        b"CTCORE01", layer, expert, MATRIX_ID[name], rows, cols, GROUP, 0,
        len(scale_bytes), len(bitmap), len(offset_bytes), len(sign), crc,
    )
    unpadded = header + payload
    padding = (-len(unpadded)) % ALIGNMENT
    record = unpadded + b"\0" * padding
    metadata = {
        "rows": rows, "cols": cols, "header_bytes": len(header),
        "scale_bytes": len(scale_bytes), "bitmap_bytes": len(bitmap),
        "row_offset_bytes": len(offset_bytes), "sign_bytes": len(sign),
        "padding_bytes": padding, "record_bytes": len(record), "crc32": crc,
    }
    return record, metadata


def tail_record(layer: int, expert: int, name: str, codes: np.ndarray):
    rows, cols = codes.shape
    entries: list[bytes] = []
    payloads: list[bytes] = []
    offset = total_bits = raw_bytes = fallback_bytes = 0
    compressed_blocks = raw_blocks = 0
    for row_start in range(0, rows, TAIL_BLOCK_ROWS):
        block = codes[row_start : row_start + TAIL_BLOCK_ROWS]
        flags = block[block < 0] == -2
        raw = pack_bits(flags)
        compressed = zlib.compress(raw, level=9)
        if len(compressed) < len(raw):
            stored, codec = compressed, 1
            compressed_blocks += 1
        else:
            stored, codec = raw, 0
            raw_blocks += 1
            fallback_bytes += len(raw)
        block_crc = binascii.crc32(raw) & 0xFFFFFFFF
        entries.append(
            TAIL_INDEX.pack(offset, len(stored), len(raw), int(flags.size), block_crc, codec)
        )
        payloads.append(stored)
        offset += len(stored)
        total_bits += int(flags.size)
        raw_bytes += len(raw)
    index = b"".join(entries)
    payload = b"".join(payloads)
    record_payload = index + payload
    crc = binascii.crc32(record_payload) & 0xFFFFFFFF
    header = TAIL_HEADER.pack(
        b"CTTAIL01", layer, expert, MATRIX_ID[name], rows, cols,
        TAIL_BLOCK_ROWS, 0, total_bits, len(entries), len(index),
        len(payload), raw_bytes, crc,
    )
    unpadded = header + record_payload
    padding = (-len(unpadded)) % ALIGNMENT
    record = unpadded + b"\0" * padding
    metadata = {
        "rows": rows, "cols": cols, "header_bytes": len(header),
        "negative_bits": total_bits, "blocks": len(entries),
        "compressed_blocks": compressed_blocks, "raw_fallback_blocks": raw_blocks,
        "raw_fallback_bytes": fallback_bytes, "index_bytes": len(index),
        "payload_bytes": len(payload), "raw_flag_bytes": raw_bytes,
        "padding_bytes": padding, "record_bytes": len(record), "crc32": crc,
    }
    return record, metadata


def prepare_expert(layer: int, expert: int, arrays: dict[str, tuple[np.ndarray, np.ndarray]]):
    prepared = []
    weights = scales = 0
    for name in MATRIX_NAMES:
        packed, scale_bits = arrays[name]
        codes = unpack_codes(packed[expert])
        if not np.logical_and(codes >= -2, codes <= 1).all():
            raise ValueError(f"code outside alphabet at {layer}:{expert}:{name}")
        scale_bytes = scale_bits[expert].astype("<u2", copy=False).tobytes()
        core_bytes, core_meta = core_record(layer, expert, name, codes, scale_bytes)
        tail_bytes, tail_meta = tail_record(layer, expert, name, codes)
        prepared.append((name, core_bytes, core_meta, tail_bytes, tail_meta))
        weights += codes.size
        scales += scale_bits[expert].size
    return expert, prepared, weights, scales


def load_layer(layer: int):
    path = SOURCE_DIR / f"layer_{layer:02d}.safetensors"
    arrays = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in MATRIX_NAMES:
            packed = handle.get_tensor(f"{name}_codes_packed").contiguous().numpy()
            scale_bits = handle.get_tensor(f"{name}_scales").contiguous().view(torch.uint16).numpy()
            arrays[name] = (packed, scale_bits)
    return path, arrays


def new_checkpoint(source_hash: str) -> dict:
    return {
        "kind": "coretail_moe_p0_full_bank_checkpoint",
        "source_verification_sha256": source_hash,
        "preregistration_sha256": sha256(PREREG),
        "completed_layers": 0,
        "core_bytes": 0,
        "tail_bytes": 0,
        "weights": 0,
        "scale_elements": 0,
        "records": [],
    }


def validate_resume(checkpoint: dict, source_hash: str) -> None:
    if checkpoint["source_verification_sha256"] != source_hash:
        raise ValueError("source verification changed since checkpoint")
    if checkpoint["preregistration_sha256"] != sha256(PREREG):
        raise ValueError("preregistration changed since checkpoint")
    if CORE_PART.stat().st_size != checkpoint["core_bytes"]:
        raise ValueError("core partial size does not match checkpoint")
    if TAIL_PART.stat().st_size != checkpoint["tail_bytes"]:
        raise ValueError("tail partial size does not match checkpoint")
    if len(checkpoint["records"]) != checkpoint["completed_layers"] * EXPERTS * 3:
        raise ValueError("checkpoint record count mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if any(path.exists() for path in (CORE_FILE, TAIL_FILE, RESULT, REPORT)):
        raise FileExistsError("refusing to overwrite completed CORETAIL full-bank outputs")
    source = json.loads(SOURCE_VERIFICATION.read_text(encoding="utf-8"))
    required = (
        source.get("status") == "full_bank_pass"
        and source.get("bank", {}).get("experts") == LAYERS * EXPERTS
        and source.get("bank", {}).get("matrices") == LAYERS * EXPERTS * 3
        and source.get("checks", {}).get("all_packed_roundtrips_exact") is True
        and source.get("checks", {}).get("all_source_weight_hashes") is True
        and source.get("checks", {}).get("all_calibration_hashes") is True
        and source.get("checks", {}).get("all_artifact_hashes") is True
    )
    if not required:
        raise ValueError("independently verified pure-GPTQ full bank is required")
    source_hash = sha256(SOURCE_VERIFICATION)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT.exists():
        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        validate_resume(checkpoint, source_hash)
    else:
        if CORE_PART.exists() or TAIL_PART.exists():
            raise FileExistsError("partial files exist without a checkpoint")
        checkpoint = new_checkpoint(source_hash)
        CORE_PART.touch()
        TAIL_PART.touch()
        atomic_json(CHECKPOINT, checkpoint)

    start_layer = checkpoint["completed_layers"]
    with CORE_PART.open("ab") as core_handle, TAIL_PART.open("ab") as tail_handle:
        for layer in range(start_layer, LAYERS):
            source_path, arrays = load_layer(layer)
            source_file_hash = sha256(source_path)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                prepared_experts = pool.map(
                    lambda expert: prepare_expert(layer, expert, arrays), range(EXPERTS)
                )
                for expert, prepared, weights, scales in prepared_experts:
                    for name, core_bytes, core_meta, tail_bytes, tail_meta in prepared:
                        core_meta["start"] = core_handle.tell()
                        tail_meta["start"] = tail_handle.tell()
                        core_handle.write(core_bytes)
                        tail_handle.write(tail_bytes)
                        checkpoint["records"].append({
                            "key": f"{layer}:{expert}:{name}",
                            "source": str(source_path.relative_to(ROOT)).replace("\\", "/"),
                            "source_sha256": source_file_hash,
                            "core": core_meta,
                            "tail": tail_meta,
                        })
                    checkpoint["weights"] += weights
                    checkpoint["scale_elements"] += scales
            core_handle.flush(); os.fsync(core_handle.fileno())
            tail_handle.flush(); os.fsync(tail_handle.fileno())
            checkpoint["completed_layers"] = layer + 1
            checkpoint["core_bytes"] = core_handle.tell()
            checkpoint["tail_bytes"] = tail_handle.tell()
            checkpoint["updated_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(CHECKPOINT, checkpoint)
            print(json.dumps({
                "layer": layer, "status": "complete",
                "core_bytes": checkpoint["core_bytes"],
                "tail_bytes": checkpoint["tail_bytes"],
                "records": len(checkpoint["records"]),
            }), flush=True)

    if checkpoint["weights"] != EXPECTED_WEIGHTS:
        raise ValueError(f"weight total mismatch: {checkpoint['weights']}")
    if checkpoint["scale_elements"] != EXPECTED_SCALES:
        raise ValueError(f"scale total mismatch: {checkpoint['scale_elements']}")
    os.replace(CORE_PART, CORE_FILE)
    os.replace(TAIL_PART, TAIL_FILE)
    core_bytes = CORE_FILE.stat().st_size
    tail_bytes = TAIL_FILE.stat().st_size
    core_gib = core_bytes / 2**30
    tail_gib = tail_bytes / 2**30
    resident_gib = core_gib + INT4_TRUNK_GIB + BF16_KV_4K_GIB + RUNTIME_RESERVE_GIB
    format_gates = {
        "actual_core_le_5_95_gib": core_gib <= 5.95,
        "actual_tail_le_0_90_gib": tail_gib <= 0.90,
        "resident_formula_le_reported_vram": resident_gib <= REPORTED_VRAM_GIB,
        "all_6144_sources_present": checkpoint["completed_layers"] * EXPERTS == 6144,
        "all_18432_records_present": len(checkpoint["records"]) == 18_432,
        "weight_count_exact": checkpoint["weights"] == EXPECTED_WEIGHTS,
        "scale_count_exact": checkpoint["scale_elements"] == EXPECTED_SCALES,
    }
    raw_fallback_bytes = sum(r["tail"]["raw_fallback_bytes"] for r in checkpoint["records"])
    payload = {
        "kind": "coretail_moe_p0_full_bank_actual_format",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "physical_format_built_pending_independent_verification",
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "source_verification": str(SOURCE_VERIFICATION.relative_to(ROOT)).replace("\\", "/"),
            "source_verification_sha256": source_hash,
        },
        "format": {
            "alignment_bytes": ALIGNMENT,
            "core_header_bytes": CORE_HEADER.size,
            "tail_header_bytes": TAIL_HEADER.size,
            "tail_block_rows": TAIL_BLOCK_ROWS,
            "tail_index_bytes_per_block": TAIL_INDEX.size,
            "tail_codec": "zlib9_or_raw",
        },
        "actual_full_bank": {
            "experts": 6144, "matrices": len(checkpoint["records"]),
            "weights": checkpoint["weights"], "scale_elements": checkpoint["scale_elements"],
            "core_file": str(CORE_FILE.relative_to(ROOT)).replace("\\", "/"),
            "core_bytes": core_bytes, "core_gib": core_gib,
            "core_bpp": core_bytes * 8 / checkpoint["weights"],
            "core_sha256": sha256(CORE_FILE),
            "tail_file": str(TAIL_FILE.relative_to(ROOT)).replace("\\", "/"),
            "tail_bytes": tail_bytes, "tail_gib": tail_gib,
            "tail_bpp": tail_bytes * 8 / checkpoint["weights"],
            "tail_sha256": sha256(TAIL_FILE),
            "total_bpp": (core_bytes + tail_bytes) * 8 / checkpoint["weights"],
            "raw_fallback_bytes_counted": raw_fallback_bytes,
        },
        "memory_gate": {
            "int4_trunk_gib": INT4_TRUNK_GIB,
            "bf16_kv_4k_gib": BF16_KV_4K_GIB,
            "runtime_reserve_gib": RUNTIME_RESERVE_GIB,
            "reported_vram_gib": REPORTED_VRAM_GIB,
            "resident_formula_gib": resident_gib,
        },
        "format_gates": format_gates,
        "records": checkpoint["records"],
        "p0_pass": False,
        "p1_authorized": False,
        "claim_boundary": "Physical full-bank format built. P0 and P1 remain closed until independent bit-exact verification passes.",
    }
    atomic_json(RESULT, payload)
    REPORT.write_text("\n".join([
        "# CORETAIL-MoE P0 — fysiek full-bankformaat", "",
        "Uitkomst: **physical_format_built_pending_independent_verification**.", "",
        f"Werkelijk gebouwd: 6.144 experts, {len(checkpoint['records']):,} matrixrecords en {checkpoint['weights']:,} codes.",
        f"Core: {core_bytes:,} bytes ({core_gib:.6f} GiB; {payload['actual_full_bank']['core_bpp']:.6f} bpp).",
        f"Tail: {tail_bytes:,} bytes ({tail_gib:.6f} GiB; {payload['actual_full_bank']['tail_bpp']:.6f} bpp).",
        f"Residentformule: {resident_gib:.6f}/{REPORTED_VRAM_GIB:.6f} GiB.", "",
        f"Fysieke formaatgates: {sum(format_gates.values())}/{len(format_gates)}. Onafhankelijke reconstructieverificatie is nog verplicht; P1 blijft voorlopig gesloten.", "",
    ]), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "format_gates": f"{sum(format_gates.values())}/{len(format_gates)}",
        "core_gib": core_gib, "tail_gib": tail_gib, "resident_gib": resident_gib,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
