from __future__ import annotations

import binascii
import hashlib
import json
import math
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from moe_lab.reporting import ROOT


PREREG = ROOT / "reports/coretail_moe/P0A_LOCKED16_FORMAT_PREREGISTRATION.md"
LOCK = ROOT / "reports/fleq_moe/p1_smoke_expert_lock.json"
SOURCE_DIR = ROOT / "reports/runs/fleq_moe/p1"
OUT_DIR = ROOT / "reports/runs/coretail_moe"
CORE_FILE = OUT_DIR / "locked16.core.bin"
TAIL_FILE = OUT_DIR / "locked16.tail.bin"
RESULT = ROOT / "reports/coretail_moe/p0a_locked16_format_result.json"
REPORT = ROOT / "reports/coretail_moe/P0A_LOCKED16_FORMAT_REPORT.md"
ALIGNMENT = 4096
TAIL_BLOCK_ROWS = 64
CORE_HEADER = struct.Struct("<8sHHB3xIIHHQQQQI")
TAIL_HEADER = struct.Struct("<8sHHB3xIIHHQIIQQI")
TAIL_INDEX = struct.Struct("<QIIIIB7x")
MATRIX_ID = {"gate": 0, "up": 1, "down": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pad(handle) -> int:
    padding = (-handle.tell()) % ALIGNMENT
    handle.write(b"\0" * padding)
    return padding


def bits(values: np.ndarray) -> bytes:
    return np.packbits(values.astype(np.uint8), bitorder="little").tobytes()


def codes_and_scales(tensors, name: str):
    weight = tensors[f"gptq_{name}_weight"].cpu()
    scales = tensors[f"gptq_{name}_scales"].cpu().contiguous()
    groups = torch.arange(weight.shape[1]) // 128
    codes = torch.round(weight.float() / scales.float()[:, groups]).to(torch.int8)
    if not bool(((codes >= -2) & (codes <= 1)).all()):
        raise ValueError("GPTQ code outside locked alphabet")
    return codes.numpy(), scales


def encode_core(handle, layer, expert, name, codes, scales):
    start = handle.tell()
    rows, cols = codes.shape
    scale_bytes = scales.view(torch.uint16).numpy().tobytes()
    bitmap_parts, sign_parts, offsets = [], [], [0]
    for row in codes:
        nonzero = row != 0
        bitmap_parts.append(bits(nonzero))
        packed_sign = bits(row[nonzero] > 0)
        sign_parts.append(packed_sign)
        offsets.append(offsets[-1] + len(packed_sign))
    bitmap = b"".join(bitmap_parts)
    sign = b"".join(sign_parts)
    offset_bytes = np.asarray(offsets, dtype="<u4").tobytes()
    payload = scale_bytes + bitmap + offset_bytes + sign
    crc = binascii.crc32(payload) & 0xFFFFFFFF
    header = CORE_HEADER.pack(b"CTCORE01", layer, expert, MATRIX_ID[name], rows, cols, 128, 0, len(scale_bytes), len(bitmap), len(offset_bytes), len(sign), crc)
    handle.write(header); handle.write(payload)
    padding = pad(handle)
    return {"start": start, "rows": rows, "cols": cols, "header_bytes": len(header), "scale_bytes": len(scale_bytes), "bitmap_bytes": len(bitmap), "row_offset_bytes": len(offset_bytes), "sign_bytes": len(sign), "padding_bytes": padding, "record_bytes": handle.tell() - start, "crc32": crc}


def encode_tail(handle, layer, expert, name, codes):
    start = handle.tell()
    rows, cols = codes.shape
    entries, payloads, offset = [], [], 0
    total_bits = 0
    for row_start in range(0, rows, TAIL_BLOCK_ROWS):
        block = codes[row_start : row_start + TAIL_BLOCK_ROWS]
        negative = block < 0
        flags = block[negative] == -2
        raw = bits(flags)
        compressed = zlib.compress(raw, level=9)
        if len(compressed) < len(raw):
            stored, codec = compressed, 1
        else:
            stored, codec = raw, 0
        crc = binascii.crc32(raw) & 0xFFFFFFFF
        entries.append(TAIL_INDEX.pack(offset, len(stored), len(raw), int(flags.size), crc, codec))
        payloads.append(stored)
        offset += len(stored)
        total_bits += int(flags.size)
    index = b"".join(entries)
    payload = b"".join(payloads)
    raw_bytes = sum(TAIL_INDEX.unpack(entry)[2] for entry in entries)
    record_payload = index + payload
    crc = binascii.crc32(record_payload) & 0xFFFFFFFF
    header = TAIL_HEADER.pack(b"CTTAIL01", layer, expert, MATRIX_ID[name], rows, cols, TAIL_BLOCK_ROWS, 0, total_bits, len(entries), len(index), len(payload), raw_bytes, crc)
    handle.write(header); handle.write(record_payload)
    padding = pad(handle)
    return {"start": start, "rows": rows, "cols": cols, "header_bytes": len(header), "negative_bits": total_bits, "blocks": len(entries), "index_bytes": len(index), "payload_bytes": len(payload), "raw_flag_bytes": raw_bytes, "padding_bytes": padding, "record_bytes": handle.tell() - start, "crc32": crc}


def decode_core(handle, offset):
    handle.seek(offset)
    fields = CORE_HEADER.unpack(handle.read(CORE_HEADER.size))
    magic, layer, expert, matrix_id, rows, cols, group, _reserved, scale_n, bitmap_n, offsets_n, sign_n, crc = fields
    if magic != b"CTCORE01" or group != 128:
        raise ValueError("invalid core header")
    payload = handle.read(scale_n + bitmap_n + offsets_n + sign_n)
    if binascii.crc32(payload) & 0xFFFFFFFF != crc:
        raise ValueError("core CRC mismatch")
    cursor = 0
    scale_raw = payload[cursor : cursor + scale_n]; cursor += scale_n
    bitmap = payload[cursor : cursor + bitmap_n]; cursor += bitmap_n
    offsets = np.frombuffer(payload[cursor : cursor + offsets_n], dtype="<u4"); cursor += offsets_n
    sign = payload[cursor : cursor + sign_n]
    codes = np.zeros((rows, cols), dtype=np.int8)
    bitmap_row_bytes = math.ceil(cols / 8)
    for row in range(rows):
        nz = np.unpackbits(np.frombuffer(bitmap[row * bitmap_row_bytes : (row + 1) * bitmap_row_bytes], dtype=np.uint8), bitorder="little")[:cols].astype(bool)
        count = int(nz.sum())
        row_sign = np.unpackbits(np.frombuffer(sign[offsets[row] : offsets[row + 1]], dtype=np.uint8), bitorder="little")[:count]
        codes[row, nz] = np.where(row_sign, 1, -1)
    return codes, scale_raw, {"layer": layer, "expert": expert, "matrix_id": matrix_id}


def apply_tail(handle, offset, core_codes):
    handle.seek(offset)
    fields = TAIL_HEADER.unpack(handle.read(TAIL_HEADER.size))
    magic, layer, expert, matrix_id, rows, cols, block_rows, _reserved, negative_bits, blocks, index_n, payload_n, raw_n, crc = fields
    record = handle.read(index_n + payload_n)
    if magic != b"CTTAIL01" or binascii.crc32(record) & 0xFFFFFFFF != crc:
        raise ValueError("invalid tail record")
    index, payload = record[:index_n], record[index_n:]
    decoded = core_codes.copy()
    observed_bits = observed_raw = 0
    for block_index in range(blocks):
        entry = TAIL_INDEX.unpack(index[block_index * TAIL_INDEX.size : (block_index + 1) * TAIL_INDEX.size])
        position, stored_n, raw_bytes, bit_count, block_crc, codec = entry
        stored = payload[position : position + stored_n]
        raw = zlib.decompress(stored) if codec == 1 else stored
        if len(raw) != raw_bytes or binascii.crc32(raw) & 0xFFFFFFFF != block_crc:
            raise ValueError("tail block mismatch")
        flags = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:bit_count].astype(bool)
        row_start = block_index * block_rows
        block = decoded[row_start : min(rows, row_start + block_rows)]
        negative = block < 0
        block[negative] -= flags.astype(np.int8)
        observed_bits += bit_count; observed_raw += raw_bytes
    if observed_bits != negative_bits or observed_raw != raw_n:
        raise ValueError("tail totals mismatch")
    return decoded, {"layer": layer, "expert": expert, "matrix_id": matrix_id}


if __name__ == "__main__":
    if any(path.exists() for path in (CORE_FILE, TAIL_FILE, RESULT, REPORT)):
        raise FileExistsError("refusing to overwrite CORETAIL P0A")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    sources = []
    for layer_text, row in lock["layers"].items():
        layer = int(layer_text)
        for expert in row["selected_experts"]:
            path = SOURCE_DIR / f"layer_{layer:02d}_expert_{expert:03d}.safetensors"
            if not path.is_file():
                raise FileNotFoundError(path)
            sources.append((layer, expert, path))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records, originals = [], {}
    with CORE_FILE.open("wb") as core_handle, TAIL_FILE.open("wb") as tail_handle:
        for layer, expert, path in sources:
            tensors = load_file(path)
            for name in ("gate", "up", "down"):
                codes, scales = codes_and_scales(tensors, name)
                key = f"{layer}:{expert}:{name}"
                originals[key] = (codes.copy(), scales.view(torch.uint16).numpy().tobytes())
                core = encode_core(core_handle, layer, expert, name, codes, scales)
                tail = encode_tail(tail_handle, layer, expert, name, codes)
                records.append({"key": key, "source": str(path.relative_to(ROOT)).replace("\\", "/"), "source_sha256": sha256(path), "core": core, "tail": tail})
    exact_codes = exact_scales = True
    with CORE_FILE.open("rb") as core_handle, TAIL_FILE.open("rb") as tail_handle:
        for record in records:
            core_codes, scale_raw, core_identity = decode_core(core_handle, record["core"]["start"])
            decoded, tail_identity = apply_tail(tail_handle, record["tail"]["start"], core_codes)
            original_codes, original_scales = originals[record["key"]]
            exact_codes &= np.array_equal(decoded, original_codes)
            exact_scales &= scale_raw == original_scales
            if core_identity != tail_identity:
                raise ValueError("core/tail identity mismatch")
    weights = sum(value[0].size for value in originals.values())
    core_bytes, tail_bytes = CORE_FILE.stat().st_size, TAIL_FILE.stat().st_size
    core_bpp, tail_bpp = core_bytes * 8 / weights, tail_bytes * 8 / weights
    full_params = 28_991_029_248
    projected_core_gib = full_params * core_bpp / 8 / 2**30
    projected_tail_gib = full_params * tail_bpp / 8 / 2**30
    resident = projected_core_gib + 0.7176275253295898 + 0.375 + 0.75
    available = len(sources)
    payload = {
        "kind": "coretail_moe_p0a_locked16_actual_format",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "p0_blocked_missing_full_bank_gptq_codes",
        "inputs": {"preregistration_sha256": sha256(PREREG), "selection_lock_sha256": sha256(LOCK), "canonical_experts": available, "required_experts": 6144, "missing_experts": 6144 - available},
        "format": {"alignment_bytes": ALIGNMENT, "core_header_bytes": CORE_HEADER.size, "tail_header_bytes": TAIL_HEADER.size, "tail_block_rows": TAIL_BLOCK_ROWS, "tail_index_bytes_per_block": TAIL_INDEX.size, "tail_codec": "zlib9_or_raw"},
        "actual_locked16": {"weights": weights, "matrices": len(records), "core_bytes": core_bytes, "tail_bytes": tail_bytes, "core_bpp": core_bpp, "tail_bpp": tail_bpp, "total_bpp": core_bpp + tail_bpp, "code_reconstruction_bit_exact": bool(exact_codes), "scale_reconstruction_bit_exact": bool(exact_scales), "core_file": str(CORE_FILE.relative_to(ROOT)).replace("\\", "/"), "core_sha256": sha256(CORE_FILE), "tail_file": str(TAIL_FILE.relative_to(ROOT)).replace("\\", "/"), "tail_sha256": sha256(TAIL_FILE)},
        "diagnostic_full_bank_linear_projection_not_gate": {"core_gib": projected_core_gib, "tail_gib": projected_tail_gib, "resident_core_plus_int4_trunk_bf16_4k_kv_reserve_gib": resident, "core_le_5_95": projected_core_gib <= 5.95, "tail_le_0_90": projected_tail_gib <= 0.90, "resident_le_7_9599609375": resident <= 7.9599609375},
        "gates": {"all_6144_canonical_sources_present": available == 6144, "locked16_codes_exact": bool(exact_codes), "locked16_scales_exact": bool(exact_scales), "official_full_bank_p0_pass": False},
        "records": records,
        "p1_authorized": False,
        "claim_boundary": "Actual physical format and exact decode for locked16 only. Linear full-bank size is diagnostic and cannot satisfy P0 coverage.",
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text("\n".join([
        "# CORETAIL-MoE P0A — locked16 werkelijk formaat", "",
        "Uitkomst: **p0_blocked_missing_full_bank_gptq_codes**.", "",
        f"Het fysieke bestand bevat {weights:,} codes in 48 matrixrecords. Core: {core_bytes:,} bytes ({core_bpp:.6f} bpp); tail: {tail_bytes:,} bytes ({tail_bpp:.6f} bpp).",
        f"Alle codes exact: {exact_codes}; alle BF16-scalebits exact: {exact_scales}.",
        f"Lineaire full-bankdiagnostiek: core {projected_core_gib:.6f} GiB, tail {projected_tail_gib:.6f} GiB, residentformule {resident:.6f} GiB.", "",
        f"De officiële P0 faalt de broncoverage: slechts {available}/6.144 canonieke GPTQ-experts bestaan. De diagnostische extrapolatie is geen gatebewijs en P1 blijft gesloten.", "",
    ]), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "canonical_experts": available, "core_bpp": core_bpp, "tail_bpp": tail_bpp, "codes_exact": exact_codes, "scales_exact": exact_scales, "projected_core_gib": projected_core_gib, "projected_tail_gib": projected_tail_gib, "projected_resident_gib": resident}, indent=2))
