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
from safetensors import safe_open

from moe_lab.reporting import ROOT


PREREG = ROOT / "reports/coretail_moe/P0_FULL_BANK_EXECUTION_PREREGISTRATION.md"
SOURCE_VERIFICATION = ROOT / "reports/qwen_gptq_bank/p0_full_bank_verification.json"
RESULT = ROOT / "reports/coretail_moe/p0_full_bank_format_result.json"
OUT_JSON = ROOT / "reports/coretail_moe/p0_full_bank_format_verification.json"
OUT_MD = ROOT / "reports/coretail_moe/P0_FULL_BANK_FORMAT_VERIFICATION.md"

LAYERS = 48
EXPERTS = 128
EXPECTED_WEIGHTS = 28_991_029_248
EXPECTED_SCALES = 226_492_416
ALIGNMENT = 4096
CORE_HEADER = struct.Struct("<8sHHB3xIIHHQQQQI")
TAIL_HEADER = struct.Struct("<8sHHB3xIIHHQIIQQI")
TAIL_INDEX = struct.Struct("<QIIIIB7x")
MATRIX_NAME = {0: "gate", 1: "up", 2: "down"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_source(packed: np.ndarray) -> np.ndarray:
    values = np.stack(tuple((packed >> shift) & 3 for shift in (0, 2, 4, 6)), axis=-1)
    return values.reshape(*packed.shape[:-1], -1).astype(np.int8) - 2


def read_core(handle, offset: int):
    handle.seek(offset)
    raw_header = handle.read(CORE_HEADER.size)
    if len(raw_header) != CORE_HEADER.size:
        raise ValueError("truncated core header")
    fields = CORE_HEADER.unpack(raw_header)
    magic, layer, expert, matrix_id, rows, cols, group, reserved, scale_n, bitmap_n, offsets_n, sign_n, crc = fields
    payload = handle.read(scale_n + bitmap_n + offsets_n + sign_n)
    cursor = 0
    scale_raw = payload[cursor : cursor + scale_n]; cursor += scale_n
    bitmap_raw = payload[cursor : cursor + bitmap_n]; cursor += bitmap_n
    offsets_raw = payload[cursor : cursor + offsets_n]; cursor += offsets_n
    signs_raw = payload[cursor : cursor + sign_n]
    row_bitmap_bytes = math.ceil(cols / 8)
    bitmap = np.unpackbits(
        np.frombuffer(bitmap_raw, dtype=np.uint8).reshape(rows, row_bitmap_bytes),
        axis=1, bitorder="little",
    )[:, :cols].astype(bool)
    offsets = np.frombuffer(offsets_raw, dtype="<u4")
    decoded = np.zeros((rows, cols), dtype=np.int8)
    offsets_valid = len(offsets) == rows + 1 and int(offsets[0]) == 0 and int(offsets[-1]) == sign_n
    if offsets_valid:
        offsets_valid = bool(np.all(offsets[1:] >= offsets[:-1]))
    for row in range(rows):
        count = int(bitmap[row].sum())
        if not offsets_valid or int(offsets[row + 1] - offsets[row]) != math.ceil(count / 8):
            offsets_valid = False
            continue
        packed = np.frombuffer(signs_raw[int(offsets[row]) : int(offsets[row + 1])], dtype=np.uint8)
        signs = np.unpackbits(packed, bitorder="little")[:count]
        decoded[row, bitmap[row]] = np.where(signs, 1, -1)
    return decoded, scale_raw, {
        "magic": magic, "layer": layer, "expert": expert, "matrix_id": matrix_id,
        "rows": rows, "cols": cols, "group": group, "reserved": reserved,
        "scale_bytes": scale_n, "bitmap_bytes": bitmap_n,
        "offset_bytes": offsets_n, "sign_bytes": sign_n,
        "crc32": crc,
        "crc_ok": len(payload) == scale_n + bitmap_n + offsets_n + sign_n
        and (binascii.crc32(payload) & 0xFFFFFFFF) == crc,
        "offsets_valid": offsets_valid,
        "unpadded_bytes": CORE_HEADER.size + len(payload),
    }


def apply_tail(handle, offset: int, decoded: np.ndarray):
    handle.seek(offset)
    raw_header = handle.read(TAIL_HEADER.size)
    if len(raw_header) != TAIL_HEADER.size:
        raise ValueError("truncated tail header")
    fields = TAIL_HEADER.unpack(raw_header)
    magic, layer, expert, matrix_id, rows, cols, block_rows, reserved, negative_bits, blocks, index_n, payload_n, raw_n, crc = fields
    record = handle.read(index_n + payload_n)
    index, payload = record[:index_n], record[index_n:]
    output = decoded.copy()
    block_crc_ok = block_layout_ok = True
    observed_bits = observed_raw = observed_payload = raw_fallback_bytes = 0
    compressed_blocks = raw_blocks = 0
    for block_id in range(blocks):
        entry_raw = index[block_id * TAIL_INDEX.size : (block_id + 1) * TAIL_INDEX.size]
        if len(entry_raw) != TAIL_INDEX.size:
            block_layout_ok = False
            continue
        position, stored_n, raw_len, bit_count, block_crc, codec = TAIL_INDEX.unpack(entry_raw)
        stored = payload[position : position + stored_n]
        try:
            raw = zlib.decompress(stored) if codec == 1 else stored
        except zlib.error:
            raw = b""
            block_crc_ok = False
        block_crc_ok &= (
            codec in (0, 1) and len(stored) == stored_n and len(raw) == raw_len
            and (binascii.crc32(raw) & 0xFFFFFFFF) == block_crc
        )
        if codec == 0:
            raw_fallback_bytes += len(raw)
            raw_blocks += 1
        elif codec == 1:
            compressed_blocks += 1
        row_start = block_id * block_rows
        block = output[row_start : min(rows, row_start + block_rows)]
        negative = block < 0
        block_layout_ok &= int(negative.sum()) == bit_count
        flags = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:bit_count].astype(bool)
        if flags.size == int(negative.sum()):
            block[negative] -= flags.astype(np.int8)
        observed_bits += bit_count
        observed_raw += raw_len
        observed_payload = max(observed_payload, position + stored_n)
    return output, {
        "magic": magic, "layer": layer, "expert": expert, "matrix_id": matrix_id,
        "rows": rows, "cols": cols, "block_rows": block_rows, "reserved": reserved,
        "blocks": blocks, "index_bytes": index_n, "payload_bytes": payload_n,
        "negative_bits": negative_bits, "raw_flag_bytes": raw_n, "crc32": crc,
        "compressed_blocks": compressed_blocks, "raw_fallback_blocks": raw_blocks,
        "record_crc_ok": len(record) == index_n + payload_n
        and (binascii.crc32(record) & 0xFFFFFFFF) == crc,
        "block_crc_ok": bool(block_crc_ok),
        "layout_ok": bool(
            block_layout_ok and index_n == blocks * TAIL_INDEX.size
            and observed_payload == payload_n and observed_bits == negative_bits and observed_raw == raw_n
        ),
        "raw_fallback_bytes": raw_fallback_bytes,
        "unpadded_bytes": TAIL_HEADER.size + len(record),
    }


def main() -> None:
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("refusing to overwrite CORETAIL full-bank verification")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    source_verification = json.loads(SOURCE_VERIFICATION.read_text(encoding="utf-8"))
    core_path = ROOT / result["actual_full_bank"]["core_file"]
    tail_path = ROOT / result["actual_full_bank"]["tail_file"]
    checks: dict[str, bool] = {
        "preregistration_hash_exact": sha256(PREREG) == result["inputs"]["preregistration_sha256"],
        "source_verification_hash_exact": sha256(SOURCE_VERIFICATION) == result["inputs"]["source_verification_sha256"],
        "source_full_bank_pass": source_verification.get("status") == "full_bank_pass",
        "core_hash_exact": sha256(core_path) == result["actual_full_bank"]["core_sha256"],
        "tail_hash_exact": sha256(tail_path) == result["actual_full_bank"]["tail_sha256"],
        "core_size_exact": core_path.stat().st_size == result["actual_full_bank"]["core_bytes"],
        "tail_size_exact": tail_path.stat().st_size == result["actual_full_bank"]["tail_bytes"],
        "record_count_exact": len(result["records"]) == LAYERS * EXPERTS * 3,
        "all_source_hashes": True,
        "all_record_order_and_identity": True,
        "all_core_headers": True,
        "all_tail_headers": True,
        "all_core_crc_and_offsets": True,
        "all_tail_record_crc": True,
        "all_tail_block_crc": True,
        "all_tail_layout": True,
        "all_record_metadata": True,
        "all_records_aligned_nonoverlapping": True,
        "all_codes_bit_exact": True,
        "all_scale_bits_exact": True,
        "all_raw_fallback_bytes_counted": True,
    }
    total_weights = total_scales = observed_raw_fallback = 0
    prior_core_end = prior_tail_end = 0
    records_by_layer = [[] for _ in range(LAYERS)]
    for record in result["records"]:
        layer = int(record["key"].split(":", 1)[0])
        records_by_layer[layer].append(record)

    with core_path.open("rb") as core_handle, tail_path.open("rb") as tail_handle:
        for layer in range(LAYERS):
            source_path = ROOT / f"reports/runs/qwen_gptq_bank/p0_bank/layer_{layer:02d}.safetensors"
            source_hash = sha256(source_path)
            layer_records = records_by_layer[layer]
            checks["all_source_hashes"] &= len(layer_records) == EXPERTS * 3
            arrays = {}
            with safe_open(source_path, framework="pt", device="cpu") as source:
                for name in MATRIX_NAME.values():
                    arrays[name] = (
                        source.get_tensor(f"{name}_codes_packed").contiguous().numpy(),
                        source.get_tensor(f"{name}_scales").contiguous().view(torch.uint16).numpy(),
                    )
            for ordinal, record in enumerate(layer_records):
                expected_expert, expected_name = divmod(ordinal, 3)[0], ("gate", "up", "down")[ordinal % 3]
                key = f"{layer}:{expected_expert}:{expected_name}"
                checks["all_record_order_and_identity"] &= record["key"] == key
                checks["all_source_hashes"] &= (
                    record["source_sha256"] == source_hash
                    and ROOT / record["source"] == source_path
                )
                core_meta_record, tail_meta_record = record["core"], record["tail"]
                core_start, tail_start = core_meta_record["start"], tail_meta_record["start"]
                checks["all_records_aligned_nonoverlapping"] &= (
                    core_start % ALIGNMENT == 0 and tail_start % ALIGNMENT == 0
                    and core_start == prior_core_end and tail_start == prior_tail_end
                )
                core_codes, scale_raw, core = read_core(core_handle, core_start)
                decoded, tail = apply_tail(tail_handle, tail_start, core_codes)
                expected_matrix_id = ("gate", "up", "down").index(expected_name)
                expected_identity = (layer, expected_expert, expected_matrix_id)
                checks["all_record_order_and_identity"] &= (
                    (core["layer"], core["expert"], core["matrix_id"]) == expected_identity
                    and (tail["layer"], tail["expert"], tail["matrix_id"]) == expected_identity
                )
                checks["all_core_headers"] &= core["magic"] == b"CTCORE01" and core["group"] == 128 and core["reserved"] == 0
                checks["all_tail_headers"] &= tail["magic"] == b"CTTAIL01" and tail["block_rows"] == 64 and tail["reserved"] == 0
                checks["all_core_crc_and_offsets"] &= core["crc_ok"] and core["offsets_valid"]
                checks["all_tail_record_crc"] &= tail["record_crc_ok"]
                checks["all_tail_block_crc"] &= tail["block_crc_ok"]
                checks["all_tail_layout"] &= tail["layout_ok"]
                core_record_bytes = core_meta_record["record_bytes"]
                tail_record_bytes = tail_meta_record["record_bytes"]
                checks["all_record_metadata"] &= (
                    core_record_bytes == core["unpadded_bytes"] + core_meta_record["padding_bytes"]
                    and tail_record_bytes == tail["unpadded_bytes"] + tail_meta_record["padding_bytes"]
                    and core_meta_record["padding_bytes"] == (-core["unpadded_bytes"]) % ALIGNMENT
                    and tail_meta_record["padding_bytes"] == (-tail["unpadded_bytes"]) % ALIGNMENT
                    and core_meta_record["rows"] == core["rows"]
                    and core_meta_record["cols"] == core["cols"]
                    and core_meta_record["scale_bytes"] == core["scale_bytes"]
                    and core_meta_record["bitmap_bytes"] == core["bitmap_bytes"]
                    and core_meta_record["row_offset_bytes"] == core["offset_bytes"]
                    and core_meta_record["sign_bytes"] == core["sign_bytes"]
                    and core_meta_record["crc32"] == core["crc32"]
                    and tail_meta_record["rows"] == tail["rows"]
                    and tail_meta_record["cols"] == tail["cols"]
                    and tail_meta_record["negative_bits"] == tail["negative_bits"]
                    and tail_meta_record["blocks"] == tail["blocks"]
                    and tail_meta_record["compressed_blocks"] == tail["compressed_blocks"]
                    and tail_meta_record["raw_fallback_blocks"] == tail["raw_fallback_blocks"]
                    and tail_meta_record["index_bytes"] == tail["index_bytes"]
                    and tail_meta_record["payload_bytes"] == tail["payload_bytes"]
                    and tail_meta_record["raw_flag_bytes"] == tail["raw_flag_bytes"]
                    and tail_meta_record["crc32"] == tail["crc32"]
                    and tail_meta_record["raw_fallback_bytes"] == tail["raw_fallback_bytes"]
                )
                prior_core_end = core_start + core_record_bytes
                prior_tail_end = tail_start + tail_record_bytes
                packed, scale_bits = arrays[expected_name]
                expected_codes = unpack_source(packed[expected_expert])
                expected_scales = scale_bits[expected_expert].astype("<u2", copy=False).tobytes()
                checks["all_codes_bit_exact"] &= np.array_equal(decoded, expected_codes)
                checks["all_scale_bits_exact"] &= scale_raw == expected_scales
                observed_raw_fallback += tail["raw_fallback_bytes"]
                total_weights += expected_codes.size
                total_scales += scale_bits[expected_expert].size
            print(json.dumps({
                "layer": layer,
                "codes_exact": checks["all_codes_bit_exact"],
                "scales_exact": checks["all_scale_bits_exact"],
            }), flush=True)

    checks["all_records_aligned_nonoverlapping"] &= (
        prior_core_end == core_path.stat().st_size and prior_tail_end == tail_path.stat().st_size
    )
    checks["all_raw_fallback_bytes_counted"] &= (
        observed_raw_fallback == result["actual_full_bank"]["raw_fallback_bytes_counted"]
    )
    checks["weight_count_exact"] = total_weights == EXPECTED_WEIGHTS
    checks["scale_count_exact"] = total_scales == EXPECTED_SCALES
    core_gib = core_path.stat().st_size / 2**30
    tail_gib = tail_path.stat().st_size / 2**30
    resident_gib = (
        core_gib + result["memory_gate"]["int4_trunk_gib"]
        + result["memory_gate"]["bf16_kv_4k_gib"]
        + result["memory_gate"]["runtime_reserve_gib"]
    )
    checks["actual_core_gate"] = core_gib <= 5.95
    checks["actual_tail_gate"] = tail_gib <= 0.90
    checks["resident_memory_gate"] = resident_gib <= result["memory_gate"]["reported_vram_gib"]
    passed = all(checks.values())
    payload = {
        "kind": "coretail_moe_p0_full_bank_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "p0_pass" if passed else "p0_fail",
        "checks": checks,
        "passed_checks": sum(bool(value) for value in checks.values()),
        "total_checks": len(checks),
        "measured": {
            "experts": LAYERS * EXPERTS, "matrices": len(result["records"]),
            "weights": total_weights, "scale_elements": total_scales,
            "core_bytes": core_path.stat().st_size, "core_gib": core_gib,
            "tail_bytes": tail_path.stat().st_size, "tail_gib": tail_gib,
            "resident_formula_gib": resident_gib,
            "raw_fallback_bytes_counted": observed_raw_fallback,
        },
        "p1_authorized": passed,
        "claim_boundary": "P0 proves physical size and exact recovery of the fixed GPTQ representation only; fused-kernel throughput and end-to-end tok/s remain unproven.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join([
        "# CORETAIL-MoE P0 — onafhankelijke full-bankverificatie", "",
        f"Uitkomst: **{payload['status']}** ({payload['passed_checks']}/{payload['total_checks']} controles).", "",
        f"Onafhankelijk gereconstrueerd: {total_weights:,} codes en {total_scales:,} BF16-scalebits over 6.144 experts en {len(result['records']):,} matrices.",
        f"Werkelijke core: {core_gib:.6f} GiB. Werkelijke tail: {tail_gib:.6f} GiB. Residentformule: {resident_gib:.6f}/{result['memory_gate']['reported_vram_gib']:.6f} GiB.", "",
        "Bij een pass opent uitsluitend P1: de exacte fused-kernelbenchmark. P0 bewijst nog geen kernelsnelheid, modelkwaliteit of end-to-end tokens per seconde.", "",
    ]), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "checks": f"{payload['passed_checks']}/{payload['total_checks']}",
        "core_gib": core_gib, "tail_gib": tail_gib, "resident_gib": resident_gib,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
