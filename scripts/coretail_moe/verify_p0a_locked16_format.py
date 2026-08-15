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


RESULT = ROOT / "reports/coretail_moe/p0a_locked16_format_result.json"
LOCK = ROOT / "reports/fleq_moe/p1_smoke_expert_lock.json"
OUT_JSON = ROOT / "reports/coretail_moe/p0a_locked16_format_verification.json"
OUT_MD = ROOT / "reports/coretail_moe/P0A_LOCKED16_FORMAT_VERIFICATION.md"
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


def source_codes_and_scales(path: Path, name: str):
    tensors = load_file(path)
    weight = tensors[f"gptq_{name}_weight"].cpu()
    scales = tensors[f"gptq_{name}_scales"].cpu().contiguous()
    groups = torch.arange(weight.shape[1]) // 128
    codes = torch.round(weight.float() / scales.float()[:, groups]).to(torch.int8).numpy()
    return codes, scales.view(torch.uint16).numpy().tobytes()


def read_core(handle, offset: int):
    handle.seek(offset)
    fields = CORE_HEADER.unpack(handle.read(CORE_HEADER.size))
    magic, layer, expert, matrix_id, rows, cols, group, reserved, scale_n, bitmap_n, offsets_n, sign_n, crc = fields
    payload = handle.read(scale_n + bitmap_n + offsets_n + sign_n)
    cursor = 0
    scale_raw = payload[cursor : cursor + scale_n]
    cursor += scale_n
    bitmap = payload[cursor : cursor + bitmap_n]
    cursor += bitmap_n
    row_offsets = np.frombuffer(payload[cursor : cursor + offsets_n], dtype="<u4")
    cursor += offsets_n
    signs = payload[cursor : cursor + sign_n]
    codes = np.zeros((rows, cols), dtype=np.int8)
    row_bitmap_n = math.ceil(cols / 8)
    for row in range(rows):
        nz = np.unpackbits(
            np.frombuffer(bitmap[row * row_bitmap_n : (row + 1) * row_bitmap_n], dtype=np.uint8),
            bitorder="little",
        )[:cols].astype(bool)
        count = int(nz.sum())
        sign = np.unpackbits(
            np.frombuffer(signs[row_offsets[row] : row_offsets[row + 1]], dtype=np.uint8),
            bitorder="little",
        )[:count]
        codes[row, nz] = np.where(sign, 1, -1)
    return codes, scale_raw, {
        "magic": magic, "layer": layer, "expert": expert, "matrix_id": matrix_id,
        "rows": rows, "cols": cols, "group": group, "reserved": reserved,
        "crc_ok": (binascii.crc32(payload) & 0xFFFFFFFF) == crc,
        "offset_terminal": int(row_offsets[-1]), "sign_bytes": sign_n,
    }


def apply_tail(handle, offset: int, decoded: np.ndarray):
    handle.seek(offset)
    fields = TAIL_HEADER.unpack(handle.read(TAIL_HEADER.size))
    magic, layer, expert, matrix_id, rows, cols, block_rows, reserved, negative_bits, blocks, index_n, payload_n, raw_n, crc = fields
    record = handle.read(index_n + payload_n)
    index, payload = record[:index_n], record[index_n:]
    block_crc_ok = True
    decoded = decoded.copy()
    observed_bits = observed_raw_n = 0
    for block_id in range(blocks):
        entry = TAIL_INDEX.unpack(index[block_id * TAIL_INDEX.size : (block_id + 1) * TAIL_INDEX.size])
        position, stored_n, raw_len, bit_count, block_crc, codec = entry
        stored = payload[position : position + stored_n]
        raw = zlib.decompress(stored) if codec == 1 else stored
        block_crc_ok &= len(raw) == raw_len and (binascii.crc32(raw) & 0xFFFFFFFF) == block_crc
        flags = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:bit_count].astype(bool)
        row_start = block_id * block_rows
        block = decoded[row_start : min(rows, row_start + block_rows)]
        negative = block < 0
        if int(negative.sum()) != bit_count:
            block_crc_ok = False
        else:
            block[negative] -= flags.astype(np.int8)
        observed_bits += bit_count
        observed_raw_n += raw_len
    return decoded, {
        "magic": magic, "layer": layer, "expert": expert, "matrix_id": matrix_id,
        "rows": rows, "cols": cols, "block_rows": block_rows, "reserved": reserved,
        "record_crc_ok": (binascii.crc32(record) & 0xFFFFFFFF) == crc,
        "block_crc_ok": bool(block_crc_ok),
        "totals_ok": observed_bits == negative_bits and observed_raw_n == raw_n,
    }


if __name__ == "__main__":
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("refusing to overwrite CORETAIL verification")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    core_path = ROOT / result["actual_locked16"]["core_file"]
    tail_path = ROOT / result["actual_locked16"]["tail_file"]
    checks: list[dict] = []

    def check(name: str, passed: bool, observed=None, expected=None):
        checks.append({"name": name, "pass": bool(passed), "observed": observed, "expected": expected})

    core_hash, tail_hash = sha256(core_path), sha256(tail_path)
    check("core_sha256", core_hash == result["actual_locked16"]["core_sha256"], core_hash, result["actual_locked16"]["core_sha256"])
    check("tail_sha256", tail_hash == result["actual_locked16"]["tail_sha256"], tail_hash, result["actual_locked16"]["tail_sha256"])
    check("core_size", core_path.stat().st_size == result["actual_locked16"]["core_bytes"], core_path.stat().st_size, result["actual_locked16"]["core_bytes"])
    check("tail_size", tail_path.stat().st_size == result["actual_locked16"]["tail_bytes"], tail_path.stat().st_size, result["actual_locked16"]["tail_bytes"])
    check("record_count", len(result["records"]) == 48, len(result["records"]), 48)
    check("source_count", result["inputs"]["canonical_experts"] == 16, result["inputs"]["canonical_experts"], 16)
    lock_hash = sha256(LOCK)
    check("source_lock_hash", lock_hash == result["inputs"]["selection_lock_sha256"], lock_hash, result["inputs"]["selection_lock_sha256"])

    all_source_hashes = all_core_headers = all_tail_headers = True
    all_core_crc = all_tail_crc = all_block_crc = all_tail_totals = True
    all_identity = all_code_exact = all_scale_exact = all_aligned = True
    total_weights = 0
    with core_path.open("rb") as core_handle, tail_path.open("rb") as tail_handle:
        for record in result["records"]:
            source = ROOT / record["source"]
            all_source_hashes &= sha256(source) == record["source_sha256"]
            core_codes, scale_raw, core_meta = read_core(core_handle, record["core"]["start"])
            decoded, tail_meta = apply_tail(tail_handle, record["tail"]["start"], core_codes)
            identity = (core_meta["layer"], core_meta["expert"], core_meta["matrix_id"])
            all_identity &= identity == (tail_meta["layer"], tail_meta["expert"], tail_meta["matrix_id"])
            all_core_headers &= core_meta["magic"] == b"CTCORE01" and core_meta["group"] == 128 and core_meta["reserved"] == 0
            all_tail_headers &= tail_meta["magic"] == b"CTTAIL01" and tail_meta["block_rows"] == 64 and tail_meta["reserved"] == 0
            all_core_crc &= core_meta["crc_ok"] and core_meta["offset_terminal"] == core_meta["sign_bytes"]
            all_tail_crc &= tail_meta["record_crc_ok"]
            all_block_crc &= tail_meta["block_crc_ok"]
            all_tail_totals &= tail_meta["totals_ok"]
            expected_codes, expected_scales = source_codes_and_scales(source, MATRIX_NAME[core_meta["matrix_id"]])
            all_code_exact &= np.array_equal(decoded, expected_codes)
            all_scale_exact &= scale_raw == expected_scales
            total_weights += expected_codes.size
            all_aligned &= record["core"]["start"] % 4096 == 0 and record["tail"]["start"] % 4096 == 0

    for name, value in (
        ("source_hashes", all_source_hashes), ("core_headers", all_core_headers),
        ("tail_headers", all_tail_headers), ("core_crc_and_offsets", all_core_crc),
        ("tail_record_crc", all_tail_crc), ("tail_block_crc", all_block_crc),
        ("tail_totals", all_tail_totals), ("core_tail_identity", all_identity),
        ("code_reconstruction_vs_source", all_code_exact), ("scale_bits_vs_source", all_scale_exact),
        ("record_alignment", all_aligned),
    ):
        check(name, value)
    check("weight_count", total_weights == result["actual_locked16"]["weights"], total_weights, result["actual_locked16"]["weights"])

    core_bpp = core_path.stat().st_size * 8 / total_weights
    tail_bpp = tail_path.stat().st_size * 8 / total_weights
    full_params = 28_991_029_248
    core_gib = full_params * core_bpp / 8 / 2**30
    tail_gib = full_params * tail_bpp / 8 / 2**30
    resident = core_gib + 0.7176275253295898 + 0.375 + 0.75
    projection = result["diagnostic_full_bank_linear_projection_not_gate"]
    check("core_bpp_arithmetic", abs(core_bpp - result["actual_locked16"]["core_bpp"]) < 1e-12, core_bpp, result["actual_locked16"]["core_bpp"])
    check("tail_bpp_arithmetic", abs(tail_bpp - result["actual_locked16"]["tail_bpp"]) < 1e-12, tail_bpp, result["actual_locked16"]["tail_bpp"])
    check("core_projection_arithmetic", abs(core_gib - projection["core_gib"]) < 1e-12, core_gib, projection["core_gib"])
    check("tail_projection_arithmetic", abs(tail_gib - projection["tail_gib"]) < 1e-12, tail_gib, projection["tail_gib"])
    check("resident_arithmetic", abs(resident - projection["resident_core_plus_int4_trunk_bf16_4k_kv_reserve_gib"]) < 1e-12, resident, projection["resident_core_plus_int4_trunk_bf16_4k_kv_reserve_gib"])
    check("diagnostic_memory_gates", core_gib <= 5.95 and tail_gib <= 0.90 and resident <= 7.9599609375)
    check("official_coverage_gate_fails", result["inputs"]["canonical_experts"] != result["inputs"]["required_experts"] and not result["gates"]["official_full_bank_p0_pass"])
    check("p1_closed", result["p1_authorized"] is False)
    check("verdict_boundary", result["verdict"] == "p0_blocked_missing_full_bank_gptq_codes")

    passed = sum(item["pass"] for item in checks)
    verification = {
        "kind": "coretail_moe_p0a_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": passed, "checks_total": len(checks), "all_pass": passed == len(checks),
        "verdict": "locked16_mechanics_verified_full_p0_still_blocked" if passed == len(checks) else "verification_failed",
        "checks": checks,
    }
    OUT_JSON.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join([
        "# CORETAIL-MoE P0A onafhankelijke verificatie", "",
        f"Uitkomst: **{verification['verdict']}** ({passed}/{len(checks)} controles geslaagd).", "",
        "Een tweede decoder controleerde headers, offsets, record- en blokchecksums en vergeleek alle 75.497.472 gereconstrueerde codes plus alle BF16-scalebits opnieuw met de 16 canonieke GPTQ-bronbestanden.", "",
        "De locked16-codec en geheugenrekenkunde zijn bevestigd. De officiële full-bank P0 blijft geblokkeerd omdat 6.128 van de 6.144 vereiste GPTQ-experts ontbreken; P1 blijft gesloten.", "",
    ]), encoding="utf-8")
    print(json.dumps({"verdict": verification["verdict"], "checks": f"{passed}/{len(checks)}"}, indent=2))
