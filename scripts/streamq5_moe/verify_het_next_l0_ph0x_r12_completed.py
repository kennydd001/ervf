#!/usr/bin/env python3
"""Independent CPU-only verifier for the completed PH0X-R12 experiment.

This file deliberately does not import any PH0/PH0X runner or common module.
It reconstructs the Q5 record and the strict BF16 oracle from frozen byte
ranges, then adjudicates immutable Intel and NVIDIA evidence.
"""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r12_independent_verification.json"
R12 = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r12_direct_noftz_cubin_nvidia/ph0x_r12_result.json"
R3 = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r3_exploratory_real_projection/ph0x_r3_result.json"
R7 = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r7_nvidia_only_lifecycle_repair/ph0x_r7_result.json"
R10 = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r10_direct_noftz_ptx_provenance_repair/ph0x_r10_result.json"
CUBIN = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r11_direct_nvrtc_noftz.cubin"
SHARD = Path(r"C:/Users/de_do/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors")
D2 = ROOT / "reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors"

EXPECTED = {
    "r12": "159d77b8fc6d1cac3d2123c09b7f256837c9691a397fa9d052309752d26955bc",
    "r3": "e5fea8e2609f11dd294733645c9a4ecb08892c9d2070de33baacbd1a74b0df7c",
    "r7": "314e08fc907965cf13b2af110b6a45424a9ac75ec5ec429b8f7bc7bf99fdba53",
    "r10": "14eb1b20b8b3f077fe5bcd73e652fe0aa4b2b6233530b637d33d73388977e51e",
    "cubin": "660c22aec2574f12c15d8eed757433d0c9a30a1146fd27957adc96dcea6aaf57",
    "source": "05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c",
    "input": "5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f",
    "codes": "20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b",
    "scales": "658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8",
    "decoded": "9fd43163f4933920168ec9d356db90615a09ecac71198bcc7d3ae373fd995c77",
    "record": "e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9",
    "oracle": "e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867",
    "r7_nvidia": "6525b36b911003ae7e746e6fea1930af61128adfd3fbc41530b6da08d0689041",
}

ROWS, COLS, GROUP = 512, 2048, 128
SOURCE_OFFSET, SOURCE_BYTES = 3_498_051_416, 2_097_152
INPUT_OFFSET, INPUT_BYTES = 155_138_788, 4_096
CODE_BYTES, SCALE_BYTES, PAD_BYTES, RECORD_BYTES = 655_360, 16_384, 4_032, 675_840
HEADER = struct.Struct("<4sHHHBBIIH2xIII28s")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_range(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"short range: {path} {offset} {size}")
    return data


def bf16_to_f32(words: np.ndarray) -> np.ndarray:
    return (np.asarray(words, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def rshift_even(number: int, shift: int) -> int:
    if shift <= 0:
        return number << (-shift)
    quotient, remainder = divmod(number, 1 << shift)
    half = 1 << (shift - 1)
    return quotient + int(remainder > half or (remainder == half and (quotient & 1)))


def finite_parts(bits: int) -> tuple[int, int]:
    sign = -1 if bits >> 31 else 1
    exponent, fraction = (bits >> 23) & 255, bits & 0x7FFFFF
    if exponent == 255:
        raise ValueError("nonfinite")
    if exponent == 0:
        return sign * fraction, -149
    return sign * ((1 << 23) | fraction), exponent - 127 - 23


def pack_exact(number: int, exponent: int) -> int:
    if number == 0:
        return 0
    sign = 0x80000000 if number < 0 else 0
    number = abs(number)
    top = number.bit_length() - 1 + exponent
    if top > 127:
        return sign | 0x7F800000
    if top >= -126:
        shift = number.bit_length() - 24
        significand = rshift_even(number, shift)
        if significand == 1 << 24:
            significand >>= 1
            shift += 1
        unbiased = exponent + shift + 23
        if unbiased > 127:
            return sign | 0x7F800000
        return sign | ((unbiased + 127) << 23) | (significand & 0x7FFFFF)
    fraction = rshift_even(number, -149 - exponent)
    if fraction == 0:
        return sign
    if fraction >= 1 << 23:
        return sign | (1 << 23)
    return sign | fraction


def soft_fma(a: int, b: int, c: int) -> int:
    an, ae = finite_parts(a)
    bn, be = finite_parts(b)
    cn, ce = finite_parts(c)
    product, pe = an * bn, ae + be
    exponent = min(pe, ce)
    return pack_exact((product << (pe - exponent)) + (cn << (ce - exponent)), exponent)


def soft_add(a: int, b: int) -> int:
    return soft_fma(a, 0x3F800000, b)


def round_bf16(bits: int) -> int:
    if (bits & 0x7F800000) == 0x7F800000 and bits & 0x7FFFFF:
        raise ValueError("nonfinite")
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def rebuild(source: bytes) -> tuple[bytes, np.ndarray, np.ndarray, dict[str, object]]:
    values = bf16_to_f32(np.frombuffer(source, "<u2")).reshape(ROWS, COLS)
    blocks = values.reshape(ROWS, COLS // GROUP, GROUP)
    maximum = np.max(np.abs(blocks), axis=-1, keepdims=True)
    scales32 = np.where(maximum > 0, np.asarray(maximum / np.float32(15), dtype=np.float32), np.float32(1))
    q = np.where(maximum > 0, np.clip(np.rint(np.asarray(blocks / scales32, dtype=np.float32)), -15, 15), 0).astype(np.int16)
    fields8 = (q + 15).astype(np.uint64).reshape(-1, 8)
    packed = np.bitwise_or.reduce(fields8 << (np.arange(8, dtype=np.uint64) * 5), axis=1)
    codes = np.stack([(packed >> (8 * index)) & 255 for index in range(5)], axis=1).astype(np.uint8).tobytes()
    scale_words = f32_to_bf16(scales32.reshape(-1))
    scales = scale_words.astype("<u2", copy=False).tobytes()
    decoded = f32_to_bf16(q.reshape(ROWS, -1).astype(np.float32) * bf16_to_f32(scale_words).reshape(ROWS, -1).repeat(GROUP, axis=1)).astype("<u2").tobytes()
    crc = zlib.crc32(scales, zlib.crc32(codes)) & 0xFFFFFFFF
    header = HEADER.pack(b"SQ5M", 1, 0, 50, 0, 5, ROWS, COLS, GROUP, len(codes), len(scales), crc, bytes(28))
    record = header + codes + scales + bytes(PAD_BYTES)
    evidence = {"codes_sha256": sha(codes), "scales_sha256": sha(scales), "decoded_sha256": sha(decoded), "record_sha256": sha(record), "crc32": crc, "record_bytes": len(record)}
    return record, q.reshape(ROWS, COLS), scale_words.reshape(ROWS, COLS // GROUP), evidence


def oracle(q: np.ndarray, scale_words: np.ndarray, input_bytes: bytes) -> np.ndarray:
    activation = np.frombuffer(input_bytes, "<u2")
    output = np.empty(ROWS, dtype="<u2")
    for row in range(ROWS):
        partial = [[0] * 32 for _ in range(8)]
        for lane in range(8):
            for virtual in range(32):
                pack, acc = lane + 8 * virtual, 0
                column = pack * 8
                scale = bf16_to_f32(np.asarray([scale_words[row, column // GROUP]], dtype=np.uint16))[0]
                for part in range(8):
                    weight = int(f32_to_bf16(np.asarray([np.float32(int(q[row, column + part])) * scale]))[0]) << 16
                    acc = soft_fma(weight, int(activation[column + part]) << 16, acc)
                partial[lane][virtual] = acc
        for distance in (16, 8, 4, 2, 1):
            for lane in range(8):
                for index in range(distance):
                    partial[lane][index] = soft_add(partial[lane][index], partial[lane][index + distance])
        lanes = [partial[index][0] for index in range(8)]
        for offset in (4, 2, 1):
            previous = lanes.copy()
            for index in range(offset):
                lanes[index] = soft_add(previous[index], previous[index + offset])
        output[row] = round_bf16(lanes[0])
    return output


def exact_nvidia_ledger(ledger: list[dict[str, object]]) -> bool:
    names = ("record", "input", "output", "counters")
    sizes = (675840, 4096, 1024, 2048)
    if len(ledger) != 24:
        return False
    if ledger[0] != {"op": "cubin_load", "cubin_sha256": EXPECTED["cubin"], "cubin_bytes": 62319, "success": True}:
        return False
    if ledger[1].get("op") != "stream_create" or ledger[1].get("code") != 0 or not ledger[1].get("pointer"):
        return False
    allocations = ledger[2:6]
    if [(x.get("name"), x.get("bytes")) for x in allocations] != list(zip(names, sizes)):
        return False
    if len({x.get("pinned_pointer") for x in allocations}) != 4 or len({x.get("device_pointer") for x in allocations}) != 4:
        return False
    if not all(x.get("pinned_pointer") and x.get("device_pointer") for x in allocations):
        return False
    expected_ops = ["memset", "memset", "H2D", "H2D", "kernel", "D2H", "D2H", "synchronize"]
    if [x.get("op") for x in ledger[6:14]] != expected_ops:
        return False
    if [(x.get("target"), x.get("bytes")) for x in ledger[6:10]] != [("output", 1024), ("counters", 2048), ("record", 675840), ("input", 4096)]:
        return False
    if ledger[10] != {"op": "kernel", "grid": [16], "block": [256]}:
        return False
    if [(x.get("target"), x.get("bytes")) for x in ledger[11:13]] != [("output", 1024), ("counters", 2048)]:
        return False
    if ledger[13] != {"op": "synchronize", "code": 0}:
        return False
    releases = [f"device_{x}" for x in reversed(names)] + [f"pinned_{x}" for x in reversed(names)] + ["stream"]
    if [(x.get("release"), x.get("code")) for x in ledger[14:23]] != [(x, 0) for x in releases]:
        return False
    return ledger[23] == {"cleanup_complete": True, "errors": []}


def exact_intel_lifecycle(intel: dict[str, object]) -> bool:
    ledger = intel.get("ledger", [])
    if len(ledger) != 14:
        return False
    expected_alloc = [("record", 675840), ("input", 4096), ("output", 1024), ("counters", 2048)]
    if [(x.get("name"), x.get("bytes")) for x in ledger[:4]] != expected_alloc:
        return False
    if not all(x.get("kind") == "usm" and x.get("pointer") for x in ledger[:4]):
        return False
    releases = ["event", "counters", "output", "input", "record", "kernel", "program", "queue", "context"]
    if [(x.get("release"), x.get("code")) for x in ledger[4:13]] != [(x, 0) for x in releases]:
        return False
    return ledger[13] == {"cleanup_complete": True, "errors": []}


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    r12, r3, r7, r10 = [json.loads(path.read_text(encoding="utf-8")) for path in (R12, R3, R7, R10)]
    source = read_range(SHARD, SOURCE_OFFSET, SOURCE_BYTES)
    input_bytes = read_range(D2, INPUT_OFFSET, INPUT_BYTES)
    record, q, scale_words, codec = rebuild(source)
    strict = oracle(q, scale_words, input_bytes)
    strict_bytes = strict.tobytes()
    intel_output = bytes.fromhex(r3["intel"]["output_hex"])
    intel_counters = np.frombuffer(bytes.fromhex(r3["intel"]["counters_hex"]), "<u4")
    nvidia_output = bytes.fromhex(r12["nvidia"]["output_hex"])
    nvidia_counters = np.frombuffer(bytes.fromhex(r12["nvidia"]["counters_hex"]), "<u4")

    dependencies = r12.get("bindings", {}).get("dependencies", {})
    dependency_checks = {}
    for raw_path, expected_hash in dependencies.items():
        path = Path(raw_path)
        dependency_checks[raw_path] = path.is_file() and file_sha(path) == expected_hash

    checks = {
        "immutable_primary_hashes": file_sha(R12) == EXPECTED["r12"] and file_sha(R3) == EXPECTED["r3"] and file_sha(R7) == EXPECTED["r7"] and file_sha(R10) == EXPECTED["r10"],
        "all_r12_dependency_hashes": bool(dependency_checks) and all(dependency_checks.values()),
        "source_range_exact": len(source) == SOURCE_BYTES and sha(source) == EXPECTED["source"],
        "input_range_exact": len(input_bytes) == INPUT_BYTES and sha(input_bytes) == EXPECTED["input"],
        "codec_codes_exact": codec["codes_sha256"] == EXPECTED["codes"],
        "codec_scales_exact": codec["scales_sha256"] == EXPECTED["scales"],
        "codec_decoded_exact": codec["decoded_sha256"] == EXPECTED["decoded"],
        "record_exact": len(record) == RECORD_BYTES and codec["record_sha256"] == EXPECTED["record"] and codec["crc32"] == 1_976_639_022,
        "independent_oracle_exact": strict.size == 512 and sha(strict_bytes) == EXPECTED["oracle"],
        "r3_intel_output_exact": len(intel_output) == 1024 and intel_output == strict_bytes,
        "r3_intel_counters_exact": intel_counters.size == 512 and bool(np.all(intel_counters == 1)),
        "r3_intel_lifecycle_exact": exact_intel_lifecycle(r3["intel"]),
        "r3_intel_transport_exact": r3["intel"].get("enqueue_calls") == 1 and r3["intel"].get("forbidden_copy_calls") == 0,
        "r3_intel_identity_exact": r3["intel"].get("identity", {}).get("pci") == "0000:00:02.0",
        "r3_controls_exact": len(r3.get("controls", [])) == 9 and all(x.get("pass") is True for x in r3["controls"]),
        "r12_positive_schema": r12.get("status") == "exploratory_direct_noftz_nvidia_positive" and r12.get("positive") is True and all(r12.get("gates", {}).values()),
        "r12_nvidia_output_exact": len(nvidia_output) == 1024 and nvidia_output == strict_bytes,
        "r12_nvidia_counters_exact": nvidia_counters.size == 512 and bool(np.all(nvidia_counters == 1)),
        "r12_nvidia_ledger_exact": exact_nvidia_ledger(r12["nvidia"]["ledger"]),
        "r12_nvidia_identity_exact": r12["nvidia"].get("identity", {}).get("count") == 1 and r12["nvidia"]["identity"].get("name") == "NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU" and r12["nvidia"]["identity"].get("pci") == "0000:01:00.0",
        "r12_intercept_exact": r12["nvidia"].get("rawmodule_intercept_calls") == 1,
        "cubin_exact": CUBIN.stat().st_size == 62319 and CUBIN.read_bytes()[:4] == b"\x7fELF" and file_sha(CUBIN) == EXPECTED["cubin"],
        "r7_development_negative_exact": r7.get("status") == "exploratory_nvidia_completion_negative" and r7.get("positive") is False and r7.get("nvidia_comparison", {}).get("different_words") == 122 and r7["nvidia_comparison"].get("output_sha256") == EXPECTED["r7_nvidia"] and r7.get("gates", {}).get("nvidia_exact") is False,
        "r10_development_failure_exact": r10.get("status") == "exploratory_direct_noftz_nvidia_failure" and "CUDA_ERROR_UNSUPPORTED_PTX_VERSION" in r10.get("error", "") and r10.get("intel_reexecuted") is False and r10.get("source_compilation_performed") is False,
        "cross_device_three_way_exact": intel_output == nvidia_output == strict_bytes,
    }
    result = {
        "kind": "ph0x_r12_independent_cpu_verification",
        "verifier_sha256": file_sha(Path(__file__)),
        "checks": checks,
        "check_count": len(checks),
        "pass_count": sum(checks.values()),
        "pass": all(checks.values()),
        "dependency_check_count": len(dependency_checks),
        "dependency_failures": [path for path, passed in dependency_checks.items() if not passed],
        "reconstruction": {
            "source_sha256": sha(source),
            "input_sha256": sha(input_bytes),
            **codec,
            "oracle_sha256": sha(strict_bytes),
            "oracle_words": int(strict.size),
        },
        "device_evidence": {
            "intel_output_sha256": sha(intel_output),
            "intel_counter_count": int(intel_counters.size),
            "nvidia_output_sha256": sha(nvidia_output),
            "nvidia_counter_count": int(nvidia_counters.size),
            "nvidia_ledger_rows": len(r12["nvidia"]["ledger"]),
            "cubin_sha256": file_sha(CUBIN),
        },
        "development_evidence": {
            "r7_result_sha256": file_sha(R7),
            "r7_formal_status": r7.get("status"),
            "r7_cpu_different_words": r7.get("nvidia_comparison", {}).get("different_words"),
            "r10_result_sha256": file_sha(R10),
            "r10_formal_status": r10.get("status"),
            "r10_error": r10.get("error"),
        },
        "claim_boundary": "Validation-only reproduction of one official real Q5 projection on one known natural activation across an independent CPU oracle, stored Intel host-USM output, and stored NVIDIA no-FTZ cubin output. No full expert, MoE, layer, model, held-out/generalized quality, cohabitation, concurrency, timing, performance, deployment, novelty, or breakthrough claim.",
    }
    OUT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"pass": result["pass"], "pass_count": result["pass_count"], "check_count": result["check_count"], "oracle_sha256": result["reconstruction"]["oracle_sha256"]}, indent=2))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
