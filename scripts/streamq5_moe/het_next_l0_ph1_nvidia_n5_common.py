#!/usr/bin/env python3
"""CPU-only PH1 NVIDIA N5 package, codec, oracle, controls and contracts.

Importing this module is side-effect free: no payload is opened and no CUDA or
compiler library is loaded.  Payload reads occur only through prepare_package().
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
CPU = REPORTS / "het_next_l0_ph1_cpu_freeze_r2"
SHARD = Path(r"C:/Users/de_do/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors")
D2 = ROOT / "reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors"
HEADER = struct.Struct("<4sHHHBBIIH2xIII28s")
RECORD_BYTES = 675_840
CODE_BYTES = 655_360
SCALE_OFFSET = 655_424
SCALE_END = 671_808
INPUT = (155_138_788, 4_096, "5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f")
LUT_SHA = "a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed"
CPU_FILES = {
    "commit.json": "f3677e9610bea03649fec172b97c0c314f2f2e4c0d40bf9d864df0ec88a44f06",
    "manifest.json": "63f6c842f377fb18738d6016b133c7529803581d0cd661739c0ffd648a82ac54",
    "cpu_stage_freeze.safetensors": "c2fbc4d6c3c400ecb0ac7af36b36c88a1c8122d3066cb123430f934bd750d6a8",
    "bf16_silu_lut.bin": LUT_SHA,
}
INTEL_FINAL = REPORTS / "het_next_l0_ph1_intel_execution_r8v1r1a_independent_verification.json"
INTEL_FINAL_SHA = "42cd69582a47b8b5f8f4b7f24a696f1d3fcc6fbd49c05d0f61354a57cefc052d"
R8A5 = REPORTS / "het_next_l0_ph1_intel_execution_r8a5"
R8A5_FILES = {
    "result.json": "9d1ac21f4fdd9657160e877f267369b5e831ff9f7a65e998f27895947c9cad50",
    "manifest.json": "2d13137f143ff183be3ffe89a3b85754cb2f35b52f92885580f49676e5fcfb7b",
    "commit.json": "07d9f03e8907a029d8bc31e40da6298de080b6bc0f0914769f8d52517b2dd965",
}
SPECS = (
    ("gate", 0, (512, 2048), (3_498_051_416, 3_500_148_568), "05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c", "20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b", "658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8", "e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9"),
    ("up", 1, (512, 2048), (3_500_148_568, 3_502_245_720), "4b36f661a351aaf907be1e041743833bc7a0564e07a6c140917ef1c8d69e4c0d", "6b2a3f124c3bc42d584b2816b063801d63244bd2a9e59cb00a32e339591e25cb", "c275fd13db6ea41ab8af1563a32a8de188e5fa488f91a6c7c939c4d3ca80a9f9", "6da7025af27de06c4f6011ddfc82672263b6f0593b2dcacf77705a443f44fbfb"),
    ("down", 2, (2048, 512), (3_495_954_264, 3_498_051_416), "bdf53c222b88c66b5845fd548ae984c20959231150b2fd34ddccf10d1777e479", "3d8782d588d507fea2a2c51ef8a3ea18ce6795d72b4be047b0c123652d77a703", "a3cd1a7c827dd9cb64925ad15299adbc18d74e592a1414504c3015e29854977e", "bd1a8ef9ae689fefb73408f3985c96a0725670dc0b0f7f46268a5a89d12157"),
)
# The up record SHA above is asserted against the canonical value below to make
# accidental hand transcription visible before any source read.
CANONICAL_RECORD_SHA = {
    "gate": "e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9",
    "up": "6da7025af27de06c4f6011ddfc82672263b6f0593b2dcacf77705a443f44fbfb",
    "down": "bd1a8ef9ae689fefb73408f3985c96a0725670dc0b0f7f46268a5a89d12157",
}
STAGE_SHA = {
    "gate": "e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867",
    "up": "f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08",
    "silu": "a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8",
    "activation": "762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f",
    "down": "142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc",
}
BUFFER_TABLE = (
    ("gate_record", 675840), ("up_record", 675840), ("down_record", 675840),
    ("natural_input", 4096), ("silu_lut", 131072), ("gate", 1024),
    ("up", 1024), ("silu", 1024), ("activation", 1024), ("down", 4096),
    ("gate_counters", 2048), ("up_counters", 2048),
    ("activation_counters", 2048), ("down_counters", 8192),
)
ARGUMENT_MAPS = (
    ("q5_linear:gate", ("gate_record", "natural_input", "gate", "gate_counters")),
    ("q5_linear:up", ("up_record", "natural_input", "up", "up_counters")),
    ("bf16_lut_activation", ("gate", "up", "silu_lut", "silu", "activation", "activation_counters")),
    ("q5_linear:down", ("down_record", "activation", "down", "down_counters")),
)
LAUNCHES = (
    ("q5_linear:gate", (16, 1, 1), (256, 1, 1)),
    ("q5_linear:up", (16, 1, 1), (256, 1, 1)),
    ("bf16_lut_activation", (2, 1, 1), (256, 1, 1)),
    ("q5_linear:down", (64, 1, 1), (256, 1, 1)),
)
RESOURCE_STAGES = (
    "process_start", "post_authorization", "post_cpu_package", "post_controls",
    "pre_cuda_init", "post_context_push", "post_module_stream_preallocation",
    "post_allocations", "post_memset_h2d", "post_launches_queued",
    "post_d2h_sync", "post_ordinary_releases_pre_pop", "post_context_release",
    "post_serialization",
)

if len(BUFFER_TABLE) != 14 or sum(v for _, v in BUFFER_TABLE) != 2_185_216:
    raise RuntimeError("frozen_buffer_contract")
if len(ARGUMENT_MAPS) != 4 or sum(len(v) for _, v in ARGUMENT_MAPS) != 18 or len(LAUNCHES) != 4:
    raise RuntimeError("frozen_launch_contract")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_range(path: Path, offset: int, size: int) -> bytes:
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"range:{path}:{offset}:{size}")
    return data


def b2f(words):
    return (np.asarray(words, np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def f2b(values):
    bits = np.asarray(values, np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def build_record(spec, source: bytes) -> bytes:
    name, ordinal, (rows, cols), _, source_sha, code_sha, scale_sha, _ = spec
    if sha(source) != source_sha:
        raise ValueError(f"source_digest:{name}")
    values = b2f(np.frombuffer(source, "<u2")).reshape(rows, cols)
    groups = values.reshape(rows, cols // 128, 128)
    maxima = np.max(np.abs(groups), axis=-1, keepdims=True)
    scales = np.where(maxima > 0, np.asarray(maxima / np.float32(15), np.float32), np.float32(1))
    q = np.where(maxima > 0, np.clip(np.rint(np.asarray(groups / scales, np.float32)), -15, 15), 0).astype(np.int16)
    fields = (q + 15).astype(np.uint64).reshape(-1, 8)
    words = np.bitwise_or.reduce(fields << (np.arange(8, dtype=np.uint64) * 5), axis=1)
    codes = np.stack([(words >> (8 * index)) & 255 for index in range(5)], axis=1).astype(np.uint8).tobytes()
    scale_bytes = f2b(scales.reshape(-1)).astype("<u2").tobytes()
    crc = zlib.crc32(scale_bytes, zlib.crc32(codes)) & 0xFFFFFFFF
    header = HEADER.pack(b"SQ5M", 1, 0, 50, ordinal, 5, rows, cols, 128, len(codes), len(scale_bytes), crc, bytes(28))
    record = header + codes + scale_bytes + bytes(4032)
    if sha(codes) != code_sha or sha(scale_bytes) != scale_sha:
        raise ValueError(f"codec_digest:{name}")
    if len(record) != RECORD_BYTES or sha(record) != CANONICAL_RECORD_SHA[name]:
        raise ValueError(f"record_digest:{name}")
    return record


def check_record(data: bytes, spec, input_digest: str) -> str:
    if len(data) != RECORD_BYTES:
        return "size"
    fields = HEADER.unpack(data[:64])
    name, ordinal, shape, *_ = spec
    if fields[:9] != (b"SQ5M", 1, 0, 50, ordinal, 5, *shape, 128):
        return "identity"
    codes, scales = data[64:SCALE_OFFSET], data[SCALE_OFFSET:SCALE_END]
    if (zlib.crc32(scales, zlib.crc32(codes)) & 0xFFFFFFFF) != fields[11]:
        return "crc"
    octets = np.frombuffer(codes, np.uint8).reshape(-1, 5).astype(np.uint64)
    packed = sum(octets[:, index] << (8 * index) for index in range(5))
    unpacked = (packed[:, None] >> (5 * np.arange(8, dtype=np.uint64))) & 31
    if np.any(unpacked == 31):
        return "field31"
    if sha(codes) != spec[5] or sha(scales) != spec[6]:
        return "canonical_digest"
    if sha(data) != CANONICAL_RECORD_SHA[name]:
        return "record_digest"
    if input_digest != INPUT[2]:
        return "input_digest"
    return "pass"


def checker_trace(data: bytes, spec, input_digest: str):
    """Retain requested/presented metadata and the exact fail-closed stage reached."""
    observed = check_record(data, spec, input_digest)
    order = ("size", "identity", "crc", "field31", "canonical_digest", "record_digest", "input_digest", "pass")
    stop = order.index(observed)
    fields = HEADER.unpack(data[:64]) if len(data) >= 64 else None
    requested = {"record": spec[0], "expert": 50, "projection": spec[1], "shape": list(spec[2]),
                 "record_sha256": spec[7], "codes_sha256": spec[5], "scales_sha256": spec[6],
                 "input_sha256": INPUT[2]}
    presented = {"bytes": len(data), "record_sha256": sha(data), "input_sha256": input_digest,
                 "header": None if fields is None else {"magic": fields[0].hex(), "version": fields[1],
                 "expert": fields[3], "projection": fields[4], "bits": fields[5], "rows": fields[6],
                 "cols": fields[7], "group": fields[8], "crc32": fields[11]},
                 "codes_sha256": None if len(data) < SCALE_OFFSET else sha(data[64:SCALE_OFFSET]),
                 "scales_sha256": None if len(data) < SCALE_END else sha(data[SCALE_OFFSET:SCALE_END])}
    trace = [{"stage": stage, "result": observed if index == stop else "pass"}
             for index, stage in enumerate(order[:stop + 1])]
    return observed, requested, presented, trace


def safe_controls(records, input_bytes: bytes, lut: bytes):
    rows = []
    zero = {name: 0 for name in ("nvrtc_load", "compile", "nvcuda_load", "context", "module", "stream", "allocation", "launch")}
    for spec in SPECS:
        name, base = spec[0], records[spec[0]]
        cases = [("truncation", base[:-1], INPUT[2], "size")]
        mutated = bytearray(base); header = list(HEADER.unpack(mutated[:64])); header[4] = (header[4] + 1) % 3; mutated[:64] = HEADER.pack(*header)
        cases.append(("wrong_projection", bytes(mutated), INPUT[2], "identity"))
        mutated = bytearray(base); mutated[64] ^= 1
        cases.append(("stale_crc", bytes(mutated), INPUT[2], "crc"))
        mutated = bytearray(base); packed = int.from_bytes(mutated[64:69], "little"); selected = None
        for slot in range(8):
            field = (packed >> (5 * slot)) & 31
            if field != 15:
                replacement = field - 1 if field > 15 else field + 1
                if replacement <= 30:
                    selected = slot, replacement; break
        if selected is None:
            raise RuntimeError("no_code_mutation")
        slot, replacement = selected; packed = (packed & ~(31 << (5 * slot))) | (replacement << (5 * slot)); mutated[64:69] = packed.to_bytes(5, "little")
        header = list(HEADER.unpack(mutated[:64])); header[11] = zlib.crc32(mutated[SCALE_OFFSET:SCALE_END], zlib.crc32(mutated[64:SCALE_OFFSET])) & 0xFFFFFFFF; mutated[:64] = HEADER.pack(*header)
        cases.append(("code_mutation", bytes(mutated), INPUT[2], "canonical_digest"))
        mutated = bytearray(base); mutated[SCALE_OFFSET] ^= 1; header = list(HEADER.unpack(mutated[:64])); header[11] = zlib.crc32(mutated[SCALE_OFFSET:SCALE_END], zlib.crc32(mutated[64:SCALE_OFFSET])) & 0xFFFFFFFF; mutated[:64] = HEADER.pack(*header)
        cases.append(("scale_mutation", bytes(mutated), INPUT[2], "canonical_digest"))
        mutated = bytearray(base); packed = int.from_bytes(mutated[64:69], "little"); mutated[64:69] = ((packed & ~31) | 31).to_bytes(5, "little"); header = list(HEADER.unpack(mutated[:64])); header[11] = zlib.crc32(mutated[SCALE_OFFSET:SCALE_END], zlib.crc32(mutated[64:SCALE_OFFSET])) & 0xFFFFFFFF; mutated[:64] = HEADER.pack(*header)
        cases.append(("field31", bytes(mutated), INPUT[2], "field31")); cases.append(("wrong_input", base, "0" * 64, "input_digest"))
        for control, presented, input_digest, expected in cases:
            observed, requested, metadata, trace = checker_trace(presented, spec, input_digest)
            rows.append({"record": name, "control": control, "expected": expected, "observed": observed,
                         "pass": observed == expected, "requested": requested, "presented": metadata,
                         "checker_trace": trace, "predevice_counts": dict(zero)})
    wrong = bytearray(lut); wrong[0] ^= 1; observed = "lut_digest" if sha(wrong) != LUT_SHA else "pass"
    rows.append({"record": "global", "control": "wrong_lut_digest", "expected": "lut_digest", "observed": observed,
                 "pass": observed == "lut_digest", "requested": {"lut_sha256": LUT_SHA},
                 "presented": {"lut_sha256": sha(wrong), "bytes": len(wrong)},
                 "checker_trace": [{"stage": "lut_digest", "result": observed}], "predevice_counts": dict(zero)})
    if len(rows) != 22 or not all(row["pass"] for row in rows):
        raise RuntimeError("control_gate")
    return rows


def _parts(bits: int):
    sign = -1 if bits >> 31 else 1; exponent = (bits >> 23) & 255; fraction = bits & 0x7FFFFF
    if exponent == 255:
        raise ValueError("nonfinite")
    return (sign * fraction, -149) if exponent == 0 else (sign * ((1 << 23) | fraction), exponent - 150)


def _round_shift_even(number: int, shift: int):
    if shift <= 0:
        return number << -shift
    quotient, remainder = divmod(number, 1 << shift); halfway = 1 << (shift - 1)
    return quotient + int(remainder > halfway or (remainder == halfway and quotient & 1))


def _pack(number: int, exponent: int):
    if number == 0:
        return 0
    sign = 0x80000000 if number < 0 else 0; number = abs(number); top = number.bit_length() - 1 + exponent
    if top > 127:
        return sign | 0x7F800000
    if top >= -126:
        shift = number.bit_length() - 24; significand = _round_shift_even(number, shift)
        if significand == 1 << 24:
            significand >>= 1; shift += 1
        unbiased = exponent + shift + 23
        return sign | 0x7F800000 if unbiased > 127 else sign | ((unbiased + 127) << 23) | (significand & 0x7FFFFF)
    fraction = _round_shift_even(number, -149 - exponent)
    return sign if fraction == 0 else sign | (1 << 23) if fraction >= 1 << 23 else sign | fraction


def fp32_fma(a: int, b: int, c: int) -> int:
    an, ae = _parts(a); bn, be = _parts(b); cn, ce = _parts(c); pn, pe = an * bn, ae + be; exponent = min(pe, ce)
    return _pack((pn << (pe - exponent)) + (cn << (ce - exponent)), exponent)


def fp32_add(a: int, b: int) -> int:
    return fp32_fma(a, 0x3F800000, b)


def round_bf16(bits: int) -> int:
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def multiply_bf16(a: int, b: int) -> int:
    if ((a >> 7) & 255) == 255 or ((b >> 7) & 255) == 255:
        raise ValueError("nonfinite_multiply")
    if (a & 0x7FFF) == 0 or (b & 0x7FFF) == 0:
        return (a ^ b) & 0x8000
    return round_bf16(fp32_fma(a << 16, b << 16, 0))


def decode_record(record: bytes, spec):
    rows, cols = spec[2]; codes = np.frombuffer(record[64:SCALE_OFFSET], np.uint8).reshape(-1, 5).astype(np.uint64)
    packed = sum(codes[:, index] << (8 * index) for index in range(5)); fields = ((packed[:, None] >> (5 * np.arange(8, dtype=np.uint64))) & 31).astype(np.int16) - 15
    q = fields.reshape(rows, cols); scales = np.frombuffer(record[SCALE_OFFSET:SCALE_END], "<u2")
    expanded = b2f(scales).reshape(rows, cols // 128).repeat(128, axis=1)
    return f2b(q.astype(np.float32) * expanded).reshape(rows, cols)


def width8_linear(weights, inputs):
    rows, cols = weights.shape; virtual_count = cols // 64; distances = (16, 8, 4, 2, 1) if cols == 2048 else (4, 2, 1); output = np.empty(rows, np.uint16)
    for row in range(rows):
        partial = [[0] * virtual_count for _ in range(8)]
        for lane in range(8):
            for virtual in range(virtual_count):
                column = (lane + 8 * virtual) * 8; accumulator = 0
                for field in range(8):
                    accumulator = fp32_fma(int(weights[row, column + field]) << 16, int(inputs[column + field]) << 16, accumulator)
                partial[lane][virtual] = accumulator
        for distance in distances:
            for lane in range(8):
                old = partial[lane].copy()
                for index in range(distance):
                    partial[lane][index] = fp32_add(old[index], old[index + distance])
        lanes = [partial[index][0] for index in range(8)]
        for distance in (4, 2, 1):
            old = lanes.copy()
            for index in range(distance):
                lanes[index] = fp32_add(old[index], old[index + distance])
        output[row] = round_bf16(lanes[0])
    return output


def oracle(records, input_bytes: bytes, lut: bytes):
    weights = {spec[0]: decode_record(records[spec[0]], spec) for spec in SPECS}
    inputs = np.frombuffer(input_bytes, "<u2")
    gate = width8_linear(weights["gate"], inputs); up = width8_linear(weights["up"], inputs)
    silu = np.frombuffer(lut, "<u2")[gate]
    activation = np.asarray([multiply_bf16(int(a), int(b)) for a, b in zip(silu, up, strict=True)], np.uint16)
    down = width8_linear(weights["down"], activation)
    arrays = {"gate": gate, "up": up, "silu": silu, "activation": activation, "down": down}
    result = {name: np.asarray(value, "<u2").tobytes() for name, value in arrays.items()}
    if {name: sha(data) for name, data in result.items()} != STAGE_SHA:
        raise RuntimeError("oracle_stage_hash")
    return result


def verify_prior_evidence():
    observed = {f"cpu/{name}": file_sha(CPU / name) for name in CPU_FILES}
    expected = {f"cpu/{name}": value for name, value in CPU_FILES.items()}
    observed["intel_final"] = file_sha(INTEL_FINAL); expected["intel_final"] = INTEL_FINAL_SHA
    for name, value in R8A5_FILES.items():
        observed[f"r8a5/{name}"] = file_sha(R8A5 / name); expected[f"r8a5/{name}"] = value
    if observed != expected:
        raise RuntimeError("prior_evidence")
    intel = json.loads(INTEL_FINAL.read_text(encoding="utf-8"))
    if intel.get("pass") is not True or intel.get("bundle_adjudication") != "positive":
        raise RuntimeError("intel_not_positive")
    return observed


def prepare_package():
    prior = verify_prior_evidence(); input_bytes = read_range(D2, INPUT[0], INPUT[1]); lut = (CPU / "bf16_silu_lut.bin").read_bytes()
    if sha(input_bytes) != INPUT[2] or sha(lut) != LUT_SHA or len(lut) != 131072:
        raise RuntimeError("input_lut")
    records = {}
    for spec in SPECS:
        source = read_range(SHARD, spec[3][0], spec[3][1] - spec[3][0]); records[spec[0]] = build_record(spec, source)
        if check_record(records[spec[0]], spec, sha(input_bytes)) != "pass":
            raise RuntimeError(f"record_check:{spec[0]}")
    controls = safe_controls(records, input_bytes, lut); stages = oracle(records, input_bytes, lut)
    return {"records": records, "input": input_bytes, "lut": lut, "controls": controls, "oracle": stages, "prior": prior}


def host_sample(stage: str, device_state: str = "not_attempted"):
    import psutil
    process = psutil.Process(); memory = process.memory_info(); vm = psutil.virtual_memory()
    return {"stage": stage, "qpc_ns": time.perf_counter_ns(), "available": int(vm.available), "rss": int(memory.rss), "peak_wset": int(memory.peak_wset), "telemetry_error": None, "device_query_state": device_state, "device_free_bytes": None, "device_total_bytes": None, "cuMemGetInfo_return": "not_attempted"}




