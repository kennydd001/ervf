#!/usr/bin/env python3
"""Create the PH1 CPU-only normative LUT and pre-device stage freeze.

This script deliberately has no OpenCL, CUDA, CuPy, compiler, model-construction,
or network path. It reads exactly three official source ranges and one D2 row.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
import uuid
import zlib
from pathlib import Path

os.environ.update(
    CUDA_VISIBLE_DEVICES="-1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    USE_HUB_KERNELS="0",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

import mpmath as mp
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
SHARD = Path(
    r"C:/Users/de_do/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/"
    r"snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors"
)
D2 = ROOT / "reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors"
LUT_PATH = REPORTS / "het_next_l0_ph1_bf16_silu_lut.bin"
MATH_LUT_PATH = REPORTS / "het_next_l0_ph1_high_precision_silu_diagnostic.bin"
RAW_PATH = REPORTS / "het_next_l0_ph1_cpu_stage_freeze.safetensors"
RESULT_PATH = REPORTS / "het_next_l0_ph1_cpu_stage_freeze.json"

SHARD_BYTES = 3_999_619_288
SHARD_SHA = "8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a"
D2_BYTES = 171_696_126
D2_SHA = "f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd"
INPUT_OFFSET = 155_138_788
INPUT_BYTES = 4_096
INPUT_SHA = "5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f"
HEADER = struct.Struct("<4sHHHBBIIH2xIII28s")
HEADER_BYTES = 64
CODE_BYTES = 655_360
SCALE_BYTES = 16_384
PAD_BYTES = 4_032
RECORD_BYTES = 675_840
GROUP = 128

RECORDS = (
    {
        "projection": "gate",
        "ordinal": 0,
        "shape": (512, 2048),
        "absolute": (3_498_051_416, 3_500_148_568),
        "source_sha256": "05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c",
        "codes_sha256": "20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b",
        "scales_sha256": "658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8",
        "decoded_sha256": "9fd43163f4933920168ec9d356db90615a09ecac71198bcc7d3ae373fd995c77",
        "crc32": 1_976_639_022,
        "record_sha256": "e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9",
    },
    {
        "projection": "up",
        "ordinal": 1,
        "shape": (512, 2048),
        "absolute": (3_500_148_568, 3_502_245_720),
        "source_sha256": "4b36f661a351aaf907be1e041743833bc7a0564e07a6c140917ef1c8d69e4c0d",
        "codes_sha256": "6b2a3f124c3bc42d584b2816b063801d63244bd2a9e59cb00a32e339591e25cb",
        "scales_sha256": "c275fd13db6ea41ab8af1563a32a8de188e5fa488f91a6c7c939c4d3ca80a9f9",
        "decoded_sha256": "ca239543f7a478e757040a994d001a15b70481c7b87bca3cc8641831305394ea",
        "crc32": 4_920_057,
        "record_sha256": "6da7025af27de06c4f6011ddfc82672263b6f0593b2dcacf77705a443f44fbfb",
    },
    {
        "projection": "down",
        "ordinal": 2,
        "shape": (2048, 512),
        "absolute": (3_495_954_264, 3_498_051_416),
        "source_sha256": "bdf53c222b88c66b5845fd548ae984c20959231150b2fd34ddccf10d1777e479",
        "codes_sha256": "3d8782d588d507fea2a2c51ef8a3ea18ce6795d72b4be047b0c123652d77a703",
        "scales_sha256": "a3cd1a7c827dd9cb64925ad15299adbc18d74e592a1414504c3015e29854977e",
        "decoded_sha256": "ef9c19383d9b1ff90a4ba0015942594c4188dd42c407103a06f26a1953d56c34",
        "crc32": 4_066_311_128,
        "record_sha256": "bd1a8ef9ae689fefebf73408f3985c96a0725670dc0b0f7f46268a5a89d12157",
    },
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_exact(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        result = handle.read(size)
    if len(result) != size:
        raise EOFError((path, offset, size, len(result)))
    return result


def atomic_new(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".inprogress")
    if path.exists() or tmp.exists():
        raise FileExistsError(path)
    with tmp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    with path.open("r+b" if os.name == "nt" else "rb") as handle:
        os.fsync(handle.fileno())


def bf16_to_f32(words: np.ndarray) -> np.ndarray:
    return (np.asarray(words, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def quantize(source: bytes, rows: int, cols: int) -> tuple[bytes, bytes, bytes]:
    values = bf16_to_f32(np.frombuffer(source, "<u2")).reshape(rows, cols)
    blocks = values.reshape(rows, cols // GROUP, GROUP)
    maximum = np.max(np.abs(blocks), axis=-1, keepdims=True)
    scale = np.where(maximum > 0, np.asarray(maximum / np.float32(15), dtype=np.float32), np.float32(1))
    q = np.where(
        maximum > 0,
        np.clip(np.rint(np.asarray(blocks / scale, dtype=np.float32)), -15, 15),
        0,
    ).astype(np.int16)
    fields = (q + 15).astype(np.uint64).reshape(-1, 8)
    packed = np.bitwise_or.reduce(fields << (np.arange(8, dtype=np.uint64) * 5), axis=1)
    codes = np.stack([(packed >> (8 * index)) & 255 for index in range(5)], axis=1).astype(np.uint8).tobytes()
    scale_words = f32_to_bf16(scale.reshape(-1))
    scales = scale_words.astype("<u2", copy=False).tobytes()
    decoded = f32_to_bf16(
        q.reshape(rows, cols).astype(np.float32)
        * bf16_to_f32(scale_words).reshape(rows, cols // GROUP).repeat(GROUP, axis=1)
    ).astype("<u2", copy=False).tobytes()
    return codes, scales, decoded


def build_record(spec: dict, source: bytes) -> tuple[bytes, np.ndarray]:
    rows, cols = spec["shape"]
    if sha(source) != spec["source_sha256"]:
        raise ValueError("source_sha256:" + spec["projection"])
    codes, scales, decoded = quantize(source, rows, cols)
    if (sha(codes), sha(scales), sha(decoded)) != (
        spec["codes_sha256"],
        spec["scales_sha256"],
        spec["decoded_sha256"],
    ):
        raise ValueError("codec:" + spec["projection"])
    crc = zlib.crc32(scales, zlib.crc32(codes)) & 0xFFFFFFFF
    header = HEADER.pack(
        b"SQ5M", 1, 0, 50, spec["ordinal"], 5, rows, cols, GROUP, CODE_BYTES, SCALE_BYTES, crc, bytes(28)
    )
    record = header + codes + scales + bytes(PAD_BYTES)
    if len(record) != RECORD_BYTES or crc != spec["crc32"] or sha(record) != spec["record_sha256"]:
        raise ValueError("record:" + spec["projection"])
    return record, np.frombuffer(decoded, "<u2").copy().reshape(rows, cols)


def finite_mpf(word: int):
    sign = -1 if word >> 15 else 1
    exponent = (word >> 7) & 255
    fraction = word & 127
    if exponent == 255:
        return None
    if exponent == 0:
        return mp.mpf(sign) * mp.mpf(fraction) * mp.power(2, -133)
    return mp.mpf(sign) * mp.mpf(128 + fraction) * mp.power(2, exponent - 127 - 7)


def nearest_bf16(value, input_word: int) -> int:
    if value == 0:
        return 0x8000 if input_word == 0x8000 else 0
    sign = 0x8000 if value < 0 else 0
    absolute = abs(value)
    exponent = int(mp.floor(mp.log(absolute, 2)))
    if exponent < -126:
        exact = absolute * mp.power(2, 133)
        rounded = int(mp.floor(exact))
        remainder = exact - rounded
        if remainder > mp.mpf("0.5") or (remainder == mp.mpf("0.5") and (rounded & 1)):
            rounded += 1
        return sign | (0x80 if rounded >= 128 else rounded)
    biased = exponent + 127
    exact = absolute / mp.power(2, exponent) * 128
    rounded = int(mp.floor(exact))
    remainder = exact - rounded
    if remainder > mp.mpf("0.5") or (remainder == mp.mpf("0.5") and (rounded & 1)):
        rounded += 1
    if rounded == 256:
        rounded = 128
        biased += 1
    return sign | 0x7F80 if biased >= 255 else sign | (biased << 7) | (rounded - 128)


def build_luts() -> tuple[np.ndarray, np.ndarray, dict]:
    words = np.arange(65536, dtype=np.uint16)
    tensor = torch.from_numpy(words.copy()).view(torch.bfloat16)
    finite = torch.isfinite(tensor)
    normative = np.zeros(65536, dtype=np.uint16)
    normative[finite.numpy()] = F.silu(tensor[finite], inplace=False).view(torch.uint16).numpy()
    mp.mp.dps = 100
    mathematical = np.zeros(65536, dtype=np.uint16)
    for word in range(65536):
        value = finite_mpf(word)
        if value is not None:
            mathematical[word] = nearest_bf16(value / (1 + mp.exp(-value)), word)
    differences = np.flatnonzero(normative != mathematical)
    evidence = {
        "normative_sha256": sha(normative.astype("<u2", copy=False).tobytes()),
        "mathematical_sha256": sha(mathematical.astype("<u2", copy=False).tobytes()),
        "different_words": int(len(differences)),
        "first_differences": [
            [int(index), int(normative[index]), int(mathematical[index])] for index in differences[:32]
        ],
        "finite_inputs": int(finite.sum()),
        "nonfinite_zero_entries": int((~finite).sum()),
        "mpmath_dps": mp.mp.dps,
    }
    return normative, mathematical, evidence


def rshift_even(number: int, shift: int) -> int:
    if shift <= 0:
        return number << (-shift)
    quotient, remainder = divmod(number, 1 << shift)
    half = 1 << (shift - 1)
    return quotient + int(remainder > half or (remainder == half and (quotient & 1)))


def finite_f32(bits: int) -> tuple[int, int]:
    sign = -1 if bits >> 31 else 1
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7FFFFF
    if exponent == 255:
        raise ValueError("nonfinite_f32")
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
        if significand == (1 << 24):
            significand >>= 1
            shift += 1
        unbiased = exponent + shift + 23
        if unbiased > 127:
            return sign | 0x7F800000
        return sign | ((unbiased + 127) << 23) | (significand & 0x7FFFFF)
    fraction = rshift_even(number, -149 - exponent)
    if fraction == 0:
        return sign
    if fraction >= (1 << 23):
        return sign | (1 << 23)
    return sign | fraction


def fma_bits(a_bits: int, b_bits: int, c_bits: int) -> int:
    a_number, a_exponent = finite_f32(a_bits)
    b_number, b_exponent = finite_f32(b_bits)
    c_number, c_exponent = finite_f32(c_bits)
    product_number, product_exponent = a_number * b_number, a_exponent + b_exponent
    common_exponent = min(product_exponent, c_exponent)
    return pack_exact(
        (product_number << (product_exponent - common_exponent))
        + (c_number << (c_exponent - common_exponent)),
        common_exponent,
    )


def add_bits(a_bits: int, b_bits: int) -> int:
    return fma_bits(a_bits, 0x3F800000, b_bits)


def f32_bits_to_bf16(bits: int) -> int:
    if (bits & 0x7F800000) == 0x7F800000:
        raise ValueError("nonfinite_round")
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def bf16_mul(a_word: int, b_word: int) -> int:
    if (a_word & 0x7F80) == 0x7F80 or (b_word & 0x7F80) == 0x7F80:
        raise ValueError("nonfinite_bf16_multiply")
    if (a_word & 0x7FFF) == 0 or (b_word & 0x7FFF) == 0:
        return ((a_word ^ b_word) & 0x8000)
    return f32_bits_to_bf16(fma_bits(a_word << 16, b_word << 16, 0))


def q5_linear(weight_words: np.ndarray, input_words: np.ndarray) -> np.ndarray:
    rows, columns = weight_words.shape
    if columns not in (512, 2048) or len(input_words) != columns:
        raise ValueError("linear_shape")
    virtual_count = columns // 64
    partial_tree = (16, 8, 4, 2, 1) if columns == 2048 else (4, 2, 1)
    result = np.empty(rows, dtype=np.uint16)
    for row in range(rows):
        partials = [[0] * virtual_count for _ in range(8)]
        for lane in range(8):
            for virtual in range(virtual_count):
                column = (lane + 8 * virtual) * 8
                accumulator = 0
                for field in range(8):
                    accumulator = fma_bits(
                        int(weight_words[row, column + field]) << 16,
                        int(input_words[column + field]) << 16,
                        accumulator,
                    )
                partials[lane][virtual] = accumulator
        for distance in partial_tree:
            for lane in range(8):
                old = partials[lane].copy()
                for index in range(distance):
                    partials[lane][index] = add_bits(old[index], old[index + distance])
        lane_values = [partials[lane][0] for lane in range(8)]
        for offset in (4, 2, 1):
            old = lane_values.copy()
            for lane in range(offset):
                lane_values[lane] = add_bits(old[lane], old[lane + offset])
        result[row] = f32_bits_to_bf16(lane_values[0])
    return result


def metric(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference_values = bf16_to_f32(reference.reshape(-1)).astype(np.float64)
    candidate_values = bf16_to_f32(candidate.reshape(-1)).astype(np.float64)
    ref_squared = 0.0
    err_squared = 0.0
    maximum = 0.0
    for ref, cand in zip(reference_values, candidate_values, strict=True):
        difference = float(cand - ref)
        ref_squared += float(ref * ref)
        err_squared += difference * difference
        maximum = max(maximum, abs(difference))
    error_norm = math.sqrt(err_squared)
    reference_norm = math.sqrt(ref_squared)
    relative = 0.0 if reference_norm == 0 and error_norm == 0 else math.inf if reference_norm == 0 else error_norm / reference_norm
    return {
        "max_abs": maximum,
        "rel_l2": relative,
        "different_words": int(np.count_nonzero(reference.reshape(-1) != candidate.reshape(-1))),
    }


def main() -> int:
    for path in (LUT_PATH, MATH_LUT_PATH, RAW_PATH, RESULT_PATH):
        if path.exists():
            raise FileExistsError(path)
    # Full payload hashing is forbidden here: immutable upstream evidence binds
    # the official file digest, while this phase opens only three allowlisted
    # source ranges and validates their individual SHA-256 values.
    if SHARD.stat().st_size != SHARD_BYTES:
        raise ValueError("shard_identity")
    if D2.stat().st_size != D2_BYTES:
        raise ValueError("d2_identity")

    try:
        psutil.Process().cpu_affinity(list(range(16)))
    except Exception as exc:
        raise RuntimeError("cpu_affinity") from exc
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mkldnn.enabled = True
    torch.set_flush_denormal(False)

    started = time.perf_counter()
    input_bytes = read_exact(D2, INPUT_OFFSET, INPUT_BYTES)
    if sha(input_bytes) != INPUT_SHA:
        raise ValueError("input_sha256")
    input_words = np.frombuffer(input_bytes, "<u2").copy()

    sources: dict[str, torch.Tensor] = {}
    decoded: dict[str, np.ndarray] = {}
    records: dict[str, bytes] = {}
    record_evidence = []
    for spec in RECORDS:
        start, end = spec["absolute"]
        source = read_exact(SHARD, start, end - start)
        record, words = build_record(spec, source)
        projection = spec["projection"]
        sources[projection] = torch.from_numpy(np.frombuffer(source, "<u2").copy()).view(torch.bfloat16).reshape(spec["shape"])
        decoded[projection] = words
        records[projection] = record
        record_evidence.append(
            {
                "projection": projection,
                "record_sha256": sha(record),
                "record_bytes": len(record),
                "absolute": list(spec["absolute"]),
            }
        )

    normative_lut, mathematical_lut, lut_evidence = build_luts()
    if lut_evidence["normative_sha256"] != "a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed":
        raise ValueError("normative_lut_sha256")

    x = torch.from_numpy(input_words.copy()).view(torch.bfloat16)
    with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
        fused = torch.cat((sources["gate"], sources["up"]), dim=0).contiguous()
        gate_up = F.linear(x.contiguous(), fused)
        source_gate, source_up = gate_up.chunk(2, dim=-1)
        source_silu = F.silu(source_gate, inplace=False)
        source_activation = source_silu * source_up
        source_down = F.linear(source_activation.contiguous(), sources["down"])

    q5_gate = q5_linear(decoded["gate"], input_words)
    q5_up = q5_linear(decoded["up"], input_words)
    if np.any(((q5_gate >> 7) & 255) == 255) or np.any(((q5_up >> 7) & 255) == 255):
        raise ValueError("nonfinite_q5_linear")
    q5_silu = normative_lut[q5_gate]
    q5_activation = np.asarray([bf16_mul(int(a), int(b)) for a, b in zip(q5_silu, q5_up, strict=True)], dtype=np.uint16)
    q5_down = q5_linear(decoded["down"], q5_activation)

    arrays = {
        "natural_input": x,
        "source_gate_up": gate_up,
        "source_gate": source_gate,
        "source_up": source_up,
        "source_silu": source_silu,
        "source_activation": source_activation,
        "source_down": source_down,
        "cpu_q5_gate": torch.from_numpy(q5_gate.copy()).view(torch.bfloat16),
        "cpu_q5_up": torch.from_numpy(q5_up.copy()).view(torch.bfloat16),
        "cpu_q5_silu": torch.from_numpy(q5_silu.copy()).view(torch.bfloat16),
        "cpu_q5_activation": torch.from_numpy(q5_activation.copy()).view(torch.bfloat16),
        "cpu_q5_down": torch.from_numpy(q5_down.copy()).view(torch.bfloat16),
    }
    if not all(bool(torch.isfinite(tensor).all()) for tensor in arrays.values()):
        raise ValueError("nonfinite_stage")
    stage_hashes = {
        name: sha(tensor.contiguous().view(torch.uint8).numpy().tobytes()) for name, tensor in arrays.items()
    }
    quality = metric(source_down.view(torch.uint16).numpy(), q5_down)
    quality_pass = bool(math.isfinite(quality["rel_l2"]) and quality["rel_l2"] <= 0.08)

    raw_tmp = RAW_PATH.with_name(RAW_PATH.name + "." + uuid.uuid4().hex + ".inprogress")
    save_file({name: tensor.contiguous() for name, tensor in arrays.items()}, raw_tmp)
    with raw_tmp.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(raw_tmp, RAW_PATH)
    atomic_new(LUT_PATH, normative_lut.astype("<u2", copy=False).tobytes())
    atomic_new(MATH_LUT_PATH, mathematical_lut.astype("<u2", copy=False).tobytes())

    dependency_paths = [
        ROOT / ".venv-next-ref/Lib/site-packages/transformers/models/qwen3_next/modeling_qwen3_next.py",
        ROOT / ".venv-next-ref/Lib/site-packages/transformers/activations.py",
        ROOT / "reports/streamq5_moe/port80b_t0r4_dependency_execution_lock.json",
    ]
    result = {
        "kind": "het_next_l0_ph1_cpu_stage_freeze",
        "status": "cpu_predevice_positive" if quality_pass else "cpu_predevice_quality_negative",
        "positive": quality_pass,
        "generator_sha256": file_sha(Path(__file__)),
        "provenance": {str(path.relative_to(ROOT)): file_sha(path) for path in dependency_paths},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "mpmath": mp.__version__,
            "cpu_identity": platform.processor(),
            "affinity": psutil.Process().cpu_affinity(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "flush_denormal": False,
            "inference_mode": True,
            "autocast": False,
        },
        "input_sha256": INPUT_SHA,
        "upstream_file_identities_not_rehashed": {"shard_sha256": SHARD_SHA, "d2_sha256": D2_SHA},
        "record_evidence": record_evidence,
        "lut": lut_evidence,
        "stage_hashes": stage_hashes,
        "quality": quality,
        "quality_threshold": {"metric": "rel_l2", "maximum": 0.08, "pass": quality_pass},
        "raw_sha256": file_sha(RAW_PATH),
        "normative_lut_file_sha256": file_sha(LUT_PATH),
        "mathematical_lut_file_sha256": file_sha(MATH_LUT_PATH),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": "CPU-only predevice freeze for one known expert/input; no device/full-MoE/model/performance claim.",
    }
    atomic_new(RESULT_PATH, json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n")
    print(json.dumps({"status": result["status"], "quality": quality, "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
    return 0 if quality_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
