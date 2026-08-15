"""NVFP4 (E2M1) and FP8-E4M3 codecs, in two independent implementations.

The N2 preregistration requires two decoders written from the published format —
one table-driven, one bit-arithmetic — that agree on every sampled code.  Two
implementations that share a helper would share its bugs, so the arithmetic path
below deliberately derives values from bit fields without consulting any table,
and the table path derives them from the format definition without doing bit
arithmetic at decode time.

Formats
-------
E2M1 (NVFP4 element): 1 sign bit, 2 exponent bits, 1 mantissa bit, exponent
bias 1.  Normals are ``2**(e-1) * (1 + m/2)``; the ``e == 0`` case is subnormal,
``2**0 * (m/2)``.  The representable magnitudes are therefore exactly
``{0, 0.5, 1, 1.5, 2, 3, 4, 6}``.  There is no infinity and no NaN.

E4M3 (block scale): 1 sign bit, 4 exponent bits, 3 mantissa bits, exponent
bias 7, in the OCP/NVIDIA variant that has no infinities and encodes NaN only at
``0x7F`` / ``0xFF``.  Normals are ``2**(e-7) * (1 + m/8)``; ``e == 0`` gives
subnormals ``2**-6 * (m/8)``.  Maximum finite magnitude is 448.

Dequantization hypothesis (ModelOpt NVFP4, group size 16):

    w = e2m1(code) * e4m3(block_scale) * f32(global_scale)

``global_scale`` is ModelOpt's ``weight_scale_2``.  This grouping, and the
nibble order below, are recorded as assumptions in N2 and must be confirmed
against the official modeling code before any full-model claim.
"""

from __future__ import annotations

import numpy as np

GROUP_SIZE = 16
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0

# Nibble order within a packed byte.  "low_first" means element 2*i is the low
# nibble and element 2*i+1 is the high nibble.  ModelOpt/TensorRT-LLM use this
# order; it is a parameter here so N2/N3 can falsify it rather than assume it.
DEFAULT_NIBBLE_ORDER = "low_first"


# --------------------------------------------------------------------------
# Implementation A: table-driven
# --------------------------------------------------------------------------

def _build_e2m1_table() -> np.ndarray:
    """16-entry table indexed by the raw 4-bit code."""
    table = np.zeros(16, dtype=np.float64)
    for code in range(16):
        sign = -1.0 if (code >> 3) & 1 else 1.0
        magnitude = E2M1_MAGNITUDES[code & 0x7]
        table[code] = sign * magnitude
    return table


def _build_e4m3_table() -> np.ndarray:
    """256-entry table indexed by the raw byte; NaN at 0x7F and 0xFF."""
    table = np.zeros(256, dtype=np.float64)
    for byte in range(256):
        sign = -1.0 if (byte >> 7) & 1 else 1.0
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0xF and mantissa == 0x7:
            table[byte] = np.nan
        elif exponent == 0:
            table[byte] = sign * (2.0 ** -6) * (mantissa / 8.0)
        else:
            table[byte] = sign * (2.0 ** (exponent - 7)) * (1.0 + mantissa / 8.0)
    return table


E2M1_TABLE = _build_e2m1_table()
E4M3_TABLE = _build_e4m3_table()


def decode_e2m1_table(codes: np.ndarray) -> np.ndarray:
    """Decode 4-bit codes (values 0..15) via lookup."""
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.size and (codes > 15).any():
        raise ValueError("E2M1 code out of range 0..15")
    return E2M1_TABLE[codes]


def decode_e4m3_table(raw: np.ndarray) -> np.ndarray:
    """Decode FP8-E4M3 bytes via lookup."""
    return E4M3_TABLE[np.asarray(raw, dtype=np.uint8)]


# --------------------------------------------------------------------------
# Implementation B: bit arithmetic, no tables consulted
# --------------------------------------------------------------------------

def decode_e2m1_bits(codes: np.ndarray) -> np.ndarray:
    """Decode 4-bit codes by evaluating the format definition directly."""
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.size and (codes > 15).any():
        raise ValueError("E2M1 code out of range 0..15")
    sign = np.where((codes >> 3) & 1, -1.0, 1.0)
    exponent = ((codes >> 1) & 0x3).astype(np.int64)
    mantissa = (codes & 0x1).astype(np.float64)

    subnormal = mantissa * 0.5                      # 2**0 * (m/2)
    normal = np.exp2((exponent - 1).astype(np.float64)) * (1.0 + mantissa * 0.5)
    magnitude = np.where(exponent == 0, subnormal, normal)
    return sign * magnitude


def decode_e4m3_bits(raw: np.ndarray) -> np.ndarray:
    """Decode FP8-E4M3 bytes by evaluating the format definition directly."""
    raw = np.asarray(raw, dtype=np.uint8)
    sign = np.where((raw >> 7) & 1, -1.0, 1.0)
    exponent = ((raw >> 3) & 0xF).astype(np.int64)
    mantissa = (raw & 0x7).astype(np.float64)

    subnormal = np.exp2(-6.0) * (mantissa / 8.0)
    normal = np.exp2((exponent - 7).astype(np.float64)) * (1.0 + mantissa / 8.0)
    value = sign * np.where(exponent == 0, subnormal, normal)
    return np.where((exponent == 0xF) & (mantissa == 0x7), np.nan, value)


# --------------------------------------------------------------------------
# Packing / unpacking
# --------------------------------------------------------------------------

def unpack_nibbles(packed: np.ndarray, nibble_order: str = DEFAULT_NIBBLE_ORDER) -> np.ndarray:
    """Expand packed bytes into 2N 4-bit codes."""
    packed = np.asarray(packed, dtype=np.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = np.empty(packed.size * 2, dtype=np.uint8)
    if nibble_order == "low_first":
        out[0::2], out[1::2] = low, high
    elif nibble_order == "high_first":
        out[0::2], out[1::2] = high, low
    else:
        raise ValueError(f"unknown nibble_order {nibble_order!r}")
    return out


def pack_nibbles(codes: np.ndarray, nibble_order: str = DEFAULT_NIBBLE_ORDER) -> np.ndarray:
    """Inverse of :func:`unpack_nibbles`."""
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.size % 2:
        raise ValueError("code count must be even to pack into bytes")
    if codes.size and (codes > 15).any():
        raise ValueError("E2M1 code out of range 0..15")
    first, second = codes[0::2], codes[1::2]
    if nibble_order == "low_first":
        return (first | (second << 4)).astype(np.uint8)
    if nibble_order == "high_first":
        return (second | (first << 4)).astype(np.uint8)
    raise ValueError(f"unknown nibble_order {nibble_order!r}")


def encode_e2m1(values: np.ndarray) -> np.ndarray:
    """Map exact E2M1-representable magnitudes back to their 4-bit codes.

    This is deliberately strict: it is used for a bit-exact round trip, so any
    value that is not exactly representable is an error rather than something to
    round.  Negative zero encodes to +0 because E2M1's two zero encodings are
    not distinguished by the magnitude table.
    """
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.abs(values)
    codes = np.zeros(values.shape, dtype=np.uint8)
    matched = np.zeros(values.shape, dtype=bool)

    for index, target in enumerate(E2M1_MAGNITUDES):
        hit = magnitude == target
        codes[hit] = index
        matched |= hit

    if not matched.all():
        bad = values[~matched]
        raise ValueError(
            f"{bad.size} value(s) are not exactly E2M1-representable, "
            f"first={bad.flat[0] if bad.size else None}"
        )

    negative = (values < 0) & (magnitude != 0.0)
    codes[negative] |= 0x8
    return codes


# --------------------------------------------------------------------------
# Dequantization
# --------------------------------------------------------------------------

def dequantize(
    packed_codes: np.ndarray,
    block_scales_fp8: np.ndarray,
    global_scale: float,
    *,
    group_size: int = GROUP_SIZE,
    nibble_order: str = DEFAULT_NIBBLE_ORDER,
    implementation: str = "table",
) -> np.ndarray:
    """Dequantize one NVFP4 matrix to float64.

    ``packed_codes`` holds ``N/2`` bytes, ``block_scales_fp8`` holds ``N/group``
    FP8-E4M3 bytes, and ``global_scale`` is the FP32 ``weight_scale_2``.
    """
    codes = unpack_nibbles(packed_codes, nibble_order)
    n_weights = codes.size

    if block_scales_fp8.size * group_size != n_weights:
        raise ValueError(
            f"scale count {block_scales_fp8.size} x group {group_size} "
            f"!= weight count {n_weights}"
        )

    if implementation == "table":
        elements = decode_e2m1_table(codes)
        scales = decode_e4m3_table(block_scales_fp8)
    elif implementation == "bits":
        elements = decode_e2m1_bits(codes)
        scales = decode_e4m3_bits(block_scales_fp8)
    else:
        raise ValueError(f"unknown implementation {implementation!r}")

    expanded = np.repeat(scales, group_size)
    return elements * expanded * float(global_scale)


def representable_set(block_scale: float, global_scale: float) -> np.ndarray:
    """Every value a single block can take, for the range-invariance check."""
    magnitudes = np.array(E2M1_MAGNITUDES, dtype=np.float64)
    signed = np.concatenate([magnitudes, -magnitudes])
    return np.unique(signed * block_scale * global_scale)
