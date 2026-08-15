"""Codec tests for the LIGHTNINGSTREAM_NEMOTRON NVFP4 implementation.

These test the codec against the published format definition only.  They do not
touch the checkpoint and make no claim about the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from moe_lab.lightningstream_nemotron import nvfp4


ALL_CODES = np.arange(16, dtype=np.uint8)
ALL_BYTES = np.arange(256, dtype=np.uint8)


def test_e2m1_magnitudes_match_published_set():
    values = nvfp4.decode_e2m1_table(ALL_CODES)
    positives = np.unique(np.abs(values))
    assert positives.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def test_e2m1_two_implementations_agree_on_every_code():
    table = nvfp4.decode_e2m1_table(ALL_CODES)
    bits = nvfp4.decode_e2m1_bits(ALL_CODES)
    assert np.array_equal(table, bits)


def test_e2m1_sign_symmetry():
    values = nvfp4.decode_e2m1_table(ALL_CODES)
    assert np.array_equal(values[:8], -values[8:])


def test_e4m3_two_implementations_agree_on_every_byte():
    table = nvfp4.decode_e4m3_table(ALL_BYTES)
    bits = nvfp4.decode_e4m3_bits(ALL_BYTES)
    both_nan = np.isnan(table) & np.isnan(bits)
    assert np.array_equal(table[~both_nan], bits[~both_nan])
    assert both_nan.sum() == 2  # 0x7F and 0xFF


def test_e4m3_max_finite_is_448():
    values = nvfp4.decode_e4m3_table(ALL_BYTES)
    assert np.nanmax(values) == nvfp4.E4M3_MAX


def test_e4m3_smallest_subnormal():
    # 0x01 -> 2**-6 * 1/8 == 2**-9
    assert nvfp4.decode_e4m3_table(np.array([1], dtype=np.uint8))[0] == 2.0 ** -9


@pytest.mark.parametrize("order", ["low_first", "high_first"])
def test_nibble_pack_unpack_round_trip(order):
    rng = np.random.default_rng(20260814)
    codes = rng.integers(0, 16, size=4096, dtype=np.uint8)
    packed = nvfp4.pack_nibbles(codes, order)
    assert packed.size == codes.size // 2
    assert np.array_equal(nvfp4.unpack_nibbles(packed, order), codes)


def test_nibble_orders_are_distinguishable():
    codes = np.array([1, 2], dtype=np.uint8)
    assert nvfp4.pack_nibbles(codes, "low_first")[0] != nvfp4.pack_nibbles(codes, "high_first")[0]


def test_encode_decode_round_trip_is_bit_exact():
    values = nvfp4.decode_e2m1_table(ALL_CODES)
    recoded = nvfp4.encode_e2m1(values)
    # -0.0 collapses onto +0 by definition; every other code must survive.
    expected = ALL_CODES.copy()
    expected[8] = 0
    assert np.array_equal(recoded, expected)


def test_encode_rejects_unrepresentable_value():
    with pytest.raises(ValueError, match="not exactly E2M1-representable"):
        nvfp4.encode_e2m1(np.array([0.75]))


def test_dequantize_shapes_and_scaling():
    rng = np.random.default_rng(7)
    n_weights = nvfp4.GROUP_SIZE * 64
    codes = rng.integers(0, 16, size=n_weights, dtype=np.uint8)
    packed = nvfp4.pack_nibbles(codes)
    # 0x38 is E4M3 for 1.0, so the block scale is a no-op here.
    scales = np.full(n_weights // nvfp4.GROUP_SIZE, 0x38, dtype=np.uint8)

    out = nvfp4.dequantize(packed, scales, 1.0)
    assert out.shape == (n_weights,)
    assert np.array_equal(out, nvfp4.decode_e2m1_table(codes))

    scaled = nvfp4.dequantize(packed, scales, 2.5)
    assert np.array_equal(scaled, out * 2.5)


def test_dequantize_implementations_agree():
    rng = np.random.default_rng(11)
    n_weights = nvfp4.GROUP_SIZE * 512
    packed = rng.integers(0, 256, size=n_weights // 2, dtype=np.uint8)
    # Exclude the two NaN encodings so equality is well defined.
    scales = rng.integers(0, 126, size=n_weights // nvfp4.GROUP_SIZE, dtype=np.uint8)

    a = nvfp4.dequantize(packed, scales, 0.125, implementation="table")
    b = nvfp4.dequantize(packed, scales, 0.125, implementation="bits")
    assert np.array_equal(a, b)


def test_dequantize_rejects_scale_count_mismatch():
    packed = np.zeros(8, dtype=np.uint8)
    with pytest.raises(ValueError, match="!= weight count"):
        nvfp4.dequantize(packed, np.zeros(3, dtype=np.uint8), 1.0)


def test_decoded_values_lie_in_representable_set():
    rng = np.random.default_rng(3)
    n_weights = nvfp4.GROUP_SIZE * 32
    packed = rng.integers(0, 256, size=n_weights // 2, dtype=np.uint8)
    scale_byte = np.uint8(0x3C)  # E4M3 1.5
    scales = np.full(n_weights // nvfp4.GROUP_SIZE, scale_byte, dtype=np.uint8)
    global_scale = 0.03125

    out = nvfp4.dequantize(packed, scales, global_scale)
    allowed = nvfp4.representable_set(
        nvfp4.decode_e4m3_table(np.array([scale_byte]))[0], global_scale
    )
    assert np.isin(out, allowed).all()


def test_code_out_of_range_rejected():
    with pytest.raises(ValueError, match="out of range"):
        nvfp4.decode_e2m1_table(np.array([16], dtype=np.uint8))
