"""C3A-v2 scale-layout correction and discriminating native preflight.

C3A-v1 inherited C1's self-invertible K-block-major outer tile order.  The
native ABI uses the row-block-major order implemented by TorchAO ``to_blocked``.
This module is additive: it patches only the in-process C3A helper function and
leaves the failed v1 source/result historically intact.
"""
from __future__ import annotations

import hashlib
from typing import Any

REVISION = "c3a_v2_torchao_to_blocked_row_block_major"
LEGACY_COMMIT = "8d922ac50c3ccc6c45777af21d690c60df3b9536"


def ceilq(x: int, q: int) -> int:
    return ((int(x) + q - 1) // q) * q


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def to_blocked_scale_u8(torch, scale_raw: bytes, rows: int, cols: int):
    """Byte-for-byte mirror of TorchAO prototype/mx_formats/utils.py::to_blocked."""
    if len(scale_raw) != rows * cols:
        raise ValueError(f"scale payload size mismatch: {len(scale_raw)} != {rows * cols}")
    nrb = ceilq(rows, 128) // 128
    ncb = ceilq(cols, 4) // 4
    padded = torch.zeros((nrb * 128, ncb * 4), dtype=torch.uint8)
    natural = torch.frombuffer(bytearray(scale_raw), dtype=torch.uint8).reshape(rows, cols)
    padded[:rows, :cols] = natural
    blocks = padded.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten().contiguous()


def repack_b_scale(torch, scale_raw: bytes, n: int, k: int):
    sfk = k // 16
    sfp = ceilq(sfk, 4)
    npad = ceilq(n, 128)
    blocked = to_blocked_scale_u8(torch, scale_raw, n, sfk)
    # Preserve the 2.12.1 shape contract already accepted by C2b/C3A-v1;
    # SWIZZLE_32_4_4 semantics live in the contiguous flattened byte order.
    return blocked.to("cuda").view(torch.float8_e4m3fn).reshape(sfp, npad).contiguous()


def install(c3lib) -> None:
    """Patch C3A's scale repacker before any native call is constructed."""
    c3lib.repack_b_scale = repack_b_scale


def _reference_blocked(raw: bytes, rows: int, cols: int, *, legacy_k_major: bool) -> bytes:
    nrb = ceilq(rows, 128) // 128
    ncb = ceilq(cols, 4) // 4
    out = bytearray(nrb * ncb * 512)
    for r in range(rows):
        rb, rr = r // 128, r % 128
        r32, g32 = rr % 32, rr // 32
        for c in range(cols):
            cb, cc = c // 4, c % 4
            outer = cb * nrb + rb if legacy_k_major else rb * ncb + cb
            off = outer * 512 + ((r32 * 4 + g32) * 4 + cc)
            out[off] = raw[r * cols + c]
    return bytes(out)


def layout_witness(torch) -> dict[str, Any]:
    rows, cols = 256, 8  # exactly 2 row blocks x 2 scale-column blocks
    raw = bytes(((r * 13 + c * 29 + 17) % 251) for r in range(rows) for c in range(cols))
    actual = bytes(to_blocked_scale_u8(torch, raw, rows, cols).tolist())
    expected = _reference_blocked(raw, rows, cols, legacy_k_major=False)
    legacy = _reference_blocked(raw, rows, cols, legacy_k_major=True)
    mismatch = sum(a != b for a, b in zip(actual, expected))
    legacy_mismatch = sum(a != b for a, b in zip(legacy, expected))
    return {
        "revision": REVISION,
        "rows": rows,
        "cols": cols,
        "n_row_blocks": 2,
        "n_col_blocks": 2,
        "canonical_blocked_shape": [64, 32],
        "input_sha256": sha(raw),
        "actual_sha256": sha(actual),
        "expected_row_major_sha256": sha(expected),
        "byte_mismatches": mismatch,
        "legacy_k_major_byte_mismatches": legacy_mismatch,
        "passes": mismatch == 0 and legacy_mismatch > 0,
    }


def nonuniform_native_smoke(torch, F, ST, SW, c3lib) -> dict[str, Any]:
    """Native smoke that C3A-v1's K-major outer permutation cannot pass."""
    m, n, k = 2, 256, 128
    sfk = k // 16  # 8 scale columns => 2 scale-column blocks
    # Four distinct 128x4 scale tiles: 0.5, 1, 2, 4 in E4M3.
    tile_raw = ((0x30, 0x38), (0x40, 0x48))
    scale = bytearray(n * sfk)
    for r in range(n):
        rb = r // 128
        for c in range(sfk):
            scale[r * sfk + c] = tile_raw[rb][c // 4]

    a = c3lib.make_a(torch, m, k)
    b = c3lib.make_b(torch, bytes([0x22]) * (n * (k // 2)), bytes(scale), 0.5, n, k)
    out = c3lib.native_call(torch, F, ST, SW, a, b)
    torch.cuda.synchronize()

    expected = torch.empty((n,), dtype=torch.bfloat16, device="cuda")
    expected[:128] = 48.0   # 0.5 * 16 * (4*0.5 + 4*1.0)
    expected[128:] = 192.0  # 0.5 * 16 * (4*2.0 + 4*4.0)
    max_abs = float((out.float() - expected.float().reshape(1, -1)).abs().max().item())
    rec = {
        "revision": REVISION,
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        "finite": bool(torch.isfinite(out).all().item()),
        "expected_first_128": 48.0,
        "expected_second_128": 192.0,
        "all_equal_expected": bool(torch.all(out == expected.reshape(1, -1)).item()),
        "max_abs_error": max_abs,
    }
    rec["passes"] = bool(rec["finite"] and rec["all_equal_expected"] and max_abs == 0.0)
    del out, expected, a, b
    torch.cuda.empty_cache()
    return rec
