"""Standalone dual-convention NVFP4 activation quantizers for C3B-v2.

TORCHAO_RNE mirrors TorchAO's traditional path.
CUDNN_CEIL changes one thing only: E4M3 block scales round upward.

The hot quantizer performs no host `.item()` calls. Diagnostics are separate.
"""
from __future__ import annotations

import math
from typing import Any

F4_MAX = 6.0
F8_MAX = 448.0


def ceilq(x: int, q: int) -> int:
    return ((int(x) + q - 1) // q) * q


# Exact RNE/ties-to-even FP32 -> E2M1 encoder, reduced from TorchAO.
def f32_to_f4_unpacked(torch, x):
    if x.dtype != torch.float32:
        x = x.float()
    ebits, mbits = 2, 1
    ebits_f32, mbits_f32 = 8, 23
    f32_bias = (1 << (ebits_f32 - 1)) - 1
    exp_bias = (1 << (ebits - 1)) - 1
    max_int = (1 << (ebits + mbits)) - 1
    sign_mask = 1 << (ebits + mbits)
    magic_adder = (1 << (mbits_f32 - mbits - 1)) - 1
    max_normal = (
        2 ** (((1 << ebits) - 1) - exp_bias)
        * (((1 << (mbits + 1)) - 1) / (2**mbits))
    )
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (f32_bias - exp_bias) + (mbits_f32 - mbits) + 1
    denorm_mask_int = denorm_exp << mbits_f32
    denorm_mask_float = torch.tensor(
        denorm_mask_int, dtype=torch.int32, device=x.device
    ).view(torch.float32)
    xi = x.view(torch.int32)
    sign = xi & -2147483648
    xi = xi ^ sign
    xp = xi.view(torch.float32)
    sat = xp >= max_normal
    den = (~sat) & (xp < min_normal)
    norm = ~(sat | den)
    den_x = (xp + denorm_mask_float).view(torch.int32) - denorm_mask_int
    den_x = den_x.to(torch.uint8)
    normal_x = xp.view(torch.int32)
    mant_odd = (normal_x >> (mbits_f32 - mbits)) & 1
    normal_x = (
        normal_x
        + (((exp_bias - f32_bias) << mbits_f32) + magic_adder)
        + mant_odd
    )
    normal_x = (normal_x >> (mbits_f32 - mbits)).to(torch.uint8)
    out = torch.full_like(xp, max_int, dtype=torch.uint8)
    out = torch.where(den, den_x, out)
    out = torch.where(norm, normal_x, out)
    sign_lp = (
        sign >> (mbits_f32 + ebits_f32 - mbits - ebits)
    ).to(torch.uint8) & sign_mask
    return out | sign_lp


def pack_uint4(torch, codes):
    shape = tuple(codes.shape)
    if shape[-1] % 2:
        raise ValueError("last dim must be even")
    flat = codes.contiguous().view(-1)
    return (flat[::2] | (flat[1::2] << 4)).view(
        *shape[:-1], shape[-1] // 2
    )


def to_blocked(torch, scales):
    """TorchAO row-block-major scale layout; CPU or CUDA."""
    rows, cols = int(scales.shape[0]), int(scales.shape[1])
    nrb, ncb = ceilq(rows, 128) // 128, ceilq(cols, 4) // 4
    pr, pc = nrb * 128, ncb * 4
    if (rows, cols) == (pr, pc):
        padded = scales
    else:
        padded = torch.zeros(
            (pr, pc), dtype=scales.dtype, device=scales.device
        )
        padded[:rows, :cols] = scales
    blocks = padded.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return (
        blocks.reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1, 32, 16)
        .flatten()
        .contiguous()
    )


def _decode_e4m3_positive(raw: int) -> float:
    exp, man = (raw >> 3) & 0xF, raw & 0x7
    if exp == 0xF and man == 0x7:
        return math.nan
    if exp == 0:
        return (2.0 ** -6) * (man / 8.0)
    return (2.0 ** (exp - 7)) * (1.0 + man / 8.0)


_POSITIVE = sorted(
    {
        (float(_decode_e4m3_positive(raw)), raw)
        for raw in range(0x00, 0x7F)
        if math.isfinite(_decode_e4m3_positive(raw))
    },
    key=lambda x: (x[0], x[1]),
)
_LUT_CACHE = {}


def _lut(torch, device):
    key = str(device)
    hit = _LUT_CACHE.get(key)
    if hit is None:
        hit = (
            torch.tensor(
                [x[0] for x in _POSITIVE],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [x[1] for x in _POSITIVE],
                dtype=torch.uint8,
                device=device,
            ),
        )
        _LUT_CACHE[key] = hit
    return hit


def e4m3_ceil(torch, values):
    """Smallest finite positive E4M3 value >= each non-negative input."""
    vals, raws = _lut(torch, values.device)
    x = values.float().clamp(min=0.0, max=F8_MAX)
    idx = torch.searchsorted(vals, x.reshape(-1), right=False)
    idx = idx.clamp(max=vals.numel() - 1)
    raw = raws[idx].reshape(x.shape).contiguous()
    return raw.view(torch.float8_e4m3fn)


def quantize_activation(torch, x, convention: str):
    if convention not in {"TORCHAO_RNE", "CUDNN_CEIL"}:
        raise ValueError(f"unknown C3B-v2 convention {convention!r}")
    if x.ndim != 2 or int(x.shape[1]) % 16:
        raise ValueError(f"expected [M,K], K%16=0; got {tuple(x.shape)}")

    x = x.float().contiguous()
    m, k = int(x.shape[0]), int(x.shape[1])
    amax = torch.amax(torch.abs(x))
    per = torch.clamp(amax / (F8_MAX * F4_MAX), min=1.0e-30)
    blocks = x.reshape(m, k // 16, 16)
    desired = (torch.amax(torch.abs(blocks), dim=-1) / F4_MAX) / per
    tiny = torch.finfo(torch.float8_e4m3fn).tiny
    desired = desired.clamp(min=tiny, max=F8_MAX)
    if convention == "TORCHAO_RNE":
        scale8 = desired.to(torch.float8_e4m3fn)
    else:
        scale8 = e4m3_ceil(torch, desired)

    reciprocal = (1.0 / per) / scale8.float()
    unbounded = blocks * reciprocal.unsqueeze(-1)
    norm = torch.clamp(
        unbounded, -F4_MAX, F4_MAX
    ).reshape(m, k)
    codes = f32_to_f4_unpacked(torch, norm)
    packed = pack_uint4(torch, codes).contiguous()
    blocked = to_blocked(torch, scale8)
    sfp = ceilq(k // 16, 4)
    scale_phys = blocked.reshape(ceilq(m, 128), sfp).contiguous()
    return {
        "u8": packed,
        "fp4": packed.view(torch.float4_e2m1fn_x2),
        "block": scale_phys,
        "global": per.reshape(1).to(torch.float32),
        "natural_scale": scale8,
        "codes": codes,
        "convention": convention,
    }


def diagnostics(torch, x, q) -> dict[str, Any]:
    """Diagnostic-only host synchronization; never call in a timed region."""
    blocks = x.float().reshape(
        int(x.shape[0]), int(x.shape[1]) // 16, 16
    )
    eff = q["natural_scale"].float() * q["global"].float()
    scaled = blocks / eff.unsqueeze(-1)
    mask = torch.abs(scaled) > F4_MAX
    return {
        "convention": q.get("convention"),
        "preclip_count_recomputed": int(mask.sum().item()),
        "preclip_fraction_recomputed": float(mask.float().mean().item()),
        "preclip_max_abs_recomputed": float(
            torch.abs(scaled).amax().item()
        ),
    }
