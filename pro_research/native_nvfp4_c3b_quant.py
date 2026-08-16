"""Dependency-light C3B NVFP4 activation quantizer and analysis helpers.

The quantization arithmetic mirrors pytorch/ao main (2026-08-16):
  torchao/prototype/mx_formats/nvfp4_tensor.py::per_tensor_amax_to_scale
  torchao/prototype/mx_formats/nvfp4_tensor.py::nvfp4_quantize
  torchao/prototype/custom_fp_utils.py::_f32_to_floatx_unpacked
  torchao/prototype/mx_formats/utils.py::to_blocked
Only PyTorch is required at runtime.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path
from typing import Any

from common import REPO

CAPTURE_DIR = REPO / "pro_research" / "results" / "native_nvfp4" / "c3b_capture"
CAPTURE = REPO / "pro_research" / "results" / "native_nvfp4" / "C3B_CAPTURE.json"
RESULT = REPO / "pro_research" / "results" / "native_nvfp4" / "C3B_W4A4_REAL_ACT.json"
VERIFY = REPO / "pro_research" / "results" / "native_nvfp4" / "C3B_W4A4_VERIFY.json"

F4_MAX = 6.0
F8_MAX = 448.0
M_VALUES = (1, 2, 4, 8)
COLD_L2_MULTIPLE = 4.0


def ceilq(x: int, q: int) -> int:
    return ((int(x) + q - 1) // q) * q


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_capture() -> dict[str, Any]:
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


def read_f32(torch, entry: dict[str, Any]):
    p = REPO / entry["path"]
    raw = p.read_bytes()
    if sha256_bytes(raw) != entry["sha256"]:
        raise RuntimeError(f"capture SHA mismatch: {entry['path']}")
    expected = math.prod(int(x) for x in entry["shape"]) * 4
    if len(raw) != expected:
        raise RuntimeError(f"capture size mismatch {entry['path']}: {len(raw)} != {expected}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(*entry["shape"]).clone()


def gpu_idle_snapshot() -> dict[str, Any]:
    """Narrow Windows/WDDM-aware idle preflight used by C3A-v2/C3B."""
    def run(cmd: list[str]) -> str:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=20)
        if p.returncode:
            raise RuntimeError(f"command failed {cmd}: {(p.stderr or p.stdout).strip()}")
        return (p.stdout or "").strip()

    apps = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits"])
    raw = [x.strip() for x in apps.splitlines() if x.strip()]
    ignored, blockers = [], []
    for line in raw:
        low = line.lower()
        if "chatgpt.exe" in low and "[n/a]" in low:
            ignored.append(line)
        else:
            blockers.append(line)
    if blockers:
        raise RuntimeError("competing CUDA process(es):\n  " + "\n  ".join(blockers))
    snap = run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"]).splitlines()[0]
    used_s, util_s = [x.strip() for x in snap.split(",")[:2]]
    used, util = int(used_s), int(util_s)
    if used > 1024 or util > 10:
        raise RuntimeError(f"GPU not idle: memory={used} MiB util={util}%")
    return {"compute_apps": raw, "ignored_wddm": ignored,
            "memory_used_mib": used, "utilization_percent": util}


# Exact RNE/ties-to-even FP32 -> E2M1 encoder, reduced from TorchAO custom_fp_utils.
def f32_to_f4_unpacked(torch, x):
    if x.dtype != torch.float32:
        x = x.float()
    ebits, mbits = 2, 1
    EBITS_F32, MBITS_F32 = 8, 23
    f32_bias = (1 << (EBITS_F32 - 1)) - 1
    exp_bias = (1 << (ebits - 1)) - 1
    max_int = (1 << (ebits + mbits)) - 1
    sign_mask = 1 << (ebits + mbits)
    magic_adder = (1 << (MBITS_F32 - mbits - 1)) - 1
    max_normal = 2 ** (((1 << ebits) - 1) - exp_bias) * (((1 << (mbits + 1)) - 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (f32_bias - exp_bias) + (MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32, device=x.device).view(torch.float32)
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
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    normal_x = normal_x + (((exp_bias - f32_bias) << MBITS_F32) + magic_adder) + mant_odd
    normal_x = (normal_x >> (MBITS_F32 - mbits)).to(torch.uint8)
    out = torch.full_like(xp, max_int, dtype=torch.uint8)
    out = torch.where(den, den_x, out)
    out = torch.where(norm, normal_x, out)
    sign_lp = (sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)).to(torch.uint8) & sign_mask
    return out | sign_lp


def pack_uint4(torch, codes):
    shape = tuple(codes.shape)
    if shape[-1] % 2:
        raise ValueError("last dim must be even")
    flat = codes.contiguous().view(-1)
    return (flat[::2] | (flat[1::2] << 4)).view(*shape[:-1], shape[-1] // 2)


def to_blocked(torch, scales):
    """TorchAO row-block-major scale layout; works on CPU or CUDA tensors."""
    rows, cols = (int(scales.shape[0]), int(scales.shape[1]))
    nrb, ncb = ceilq(rows, 128) // 128, ceilq(cols, 4) // 4
    pr, pc = nrb * 128, ncb * 4
    if (rows, cols) == (pr, pc):
        padded = scales
    else:
        padded = torch.zeros((pr, pc), dtype=scales.dtype, device=scales.device)
        padded[:rows, :cols] = scales
    blocks = padded.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten().contiguous()


def quantize_activation(torch, x):
    """Dynamic two-level NVFP4 quantization, TorchAO traditional path semantics."""
    if x.ndim != 2 or x.shape[1] % 16:
        raise ValueError(f"expected [M,K], K%16=0; got {tuple(x.shape)}")
    x = x.float().contiguous()
    m, k = int(x.shape[0]), int(x.shape[1])
    amax = torch.amax(torch.abs(x))
    per = amax / (F8_MAX * F4_MAX)
    # C3B uses real non-zero activations. Keep the upstream arithmetic exact;
    # do not invent an alternate all-zero special case here.
    blocks = x.reshape(m, k // 16, 16)
    block_scale = torch.amax(torch.abs(blocks), dim=-1) / F4_MAX
    scaled = block_scale.float() / per
    tiny = torch.finfo(torch.float8_e4m3fn).tiny
    scale8 = torch.clamp(scaled, min=tiny, max=F8_MAX).to(torch.float8_e4m3fn)
    scale32 = scale8.float()
    reciprocal = (1.0 / per) / scale32
    norm = torch.clamp(blocks * reciprocal.unsqueeze(-1), -F4_MAX, F4_MAX).reshape(m, k)
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
    }


def dequantize_activation(torch, q, k: int):
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       device=q["codes"].device, dtype=torch.float32)
    vals = lut[q["codes"].long()]
    scales = q["natural_scale"].float().repeat_interleave(16, dim=-1)
    return vals * scales * q["global"].float()


def tensor_metrics(torch, actual, ref) -> dict[str, float]:
    a, b = actual.float(), ref.float()
    d = a - b
    rmse = torch.sqrt(torch.mean(d * d))
    rrms = torch.sqrt(torch.mean(b * b))
    dot = torch.sum(a * b)
    an = torch.sqrt(torch.sum(a * a)); bn = torch.sqrt(torch.sum(b * b))
    cos = dot / torch.clamp(an * bn, min=1e-30)
    max_abs = torch.amax(torch.abs(d)); ref_max = torch.amax(torch.abs(b))
    return {"rmse": float(rmse.item()), "reference_rms": float(rrms.item()),
            "normalized_rmse": float((rmse / torch.clamp(rrms, min=1e-30)).item()),
            "cosine": float(cos.item()), "max_abs_error": float(max_abs.item()),
            "reference_max_abs": float(ref_max.item()),
            "normalized_max_abs_error": float((max_abs / torch.clamp(ref_max, min=1e-30)).item())}


def lm_distribution_metrics(torch, native, ref) -> dict[str, Any]:
    a, b = native.float(), ref.float()
    top1_a = torch.argmax(a, dim=-1); top1_b = torch.argmax(b, dim=-1)
    agree = top1_a == top1_b
    k = 5
    ta = torch.topk(a, k, dim=-1).indices; tb = torch.topk(b, k, dim=-1).indices
    overlap = (ta.unsqueeze(-1) == tb.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float() / k
    logp_b = torch.log_softmax(b, dim=-1); logp_a = torch.log_softmax(a, dim=-1)
    p_b = torch.exp(logp_b)
    kl = torch.sum(p_b * (logp_b - logp_a), dim=-1)
    return {
        "rows": int(a.shape[0]),
        "top1_agree": int(agree.sum().item()),
        "top1_agreement_fraction": float(agree.float().mean().item()),
        "reference_top1_ids": [int(x) for x in top1_b.cpu().tolist()],
        "native_top1_ids": [int(x) for x in top1_a.cpu().tolist()],
        "mean_top5_overlap": float(overlap.mean().item()),
        "min_top5_overlap": float(overlap.min().item()),
        "mean_kl_ref_to_native": float(kl.mean().item()),
        "max_kl_ref_to_native": float(kl.max().item()),
    }


def event_p50(torch, fn, reps: int, rounds: int = 5) -> dict[str, Any]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(rounds):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps):
            fn()
        en.record(); en.synchronize()
        vals.append(float(st.elapsed_time(en)) / reps)
    s = sorted(vals)
    return {"samples_ms": vals, "p50_ms": s[len(s)//2], "min_ms": s[0], "max_ms": s[-1], "reps": reps}


def parse_f32_scalar(raw: bytes) -> float:
    if len(raw) != 4:
        raise ValueError(f"expected 4-byte F32 scalar, got {len(raw)}")
    return float(struct.unpack("<f", raw)[0])
