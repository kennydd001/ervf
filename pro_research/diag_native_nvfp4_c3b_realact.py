"""C3B: real causal activations -> auditable NVFP4 A -> native SM120 scaled_mm.

The A quantizer is intentionally written from ordinary PyTorch tensor ops. Its
numerics are the experiment; its timing is an explicit upper bound, not a
production implementation. C3C owns fusion/preallocation.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import struct
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from diag_native_nvfp4_c3a_real_weight_v2 import require_gpu_idle_wddm
import native_nvfp4_c3a_lib as c3lib
import native_nvfp4_c3a_layout_v2 as c3v2

CAP_DIR = REPO / "pro_research" / "results" / "native_nvfp4" / "c3b_capture"
CAP_META = CAP_DIR / "C3B_REAL_ACTIVATIONS_META.json"
C3A = REPO / "pro_research" / "results" / "native_nvfp4" / "C3A_REAL_WEIGHT.json"
C3A_V2 = REPO / "pro_research" / "results" / "native_nvfp4" / "C3A_V2_LAYOUT_PREFLIGHT.json"
OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "C3B_REAL_ACTIVATION.json"
PREREG = REPO / "pro_research" / "S100_NATIVE_NVFP4_C3B_REALACT_PREREGISTRATION.md"

M_VALUES = (1, 2, 4, 8)
CAL_N = 32
HELD_N = 32
STATIC_MARGIN = 1.10
NRMSE_MAX = 0.080
COSINE_MIN = 0.9950
NMAX_MAX = 0.200
LM_TOP1_MIN = 0.90
LM_TOP1_IN_TOP5_MIN = 0.97
COLD_L2_MULTIPLE = 4.0
M8_OVER_M1_MAX = 1.20
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def ceilq(x: int, q: int) -> int:
    return ((int(x) + q - 1) // q) * q


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_tensor(torch, rec: dict[str, Any]):
    path = REPO / rec["path"]
    raw = bytearray(path.read_bytes())
    dt = str(rec["dtype"])
    if dt == "<f4":
        dtype = torch.float32
    elif dt == "<i4":
        dtype = torch.int32
    else:
        raise ValueError(f"unsupported capture dtype {dt}")
    t = torch.frombuffer(raw, dtype=dtype).clone().reshape(tuple(int(x) for x in rec["shape"]))
    return t


def fp4_rne_codes(torch, x):
    """Finite E2M1 RNE, encoded low 3 magnitude bits + sign bit."""
    a = x.abs().clamp_(0.0, 6.0)
    c = torch.empty_like(a, dtype=torch.uint8)
    c.fill_(7)
    c = torch.where(a <= 5.0, torch.full_like(c, 6), c)
    c = torch.where(a < 3.5, torch.full_like(c, 5), c)   # tie 3.5 -> 4 (code 6)
    c = torch.where(a <= 2.5, torch.full_like(c, 4), c)  # tie 2.5 -> 2 (code 4)
    c = torch.where(a < 1.75, torch.full_like(c, 3), c)  # tie 1.75 -> 2
    c = torch.where(a <= 1.25, torch.full_like(c, 2), c) # tie 1.25 -> 1
    c = torch.where(a < 0.75, torch.full_like(c, 1), c)  # tie 0.75 -> 1
    c = torch.where(a <= 0.25, torch.zeros_like(c), c)   # tie 0.25 -> 0
    sign = (x < 0).to(torch.uint8) << 3
    return c | sign


def swizzle_a_scale(torch, sf, m: int, k: int):
    sfk, sfp, mpad = k // 16, ceilq(k // 16, 4), ceilq(m, 128)
    padded = torch.zeros((mpad, sfp), dtype=torch.float8_e4m3fn, device=sf.device)
    padded[:m, :sfk] = sf
    nrb, ncb = mpad // 128, sfp // 4
    blocked = padded.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    blocked = blocked.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten().contiguous()
    # Preserve the A-side shape contract already accepted by C2/C3A make_a.
    return blocked.reshape(mpad, sfp).contiguous()


def quantize_a_reference(torch, x, tensor_scale=None):
    if x.dtype != torch.float32 or x.ndim != 2 or not x.is_cuda:
        raise ValueError("C3B quantizer expects CUDA float32 [M,K]")
    m, k = (int(x.shape[0]), int(x.shape[1]))
    if k % 16:
        raise ValueError("K must be divisible by 16")
    sfk = k // 16
    if tensor_scale is None:
        g = (x.abs().amax() / float(448 * 6)).clamp_min(1.0e-12).reshape(1)
        policy = "dynamic"
    else:
        g = tensor_scale.reshape(1)
        policy = "static"
    blocks = x.reshape(m, sfk, 16)
    sf = ((blocks.abs().amax(dim=-1) / 6.0) / g).clamp(min=2.0 ** -6, max=448.0)
    sfq = sf.to(torch.float8_e4m3fn)
    denom = g * sfq.float()
    scaled = (blocks / denom.unsqueeze(-1)).clamp(min=-6.0, max=6.0)
    codes = fp4_rne_codes(torch, scaled)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(m, k // 2).contiguous()
    blocked = swizzle_a_scale(torch, sfq, m, k)
    # Do not compute diagnostic saturation here: a host .item() would serialize
    # the CUDA stream and poison quantizer timing. Quality code computes it
    # separately, outside timed regions.
    return {"u8": packed, "fp4": packed.view(torch.float4_e2m1fn_x2),
            "block": blocked, "global": g.contiguous(), "policy": policy}


def saturation_count(x, a) -> int:
    # Diagnostic only; intentionally synchronizes outside timed regions.
    return int((x.abs() > (a["global"] * float(448 * 6))).sum().item())


def native_call(torch, F, ST, SW, a, b):
    return F.scaled_mm(
        a["fp4"], b["fp4"],
        scale_a=[a["block"], a["global"]],
        scale_recipe_a=[ST.BlockWise1x16, ST.TensorWise],
        scale_b=[b["block"], b["global"]],
        scale_recipe_b=[ST.BlockWise1x16, ST.TensorWise],
        swizzle_a=[SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE],
        swizzle_b=[SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE],
        output_dtype=torch.bfloat16, use_fast_accum=False)


def event_p50_ms(torch, fn, reps: int, rounds: int = 5) -> dict[str, Any]:
    for _ in range(min(6, reps)):
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
    return {"samples_ms": vals, "p50_ms": s[len(s)//2], "min_ms": s[0], "max_ms": s[-1]}


def e4_lut(torch):
    vals = []
    for i in range(256):
        v = c3lib.e4m3(i)
        vals.append(0.0 if not math.isfinite(v) else float(v))
    return torch.tensor(vals, dtype=torch.float32)


def dequant_b_rows(torch, wr: bytes, sr: bytes, g: float, n: int, k: int, rows: list[int]):
    ridx = torch.tensor(rows, dtype=torch.int64)
    wb = torch.frombuffer(bytearray(wr), dtype=torch.uint8).reshape(n, k // 2)[ridx]
    sb = torch.frombuffer(bytearray(sr), dtype=torch.uint8).reshape(n, k // 16)[ridx]
    lut = torch.tensor(E2M1, dtype=torch.float32)
    w = torch.empty((len(rows), k), dtype=torch.float32)
    w[:, 0::2] = lut[(wb & 15).long()]
    w[:, 1::2] = lut[(wb >> 4).long()]
    sc = e4_lut(torch)[sb.long()] * float(g)
    w.mul_(sc.repeat_interleave(16, dim=1))
    return w


def metrics(torch, actual, ref) -> dict[str, float]:
    a, b = actual.double().reshape(-1), ref.double().reshape(-1)
    d = a - b
    rmse = float(torch.sqrt(torch.mean(d*d)).item())
    rrms = float(torch.sqrt(torch.mean(b*b)).item())
    an, bn = float(torch.linalg.vector_norm(a).item()), float(torch.linalg.vector_norm(b).item())
    dot = float(torch.dot(a, b).item())
    cos = dot / (an * bn) if an and bn else (1.0 if an == bn else 0.0)
    ma = float(d.abs().max().item()); rmax = float(b.abs().max().item())
    return {"rmse": rmse, "reference_rms": rrms,
            "normalized_rmse": rmse / max(rrms, 1e-12), "cosine": cos,
            "max_abs_error": ma, "reference_max_abs": rmax,
            "normalized_max_abs_error": ma / max(rmax, 1e-12)}


def checkpoint_spec(entries, headers, base: str, label: str) -> dict[str, Any]:
    w, s, g = base + ".weight", base + ".weight_scale", base + ".weight_scale_2"
    for name in (w, s, g):
        if name not in entries:
            raise KeyError(f"missing checkpoint tensor {name}")
    sr = c3lib.rec(s, entries, headers); wr = c3lib.rec(w, entries, headers)
    n, sfk = [int(x) for x in sr["shape"]]; k = sfk * 16
    if [int(x) for x in wr["shape"]] != [n, k // 2]:
        raise ValueError(f"{label}: weight/scale shape mismatch")
    return {"label": label, "base": base, "weight": w, "scale": s, "global": g, "N": n, "K": k}


def build_rotation(torch, weight_raw: bytes, scale_raw: bytes, g: float,
                   n: int, k: int, l2: int):
    base = c3lib.make_b(torch, weight_raw, scale_raw, g, n, k)
    one = len(weight_raw) + int(base["block"].numel())
    cycle = max(1, math.ceil((COLD_L2_MULTIPLE * l2) / max(one, 1)))
    free, _ = torch.cuda.mem_get_info()
    estimated = cycle * one + one + 128 * 1024 * 1024
    if estimated > int(free * 0.70):
        return [base], {"status": "not_run_memory_gate", "cycle": 1,
                        "one_b_bytes": one, "estimated_peak_bytes": int(estimated),
                        "free_bytes_before": int(free)}
    bs = [base]
    for _ in range(1, cycle):
        u = base["u8"].clone(); block = base["block"].clone()
        bs.append({"u8": u, "fp4": u.view(torch.float4_e2m1fn_x2).t(),
                   "block": block, "global": base["global"]})
    return bs, {"status": "measured", "cycle": cycle, "one_b_bytes": one,
                "working_set_bytes": cycle * one,
                "working_set_over_l2": (cycle * one) / max(l2, 1),
                "free_bytes_before": int(free), "estimated_peak_bytes": int(estimated)}


def quant_preflight(torch, F, ST, SW) -> dict[str, Any]:
    # Witness 1: nibble order / RNE. [0.5, 6] with g=1 has sf=1 and packs
    # low=0.5 (code 1), high=6 (code 7) => 0x71.
    k, n = 128, 128
    nib = torch.tensor([0.5, 6.0] * (k // 2), dtype=torch.float32, device="cuda").reshape(1, k)
    g = torch.ones(1, dtype=torch.float32, device="cuda")
    an = quantize_a_reference(torch, nib, g)
    code = an["u8"].detach().cpu().flatten().tolist()
    raw_sf = an["block"].view(torch.uint8).detach().cpu().flatten().tolist()
    nibble_ok = all(int(x) == 0x71 for x in code) and int(raw_sf[0]) == 0x38
    del an, nib

    # Witness 2: 2 row-blocks x 2 scale-column-blocks. This is deliberately
    # non-uniform so the legacy C3A-v1 K-major outer-tile order cannot pass by
    # accident. Each 16-value block is a constant 6*sf, hence FP4=6 exactly.
    m = 256
    x = torch.empty((m, k), dtype=torch.float32, device="cuda")
    x[:128, :64] = 3.0    # sf=.5  raw 0x30
    x[:128, 64:] = 6.0    # sf=1   raw 0x38
    x[128:, :64] = 12.0   # sf=2   raw 0x40
    x[128:, 64:] = 24.0   # sf=4   raw 0x48
    a = quantize_a_reference(torch, x, g)
    b = c3lib.make_b(torch, bytes([0x22]) * (n * (k // 2)),
                     bytes([0x38]) * (n * (k // 16)), 1.0, n, k)
    out = native_call(torch, F, ST, SW, a, b); torch.cuda.synchronize()
    expected = torch.empty((m,), dtype=torch.float32, device="cuda")
    expected[:128] = 64.0 * (3.0 + 6.0)     # 576
    expected[128:] = 64.0 * (12.0 + 24.0)  # 2304
    max_abs = float((out.float()[:, 0] - expected).abs().max().item())
    row0_spread = float((out.float()[0] - out.float()[0, 0]).abs().max().item())
    row255_spread = float((out.float()[-1] - out.float()[-1, 0]).abs().max().item())
    rec = {"nibble_witness_packed_all_0x71_and_sf_0x38": nibble_ok,
           "layout_shape": [m, n, k], "native_finite": bool(torch.isfinite(out).all().item()),
           "expected_rows_0_127": 576.0, "expected_rows_128_255": 2304.0,
           "layout_max_abs_error": max_abs,
           "row0_output_spread": row0_spread, "row255_output_spread": row255_spread}
    rec["passes"] = bool(nibble_ok and rec["native_finite"] and max_abs == 0.0
                         and row0_spread == 0.0 and row255_spread == 0.0)
    del a, b, out, x, expected, g
    torch.cuda.empty_cache()
    return rec


def family_run(torch, F, ST, SW, spec, x_all, entries, headers, l2: int):
    label, n, k = spec["label"], int(spec["N"]), int(spec["K"])
    if int(x_all.shape[1]) != k:
        raise ValueError(f"{label}: captured K={x_all.shape[1]} != checkpoint K={k}")
    wr = c3lib.tensor_raw(spec["weight"], entries, headers)
    sr = c3lib.tensor_raw(spec["scale"], entries, headers)
    gr = c3lib.tensor_raw(spec["global"], entries, headers)
    wg = float(struct.unpack("<f", gr)[0])
    cal, held = x_all[:CAL_N], x_all[CAL_N:CAL_N+HELD_N]
    static_value = max(float(cal.abs().max().item()) * STATIC_MARGIN / float(448*6), 1e-12)
    static_g = torch.tensor([static_value], dtype=torch.float32, device="cuda")

    rows = c3lib.sample_rows(n, 64)
    b_sample = dequant_b_rows(torch, wr, sr, wg, n, k, rows)
    xq_cpu = held[:8].contiguous()
    ref = xq_cpu.double() @ b_sample.double().t()
    xq = xq_cpu.to("cuda")
    b_one = c3lib.make_b(torch, wr, sr, wg, n, k)
    quality = {}
    samples = {}
    finite_all = True
    for arm in ("dynamic", "static_1p10"):
        a = quantize_a_reference(torch, xq, None if arm == "dynamic" else static_g)
        out = native_call(torch, F, ST, SW, a, b_one); torch.cuda.synchronize()
        act = out[:, rows].float().cpu()
        mm = metrics(torch, act, ref)
        finite = bool(torch.isfinite(out).all().item())
        finite_all = finite_all and finite
        quality[arm] = {"metrics": mm, "finite": finite,
                        "saturation_values_M8": saturation_count(xq, a),
                        "tensor_scale": float(a["global"].detach().cpu().item())}
        samples[arm] = {"actual": act.tolist(), "reference": ref.float().tolist(), "sampled_output_rows": rows}
        del a, out, act
    del b_one
    torch.cuda.empty_cache()

    bs, cold = build_rotation(torch, wr, sr, wg, n, k, l2)
    timing = {"dynamic": {}, "static_1p10": {}}
    quant_upper = {"dynamic": {}, "static_1p10": {}}
    combined_upper = {"dynamic": {}, "static_1p10": {}}
    if cold.get("status") == "measured":
        reps = max(int(cold["cycle"]) * 3, 40)
        cold["reps_per_round"] = reps
        for m in M_VALUES:
            x = held[:m].contiguous().to("cuda")
            for arm in ("dynamic", "static_1p10"):
                sg = None if arm == "dynamic" else static_g
                a_pre = quantize_a_reference(torch, x, sg)
                counter = {"i": 0}
                def gemm_only():
                    i = counter["i"]; counter["i"] = i + 1
                    return native_call(torch, F, ST, SW, a_pre, bs[i % len(bs)])
                timing[arm][f"M{m}"] = event_p50_ms(torch, gemm_only, reps)

                qreps = 40
                quant_upper[arm][f"M{m}"] = event_p50_ms(
                    torch, lambda x=x, sg=sg: quantize_a_reference(torch, x, sg), qreps, rounds=3)
                counter2 = {"i": 0}
                def combined():
                    aa = quantize_a_reference(torch, x, sg)
                    i = counter2["i"]; counter2["i"] = i + 1
                    return native_call(torch, F, ST, SW, aa, bs[i % len(bs)])
                combined_upper[arm][f"M{m}"] = event_p50_ms(torch, combined, reps, rounds=3)
                del a_pre
            del x
            torch.cuda.synchronize()
        for arm in timing:
            m1 = float(timing[arm]["M1"]["p50_ms"]); m8 = float(timing[arm]["M8"]["p50_ms"])
            timing[arm]["M8_over_M1"] = m8 / m1 if m1 else None
    del bs
    torch.cuda.empty_cache(); gc.collect()
    return {"label": label, "selected": spec,
            "checkpoint": {"weight_sha256": sha_bytes(wr), "scale_sha256": sha_bytes(sr),
                           "global_sha256": sha_bytes(gr), "weight_global_scale": wg},
            "activation": {"calibration_amax": float(cal.abs().max().item()),
                           "static_margin": STATIC_MARGIN, "static_tensor_scale": static_value,
                           "heldout_amax": float(held.abs().max().item())},
            "quality": quality, "quality_samples": samples, "finite_all": finite_all,
            "cold": cold, "prequantized_native_timing": timing,
            "reference_quantizer_timing_upper_bound": quant_upper,
            "reference_quantizer_plus_native_upper_bound": combined_upper}


def lm_quality(torch, F, ST, SW, spec, x_all, exact_ids, exact_top5, entries, headers):
    n, k = int(spec["N"]), int(spec["K"])
    wr = c3lib.tensor_raw(spec["weight"], entries, headers)
    sr = c3lib.tensor_raw(spec["scale"], entries, headers)
    wg = float(struct.unpack("<f", c3lib.tensor_raw(spec["global"], entries, headers))[0])
    b = c3lib.make_b(torch, wr, sr, wg, n, k)
    cal, held = x_all[:CAL_N], x_all[CAL_N:CAL_N+HELD_N]
    static_value = max(float(cal.abs().max().item()) * STATIC_MARGIN / float(448*6), 1e-12)
    static_g = torch.tensor([static_value], dtype=torch.float32, device="cuda")
    ex = exact_ids[CAL_N:CAL_N+HELD_N].long()
    ex5 = exact_top5[CAL_N:CAL_N+HELD_N].long()
    rec = {}
    for arm in ("dynamic", "static_1p10"):
        tops = []
        tops5 = []
        finite = True
        sats = 0
        for s in range(0, HELD_N, 8):
            x = held[s:s+8].contiguous().to("cuda")
            a = quantize_a_reference(torch, x, None if arm == "dynamic" else static_g)
            out = native_call(torch, F, ST, SW, a, b); torch.cuda.synchronize()
            finite = finite and bool(torch.isfinite(out).all().item())
            tops.extend([int(v) for v in torch.argmax(out, dim=1).cpu().tolist()])
            tops5.extend([[int(z) for z in row] for row in torch.topk(out.float(), 5, dim=1).indices.cpu().tolist()])
            sats += saturation_count(x, a)
            del x, a, out
        top = torch.tensor(tops, dtype=torch.int64)
        eq = float((top == ex).float().mean().item())
        in5 = float(torch.tensor([int(top[i].item()) in set(int(z) for z in ex5[i].tolist())
                                  for i in range(HELD_N)], dtype=torch.float32).mean().item())
        overlap = sum(len(set(tops5[i]).intersection(set(int(z) for z in ex5[i].tolist()))) / 5.0
                      for i in range(HELD_N)) / HELD_N
        rec[arm] = {"finite": finite, "top1_retention": eq,
                    "native_top1_in_ervf_top5": in5, "mean_top5_overlap": overlap,
                    "saturation_values": sats, "native_top1_ids": tops,
                    "native_top5_ids": tops5, "exact_top1_ids": [int(x) for x in ex.tolist()],
                    "exact_top5_ids": [[int(z) for z in row] for row in ex5.tolist()]}
    del b
    torch.cuda.empty_cache()
    return rec


def main() -> int:
    payload: dict[str, Any] = {
        "kind": "s100_native_nvfp4_c3b_real_activation",
        "status": "started", "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": ("real causal A activations + real checkpoint B + native SM120 geometry; "
                           "reference quantizer timing is an upper bound; no integrated or tok/s claim"),
        "thresholds": {"normalized_rmse_max": NRMSE_MAX, "cosine_min": COSINE_MIN,
            "normalized_max_abs_error_max": NMAX_MAX, "lm_top1_retention_min": LM_TOP1_MIN,
            "lm_native_top1_in_ervf_top5_min": LM_TOP1_IN_TOP5_MIN,
            "cold_working_set_over_l2_min": COLD_L2_MULTIPLE,
            "M8_over_M1_max": M8_OVER_M1_MAX, "static_margin": STATIC_MARGIN},
    }
    try:
        payload["gpu_idle_preflight"] = require_gpu_idle_wddm()
        import torch
        import torch.nn.functional as F
        ST, SW = F.ScalingType, F.SwizzleType
        cap = tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else (-1, -1)
        if not (str(torch.__version__).startswith("2.12.1") and str(torch.version.cuda).startswith("13.2")
                and torch.cuda.is_available() and cap >= (12, 0)):
            raise RuntimeError(f"frozen SM120 environment not met: torch={torch.__version__} cuda={torch.version.cuda} cap={cap}")

        pre = json.loads(C3A_V2.read_text(encoding="utf-8"))
        c3a = json.loads(C3A.read_text(encoding="utf-8"))
        parent_ok = pre.get("status") == "layout_v2_preflight_pass" and c3a.get("status") in {
            "real_weight_representation_and_geometry_candidate", "real_weight_representation_green_perf_miss"}
        capmeta = json.loads(CAP_META.read_text(encoding="utf-8"))
        capture_ok = capmeta.get("status") == "captured"
        hash_ok = capture_ok and all(sha_file(REPO / r["path"]) == r["sha256"]
                                     for r in (capmeta.get("arrays") or {}).values())
        if not (parent_ok and capture_ok and hash_ok):
            payload.update({"status": "precondition_failed", "gates": {
                "C3B_G1_C3A_v2_parent_green": parent_ok,
                "C3B_G2_capture_hashes": hash_ok}, "completed_utc": utc_now()})
            write_json_atomic(OUT, payload, archive=True); print(json.dumps(payload, indent=2)); return 0

        c3v2.install(c3lib)
        arrays = capmeta["arrays"]
        moe = load_raw_tensor(torch, arrays["moe_normed"])
        shared = load_raw_tensor(torch, arrays["shared_act"])
        lm_in = load_raw_tensor(torch, arrays["lm_head_in"])
        exact = load_raw_tensor(torch, arrays["exact_token_ids"])
        exact5 = load_raw_tensor(torch, arrays["ervf_top5_ids"])
        if int(moe.shape[0]) != 64 or int(shared.shape[0]) != 64 or int(lm_in.shape[0]) != 64:
            raise RuntimeError("C3B requires exactly 64 captured rows")

        entries, headers = c3lib.load_index_headers()
        layer = int(capmeta["capture"]["target_moe_layer"])
        p = f"backbone.layers.{layer}.mixer"
        specs = [
            checkpoint_spec(entries, headers, "lm_head", "lm_head"),
            checkpoint_spec(entries, headers, f"{p}.shared_experts.up_proj", "shared_up"),
            checkpoint_spec(entries, headers, f"{p}.shared_experts.down_proj", "shared_down"),
            checkpoint_spec(entries, headers, f"{p}.experts.0.up_proj", "routed_up"),
        ]
        x_by = {"lm_head": lm_in, "shared_up": moe, "shared_down": shared, "routed_up": moe}
        props = torch.cuda.get_device_properties(0)
        l2 = int(getattr(props, "L2_cache_size", 0) or getattr(props, "l2_cache_size", 0) or 0)
        if l2 <= 0:
            raise RuntimeError("L2 cache size unavailable")
        smoke = quant_preflight(torch, F, ST, SW)
        fams = [family_run(torch, F, ST, SW, s, x_by[s["label"]], entries, headers, l2) for s in specs]
        lm_spec = next(s for s in specs if s["label"] == "lm_head")
        lm = lm_quality(torch, F, ST, SW, lm_spec, lm_in, exact, exact5, entries, headers)

        finite = bool(smoke.get("passes")) and all(f["finite_all"] for f in fams) and all(v["finite"] for v in lm.values())
        def local_ok(arm):
            return all(f["quality"][arm]["metrics"]["normalized_rmse"] <= NRMSE_MAX
                       and f["quality"][arm]["metrics"]["cosine"] >= COSINE_MIN
                       and f["quality"][arm]["metrics"]["normalized_max_abs_error"] <= NMAX_MAX
                       for f in fams)
        def lmq_ok(arm):
            return lm[arm]["top1_retention"] >= LM_TOP1_MIN and lm[arm]["native_top1_in_ervf_top5"] >= LM_TOP1_IN_TOP5_MIN
        measured = [f for f in fams if f["cold"].get("status") == "measured"]
        cold_ok = len(measured) == len(fams) and all(f["cold"]["working_set_over_l2"] >= COLD_L2_MULTIPLE for f in measured)
        def perf_ok(arm):
            ratios = [float(f["prequantized_native_timing"][arm]["M8_over_M1"]) for f in measured]
            count = sum(r <= M8_OVER_M1_MAX for r in ratios)
            lm_ratio = next(float(f["prequantized_native_timing"][arm]["M8_over_M1"]) for f in measured if f["label"] == "lm_head")
            return count >= 3 and lm_ratio <= M8_OVER_M1_MAX, count, {f["label"]: float(f["prequantized_native_timing"][arm]["M8_over_M1"]) for f in measured}
        pd, nd, rd = perf_ok("dynamic") if cold_ok else (False, 0, {})
        ps, ns, rs = perf_ok("static_1p10") if cold_ok else (False, 0, {})
        ld, ls = local_ok("dynamic"), local_ok("static_1p10")
        md, ms = lmq_ok("dynamic"), lmq_ok("static_1p10")
        gates = {
            "C3B_G1_C3A_v2_parent_green": parent_ok,
            "C3B_G2_capture_hashes": hash_ok,
            "C3B_G3_quant_layout_native_preflight_exact": bool(smoke.get("passes")),
            "C3B_G4_all_native_outputs_finite": finite,
            "C3B_G5_DYNAMIC_LOCAL": ld,
            "C3B_G6_STATIC_LOCAL": ls,
            "C3B_G7_DYNAMIC_LM": md,
            "C3B_G8_STATIC_LM": ms,
            "C3B_P1_cold_rotation_ge_4x_L2": cold_ok,
            "C3B_P2_DYNAMIC_M8_geometry": pd,
            "C3B_P3_STATIC_M8_geometry": ps,
        }
        dynamic_candidate = bool(finite and ld and md and cold_ok and pd)
        static_candidate = bool(finite and ls and ms and cold_ok and ps)
        selected = "static_1p10" if static_candidate else "dynamic" if dynamic_candidate else None
        payload.update({"environment": environment_snapshot((Path(__file__), PREREG, CAP_META, C3A, C3A_V2)),
                        "api": {"torch": str(torch.__version__), "cuda": str(torch.version.cuda),
                                "gpu": torch.cuda.get_device_name(0), "capability": list(cap), "l2_bytes": l2},
                        "capture_manifest": capmeta, "quant_preflight": smoke,
                        "families": fams, "lm_head_quality": lm, "gates": gates,
                        "summary": {"dynamic_candidate": dynamic_candidate,
                                    "static_1p10_candidate": static_candidate,
                                    "selected_candidate_arm": selected,
                                    "dynamic_M8_ratio_pass_count": nd,
                                    "static_M8_ratio_pass_count": ns,
                                    "dynamic_M8_over_M1": rd, "static_M8_over_M1": rs,
                                    "lm_top1_retention": {k: v["top1_retention"] for k, v in lm.items()},
                                    "lm_top1_in_ervf_top5": {k: v["native_top1_in_ervf_top5"] for k, v in lm.items()}},
                        "status": "real_activation_native_candidate" if selected else "real_activation_gate_failed",
                        "completed_utc": utc_now()})
    except Exception as exc:
        payload.update({"status": "technical_failure",
                        "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                        "completed_utc": utc_now()})
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({"status": payload.get("status"), "summary": payload.get("summary"),
                      "gates": payload.get("gates"), "error": (payload.get("error") or {}).get("message"),
                      "output": str(OUT)}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
