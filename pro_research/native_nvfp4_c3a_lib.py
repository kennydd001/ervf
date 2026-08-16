"""Dependency-light helpers for C3A. The independent verifier does not import this."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from common import REPO

MODEL_DIR = REPO / "models" / "nemotron_3_5_lightning_v35"
INDEX = MODEL_DIR / "model.safetensors.index.json"
REFERENCE_ROWS = 64
COLD_L2_MULTIPLE = 4.0
COLD_ROUNDS = 5
M_VALUES = (1, 8)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def ceilq(x: int, q: int) -> int:
    return ((int(x) + q - 1) // q) * q


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_index_headers() -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    idx = json.loads(INDEX.read_text(encoding="utf-8"))
    entries = idx["weight_map"]
    headers: dict[str, dict[str, Any]] = {}
    for shard in sorted(set(entries.values())):
        p = MODEL_DIR / shard
        with p.open("rb") as fh:
            hlen = int.from_bytes(fh.read(8), "little")
            headers[shard] = json.loads(fh.read(hlen))
    return entries, headers


def rec(name: str, entries: dict[str, str], headers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return headers[entries[name]][name]


def tensor_raw(name: str, entries: dict[str, str], headers: dict[str, dict[str, Any]]) -> bytes:
    shard = entries[name]
    rr = headers[shard][name]
    a, b = (int(x) for x in rr["data_offsets"])
    p = MODEL_DIR / shard
    with p.open("rb") as fh:
        hlen = int.from_bytes(fh.read(8), "little")
        fh.seek(8 + hlen + a)
        raw = fh.read(b - a)
    if len(raw) != b - a:
        raise IOError(f"short read {name}: {len(raw)} != {b-a}")
    return raw


def all_nvfp4_triples(entries: dict[str, str], headers: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    triples: list[dict[str, Any]] = []
    for sname in sorted(n for n in entries if n.endswith(".weight_scale")):
        base = sname[:-len(".weight_scale")]
        wname, gname = base + ".weight", base + ".weight_scale_2"
        if wname not in entries or gname not in entries:
            continue
        sr, wr, gr = rec(sname, entries, headers), rec(wname, entries, headers), rec(gname, entries, headers)
        ss = [int(x) for x in sr.get("shape", [])]
        ws = [int(x) for x in wr.get("shape", [])]
        gs = [int(x) for x in gr.get("shape", [])]
        if len(ss) != 2 or len(ws) != 2:
            continue
        n, sfk = ss
        k = sfk * 16
        if not (ws == [n, k // 2] and wr.get("dtype") == "U8"
                and sr.get("dtype") == "F8_E4M3" and gr.get("dtype") == "F32"
                and gs in ([], [1])):
            continue
        triples.append({
            "base": base, "weight": wname, "scale": sname, "global": gname,
            "N": n, "K": k, "SFK": sfk,
            "weight_shape": ws, "scale_shape": ss, "global_shape": gs,
            "weight_dtype": wr.get("dtype"), "scale_dtype": sr.get("dtype"), "global_dtype": gr.get("dtype"),
        })
    return triples


def choose_representatives(triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def pick(label: str, n: int, k: int, include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> dict[str, Any]:
        cands = [x for x in triples if x["N"] == n and x["K"] == k]
        good = [x for x in cands if all(t in x["base"].lower() for t in include)
                and not any(t in x["base"].lower() for t in exclude)]
        if not good:
            raise RuntimeError(f"no representative for {label} N={n} K={k}; candidates={[x['base'] for x in cands[:8]]}")
        out = dict(sorted(good, key=lambda x: x["base"])[0]); out["label"] = label
        return out
    exact = [x for x in triples if x["base"] == "lm_head"]
    if not exact:
        raise RuntimeError("lm_head NVFP4 triple not found")
    lm = dict(exact[0]); lm["label"] = "lm_head"
    return [lm,
            pick("shared_up", 3712, 2688, ("shared", "up_proj")),
            pick("shared_down", 2688, 3712, ("shared", "down_proj")),
            pick("routed_up", 1856, 2688, ("up_proj",), ("shared",))]


def e4m3(raw: int) -> float:
    sign = -1.0 if (raw >> 7) & 1 else 1.0
    exp, man = (raw >> 3) & 0xF, raw & 0x7
    if exp == 0:
        return sign * (2.0 ** -6) * (man / 8.0)
    if exp == 0xF and man == 0x7:
        return math.nan
    return sign * (2.0 ** (exp - 7)) * (1.0 + man / 8.0)


def reference_row(weight_raw: bytes, scale_raw: bytes, global_scale: float, row: int, n: int, k: int) -> float:
    pk, sfk = k // 2, k // 16
    if len(weight_raw) != n * pk or len(scale_raw) != n * sfk:
        raise ValueError("reference payload size mismatch")
    wb = memoryview(weight_raw)[row * pk:(row + 1) * pk]
    sb = memoryview(scale_raw)[row * sfk:(row + 1) * sfk]
    terms: list[float] = []
    for j, byte in enumerate(wb):
        s = e4m3(sb[(2 * j) // 16])
        terms.append(E2M1[byte & 0xF] * s * global_scale)
        terms.append(E2M1[(byte >> 4) & 0xF] * s * global_scale)
    return math.fsum(terms)


def sample_rows(n: int, count: int = REFERENCE_ROWS) -> list[int]:
    if n <= count:
        return list(range(n))
    rows = {0, 1, n - 2, n - 1, n // 2}
    for i in range(count):
        rows.add((i * (n - 1)) // max(count - 1, 1))
    return sorted(rows)[:count]


def metrics(actual: list[float], ref: list[float]) -> dict[str, float]:
    if len(actual) != len(ref) or not actual:
        raise ValueError("metric vectors must be same nonzero length")
    rmse = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(actual, ref)) / len(ref))
    rrms = math.sqrt(math.fsum(b * b for b in ref) / len(ref))
    dot = math.fsum(a * b for a, b in zip(actual, ref))
    an = math.sqrt(math.fsum(a * a for a in actual)); bn = math.sqrt(math.fsum(b * b for b in ref))
    cos = dot / (an * bn) if an and bn else (1.0 if an == bn else 0.0)
    ma = max(abs(a - b) for a, b in zip(actual, ref)); rmax = max(abs(x) for x in ref)
    return {"rmse": rmse, "reference_rms": rrms, "normalized_rmse": rmse / max(rrms, 1e-12),
            "cosine": cos, "max_abs_error": ma, "reference_max_abs": rmax,
            "normalized_max_abs_error": ma / max(rmax, 1e-12)}


def _swizzle_offset(torch, mm, ss, rows: int):
    nmb = ceilq(rows, 128) // 128
    mb, r = mm // 128, mm % 128
    r32, g32, kb, sf4 = r % 32, r // 32, ss // 4, ss % 4
    return (kb * nmb + mb) * 512 + ((r32 * 4 + g32) * 4 + sf4)


def repack_b_scale(torch, scale_raw: bytes, n: int, k: int):
    sfk, sfp, npad = k // 16, ceilq(k // 16, 4), ceilq(n, 128)
    natural = torch.frombuffer(bytearray(scale_raw), dtype=torch.uint8).reshape(n, sfk)
    dst = torch.zeros(sfp * npad, dtype=torch.uint8)
    ss = torch.arange(sfk, dtype=torch.int64).reshape(1, -1)
    for r0 in range(0, n, 256):
        r1 = min(n, r0 + 256)
        mm = torch.arange(r0, r1, dtype=torch.int64).reshape(-1, 1)
        dst[_swizzle_offset(torch, mm, ss, n).reshape(-1)] = natural[r0:r1].reshape(-1)
    return dst.to("cuda").view(torch.float8_e4m3fn).reshape(sfp, npad).contiguous()


def make_a(torch, m: int, k: int):
    au8 = torch.full((m, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    sfp = ceilq(k // 16, 4)
    return {"u8": au8, "fp4": au8.view(torch.float4_e2m1fn_x2),
            "block": torch.ones((ceilq(m, 128), sfp), dtype=torch.float8_e4m3fn, device="cuda"),
            "global": torch.ones((1,), dtype=torch.float32, device="cuda")}


def make_b(torch, weight_raw: bytes, scale_raw: bytes, global_scale: float, n: int, k: int):
    bu8 = torch.frombuffer(bytearray(weight_raw), dtype=torch.uint8).reshape(n, k // 2).to("cuda")
    return {"u8": bu8, "fp4": bu8.view(torch.float4_e2m1fn_x2).t(),
            "block": repack_b_scale(torch, scale_raw, n, k),
            "global": torch.tensor([global_scale], dtype=torch.float32, device="cuda")}


def native_call(torch, F, ST, SW, a, b):
    return F.scaled_mm(
        a["fp4"], b["fp4"],
        scale_a=[a["block"], a["global"]], scale_recipe_a=[ST.BlockWise1x16, ST.TensorWise],
        scale_b=[b["block"], b["global"]], scale_recipe_b=[ST.BlockWise1x16, ST.TensorWise],
        swizzle_a=[SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE],
        swizzle_b=[SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE],
        output_dtype=torch.bfloat16, use_fast_accum=False)


def two_level_smoke(torch, F, ST, SW) -> dict[str, Any]:
    m, n, k = 2, 128, 256
    a = make_a(torch, m, k)
    b = make_b(torch, bytes([0x22]) * (n * (k // 2)), bytes([0x38]) * (n * (k // 16)), 0.5, n, k)
    out = native_call(torch, F, ST, SW, a, b); torch.cuda.synchronize()
    expected = torch.tensor(k * 0.5, dtype=torch.bfloat16, device="cuda")
    rr = {"output_shape": list(out.shape), "output_dtype": str(out.dtype),
          "finite": bool(torch.isfinite(out).all().item()), "expected_bf16": float(expected.float().item()),
          "all_equal_expected": bool(torch.all(out == expected).item()),
          "max_abs_error": float((out.float() - expected.float()).abs().max().item())}
    del out, a, b; torch.cuda.empty_cache()
    return rr


def event_p50_ms(torch, fn, reps: int, rounds: int = COLD_ROUNDS) -> dict[str, Any]:
    for _ in range(min(8, reps)):
        fn()
    torch.cuda.synchronize(); vals: list[float] = []
    for _ in range(rounds):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps):
            fn()
        en.record(); en.synchronize(); vals.append(float(st.elapsed_time(en)) / reps)
    ss = sorted(vals)
    return {"samples_ms": vals, "p50_ms": ss[len(ss)//2], "min_ms": ss[0], "max_ms": ss[-1]}


def cold_timing(torch, F, ST, SW, weight_raw: bytes, scale_raw: bytes,
                global_scale: float, n: int, k: int, l2_bytes: int) -> dict[str, Any]:
    one_w, one_s = len(weight_raw), ceilq(k // 16, 4) * ceilq(n, 128)
    one_b = one_w + one_s
    cycle = max(1, math.ceil((COLD_L2_MULTIPLE * l2_bytes) / one_b))
    target = cycle * one_b
    free, _ = torch.cuda.mem_get_info(); estimated = target + one_b + 128 * 1024 * 1024
    rr: dict[str, Any] = {"l2_bytes": int(l2_bytes), "one_b_weight_bytes": one_w,
        "one_b_scale_physical_bytes": one_s, "one_b_bytes": one_b, "cycle": cycle,
        "rotation_working_set_bytes": target, "working_set_over_l2": target / l2_bytes,
        "free_bytes_before": int(free), "estimated_peak_bytes": int(estimated)}
    if estimated > int(free * 0.70):
        rr["status"] = "not_run_memory_gate"; return rr
    base = make_b(torch, weight_raw, scale_raw, global_scale, n, k); bs = [base]
    for _ in range(1, cycle):
        u = base["u8"].clone(); block = base["block"].clone()
        bs.append({"u8": u, "fp4": u.view(torch.float4_e2m1fn_x2).t(), "block": block, "global": base["global"]})
    reps = max(cycle * 3, 40); rr["reps_per_round"] = reps; timings = {}
    for m in M_VALUES:
        a = make_a(torch, m, k); counter = {"i": 0}
        def one_call():
            i = counter["i"]; counter["i"] = i + 1
            return native_call(torch, F, ST, SW, a, bs[i % cycle])
        timings[f"M{m}"] = event_p50_ms(torch, one_call, reps)
        del a; torch.cuda.synchronize()
    m1, m8 = float(timings["M1"]["p50_ms"]), float(timings["M8"]["p50_ms"])
    rr.update({"status": "measured", "timing": timings, "M8_over_M1": m8 / m1 if m1 else None})
    del bs, base; torch.cuda.empty_cache(); return rr


def run_family(torch, F, ST, SW, spec: dict[str, Any], entries, headers, l2_bytes: int) -> dict[str, Any]:
    label, n, k = spec["label"], int(spec["N"]), int(spec["K"])
    wr = tensor_raw(spec["weight"], entries, headers); sr = tensor_raw(spec["scale"], entries, headers)
    gr = tensor_raw(spec["global"], entries, headers)
    if len(gr) != 4:
        raise ValueError(f"{label}: global scale is {len(gr)} bytes, expected 4")
    g = float(struct.unpack("<f", gr)[0]); rows = sample_rows(n)
    ref = [reference_row(wr, sr, g, r, n, k) for r in rows]
    if not all(math.isfinite(x) for x in ref):
        raise ValueError(f"{label}: non-finite independent reference")
    b = make_b(torch, wr, sr, g, n, k); outputs = {}; actual_m1: list[float] | None = None; m8_nmax = None
    for m in M_VALUES:
        a = make_a(torch, m, k); out = native_call(torch, F, ST, SW, a, b); torch.cuda.synchronize()
        sampled = out[:, rows].float().cpu(); finite = bool(torch.isfinite(out).all().item())
        if m == 1:
            actual_m1 = [float(x) for x in sampled[0].tolist()]; mm = metrics(actual_m1, ref)
        else:
            assert actual_m1 is not None
            mm = metrics([float(x) for x in sampled[0].tolist()], ref)
            m8_nmax = float((sampled - sampled[0:1]).abs().max().item()) / max(max(abs(x) for x in actual_m1), 1e-12)
        outputs[f"M{m}"] = {"finite": finite, "shape": list(out.shape), "dtype": str(out.dtype), "reference_metrics_first_row": mm}
        del out, sampled, a
    assert actual_m1 is not None and m8_nmax is not None
    del b; torch.cuda.empty_cache()
    return {"label": label, "selected": spec,
            "payload": {"weight_bytes": len(wr), "scale_bytes": len(sr), "global_bytes": len(gr),
                        "global_scale_f32": g, "weight_sha256": sha(wr), "scale_sha256": sha(sr), "global_sha256": sha(gr)},
            "reference_samples": {"rows": rows, "values": ref}, "native": outputs,
            "M8_identical_rows_normalized_max_diff": m8_nmax,
            "cold_timing": cold_timing(torch, F, ST, SW, wr, sr, g, n, k, l2_bytes)}
