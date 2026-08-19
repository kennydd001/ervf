"""S100-D4 representative weight-only NVFP4 diagnostic.

This is deliberately self-contained and uses the production CuPy runtime. It
captures real causal inputs, re-encodes selected BF16/FP8 dense matrices
offline, compares outputs, and measures cold >=4x-L2 current-vs-NVFP4 W4A32
kernels. No integration and no tok/s projection.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "pro_research"))

from common import environment_snapshot, require_model_dir, write_json_atomic

OUT = REPO / "pro_research" / "results" / "S100_D4_WEIGHT_ONLY_DENSE.json"
PREREG = REPO / "pro_research" / "S100_D4_WEIGHT_ONLY_DENSE_PREREGISTRATION.md"

E2 = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
F4MAX = 6.0
F8MAX = 448.0
ROWS = 24
L2_MULTIPLE = 4.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def e4_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals, raws = [], []
    full = np.zeros(256, dtype=np.float32)
    for raw in range(256):
        sign = -1.0 if raw & 0x80 else 1.0
        exp, man = (raw >> 3) & 0xF, raw & 7
        if exp == 0xF and man == 7:
            v = np.nan
        elif exp == 0:
            v = sign * (2.0 ** -6) * (man / 8.0)
        else:
            v = sign * (2.0 ** (exp - 7)) * (1.0 + man / 8.0)
        full[raw] = v
        if raw < 0x7F and np.isfinite(v) and v >= 0:
            vals.append(v); raws.append(raw)
    order = np.argsort(np.asarray(vals), kind="stable")
    return np.asarray(vals, np.float32)[order], np.asarray(raws, np.uint8)[order], full


E4_VALUES, E4_RAWS, E4_FULL = e4_table()


def e4_encode(x: np.ndarray, convention: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, F8MAX)
    hi = np.searchsorted(E4_VALUES, x, side="left")
    hi = np.clip(hi, 0, len(E4_VALUES) - 1)
    if convention == "CEIL":
        idx = hi
    elif convention == "RNE":
        lo = np.maximum(hi - 1, 0)
        dl = np.abs(x - E4_VALUES[lo])
        dh = np.abs(E4_VALUES[hi] - x)
        tie = np.isclose(dl, dh, rtol=0.0, atol=np.finfo(np.float32).eps * 4)
        choose_hi = (dh < dl) | (tie & ((E4_RAWS[hi] & 1) == 0))
        idx = np.where(choose_hi, hi, lo)
    else:
        raise ValueError(convention)
    return E4_RAWS[idx], E4_VALUES[idx]


def fp4_codes(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    c = np.full(a.shape, 7, dtype=np.uint8)
    c[a <= 5.0] = 6
    c[a < 3.5] = 5
    c[a <= 2.5] = 4
    c[a < 1.75] = 3
    c[a <= 1.25] = 2
    c[a < 0.75] = 1
    c[a <= 0.25] = 0
    return c | ((x < 0).astype(np.uint8) << 3)


def quantize_matrix(w: np.ndarray, convention: str, chunk_rows: int = 128) -> dict[str, Any]:
    if w.ndim != 2 or w.shape[1] % 16:
        raise ValueError(f"matrix shape must be [N,K], K%16=0: {w.shape}")
    n, k = map(int, w.shape)
    amax = float(np.max(np.abs(w)))
    global_scale = max(amax / (F8MAX * F4MAX), 1.0e-30)
    codes = np.empty((n, k // 2), dtype=np.uint8)
    scales = np.empty((n, k // 16), dtype=np.uint8)
    clipped = 0
    max_preclip = 0.0
    for r0 in range(0, n, chunk_rows):
        r1 = min(n, r0 + chunk_rows)
        b = np.ascontiguousarray(w[r0:r1], dtype=np.float32).reshape(r1-r0, k//16, 16)
        desired = np.max(np.abs(b), axis=-1) / F4MAX / global_scale
        sraw, sval = e4_encode(desired, convention)
        eff = sval[..., None] * global_scale
        eff = np.where(eff == 0.0, 1.0, eff)
        norm = b / eff
        clipped += int(np.count_nonzero(np.abs(norm) > F4MAX))
        max_preclip = max(max_preclip, float(np.max(np.abs(norm))))
        norm = np.clip(norm, -F4MAX, F4MAX)
        c = fp4_codes(norm)
        codes[r0:r1] = (c[..., 0::2] | (c[..., 1::2] << 4)).reshape(r1-r0, k//2)
        scales[r0:r1] = sraw
    return {
        "codes": codes,
        "scales": scales,
        "global": global_scale,
        "clipped_values": clipped,
        "clipped_fraction": clipped / float(n*k),
        "max_preclip_abs": max_preclip,
        "bytes": int(codes.nbytes + scales.nbytes + 4),
    }


def quantize_fp8_tensor_matrix(w: np.ndarray) -> dict[str, Any]:
    """Tensor-wise E4M3 weight-only quantization for the existing W8A32 kernel."""
    if w.ndim != 2:
        raise ValueError(w.shape)
    amax = float(np.max(np.abs(w)))
    global_scale = max(amax / F8MAX, 1.0e-30)
    normalized = np.asarray(w, dtype=np.float32) / global_scale
    pos_raw, pos_val = e4_encode(np.abs(normalized), "RNE")
    raw = pos_raw | ((normalized < 0).astype(np.uint8) << 7)
    clipped = int(np.count_nonzero(np.abs(normalized) > F8MAX))
    return {
        "codes": np.ascontiguousarray(raw),
        "global": global_scale,
        "clipped_values": clipped,
        "clipped_fraction": clipped / float(w.size),
        "max_preclip_abs": float(np.max(np.abs(normalized))),
        "bytes": int(raw.nbytes + 4),
    }


def candidate_call(rt, spec, candidate, out, x, codes, scales=None):
    if candidate["_kind"] == "fp8_tensor":
        rt.k.mv_fp8_tensor(out, codes, x, candidate["_global"],
                           spec["rows"], spec["cols"])
    elif candidate["_kind"] == "nvfp4":
        rt.fused.gemv_into(out, codes, scales, x, candidate["_global"],
                           spec["rows"], spec["cols"])
    else:
        raise ValueError(candidate["_kind"])


def bf16_to_f32(raw_u16: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    u32 = raw_u16.reshape(shape).astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def decode_current(cp, spec: dict[str, Any]) -> np.ndarray:
    shape = (spec["rows"], spec["cols"])
    if spec["kind"] == "bf16":
        return bf16_to_f32(cp.asnumpy(spec["weight"]).astype(np.uint16, copy=False), shape)
    if spec["kind"] == "fp8_tensor":
        raw = cp.asnumpy(spec["weight"]).astype(np.uint8, copy=False).reshape(shape)
        return E4_FULL[raw.astype(np.int32)] * float(spec["scale"])
    raise ValueError(f"unsupported current kind {spec['kind']}")


def metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    aa = np.asarray(a, np.float64).reshape(-1)
    bb = np.asarray(b, np.float64).reshape(-1)
    d = aa - bb
    rmse = float(np.sqrt(np.mean(d*d)))
    rrms = float(np.sqrt(np.mean(bb*bb)))
    an, bn = float(np.linalg.norm(aa)), float(np.linalg.norm(bb))
    cos = float(np.dot(aa, bb) / max(an*bn, 1e-30))
    ma, rmax = float(np.max(np.abs(d))), float(np.max(np.abs(bb)))
    return {
        "normalized_rmse": rmse / max(rrms, 1e-30),
        "cosine": cos,
        "normalized_max_abs_error": ma / max(rmax, 1e-30),
        "rmse": rmse,
        "max_abs_error": ma,
    }


def timed(cp, fn, reps: int, rounds: int = 6) -> dict[str, Any]:
    for _ in range(4):
        fn()
    cp.cuda.get_current_stream().synchronize()
    vals = []
    for _ in range(rounds):
        a, b = cp.cuda.Event(), cp.cuda.Event()
        a.record()
        for _ in range(reps):
            fn()
        b.record(); b.synchronize()
        vals.append(float(cp.cuda.get_elapsed_time(a, b)) / reps)
    s = sorted(vals)
    return {"samples_ms": vals, "p50_ms": s[len(s)//2], "min_ms": s[0], "max_ms": s[-1]}


def current_call(rt, spec, out, x, weight):
    if spec["kind"] == "bf16":
        rt.k.mv_bf16(out, weight, x, spec["rows"], spec["cols"])
    elif spec["kind"] == "fp8_tensor":
        rt.k.mv_fp8_tensor(out, weight, x, spec["scale"], spec["rows"], spec["cols"])
    else:
        raise ValueError(spec["kind"])


def build_specs(rt) -> list[dict[str, Any]]:
    specs = []
    ai = int(rt.attn_layers[0])
    ad = rt.layer[ai]
    specs.extend([
        {"label": "attention_q", "layer": ai, "kind": "bf16", "weight": ad["q_proj"],
         "rows": rt.n_heads * rt.head_dim, "cols": rt.hidden, "activation": "attn_normed"},
        {"label": "attention_o", "layer": ai, "kind": "bf16", "weight": ad["o_proj"],
         "rows": rt.hidden, "cols": rt.n_heads * rt.head_dim, "activation": "attn_ctx"},
    ])

    for which in ("in", "out"):
        chosen = None
        for i in rt.mamba_layers:
            d = rt.layer[i]
            kind = d[f"{which}_k"]
            if kind == "nvfp4":
                continue
            if which == "in":
                rows, cols = int(rt.proj.size), rt.hidden
                weight = d["in_w8"] if kind == "fp8_tensor" else d["in_w"]
                scale = d.get("in_s")
                activation = "mamba_normed"
            else:
                rows, cols = rt.hidden, rt.d_inner
                weight = d["out_w8"] if kind == "fp8_tensor" else d["out_w"]
                scale = d.get("out_s")
                activation = "mamba_gn"
            chosen = {"label": f"mamba_{which}", "layer": int(i), "kind": kind,
                      "weight": weight, "scale": scale, "rows": rows, "cols": cols,
                      "activation": activation}
            break
        if chosen is not None:
            specs.append(chosen)
    return specs


def capture(rt, specs, count: int) -> dict[str, np.ndarray]:
    import cupy as cp
    wanted_attn = {s["layer"] for s in specs if s["label"].startswith("attention_")}
    wanted_mamba = {s["layer"] for s in specs if s["label"].startswith("mamba_")}
    data = {s["label"]: [] for s in specs}

    orig_attn = rt._attention
    orig_mamba = rt._mamba
    enabled = {"v": False}

    def attn_wrap(self, i, out):
        take = enabled["v"] and i in wanted_attn
        before = cp.asnumpy(self.normed).astype(np.float32, copy=True) if take else None
        result = orig_attn(i, out)
        if take:
            for s in specs:
                if s["layer"] == i and s["label"] == "attention_q":
                    data[s["label"]].append(before)
                elif s["layer"] == i and s["label"] == "attention_o":
                    data[s["label"]].append(cp.asnumpy(self.ctx).astype(np.float32, copy=True))
        return result

    def mamba_wrap(self, i, out):
        take = enabled["v"] and i in wanted_mamba
        before = cp.asnumpy(self.normed).astype(np.float32, copy=True) if take else None
        result = orig_mamba(i, out)
        if take:
            for s in specs:
                if s["layer"] != i:
                    continue
                if s["label"] == "mamba_in":
                    data[s["label"]].append(before)
                elif s["label"] == "mamba_out":
                    data[s["label"]].append(cp.asnumpy(self.gn).astype(np.float32, copy=True))
        return result

    rt._attention = types.MethodType(attn_wrap, rt)
    rt._mamba = types.MethodType(mamba_wrap, rt)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(require_model_dir()), local_files_only=True, trust_remote_code=True, use_fast=True
    )
    prompt = tok.encode("The history of computing began when", add_special_tokens=False)
    rt.reset()
    nxt = None
    for t in prompt:
        nxt = int(rt.step(int(t)))
    if nxt is None:
        raise RuntimeError("empty prompt")
    enabled["v"] = True
    cur = nxt
    for _ in range(count):
        cur = int(rt.step(cur))
    enabled["v"] = False
    cp.cuda.get_current_stream().synchronize()

    rt._attention = orig_attn
    rt._mamba = orig_mamba
    out = {}
    for s in specs:
        rows = data[s["label"]]
        if len(rows) != count:
            raise RuntimeError(f"{s['label']} captured {len(rows)} != {count}")
        out[s["label"]] = np.stack(rows)
    return out


def family_run(rt, spec, activations: np.ndarray, l2: int) -> dict[str, Any]:
    import cupy as cp

    w = decode_current(cp, spec)
    reference_outputs = []
    out = cp.empty(spec["rows"], dtype=cp.float32)
    for xh in activations:
        x = cp.asarray(xh)
        current_call(rt, spec, out, x, spec["weight"])
        cp.cuda.get_current_stream().synchronize()
        reference_outputs.append(cp.asnumpy(out).copy())
    ref = np.stack(reference_outputs)

    definitions: dict[str, dict[str, Any]] = {}
    if spec["kind"] == "bf16":
        q8 = quantize_fp8_tensor_matrix(w)
        definitions["FP8_RNE"] = {
            "_kind": "fp8_tensor",
            "_codes_host": q8["codes"],
            "_scales_host": None,
            "_global": q8["global"],
            "quantization": {k: v for k, v in q8.items() if k != "codes"},
        }
    for conv in ("RNE", "CEIL"):
        q4 = quantize_matrix(w, conv)
        definitions[f"NVFP4_{conv}"] = {
            "_kind": "nvfp4",
            "_codes_host": q4["codes"],
            "_scales_host": q4["scales"],
            "_global": q4["global"],
            "quantization": {k: v for k, v in q4.items() if k not in {"codes", "scales"}},
        }

    candidates: dict[str, dict[str, Any]] = {}
    for arm, c in definitions.items():
        codes = cp.asarray(c["_codes_host"])
        scales = cp.asarray(c["_scales_host"]) if c["_scales_host"] is not None else None
        got = []
        for xh in activations:
            x = cp.asarray(xh)
            candidate_call(rt, spec, c, out, x, codes, scales)
            cp.cuda.get_current_stream().synchronize()
            got.append(cp.asnumpy(out).copy())
        actual = np.stack(got)

        x0 = cp.asarray(activations[0])
        candidate_call(rt, spec, c, out, x0, codes, scales)
        cp.cuda.get_current_stream().synchronize()
        repeat_a = cp.asnumpy(out).copy()
        candidate_call(rt, spec, c, out, x0, codes, scales)
        cp.cuda.get_current_stream().synchronize()
        repeat_b = cp.asnumpy(out).copy()
        deterministic = bool(np.array_equal(repeat_a.view(np.uint32),
                                             repeat_b.view(np.uint32)))

        ctrl_codes_h = np.roll(c["_codes_host"], 1, axis=0).copy()
        ctrl_codes = cp.asarray(ctrl_codes_h)
        if c["_scales_host"] is not None:
            ctrl_scales = cp.asarray(np.roll(c["_scales_host"], 1, axis=0).copy())
        else:
            ctrl_scales = None
        candidate_call(rt, spec, c, out, x0, ctrl_codes, ctrl_scales)
        cp.cuda.get_current_stream().synchronize()
        control = cp.asnumpy(out).copy()
        control_diverged = bool(np.count_nonzero(
            control.view(np.uint32) != repeat_a.view(np.uint32)
        ) > 0)
        del ctrl_codes, ctrl_scales, repeat_a, repeat_b, control, x0

        candidates[arm] = {
            "kind": c["_kind"],
            "quantization": c["quantization"],
            "output_metrics": metrics(actual, ref),
            "finite": bool(np.isfinite(actual).all()),
            "deterministic_repeat": deterministic,
            "control_row_rotation_diverged": control_diverged,
            "_kind": c["_kind"],
            "_codes": codes,
            "_scales": scales,
            "_global": c["_global"],
        }

    # Cold A/C/C/B. The original and each candidate independently rotate >=4x L2.
    orig_bytes = int(spec["weight"].nbytes)
    co = max(1, math.ceil(L2_MULTIPLE * l2 / max(orig_bytes, 1)))
    free, _ = cp.cuda.runtime.memGetInfo()
    original_estimate = co * orig_bytes + 128 * 1024 * 1024
    timing: dict[str, Any] = {
        "free_before": int(free),
        "original_estimated_bytes": int(original_estimate),
    }
    if original_estimate > int(free * 0.55):
        timing["status"] = "not_run_memory_gate"
    else:
        origs = [spec["weight"]] + [spec["weight"].copy() for _ in range(1, co)]
        measured_arms = 0
        for arm, c in candidates.items():
            cand_bytes = int(c["_codes"].nbytes + (c["_scales"].nbytes if c["_scales"] is not None else 0))
            cc = max(1, math.ceil(L2_MULTIPLE * l2 / max(cand_bytes, 1)))
            estimate = co * orig_bytes + cc * cand_bytes + 128 * 1024 * 1024
            if estimate > int(free * 0.70):
                timing[arm] = {
                    "status": "not_run_memory_gate",
                    "estimated_bytes": int(estimate),
                    "current_rotation_over_l2": co * orig_bytes / l2,
                    "candidate_rotation_over_l2": cc * cand_bytes / l2,
                }
                continue
            code_rots = [c["_codes"]] + [c["_codes"].copy() for _ in range(1, cc)]
            if c["_scales"] is not None:
                scale_rots = [c["_scales"]] + [c["_scales"].copy() for _ in range(1, cc)]
            else:
                scale_rots = [None] * cc
            x = cp.asarray(activations[0])
            out_ref = cp.empty(spec["rows"], dtype=cp.float32)
            out_cand = cp.empty(spec["rows"], dtype=cp.float32)
            io, ic = {"i": 0}, {"i": 0}

            def ref_call():
                i = io["i"]; io["i"] += 1
                current_call(rt, spec, out_ref, x, origs[i % co])

            def cand_call():
                i = ic["i"]; ic["i"] += 1
                j = i % cc
                candidate_call(rt, spec, c, out_cand, x,
                               code_rots[j], scale_rots[j])

            reps = max(co, cc, 14) * 3
            ra = timed(cp, ref_call, reps)
            ca = timed(cp, cand_call, reps)
            cb = timed(cp, cand_call, reps)
            rb = timed(cp, ref_call, reps)
            ref_mid = 0.5 * (ra["p50_ms"] + rb["p50_ms"])
            cand_mid = 0.5 * (ca["p50_ms"] + cb["p50_ms"])
            timing[arm] = {
                "status": "measured",
                "reference_a": ra, "candidate_a": ca,
                "candidate_b": cb, "reference_b": rb,
                "reference_mid_ms": ref_mid,
                "candidate_mid_ms": cand_mid,
                "speedup": ref_mid / cand_mid if cand_mid else None,
                "reference_drift_fraction": abs(ra["p50_ms"] - rb["p50_ms"]) / ref_mid,
                "current_rotation_over_l2": co * orig_bytes / l2,
                "candidate_rotation_over_l2": cc * cand_bytes / l2,
                "current_physical_bytes": orig_bytes,
                "candidate_physical_bytes": cand_bytes,
                "estimated_bytes": int(estimate),
            }
            measured_arms += 1
            del code_rots, scale_rots, x, out_ref, out_cand
            cp.get_default_memory_pool().free_all_blocks()
        timing["status"] = "measured" if measured_arms else "not_run_memory_gate"
        del origs

    clean = {}
    for arm, rec in candidates.items():
        clean[arm] = {k: v for k, v in rec.items() if not k.startswith("_")}
    del w, ref, reference_outputs, candidates, definitions
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return {
        "label": spec["label"], "layer": spec["layer"], "current_kind": spec["kind"],
        "shape": [spec["rows"], spec["cols"]], "activation_rows": len(activations),
        "candidates": clean, "timing": timing,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload = {
        "kind": "s100_d4_weight_only_dense",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "representative real-activation weight-only W8A32/W4A32 microbenchmark; no integration or tok/s claim",
    }
    try:
        from diag_fp4_activation_quality import _require_gpu_idle_wddm
        payload["gpu_idle_preflight"] = _require_gpu_idle_wddm()
        import cupy as cp
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

        rt = LightningRuntime(require_model_dir(), contexts_max=4096,
                              embed_on_host=True, fp8_kv=True, verbose=False)
        rt.enable_cache(72)
        rt.load_routed_bank()
        rt.device_cache = True
        rt.deterministic_accum = True
        specs = build_specs(rt)
        captures = capture(rt, specs, ROWS if args.mode == "full" else 8)

        # Capture is complete; release the 4.3 GiB expert cache before cold rotations.
        rt.cache = {}
        rt._dev_cache = {}
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        props = cp.cuda.runtime.getDeviceProperties(0)
        if isinstance(props, dict):
            l2 = int(props.get("l2CacheSize", props.get(b"l2CacheSize", 0)) or 0)
        else:
            l2 = 0
        if l2 <= 0:
            # The target machine is known to have 32 MiB; fail closed rather than
            # silently call a warm set cold when the API spelling changes.
            raise RuntimeError("unable to read L2 cache size from CuPy device properties")

        families = [family_run(rt, s, captures[s["label"]], l2) for s in specs]
        gates = {}
        family_joint_pass = {}
        family_key_pass = {}
        for f in families:
            any_joint = False
            any_key = False
            for arm, q in f["candidates"].items():
                m = q["output_metrics"]
                quality = bool(
                    q["finite"] and q["deterministic_repeat"]
                    and q["control_row_rotation_diverged"]
                    and m["cosine"] >= 0.995
                    and m["normalized_rmse"] <= 0.100
                    and m["normalized_max_abs_error"] <= 0.250
                )
                t = f["timing"].get(arm) or {}
                speed = t.get("speedup")
                speed150 = bool(
                    f["timing"].get("status") == "measured"
                    and speed is not None and speed >= 1.50
                    and float(t.get("reference_drift_fraction", 1.0)) <= 0.05
                )
                speed135 = bool(
                    f["timing"].get("status") == "measured"
                    and speed is not None and speed >= 1.35
                    and float(t.get("reference_drift_fraction", 1.0)) <= 0.05
                )
                gates[f"{f['label']}_{arm}_quality"] = quality
                gates[f"{f['label']}_{arm}_speed_ge_1_50"] = speed150
                gates[f"{f['label']}_{arm}_joint"] = quality and speed150
                any_joint = any_joint or (quality and speed150)
                any_key = any_key or (quality and speed135)
            family_joint_pass[f["label"]] = any_joint
            family_key_pass[f["label"]] = any_key

        minimum_families = min(3, len(families))
        family_count = sum(bool(v) for v in family_joint_pass.values())
        key_ok = all(
            family_key_pass.get(name, True)
            for name in ("attention_q", "mamba_in")
            if name in family_key_pass
        )
        gates["D4_P_family_joint_pass_count"] = family_count >= minimum_families
        gates["D4_P_attention_q_and_mamba_in_ge_1_35"] = key_ok
        payload.update({
            "environment": environment_snapshot((Path(__file__), PREREG)),
            "l2_bytes": l2,
            "representatives": [{k: v for k, v in s.items() if k not in {"weight"}}
                                for s in specs],
            "families": families,
            "gates": gates,
            "family_joint_pass": family_joint_pass,
            "family_key_pass": family_key_pass,
            "status": "micro_candidate" if (family_count >= minimum_families and key_ok) else "micro_below_gate",
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        import traceback
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "gates": payload.get("gates"),
        "families": [
            {"label": f["label"],
             "kind": f["current_kind"],
             "quality": {a: r["output_metrics"] for a, r in f["candidates"].items()},
             "timing": f["timing"]}
            for f in payload.get("families", [])
        ],
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2, allow_nan=False))
    return 2 if payload.get("status") == "technical_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())


