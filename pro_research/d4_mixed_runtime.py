"""Selective D4 weight conversion and reversible runtime installation.

This module changes only resident dense weight representations. It does not
change routed experts, routing, state recurrence, KV arithmetic or MoE order.
"""
from __future__ import annotations

import gc
import hashlib
import types
from typing import Any

import numpy as np

from diag_s100_d4_weight_only_dense import (
    E4_FULL,
    quantize_fp8_tensor_matrix,
    quantize_matrix,
)

PROFILES = ("mamba", "safe", "fast")


def _bf16_to_f32(raw: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    u16 = np.asarray(raw).view(np.uint16).reshape(shape)
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def _source_matrix(rt, base: str, kind: str,
                   shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    raw = np.asarray(rt.index.read_raw(base + ".weight"))
    if kind == "bf16":
        return _bf16_to_f32(raw, shape), int(raw.nbytes)
    if kind == "fp8_tensor":
        scale = float(rt.index.get_scalar(base + ".weight_scale"))
        codes = raw.astype(np.uint8, copy=False).reshape(shape)
        return E4_FULL[codes.astype(np.int32)] * scale, int(raw.nbytes + 4)
    raise ValueError(f"{base}: unsupported source kind {kind!r}")


def _sha_parts(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def _clear_mamba(d: dict[str, Any], side: str) -> None:
    for key in (
        f"{side}_codes", f"{side}_scales", f"{side}_g",
        f"{side}_w8", f"{side}_s", f"{side}_w",
        f"{side}_k", f"{side}_q",
    ):
        d.pop(key, None)


def _clear_attention(d: dict[str, Any], name: str) -> None:
    d.pop(f"{name}_proj", None)
    for key in (
        f"{name}_kind", f"{name}_codes", f"{name}_scales",
        f"{name}_g", f"{name}_w8", f"{name}_s",
    ):
        d.pop(key, None)


def _install_mamba_nvfp4(rt, layer: int, side: str,
                         convention: str = "CEIL") -> dict[str, Any]:
    cp = rt.cp
    d = rt.layer[layer]
    base = f"backbone.layers.{layer}.mixer.{side}_proj"
    kind = str(d[f"{side}_k"])
    rows, cols = (
        (int(rt.proj.size), int(rt.hidden))
        if side == "in"
        else (int(rt.hidden), int(rt.d_inner))
    )
    if kind == "nvfp4":
        return {
            "layer": layer, "family": f"mamba_{side}",
            "source_kind": kind, "candidate_kind": kind,
            "shape": [rows, cols], "changed": False,
        }

    w, current_bytes = _source_matrix(rt, base, kind, (rows, cols))
    q = quantize_matrix(w, convention)
    codes_h = q["codes"]
    scales_h = q["scales"]
    candidate_bytes = int(q["bytes"])
    digest = _sha_parts(codes_h, scales_h)

    rt._graph = None
    cp.cuda.get_current_stream().synchronize()
    _clear_mamba(d, side)
    cp.get_default_memory_pool().free_all_blocks()

    d[f"{side}_k"] = "nvfp4"
    d[f"{side}_q"] = True
    d[f"{side}_codes"] = cp.asarray(codes_h).reshape(-1)
    d[f"{side}_scales"] = cp.asarray(scales_h).reshape(-1)
    d[f"{side}_g"] = float(q["global"])

    del w, q, codes_h, scales_h
    gc.collect()
    return {
        "layer": layer, "family": f"mamba_{side}",
        "base": base, "source_kind": kind,
        "candidate_kind": "nvfp4_ceil",
        "shape": [rows, cols], "changed": True,
        "current_bytes": current_bytes,
        "candidate_bytes": candidate_bytes,
        "bytes_saved": current_bytes - candidate_bytes,
        "candidate_sha256": digest,
    }


def _install_attention(rt, layer: int, name: str,
                       candidate_kind: str) -> dict[str, Any]:
    cp = rt.cp
    d = rt.layer[layer]
    base = f"backbone.layers.{layer}.mixer.{name}_proj"
    rows, cols = (
        (int(rt.n_heads * rt.head_dim), int(rt.hidden))
        if name == "q"
        else (int(rt.hidden), int(rt.n_heads * rt.head_dim))
    )
    w, current_bytes = _source_matrix(rt, base, "bf16", (rows, cols))

    if candidate_kind == "fp8_tensor":
        q = quantize_fp8_tensor_matrix(w)
        codes_h = q["codes"]
        candidate_bytes = int(q["bytes"])
        digest = _sha_parts(codes_h)
        scales_h = None
    elif candidate_kind == "nvfp4_ceil":
        q = quantize_matrix(w, "CEIL")
        codes_h = q["codes"]
        scales_h = q["scales"]
        candidate_bytes = int(q["bytes"])
        digest = _sha_parts(codes_h, scales_h)
    else:
        raise ValueError(candidate_kind)

    rt._graph = None
    cp.cuda.get_current_stream().synchronize()
    _clear_attention(d, name)
    cp.get_default_memory_pool().free_all_blocks()

    d[f"{name}_kind"] = (
        "fp8_tensor" if candidate_kind == "fp8_tensor" else "nvfp4"
    )
    if candidate_kind == "fp8_tensor":
        d[f"{name}_w8"] = cp.asarray(codes_h).reshape(-1)
        d[f"{name}_s"] = float(q["global"])
    else:
        d[f"{name}_codes"] = cp.asarray(codes_h).reshape(-1)
        d[f"{name}_scales"] = cp.asarray(scales_h).reshape(-1)
        d[f"{name}_g"] = float(q["global"])

    del w, q, codes_h, scales_h
    gc.collect()
    return {
        "layer": layer, "family": f"attention_{name}",
        "base": base, "source_kind": "bf16",
        "candidate_kind": candidate_kind,
        "shape": [rows, cols], "changed": True,
        "current_bytes": current_bytes,
        "candidate_bytes": candidate_bytes,
        "bytes_saved": current_bytes - candidate_bytes,
        "candidate_sha256": digest,
    }


def _mixed_linear(self, d, name: str, out, x,
                  rows: int, cols: int) -> None:
    kind = d[f"{name}_kind"]
    if kind == "fp8_tensor":
        self.k.mv_fp8_tensor(
            out, d[f"{name}_w8"], x, d[f"{name}_s"], rows, cols
        )
    elif kind == "nvfp4":
        self.fused.gemv_into(
            out, d[f"{name}_codes"], d[f"{name}_scales"], x,
            d[f"{name}_g"], rows, cols
        )
    else:
        raise RuntimeError(f"unexpected mixed attention kind {kind!r}")


def _attention_mixed(self, i, out):
    """Runtime._attention with q representation dispatch only."""
    import numpy as _np

    k, d = self.k, self.layer[i]
    self._d4_attention_capture_calls += 1
    _mixed_linear(
        self, d, "q", self.qv, self.normed,
        self.n_heads * self.head_dim, self.hidden
    )
    k.mv_bf16(
        self.kv_, d["k_proj"], self.normed,
        self.kv_dim, self.hidden
    )
    k.mv_bf16(
        self.vv, d["v_proj"], self.normed,
        self.kv_dim, self.hidden
    )

    scale = 1.0 / float(_np.sqrt(self.head_dim))
    if self.fp8_kv and self.graph_mode:
        k.kv_write_fp8_dp(
            self.kc[i], self.kv_, self._pos_dev,
            self.n_kv, self.head_dim, self.max_ctx
        )
        k.kv_write_fp8_dp(
            self.vc[i], self.vv, self._pos_dev,
            self.n_kv, self.head_dim, self.max_ctx
        )
        k.attention_fp8_gqa4_dp(
            self.ctx, self.qv, self.kc[i], self.vc[i],
            self._pos_dev, self.n_heads, self.head_dim,
            self.groups, self.max_ctx, scale,
            self.part_acc, self.part_ml
        )
        k.mv_bf16(
            out, d["o_proj"], self.ctx,
            self.hidden, self.n_heads * self.head_dim
        )
        return

    t = self.pos + 1
    if self.fp8_kv:
        k.kv_write_fp8(
            self.kc[i], self.kv_, self.pos,
            self.n_kv, self.head_dim, self.max_ctx
        )
        k.kv_write_fp8(
            self.vc[i], self.vv, self.pos,
            self.n_kv, self.head_dim, self.max_ctx
        )
        self.attn(
            self.ctx, self.qv, self.kc[i], self.vc[i], t,
            self.n_heads, self.head_dim, self.groups,
            self.max_ctx, scale, self.part_acc, self.part_ml
        )
    else:
        k.kv_write(
            self.kc[i], self.kv_, self.pos,
            self.n_kv, self.head_dim, self.max_ctx
        )
        k.kv_write(
            self.vc[i], self.vv, self.pos,
            self.n_kv, self.head_dim, self.max_ctx
        )
        k.attention(
            self.ctx, self.qv, self.kc[i], self.vc[i], t,
            self.n_heads, self.head_dim, self.groups, self.max_ctx,
            scale, self.part_acc, self.part_ml
        )
    k.mv_bf16(
        out, d["o_proj"], self.ctx,
        self.hidden, self.n_heads * self.head_dim
    )


def apply_profile(rt, profile: str) -> dict[str, Any]:
    profile = profile.lower()
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")

    cp = rt.cp
    rt._graph = None
    cp.cuda.get_current_stream().synchronize()
    manifest: list[dict[str, Any]] = []

    for layer in rt.mamba_layers:
        manifest.append(_install_mamba_nvfp4(rt, int(layer), "in"))
        manifest.append(_install_mamba_nvfp4(rt, int(layer), "out"))

    original_attention = rt._attention
    rt._d4_attention_capture_calls = 0
    if profile != "mamba":
        q_kind = "fp8_tensor" if profile == "safe" else "nvfp4_ceil"
        for layer in rt.attn_layers:
            manifest.append(
                _install_attention(rt, int(layer), "q", q_kind)
            )
        rt._attention = types.MethodType(_attention_mixed, rt)

    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    changed = [m for m in manifest if m.get("changed")]
    return {
        "profile": profile,
        "manifest": manifest,
        "changed_matrix_count": len(changed),
        "current_bytes": sum(int(m.get("current_bytes", 0)) for m in changed),
        "candidate_bytes": sum(
            int(m.get("candidate_bytes", 0)) for m in changed
        ),
        "bytes_saved": sum(int(m.get("bytes_saved", 0)) for m in changed),
        "_original_attention": original_attention,
    }


def restore_checkpoint_dense(rt, install: dict[str, Any]) -> None:
    cp = rt.cp
    rt._graph = None
    cp.cuda.get_current_stream().synchronize()
    rt._attention = install["_original_attention"]

    for layer in rt.mamba_layers:
        d = rt.layer[layer]
        for side in ("in", "out"):
            _clear_mamba(d, side)
            base = f"backbone.layers.{layer}.mixer.{side}_proj"
            kind = rt.index.quant_kind(base)
            d[f"{side}_k"] = kind
            d[f"{side}_q"] = kind == "nvfp4"
            if kind == "nvfp4":
                d[f"{side}_codes"] = cp.asarray(
                    rt.index.read_raw(base + ".weight")
                )
                d[f"{side}_scales"] = cp.asarray(
                    rt.index.read_raw(base + ".weight_scale")
                )
                d[f"{side}_g"] = float(
                    rt.index.get_scalar(base + ".weight_scale_2")
                )
            elif kind == "fp8_tensor":
                d[f"{side}_w8"] = cp.asarray(
                    rt.index.read_raw(base + ".weight")
                )
                d[f"{side}_s"] = float(
                    rt.index.get_scalar(base + ".weight_scale")
                )
            else:
                d[f"{side}_w"] = cp.asarray(
                    rt.index.read_raw(base + ".weight").view(np.uint16)
                )

    if install["profile"] != "mamba":
        for layer in rt.attn_layers:
            d = rt.layer[layer]
            _clear_attention(d, "q")
            base = f"backbone.layers.{layer}.mixer.q_proj.weight"
            d["q_proj"] = cp.asarray(
                rt.index.read_raw(base).view(np.uint16)
            )

    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()


def public_install_record(install: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in install.items() if not k.startswith("_")}
