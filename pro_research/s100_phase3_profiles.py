"""Frozen phase-3 candidate profiles."""
from __future__ import annotations

import gc
import types
from typing import Any

import numpy as np

from d4_mixed_runtime import (
    _attention_mixed,
    _clear_attention,
    _install_attention,
    apply_profile as apply_d4_profile,
    public_install_record as public_d4_record,
    restore_checkpoint_dense as restore_d4_profile,
)

TIMING_PROFILES = ("qfast", "mamba", "fast", "k5", "k4", "fast_k5", "fast_k4")
FIDELITY_PROFILES = (
    "qfast", "mamba", "fast", "k5", "k4", "fast_k5", "fast_k4", "k1_control"
)
_PROFILE = {
    "qfast": {"weights": "qfast", "top_k": 6},
    "mamba": {"weights": "mamba", "top_k": 6},
    "fast": {"weights": "fast", "top_k": 6},
    "k5": {"weights": None, "top_k": 5},
    "k4": {"weights": None, "top_k": 4},
    "fast_k5": {"weights": "fast", "top_k": 5},
    "fast_k4": {"weights": "fast", "top_k": 4},
    "k1_control": {"weights": None, "top_k": 1},
}


def profile_spec(name: str) -> dict[str, Any]:
    try:
        return dict(_PROFILE[name.lower()])
    except KeyError as exc:
        raise ValueError(f"unknown phase-3 profile {name!r}") from exc


def _restore_qfast(rt, applied: dict[str, Any]) -> None:
    cp = rt.cp
    rt._graph = None
    cp.cuda.get_current_stream().synchronize()
    rt._attention = applied["_original_attention"]
    for layer in rt.attn_layers:
        d = rt.layer[layer]
        _clear_attention(d, "q")
        name = f"backbone.layers.{layer}.mixer.q_proj.weight"
        d["q_proj"] = cp.asarray(rt.index.read_raw(name).view(np.uint16))
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()


def _apply_qfast(rt) -> dict[str, Any]:
    manifest = []
    original_attention = rt._attention
    rt._d4_attention_capture_calls = 0
    for layer in rt.attn_layers:
        manifest.append(_install_attention(rt, int(layer), "q", "nvfp4_ceil"))
    rt._attention = types.MethodType(_attention_mixed, rt)
    changed = [m for m in manifest if m.get("changed")]
    return {
        "kind": "qfast",
        "manifest": manifest,
        "changed_matrix_count": len(changed),
        "current_bytes": sum(int(x.get("current_bytes", 0)) for x in changed),
        "candidate_bytes": sum(int(x.get("candidate_bytes", 0)) for x in changed),
        "bytes_saved": sum(int(x.get("bytes_saved", 0)) for x in changed),
        "_original_attention": original_attention,
    }


def apply_phase3_profile(rt, name: str) -> dict[str, Any]:
    name = name.lower()
    spec = profile_spec(name)
    original_top_k = int(rt.top_k)
    applied: dict[str, Any] = {
        "profile": name,
        "spec": spec,
        "original_top_k": original_top_k,
        "candidate_top_k": int(spec["top_k"]),
        "top_k_changed": int(spec["top_k"]) != original_top_k,
        "_weight_install": None,
        "_weight_kind": spec["weights"],
    }
    if spec["weights"] == "qfast":
        applied["_weight_install"] = _apply_qfast(rt)
        applied["weights"] = {
            k: v for k, v in applied["_weight_install"].items()
            if not k.startswith("_")
        }
    elif spec["weights"] in {"mamba", "fast"}:
        install = apply_d4_profile(rt, str(spec["weights"]))
        applied["_weight_install"] = install
        applied["weights"] = public_d4_record(install)
    else:
        applied["weights"] = {
            "kind": "exact_checkpoint", "changed_matrix_count": 0, "bytes_saved": 0
        }
    rt.top_k = int(spec["top_k"])
    return applied


def restore_phase3_profile(rt, applied: dict[str, Any]) -> None:
    kind = applied.get("_weight_kind")
    install = applied.get("_weight_install")
    if kind == "qfast" and install is not None:
        _restore_qfast(rt, install)
    elif kind in {"mamba", "fast"} and install is not None:
        restore_d4_profile(rt, install)
    rt.top_k = int(applied["original_top_k"])


def public_profile_record(applied: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in applied.items() if not k.startswith("_")}
