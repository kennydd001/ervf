"""Capture a post-hoc BF16-activation proxy trace for Phase38 DFlash."""
from __future__ import annotations

import hashlib
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, sha256_file, utc_now, write_json_atomic
from diag_native_nvfp4_c3a_real_weight_v2 import require_gpu_idle_wddm
from s100_phase21_common import identity_gate, load_trace, make_rt, release
from s100_phase38_dflash_capture import CAPTURE_LAYERS, COUNT, _checkpoint_identity, _write_raw

RESULTS = REPO / "pro_research" / "results" / "s100_phase38"
META = RESULTS / "S100_PHASE38_DFLASH_BF16_PROXY_CAPTURE.json"
HIDDEN_RAW = RESULTS / "target_aux_hidden_bf16_residual_proxy.f32"
TOKENS_RAW = RESULTS / "tokens_bf16_residual_proxy.i32"
PREREG = REPO / "pro_research" / "S100_PHASE38_DFLASH_PREREGISTRATION.md"
DFLASH_SNAPSHOT = (
    Path(r"C:\Users\de_do\.cache\huggingface\hub")
    / "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash"
    / "snapshots"
    / "7fc1f1ff4b82b917efbd0710df0872c2bb89caa5"
)


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return None


def _capture_proxy(
    rt: Any,
    prompt_tokens: list[int],
    count: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    import cupy as cp

    layer_to_slot = {layer: slot for slot, layer in enumerate(CAPTURE_LAYERS)}
    device = cp.empty((count, len(CAPTURE_LAYERS), int(rt.hidden)), dtype=cp.float32)
    round_bf16_rne = cp.ElementwiseKernel(
        "float32 x",
        "float32 y",
        r"""
        unsigned int bits = __float_as_uint(x);
        unsigned int exponent = bits & 0x7f800000u;
        if (exponent == 0x7f800000u) {
            y = x;
        } else {
            unsigned int lsb = (bits >> 16) & 1u;
            bits = (bits + 0x7fffu + lsb) & 0xffff0000u;
            y = __uint_as_float(bits);
        }
        """,
        "phase38_round_bf16_rne",
    )

    predictions: list[int] = []
    input_tokens: list[int] = []
    cp_mod, kernels = rt.cp, rt.k
    prompt_length = len(prompt_tokens)
    rt.reset()

    for token_index in range(count):
        token_id = (
            int(prompt_tokens[token_index])
            if token_index < prompt_length
            else int(predictions[-1])
        )
        input_tokens.append(token_id)
        if rt.embed_on_host:
            row = cp_mod.asarray(
                rt.embed_host[token_id * rt.hidden:(token_id + 1) * rt.hidden]
            )
        else:
            row = rt.embed[token_id * rt.hidden:(token_id + 1) * rt.hidden]
        rt.h[:] = (row.astype(cp_mod.uint32) << cp_mod.uint32(16)).view(cp_mod.float32)

        for layer_index, layer_type in enumerate(rt.pattern):
            layer = rt.layer[layer_index]
            kernels.norm(rt.normed, rt.h, layer["norm"], rt.hidden, rt.eps)
            round_bf16_rne(rt.normed, rt.normed)
            if layer_type == "M":
                rt._mamba(layer_index, rt.acc)
            elif layer_type == "*":
                rt._attention(layer_index, rt.acc)
            else:
                rt._moe(layer_index, rt.acc)
            round_bf16_rne(rt.acc, rt.acc)
            kernels.add_(rt.h, rt.acc, rt.hidden)
            round_bf16_rne(rt.h, rt.h)
            slot = layer_to_slot.get(layer_index)
            if slot is not None:
                cp_mod.copyto(device[token_index, slot], rt.h)

        kernels.norm(rt.normed, rt.h, rt.norm_f, rt.hidden, rt.eps)
        round_bf16_rne(rt.normed, rt.normed)
        if rt.lm_head_kind == "nvfp4":
            rt.fused.gemv_into(
                rt.logits,
                rt.lm_head_codes,
                rt.lm_head_scales,
                rt.normed,
                rt.lm_head_g,
                rt.vocab,
                rt.hidden,
            )
        else:
            kernels.mv_bf16(rt.logits, rt.lm_head, rt.normed, rt.vocab, rt.hidden)
        rt.pos += 1
        predictions.append(int(cp_mod.argmax(rt.logits)))

    cp.cuda.get_current_stream().synchronize()
    host = cp.asnumpy(device)
    tokens = np.asarray(input_tokens + [predictions[-1]], dtype="<i4")
    if not np.isfinite(host).all():
        raise RuntimeError("BF16 proxy hidden capture contains non-finite values")
    if not np.array_equal(
        np.asarray(predictions[prompt_length - 1:], dtype=np.int32),
        tokens[prompt_length:],
    ):
        raise RuntimeError("BF16 proxy continuation is not self-consistent")
    return host, tokens, predictions


def main() -> int:
    payload: dict[str, Any] = {
        "kind": "s100_phase38_dflash_bf16_residual_proxy_capture",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "post-hoc BF16 activation sensitivity proxy; not official vLLM target parity",
    }
    rt = None
    try:
        model_dir, target_config, _p20s, _p20b = identity_gate()
        trace = load_trace()
        canonical = np.asarray(trace["tokens"][:COUNT + 1], dtype="<i4")
        prompt_length = int(trace.get("prompt_length", 0))
        if prompt_length < 1 or canonical.size != COUNT + 1:
            raise RuntimeError("invalid canonical prompt/trace for BF16 proxy")
        payload["gpu_idle_preflight"] = require_gpu_idle_wddm()
        payload["dflash_checkpoint"] = _checkpoint_identity(DFLASH_SNAPSHOT)

        rt, keep = make_rt(COUNT, "v6_device_rows")
        hidden, tokens, predictions = _capture_proxy(
            rt, canonical[:prompt_length].tolist(), COUNT
        )
        continuation_equal = tokens[prompt_length:] == canonical[prompt_length:]
        mismatch_offsets = np.flatnonzero(~continuation_equal)
        first_divergence = (
            int(prompt_length + mismatch_offsets[0]) if mismatch_offsets.size else None
        )
        matching_prefix = (
            int(mismatch_offsets[0]) if mismatch_offsets.size else int(continuation_equal.size)
        )
        arrays = {
            "target_aux_hidden": _write_raw(HIDDEN_RAW, hidden, "<f4"),
            "tokens": _write_raw(TOKENS_RAW, tokens, "<i4"),
        }
        payload.update({
            "status": "captured",
            "completed_utc": utc_now(),
            "git_head": _git_head(),
            "environment": environment_snapshot((Path(__file__), PREREG)),
            "target": {
                "snapshot": model_dir.name,
                "config_sha256": sha256_file(model_dir / "config.json"),
                "model_type": target_config.get("model_type"),
                "runtime_arm": "v6_device_rows_bf16_activation_proxy",
            },
            "trace": {
                "prompt_length": prompt_length,
                "generated_inputs": COUNT,
                "self_consistent_continuation": True,
                "canonical_matching_continuation_prefix_tokens": matching_prefix,
                "first_canonical_divergence_position": first_divergence,
                "proxy_token_ids_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
                "prediction_ids_sha256": hashlib.sha256(
                    np.asarray(predictions, dtype="<i4").tobytes()
                ).hexdigest(),
            },
            "capture": {
                "target_layer_ids_zero_based": list(CAPTURE_LAYERS),
                "shape": list(hidden.shape),
                "rounding": "BF16 RNE then widen FP32 after normalized layer input, branch output, residual add and final norm",
                "limitations": "internal Mamba/MoE temporaries and recurrent states retain custom-runtime storage",
                "all_finite": True,
            },
            "arrays": arrays,
        })
        del keep
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "completed_utc": utc_now(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        })
    finally:
        if rt is not None:
            release(rt)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json_atomic(META, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "trace": payload.get("trace"),
        "arrays": payload.get("arrays"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(META),
    }, indent=2))
    return 0 if payload.get("status") == "captured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
