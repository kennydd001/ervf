"""Capture exact target auxiliary hidden states for official DFlash evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, sha256_file, utc_now, write_json_atomic
from diag_native_nvfp4_c3a_real_weight_v2 import require_gpu_idle_wddm
from s100_phase21_common import SNAPSHOT, identity_gate, load_trace, make_rt, release

COUNT = 512
CAPTURE_LAYERS = (1, 5, 19, 29, 41, 51)
RESULTS = REPO / "pro_research" / "results" / "s100_phase38"
META = RESULTS / "S100_PHASE38_TARGET_CAPTURE.json"
HIDDEN_RAW = RESULTS / "target_aux_hidden.f32"
TOKENS_RAW = RESULTS / "tokens.i32"
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


def _write_raw(path: Path, array: np.ndarray, dtype: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(array.astype(np.dtype(dtype), copy=False))
    contiguous.tofile(path)
    return {
        "path": str(path.relative_to(REPO)),
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _checkpoint_identity(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    model_path = model_dir / "model.safetensors"
    if model_dir.name != "7fc1f1ff4b82b917efbd0710df0872c2bb89caa5":
        raise RuntimeError(f"wrong DFlash snapshot directory: {model_dir.name}")
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"incomplete DFlash snapshot: {model_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("architectures") != ["DFlashDraftModel"]:
        raise RuntimeError("checkpoint is not the official DFlashDraftModel")
    if tuple(config.get("target_layer_ids", ())) != CAPTURE_LAYERS:
        raise RuntimeError("DFlash target-layer contract changed")
    if int(config.get("mask_token_id", -1)) != 990:
        raise RuntimeError("DFlash mask token contract changed")
    return {
        "snapshot": model_dir.name,
        "config_sha256": sha256_file(config_path),
        "model_bytes": int(model_path.stat().st_size),
        # The 1.18 GB model hash is intentionally deferred to Stage B, where the
        # file is read anyway. Config + snapshot identity gates Stage A.
    }


def _capture(
    rt: Any, trace_tokens: list[int], count: int, prompt_length: int
) -> tuple[np.ndarray, list[int], int]:
    import cupy as cp

    if tuple(CAPTURE_LAYERS) != tuple(sorted(CAPTURE_LAYERS)):
        raise RuntimeError("capture layers must be ordered")
    layer_to_slot = {layer: slot for slot, layer in enumerate(CAPTURE_LAYERS)}
    device = cp.empty((count, len(CAPTURE_LAYERS), int(rt.hidden)), dtype=cp.float32)
    predictions: list[int] = []
    cp_mod, kernels = rt.cp, rt.k

    rt.reset()
    for token_index in range(count):
        token_id = int(trace_tokens[token_index])
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
            if layer_type == "M":
                rt._mamba(layer_index, rt.acc)
            elif layer_type == "*":
                rt._attention(layer_index, rt.acc)
            else:
                rt._moe(layer_index, rt.acc)
            kernels.add_(rt.h, rt.acc, rt.hidden)
            slot = layer_to_slot.get(layer_index)
            if slot is not None:
                cp_mod.copyto(device[token_index, slot], rt.h)

        kernels.norm(rt.normed, rt.h, rt.norm_f, rt.hidden, rt.eps)
        if rt.lm_head_kind == "nvfp4":
            rt.fused.gemv_into(
                rt.logits, rt.lm_head_codes, rt.lm_head_scales,
                rt.normed, rt.lm_head_g, rt.vocab, rt.hidden,
            )
        else:
            kernels.mv_bf16(rt.logits, rt.lm_head, rt.normed, rt.vocab, rt.hidden)
        rt.pos += 1
        predicted = int(cp_mod.argmax(rt.logits))
        expected = int(trace_tokens[token_index + 1])
        # Inputs before the final prompt token are externally supplied prompt
        # pieces, not greedy target predictions. The continuation begins with
        # the argmax emitted after input prompt_length - 1.
        if token_index >= prompt_length - 1 and predicted != expected:
            raise RuntimeError(
                f"canonical continuation diverged at input {token_index}: "
                f"predicted={predicted} expected={expected}"
            )
        predictions.append(predicted)

    cp.cuda.get_current_stream().synchronize()
    host = cp.asnumpy(device)
    if not np.isfinite(host).all():
        raise RuntimeError("captured target hidden states contain non-finite values")
    verified = count - (prompt_length - 1)
    return host, predictions, verified


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=COUNT)
    args = parser.parse_args()
    payload: dict[str, Any] = {
        "kind": "s100_phase38_official_dflash_target_capture",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "untimed exact target-state capture; no DFlash acceptance or throughput claim",
    }
    rt = None
    try:
        if args.tokens != COUNT:
            raise ValueError(f"Phase38 preregistration freezes --tokens {COUNT}")
        model_dir, target_config, _p20s, _p20b = identity_gate()
        trace = load_trace()
        trace_tokens = [int(x) for x in trace["tokens"]]
        prompt_length = int(trace.get("prompt_length", 0))
        if prompt_length < 1 or prompt_length >= COUNT:
            raise RuntimeError(f"invalid canonical prompt_length={prompt_length}")
        if len(trace_tokens) < COUNT + 1:
            raise RuntimeError("canonical trace is too short for frozen capture")
        payload["gpu_idle_preflight"] = require_gpu_idle_wddm()
        payload["dflash_checkpoint"] = _checkpoint_identity(DFLASH_SNAPSHOT)

        rt, keep = make_rt(COUNT, "v6_device_rows")
        hidden, predictions, verified = _capture(rt, trace_tokens, COUNT, prompt_length)
        token_array = np.asarray(trace_tokens[:COUNT + 1], dtype="<i4")
        arrays = {
            "target_aux_hidden": _write_raw(HIDDEN_RAW, hidden, "<f4"),
            "tokens": _write_raw(TOKENS_RAW, token_array, "<i4"),
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
                "num_hidden_layers": int(target_config.get("num_hidden_layers", 0)),
                "runtime_arm": "v6_device_rows",
            },
            "trace": {
                "path": str(Path(trace.get("trace_path", ""))) if trace.get("trace_path") else str(
                    (REPO / "pro_research" / "results" / "s100_phase20b" /
                     "S100_PHASE20B_CANONICAL_TRACE.json").relative_to(REPO)
                ),
                "source_sha256": sha256_file(
                    REPO / "pro_research" / "results" / "s100_phase20b" /
                    "S100_PHASE20B_CANONICAL_TRACE.json"
                ),
                "captured_inputs": COUNT,
                "prompt_length": prompt_length,
                "teacher_forced_prompt_inputs": prompt_length - 1,
                "exact_continuation_predictions": verified,
                "prediction_ids_sha256": hashlib.sha256(
                    np.asarray(predictions, dtype="<i4").tobytes()
                ).hexdigest(),
            },
            "capture": {
                "target_layer_ids_zero_based": list(CAPTURE_LAYERS),
                "shape": list(hidden.shape),
                "source_dtype": "float32 target residual after layer residual-add",
                "dflash_activation_contract": "round captured residuals to bfloat16 in Stage B",
                "all_finite": True,
                "canonical_replay_exact": True,
            },
            "arrays": arrays,
        })
        # Keep adapter objects live until all GPU work and host copies complete.
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
        "capture": payload.get("capture"),
        "arrays": payload.get("arrays"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(META),
    }, indent=2))
    return 0 if payload.get("status") == "captured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
