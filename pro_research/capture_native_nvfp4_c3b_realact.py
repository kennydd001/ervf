"""C3B capture: real causal activations from the arithmetic-equivalent V18 stack.

Capture is deliberately eager and untimed. V18's H-SCALE/B3 mechanisms only move
bytes / overlap transfers; V6 supplies the same arithmetic while allowing a
non-invasive Python wrapper around the live MoE call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
import types
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, require_gpu_free, utc_now, write_json_atomic
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _load_prompt_set, _new_runtime
from layer_capacity import apply_nonuniform_capacity
from moe_dev_batched import install_batched_moe_dev
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

OUT_DIR = REPO / "pro_research" / "results" / "native_nvfp4" / "c3b_capture"
META = OUT_DIR / "C3B_REAL_ACTIVATIONS_META.json"
PREREG = REPO / "pro_research" / "S100_NATIVE_NVFP4_C3B_REALACT_PREREGISTRATION.md"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return None


def _write_raw(name: str, arr: np.ndarray, dtype: str) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = np.ascontiguousarray(arr.astype(np.dtype(dtype), copy=False))
    p = OUT_DIR / name
    a.tofile(p)
    return {"path": str(p.relative_to(REPO)), "dtype": a.dtype.str,
            "shape": list(a.shape), "bytes": int(p.stat().st_size), "sha256": _sha(p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=64)
    args = ap.parse_args()
    payload: dict[str, Any] = {
        "kind": "s100_native_nvfp4_c3b_real_activation_capture",
        "status": "started", "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "untimed real causal activation capture only; no kernel or tok/s claim",
    }
    try:
        if args.tokens != 64:
            raise ValueError("C3B preregistration freezes --tokens 64")
        require_gpu_free()
        import cupy as cp

        prompts, _expected, _n, capacity = _load_prompt_set("smoke")
        if not prompts or not prompts[0].get("prompt_ids"):
            raise RuntimeError("registered anchor prompt unavailable")
        prompt = prompts[0]

        rt = _new_runtime(capacity)
        dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
        # Mirror V18's V6 parent setup exactly through the arithmetic patches.
        rt.enable_cache(capacity)
        apply_nonuniform_capacity(rt)
        rt.device_cache = True
        rt.deterministic_accum = True
        restore_sel, _ = _install_selective(rt, dense)
        install_batched_moe_dev(rt, down, up)

        target_layer = int(rt.moe_layers[0])
        orig_moe = rt._moe_dev
        enabled = {"v": False}
        moe_normed: list[np.ndarray] = []
        shared_act: list[np.ndarray] = []

        def wrapped_moe(self, i, out):
            take = bool(enabled["v"] and int(i) == target_layer)
            if take:
                before = cp.asnumpy(self.normed).astype(np.float32, copy=True)
            result = orig_moe(i, out)
            if take:
                moe_normed.append(before)
                shared_act.append(cp.asnumpy(self._act_shared).astype(np.float32, copy=True))
            return result

        rt._moe_dev = types.MethodType(wrapped_moe, rt)

        rt.reset()
        nxt = None
        for token in prompt["prompt_ids"]:
            nxt = int(rt.step(int(token)))
        if nxt is None:
            raise RuntimeError("prompt produced no next token")

        lm_head_in: list[np.ndarray] = []
        token_ids: list[int] = []
        top5_ids: list[np.ndarray] = []
        top5_logits: list[np.ndarray] = []
        cur = int(nxt)
        enabled["v"] = True
        for _ in range(args.tokens):
            out_id = int(rt.step(cur))
            # step()'s argmax synchronizes the work; normed/logits now correspond
            # to out_id from the same causal step.
            lm_head_in.append(cp.asnumpy(rt.normed).astype(np.float32, copy=True))
            token_ids.append(out_id)
            idx = cp.argpartition(rt.logits, -5)[-5:]
            idx = idx[cp.argsort(-rt.logits[idx])]
            top5_ids.append(cp.asnumpy(idx).astype(np.int32, copy=True))
            top5_logits.append(cp.asnumpy(rt.logits[idx]).astype(np.float32, copy=True))
            cur = out_id
        enabled["v"] = False
        cp.cuda.Device(0).synchronize()

        if not (len(moe_normed) == len(shared_act) == len(lm_head_in) == len(token_ids) == args.tokens):
            raise RuntimeError(f"capture count mismatch: moe={len(moe_normed)} shared={len(shared_act)} "
                               f"lm={len(lm_head_in)} ids={len(token_ids)}")

        arrays = {
            "moe_normed": _write_raw("moe_normed.f32", np.stack(moe_normed), "<f4"),
            "shared_act": _write_raw("shared_act.f32", np.stack(shared_act), "<f4"),
            "lm_head_in": _write_raw("lm_head_in.f32", np.stack(lm_head_in), "<f4"),
            "exact_token_ids": _write_raw("exact_token_ids.i32", np.asarray(token_ids), "<i4"),
            "ervf_top5_ids": _write_raw("ervf_top5_ids.i32", np.stack(top5_ids), "<i4"),
            "ervf_top5_logits": _write_raw("ervf_top5_logits.f32", np.stack(top5_logits), "<f4"),
        }
        payload.update({
            "status": "captured",
            "completed_utc": utc_now(),
            "git_head": _git_head(),
            "environment": environment_snapshot((Path(__file__), PREREG)),
            "capture": {
                "tokens": args.tokens, "calibration_rows": [0, 32], "heldout_rows": [32, 64],
                "target_moe_layer": target_layer,
                "prompt_kind": prompt.get("kind"),
                "prompt_token_count": len(prompt["prompt_ids"]),
                "prompt_ids_sha256": hashlib.sha256(np.asarray(prompt["prompt_ids"], dtype="<i4").tobytes()).hexdigest(),
                "hidden": int(rt.hidden), "shared_inter": int(rt.shared_inter),
                "moe_inter": int(rt.moe_inter), "vocab": int(rt.vocab),
                "arithmetic_stack": "V6 parent of V18: selective ERVF + batched MoE; eager submission; H-SCALE/B3 omitted because bitexact byte-placement/scheduling only",
            },
            "arrays": arrays,
        })
        restore_sel()
    except Exception as exc:
        payload.update({"status": "technical_failure",
                        "error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()},
                        "completed_utc": utc_now()})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(META, payload, archive=True)
    print(json.dumps({"status": payload.get("status"), "capture": payload.get("capture"),
                      "arrays": payload.get("arrays"), "error": (payload.get("error") or {}).get("message"),
                      "output": str(META)}, indent=2))
    return 0 if payload.get("status") == "captured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
