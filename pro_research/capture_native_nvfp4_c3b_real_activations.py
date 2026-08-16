"""C3B phase 1: capture real V18 activation/reference tensors without timing capture."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import traceback
import types
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, require_model_dir, utc_now, write_json_atomic
from native_nvfp4_c3b_quant import CAPTURE, CAPTURE_DIR, gpu_idle_snapshot, sha256_file

TOKENS_PER_PROMPT = 8
TARGET_LAYER = 1
TARGET_EXPERT = 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_array(name: str, arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype="<f4", order="C")
    path = CAPTURE_DIR / f"{name}.f32"
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    return {"path": str(path.relative_to(REPO)), "dtype": "float32-le",
            "shape": [int(x) for x in arr.shape], "bytes": int(path.stat().st_size),
            "sha256": _sha(path)}


def main() -> int:
    payload: dict[str, Any] = {
        "kind": "s100_native_nvfp4_c3b_capture",
        "status": "started",
        "started_utc": utc_now(),
        "tokens_per_prompt": TOKENS_PER_PROMPT,
        "claim_boundary": "real V18 arithmetic-stack activation/reference capture only; eager submission for observability; no timing claim",
    }
    rt = None
    restore_sel = None
    restore_combined = None
    try:
        payload["gpu_idle_preflight"] = gpu_idle_snapshot()
        sys.path.insert(0, str(REPO / "src"))
        sys.path.insert(0, str(REPO / "pro_research"))
        import cupy as cp
        from down_proj_batch_kernels import DownProjBatchKernels
        from ervf_dense import DenseERVF
        from graph_e1f22 import _load_prompt_set, _new_runtime
        from layer_capacity import apply_nonuniform_capacity
        from moe_dev_batched import install_batched_moe_dev
        from moe_dev_combined import install_combined_moe_dev
        from moe_dev_scale_resident import planned_plane_bytes
        from scale_resident_kernels import ScaleResidentKernels
        from selective_ervf_v3 import _install_selective
        from up_proj_batch_kernels import UpProjBatchKernels

        prompts, _expected, _n, capacity = _load_prompt_set("full")
        if len(prompts) != 3:
            raise RuntimeError(f"frozen full prompt set must contain 3 prompts, got {len(prompts)}")

        rt = _new_runtime(capacity)
        apply_nonuniform_capacity(rt)
        rt.device_cache = True
        rt.deterministic_accum = True
        dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
        restore_sel, _counters = _install_selective(rt, dense)
        install_batched_moe_dev(rt, down, up)
        cp.get_default_memory_pool().free_all_blocks()
        planned = planned_plane_bytes(rt)
        free_before = int(cp.cuda.Device(0).mem_info[0])
        if planned > free_before:
            raise RuntimeError(f"V18 H-SCALE planes do not fit: planned={planned} free={free_before}")
        sres = ScaleResidentKernels()
        restore_combined = install_combined_moe_dev(rt, down, up, sres)

        if not rt.moe_layers or int(rt.moe_layers[0]) != TARGET_LAYER:
            raise RuntimeError(f"first MoE layer must be {TARGET_LAYER}; got {rt.moe_layers[:3]}")
        d = rt.layer[TARGET_LAYER]
        if int(rt.hidden) != 2688 or int(rt.shared_inter) != 3712 or int(rt.moe_inter) != 1856:
            raise RuntimeError(f"unexpected model dims hidden/shared/moe={rt.hidden}/{rt.shared_inter}/{rt.moe_inter}")

        ep = f"backbone.layers.{TARGET_LAYER}.mixer.experts.{TARGET_EXPERT}.up_proj"
        exp_codes = cp.asarray(rt.index.read_raw(ep + ".weight"))
        exp_scales = cp.asarray(rt.index.read_raw(ep + ".weight_scale"))
        exp_global = float(rt.index.get_scalar(ep + ".weight_scale_2"))

        ref_su = cp.empty(rt.shared_inter, dtype=cp.float32)
        ref_sd = cp.empty(rt.hidden, dtype=cp.float32)
        ref_ru = cp.empty(rt.moe_inter, dtype=cp.float32)
        captures = {k: [] for k in (
            "moe_normed", "shared_up_ref", "shared_down_input", "shared_down_ref",
            "routed_up_ref", "lm_head_input", "lm_head_ref")}
        row_meta: list[dict[str, Any]] = []
        state = {"active": False, "moe_done": False}
        orig_moe = rt._moe

        def wrapped_moe(self, i, out):
            do = bool(state["active"] and int(i) == TARGET_LAYER)
            x = self.normed.copy() if do else None
            result = orig_moe(i, out)
            if do:
                if state["moe_done"]:
                    raise RuntimeError("target MoE captured more than once in one token")
                sh_in = self._act_shared.copy()
                self.fused.gemv_into(ref_su, d["sh_up_c"], d["sh_up_s"], x,
                                     d["sh_up_g"], self.shared_inter, self.hidden,
                                     apply_relu2=False)
                self.fused.gemv_into(ref_sd, d["sh_dn_c"], d["sh_dn_s"], sh_in,
                                     d["sh_dn_g"], self.hidden, self.shared_inter,
                                     apply_relu2=False)
                self.fused.gemv_into(ref_ru, exp_codes, exp_scales, x, exp_global,
                                     self.moe_inter, self.hidden, apply_relu2=False)
                cp.cuda.get_current_stream().synchronize()
                captures["moe_normed"].append(cp.asnumpy(x))
                captures["shared_up_ref"].append(cp.asnumpy(ref_su))
                captures["shared_down_input"].append(cp.asnumpy(sh_in))
                captures["shared_down_ref"].append(cp.asnumpy(ref_sd))
                captures["routed_up_ref"].append(cp.asnumpy(ref_ru))
                state["moe_done"] = True
            return result

        rt._moe = types.MethodType(wrapped_moe, rt)

        generated_by_prompt = []
        for pi, p in enumerate(prompts):
            ids = [int(x) for x in p["prompt_ids"]]
            if not ids:
                raise RuntimeError("empty frozen prompt")
            rt.reset()
            state["active"] = False
            nxt = None
            for tid in ids[:-1]:
                nxt = int(rt.step(tid))
            cur_input = ids[-1]
            out_ids = []
            for pos in range(TOKENS_PER_PROMPT):
                state["active"] = True
                state["moe_done"] = False
                out_id = int(rt.step(int(cur_input)))
                if not state["moe_done"]:
                    raise RuntimeError("target MoE hook did not execute")
                cp.cuda.get_current_stream().synchronize()
                captures["lm_head_input"].append(cp.asnumpy(rt.normed.copy()))
                captures["lm_head_ref"].append(cp.asnumpy(rt.logits.copy()))
                row_meta.append({"row": len(row_meta), "prompt_index": pi,
                                 "prompt": p["prompt"], "kind": p["kind"],
                                 "position": pos, "input_token": int(cur_input),
                                 "output_token": out_id})
                out_ids.append(out_id)
                cur_input = out_id
                state["active"] = False
            generated_by_prompt.append({"prompt_index": pi, "prompt": p["prompt"],
                                        "kind": p["kind"], "ids": out_ids})

        expected_rows = len(prompts) * TOKENS_PER_PROMPT
        if len(row_meta) != expected_rows or any(len(v) != expected_rows for v in captures.values()):
            raise RuntimeError("capture row count mismatch")
        arrays = {}
        finite = True
        for name, rows in captures.items():
            arr = np.stack(rows).astype(np.float32, copy=False)
            finite = finite and bool(np.isfinite(arr).all())
            arrays[name] = _write_array(name, arr)

        source_files = [
            Path(__file__), REPO / "pro_research" / "combined_v18.py",
            REPO / "pro_research" / "moe_dev_combined.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
        ]
        payload.update({
            "status": "capture_pass" if finite else "capture_nonfinite",
            "environment": environment_snapshot(source_files),
            "v18": {"capacity": int(capacity), "target_layer": TARGET_LAYER,
                    "target_expert": TARGET_EXPERT, "moe_layer_count": len(rt.moe_layers),
                    "top_k": int(rt.top_k), "hscale_planned_bytes": int(planned),
                    "free_before_hscale_bytes": int(free_before), "arithmetic_stack": "V18",
                    "submission": "eager_for_observability"},
            "rows": row_meta,
            "generated_by_prompt": generated_by_prompt,
            "arrays": arrays,
            "reuse_contract": {"shared_up_input": "moe_normed",
                               "routed_up_input": "moe_normed",
                               "identical_by_construction": True,
                               "quantizer_cost_model": "one moe_normed quantization per MoE layer reused across shared_up + top_k routed-up weights"},
            "finite": finite,
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({"status": "technical_failure",
                        "error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()},
                        "completed_utc": utc_now()})
    finally:
        try:
            if restore_combined is not None: restore_combined()
        except Exception:
            pass
        try:
            if restore_sel is not None: restore_sel()
        except Exception:
            pass
        try:
            del rt
            gc.collect()
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    write_json_atomic(CAPTURE, payload, archive=True)
    print(json.dumps({"status": payload.get("status"), "rows": len(payload.get("rows") or []),
                      "v18": payload.get("v18"), "arrays": payload.get("arrays"),
                      "error": (payload.get("error") or {}).get("message"),
                      "output": str(CAPTURE)}, indent=2))
    return 0 if payload.get("status") == "capture_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
