"""C0: audit native Blackwell NVFP4 prerequisites without making a speed claim."""
from __future__ import annotations

import ctypes
import ctypes.util
import importlib.metadata as md
import importlib.util
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

from common import REPO, environment_snapshot, require_gpu_free, require_model_dir, utc_now, write_json_atomic

OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "C0_CAPABILITIES.json"
PREREG = REPO / "pro_research" / "S100_NATIVE_NVFP4_CAPABILITY_PREREGISTRATION.md"


def _dist_version(*names: str) -> str | None:
    for name in names:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            pass
    return None


def _load_cublaslt() -> tuple[object | None, str | None, list[str]]:
    candidates: list[str] = []
    for name in ("cublasLt64_12.dll", "cublasLt64_13.dll", "libcublasLt.so.12", "libcublasLt.so"):
        p = shutil.which(name)
        if p:
            candidates.append(p)
        candidates.append(name)
    found = ctypes.util.find_library("cublasLt")
    if found:
        candidates.insert(0, found)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        for name in ("cublasLt64_12.dll", "cublasLt64_13.dll"):
            p = Path(cuda_path) / "bin" / name
            if p.exists():
                candidates.insert(0, str(p))

    # NVIDIA pip wheels commonly place DLLs below site-packages/nvidia/cublas/bin.
    for root in (Path(sys.prefix), Path(sys.base_prefix)):
        for suffix in (
            Path("Lib/site-packages/nvidia/cublas/bin/cublasLt64_12.dll"),
            Path("Lib/site-packages/nvidia/cublas/bin/cublasLt64_13.dll"),
        ):
            p = root / suffix
            if p.exists():
                candidates.insert(0, str(p))

    seen: set[str] = set()
    errors: list[str] = []
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        try:
            loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
            return loader(cand), cand, errors
        except OSError as exc:
            errors.append(f"{cand}: {exc}")
    return None, None, errors


def _find_representative_scale_names(entries: dict[str, object]) -> dict[str, str | None]:
    names = sorted(entries)

    def first(pred):
        return next((n for n in names if pred(n)), None)

    return {
        "lm_head": "lm_head.weight_scale" if "lm_head.weight_scale" in entries else None,
        "shared_up": first(lambda n: n.endswith("shared_experts.up_proj.weight_scale")),
        "shared_down": first(lambda n: n.endswith("shared_experts.down_proj.weight_scale")),
        "routed_up": first(lambda n: ".experts.0.up_proj.weight_scale" in n),
        "routed_down": first(lambda n: ".experts.0.down_proj.weight_scale" in n),
    }


def _sample_scale(idx, name: str) -> dict:
    e = idx.entries[name]
    raw = np.asarray(idx.read_raw(name)).view(np.uint8).reshape(-1)
    weight_name = name[:-len("_scale")]  # *.weight_scale -> *.weight
    w = idx.entries.get(weight_name)
    expected = None
    weight_shape = None
    if w is not None:
        weight_shape = [int(x) for x in w.shape]
        elems = int(np.prod(np.asarray(weight_shape, dtype=np.int64)))
        expected = elems // 16 if elems % 16 == 0 else None
    sign = int(np.count_nonzero(raw & np.uint8(0x80)))
    return {
        "name": name,
        "dtype": str(e.dtype),
        "shape": [int(x) for x in e.shape],
        "nbytes": int(e.nbytes),
        "raw_count": int(raw.size),
        "signbit_set_count": sign,
        "signbit_clear_fraction": float((raw.size - sign) / raw.size) if raw.size else None,
        "min_byte": int(raw.min()) if raw.size else None,
        "max_byte": int(raw.max()) if raw.size else None,
        "weight_name": weight_name if w is not None else None,
        "weight_shape": weight_shape,
        "expected_group16_scale_count": expected,
        "count_matches_group16": bool(expected is not None and int(raw.size) == int(expected)),
        "global_scale_name": name + "_2" if (name + "_2") in idx.entries else None,
    }


def _quant_groups(config: dict) -> list[dict]:
    q = config.get("quantization_config") or {}
    groups = q.get("config_groups") or {}
    out = []
    if isinstance(groups, dict):
        for name, g in groups.items():
            if not isinstance(g, dict):
                continue
            w = g.get("weights") or {}
            a = g.get("input_activations") or {}
            out.append({
                "name": str(name),
                "format": g.get("format"),
                "targets": g.get("targets"),
                "weights": {
                    "num_bits": w.get("num_bits"), "type": w.get("type"),
                    "group_size": w.get("group_size"), "dynamic": w.get("dynamic"),
                },
                "input_activations": {
                    "num_bits": a.get("num_bits"), "type": a.get("type"),
                    "group_size": a.get("group_size"), "dynamic": a.get("dynamic"),
                    "strategy": a.get("strategy"),
                },
            })
    return out


def main() -> int:
    payload = {
        "kind": "s100_native_nvfp4_c0",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "capability/format audit only; no native matmul, numerical-equivalence or speed claim",
    }
    try:
        require_gpu_free()
        sys.path.insert(0, str(REPO / "src"))
        from moe_lab.lightningstream_nemotron.loader import ShardIndex
        import cupy as cp

        model = require_model_dir()
        idx = ShardIndex(model)
        props = cp.cuda.runtime.getDeviceProperties(0)
        major, minor = int(props["major"]), int(props["minor"])
        runtime = int(cp.cuda.runtime.runtimeGetVersion())
        driver = int(cp.cuda.runtime.driverGetVersion())

        lib, lib_name, load_errors = _load_cublaslt()
        symbols = (
            "cublasLtCreate", "cublasLtDestroy", "cublasLtMatmul",
            "cublasLtMatmulDescCreate", "cublasLtMatrixLayoutCreate",
            "cublasLtMatmulPreferenceCreate", "cublasLtMatmulAlgoGetHeuristic",
        )
        symbol_map = {s: bool(lib is not None and hasattr(lib, s)) for s in symbols}

        reps = _find_representative_scale_names(idx.entries)
        scale_samples = {k: (_sample_scale(idx, v) if v else None) for k, v in reps.items()}
        present_samples = [x for x in scale_samples.values() if x is not None]

        groups = _quant_groups(idx.config)
        qcfg = idx.config.get("quantization_config") or {}
        hf_quant = None
        hq = model / "hf_quant_config.json"
        if hq.exists():
            try:
                hf_quant = json.loads(hq.read_text(encoding="utf-8"))
            except Exception as exc:
                hf_quant = {"read_error": f"{type(exc).__name__}: {exc}"}

        group16_nvfp4 = any(
            g["weights"].get("num_bits") == 4
            and str(g["weights"].get("type")).lower() == "float"
            and g["weights"].get("group_size") == 16
            for g in groups
        )
        if not group16_nvfp4 and isinstance(hf_quant, dict):
            q = hf_quant.get("quantization") or hf_quant.get("quantization_config") or {}
            group16_nvfp4 = str(q.get("quant_algo", "")).upper() == "NVFP4" and int(q.get("group_size", 0) or 0) == 16

        torch_info = {
            "distribution_version": _dist_version("torch"),
            "module_present": importlib.util.find_spec("torch") is not None,
            "float4_e2m1fn_x2": False,
        }
        if torch_info["module_present"]:
            try:
                import torch
                torch_info["module_version"] = str(torch.__version__)
                torch_info["float4_e2m1fn_x2"] = hasattr(torch, "float4_e2m1fn_x2")
                torch_info["scaled_mm_present"] = hasattr(torch, "_scaled_mm")
            except Exception as exc:
                torch_info["import_error"] = f"{type(exc).__name__}: {exc}"

        py_api = {
            "cupy": cp.__version__,
            "torch": torch_info,
            "nvmath_distribution": _dist_version("nvmath-python", "nvidia-nvmath-python", "nvmath"),
            "nvmath_module_present": importlib.util.find_spec("nvmath") is not None,
            "cutlass_distribution": _dist_version("cutlass", "nvidia-cutlass"),
        }

        mandatory_sample_keys = ("shared_up", "shared_down", "routed_up", "routed_down")
        rep_found = all(scale_samples.get(k) is not None for k in mandatory_sample_keys)
        # lm_head is recorded but not made mandatory because some official NVFP4
        # variants intentionally exclude it; this exact Lightning checkpoint may include it.
        sign_clear = bool(present_samples) and all(int(x["signbit_set_count"]) == 0 for x in present_samples)
        group16_counts = bool(present_samples) and all(bool(x["count_matches_group16"]) for x in present_samples)

        gates = {
            "C0_GPU_SM120_OR_NEWER": major >= 12,
            "C0_CUDA_RUNTIME_GE_12080": runtime >= 12080,
            "C0_CUBLASLT_LOADABLE": lib is not None,
            "C0_CUBLASLT_CORE_SYMBOLS": all(symbol_map[s] for s in (
                "cublasLtCreate", "cublasLtDestroy", "cublasLtMatmul",
                "cublasLtMatmulDescCreate", "cublasLtMatrixLayoutCreate")),
            "C0_MODEL_GROUP16_NVFP4": bool(group16_nvfp4),
            "C0_SCALE_SIGNBIT_CLEAR_REPRESENTATIVE": sign_clear,
            "C0_SCALE_COUNTS_MATCH_GROUP16": group16_counts,
            "C0_REPRESENTATIVE_TENSORS_FOUND": rep_found,
        }
        all_mandatory = all(gates.values())
        payload.update({
            "environment": environment_snapshot((Path(__file__), PREREG)),
            "cuda": {
                "compute_capability": f"{major}.{minor}",
                "sm": major * 10 + minor,
                "runtime_version": runtime,
                "driver_version": driver,
            },
            "cublaslt": {
                "library": lib_name,
                "load_errors": load_errors,
                "symbols": symbol_map,
            },
            "python_apis": py_api,
            "model": {
                "path": str(model),
                "quantization_config_format": qcfg.get("format"),
                "quant_method": qcfg.get("quant_method"),
                "quant_groups": groups,
                "hf_quant_config_present": hq.exists(),
                "hf_quant_summary": hf_quant,
                "representative_scale_names": reps,
                "scale_samples": scale_samples,
            },
            "gates": gates,
            "all_mandatory_gates": all_mandatory,
            "interpretation": (
                "C0_pass_value_format_plausible_native_fp4_C1_allowed"
                if all_mandatory else "C0_fail_do_not_build_native_fp4_C1"
            ),
            "status": "pass" if all_mandatory else "gate_failed",
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "cuda": payload.get("cuda"),
        "cublaslt": payload.get("cublaslt"),
        "python_apis": payload.get("python_apis"),
        "gates": payload.get("gates"),
        "interpretation": payload.get("interpretation"),
        "output": str(OUT),
        "error": (payload.get("error") or {}).get("message"),
    }, indent=2))
    return 0 if payload.get("status") in {"pass", "gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
