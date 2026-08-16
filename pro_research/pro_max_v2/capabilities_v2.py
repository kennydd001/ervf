"""PV2-21: post-V6 CUDA graph/TMA/toolchain capability census."""
from __future__ import annotations

import ctypes
import json
import shutil
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shared import environment, result_path, utc_now, write_json
from child_epoch_v11 import _load_cudart

OUT = result_path("PV2_21_CAPABILITIES.json")


def main() -> int:
    payload = {"kind": "pv2_capabilities", "status": "started",
               "started_utc": utc_now(), "preregistration": "PREREGISTRATION.md"}
    try:
        import cupy as cp
        payload["environment"] = environment((Path(__file__), HERE / "PREREGISTRATION.md"))
        props = cp.cuda.runtime.getDeviceProperties(0)
        major, minor = int(props["major"]), int(props["minor"])
        runtime = int(cp.cuda.runtime.runtimeGetVersion())
        driver = int(cp.cuda.runtime.driverGetVersion())
        payload["cuda"] = {
            "cupy_version": cp.__version__, "runtime_version": runtime,
            "driver_version": driver, "compute_capability": f"{major}.{minor}",
            "sm": major * 10 + minor,
            "cupy_graph_runtime_functions": sorted(
                x for x in dir(cp.cuda.runtime) if "graph" in x.lower()),
            "graph_object_attributes": ["graph", "graphExec"],
        }
        lib, lib_name, load_errors = _load_cudart(cp)
        symbols = [
            "cudaGraphCreate", "cudaGraphAddChildGraphNode",
            "cudaGraphAddMemcpyNode1D", "cudaGraphInstantiateWithFlags",
            "cudaGraphLaunch", "cudaGraphUpload",
            "cudaGraphExecUpdate", "cudaGraphConditionalHandleCreate",
            "cudaGraphAddNode", "cudaGraphSetConditional",
        ]
        payload["cudart"] = {
            "library": lib_name, "load_errors": load_errors,
            "symbols": {name: hasattr(lib, name) for name in symbols},
        }
        payload["tools"] = {
            "nvcc": shutil.which("nvcc"),
            "nsys": shutil.which("nsys"),
            "ncu": shutil.which("ncu"),
            "compute_sanitizer": shutil.which("compute-sanitizer"),
        }
        payload["interpretation"] = {
            "child_graph_api_ready": all(payload["cudart"]["symbols"].get(x, False)
                                         for x in ("cudaGraphCreate",
                                                   "cudaGraphAddChildGraphNode",
                                                   "cudaGraphInstantiateWithFlags",
                                                   "cudaGraphLaunch")),
            "conditional_graph_symbols_present": bool(
                payload["cudart"]["symbols"].get("cudaGraphConditionalHandleCreate")
                and payload["cudart"]["symbols"].get("cudaGraphAddNode")),
            "tma_architecture_prerequisite": major >= 9,
            "mapped_host_to_smem_tma_proven": False,
            "mapped_host_to_smem_note": (
                "Architecture/toolchain capability is not evidence that TMA can "
                "efficiently consume this mapped pinned host expert layout. A "
                "separate byte-exact bandwidth microbenchmark is required."
            ),
        }
        payload["status"] = "pass"
        payload["completed_utc"] = utc_now()
    except Exception as exc:
        payload.update(status="technical_failure", completed_utc=utc_now(),
                       error={"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()})
    write_json(OUT, payload)
    print(json.dumps({"status": payload.get("status"),
                      "interpretation": payload.get("interpretation"),
                      "output": str(OUT)}, indent=2))
    return 0 if payload.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
