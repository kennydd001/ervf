"""PV2-20: exact low-level child-graph epochs, avoiding unsupported nested capture."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shared import (
    capture_v6, environment, load_json, new_v6_bundle, percentiles, prompt_set,
    result_path, utc_now, write_json,
)
from addnorm_v7 import AddNorm, install as install_addnorm
from qkv_v8 import QKV, install as install_qkv
from lmhead_argmax_v9 import LMArgmax, install as install_lm

OUT = result_path("PV2_20_CHILD_EPOCH.json")


def _adopted() -> set[str]:
    mapping = {
        "addnorm": result_path("PV2_10_ADDNORM.json"),
        "qkv": result_path("PV2_11_QKV.json"),
        "lmhead_argmax": result_path("PV2_12_LMHEAD_ARGMAX.json"),
    }
    out = set()
    for name, path in mapping.items():
        if path.exists() and bool(load_json(path).get("adopt")):
            out.add(name)
    return out


def _load_cudart(cp):
    candidates: list[str] = []
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.extend(glob.glob(str(Path(cuda_path) / "bin" / "cudart64_*.dll")))
    root = Path(cp.__file__).resolve().parent.parent
    try:
        candidates.extend(str(x) for x in root.rglob("cudart64_*.dll"))
    except Exception:
        pass
    candidates.extend(["cudart64_130.dll", "cudart64_12.dll", "cudart64_110.dll"])
    errors = []
    for name in dict.fromkeys(candidates):
        try:
            return ctypes.WinDLL(name), name, errors
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load cudart DLL; " + " | ".join(errors[-8:]))


class ParentEpoch:
    def __init__(self, cp, child_graph_handle: int, tok_ptr: int, k: int):
        self.cp = cp
        self.k = int(k)
        self.lib, self.lib_name, self.load_errors = _load_cudart(cp)
        cvoid = ctypes.c_void_p
        csize = ctypes.c_size_t
        cint = ctypes.c_int
        cuint = ctypes.c_uint
        culonglong = ctypes.c_ulonglong

        self.lib.cudaGetErrorString.argtypes = [cint]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaGraphCreate.argtypes = [ctypes.POINTER(cvoid), cuint]
        self.lib.cudaGraphCreate.restype = cint
        self.lib.cudaGraphAddChildGraphNode.argtypes = [
            ctypes.POINTER(cvoid), cvoid, ctypes.POINTER(cvoid), csize, cvoid]
        self.lib.cudaGraphAddChildGraphNode.restype = cint
        self.lib.cudaGraphAddMemcpyNode1D.argtypes = [
            ctypes.POINTER(cvoid), cvoid, ctypes.POINTER(cvoid), csize,
            cvoid, cvoid, csize, cint]
        self.lib.cudaGraphAddMemcpyNode1D.restype = cint
        if not hasattr(self.lib, "cudaGraphInstantiateWithFlags"):
            raise RuntimeError("cudart lacks cudaGraphInstantiateWithFlags")
        self.lib.cudaGraphInstantiateWithFlags.argtypes = [ctypes.POINTER(cvoid), cvoid, culonglong]
        self.lib.cudaGraphInstantiateWithFlags.restype = cint
        self.lib.cudaGraphLaunch.argtypes = [cvoid, cvoid]
        self.lib.cudaGraphLaunch.restype = cint
        self.lib.cudaGraphExecDestroy.argtypes = [cvoid]
        self.lib.cudaGraphExecDestroy.restype = cint
        self.lib.cudaGraphDestroy.argtypes = [cvoid]
        self.lib.cudaGraphDestroy.restype = cint

        self.graph = cvoid()
        self.exec = cvoid()
        self.ring = cp.empty(self.k, dtype=cp.int32)
        self._check(self.lib.cudaGraphCreate(ctypes.byref(self.graph), 0), "cudaGraphCreate")
        prev = cvoid()
        for j in range(self.k):
            child = cvoid()
            deps = None if not prev.value else (cvoid * 1)(prev)
            ndeps = 0 if deps is None else 1
            self._check(self.lib.cudaGraphAddChildGraphNode(
                ctypes.byref(child), self.graph, deps, ndeps, cvoid(child_graph_handle)),
                f"cudaGraphAddChildGraphNode[{j}]")
            copy = cvoid()
            dep2 = (cvoid * 1)(child)
            dst = cvoid(int(self.ring.data.ptr) + 4 * j)
            self._check(self.lib.cudaGraphAddMemcpyNode1D(
                ctypes.byref(copy), self.graph, dep2, 1, dst, cvoid(tok_ptr), 4, 3),
                f"cudaGraphAddMemcpyNode1D[{j}]")
            prev = copy
        self._check(self.lib.cudaGraphInstantiateWithFlags(
            ctypes.byref(self.exec), self.graph, 0), "cudaGraphInstantiateWithFlags")

    def _check(self, code: int, what: str):
        if int(code) != 0:
            msg = self.lib.cudaGetErrorString(int(code))
            text = msg.decode(errors="replace") if msg else f"CUDA error {code}"
            raise RuntimeError(f"{what}: {text} ({code})")

    def launch(self, stream) -> list[int]:
        self._check(self.lib.cudaGraphLaunch(self.exec, ctypes.c_void_p(int(stream.ptr))),
                    "cudaGraphLaunch(parent)")
        stream.synchronize()
        return [int(x) for x in self.cp.asnumpy(self.ring)]

    def close(self):
        if getattr(self, "exec", None) and self.exec.value:
            self.lib.cudaGraphExecDestroy(self.exec); self.exec = ctypes.c_void_p()
        if getattr(self, "graph", None) and self.graph.value:
            self.lib.cudaGraphDestroy(self.graph); self.graph = ctypes.c_void_p()


def prefill(rt, prompt_ids: list[int]) -> int:
    rt.reset()
    start = int(rt._ring_i)
    for token in prompt_ids:
        rt.step_graph(int(token))
        rt._graph_stream.synchronize()
    slot = (start + len(prompt_ids) - 1) % int(rt._ring_size)
    return int(rt.ring_harvest(slot, 1)[0])


def separate_batch(rt, k: int, ring, stream) -> list[int]:
    cp = rt.cp
    for j in range(k):
        rt._graph.launch(stream)
        cp.cuda.runtime.memcpyAsync(int(ring.data.ptr) + 4 * j,
                                    int(rt._tok_dev.data.ptr), 4,
                                    cp.cuda.runtime.memcpyDeviceToDevice,
                                    int(stream.ptr))
    stream.synchronize()
    return [int(x) for x in cp.asnumpy(ring)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload: dict[str, Any] = {"kind": "pv2_child_epoch", "status": "started",
                               "mode": args.mode, "started_utc": utc_now(),
                               "preregistration": "PREREGISTRATION.md",
                               "claim_boundary": "queued exact throughput; not first-token latency"}
    bundle = None
    restores = []
    parents: list[ParentEpoch] = []
    try:
        from shared import require_gpu_free
        require_gpu_free()
        prompts, _expected, _n, _capacity = prompt_set("smoke")
        prompt = prompts[0]
        payload["environment"] = environment((Path(__file__), HERE / "PREREGISTRATION.md"))
        bundle = new_v6_bundle(); rt = bundle.rt
        selected = _adopted(); payload["selected_finale_candidates"] = sorted(selected)
        if "addnorm" in selected:
            restores.append(install_addnorm(rt, AddNorm()))
        if "qkv" in selected:
            restores.append(install_qkv(rt, QKV()))
        if "lmhead_argmax" in selected:
            restores.append(install_lm(rt, LMArgmax(rt.vocab)))
        capture_v6(bundle)

        cp = rt.cp
        child_handle = int(rt._graph.graph)
        payload["graph_handles"] = {"graph": child_handle, "graphExec": int(rt._graph.graphExec)}
        ks = [2, 4] if args.mode == "smoke" else [2, 4, 8, 16]
        rounds = 4 if args.mode == "smoke" else 20
        payload["epochs"] = {}
        for k in ks:
            rec: dict[str, Any] = {"k": k, "status": "started"}
            parent = None
            try:
                parent = ParentEpoch(cp, child_handle, int(rt._tok_dev.data.ptr), k)
                parents.append(parent)
                stream = cp.cuda.Stream(non_blocking=True)
                separate_ring = cp.empty(k, dtype=cp.int32)

                prefill(rt, [int(x) for x in prompt["prompt_ids"]])
                separate_ids = separate_batch(rt, k, separate_ring, stream)
                prefill(rt, [int(x) for x in prompt["prompt_ids"]])
                parent_ids = parent.launch(stream)

                sep_ms, par_ms = [], []
                for _ in range(rounds):
                    prefill(rt, [int(x) for x in prompt["prompt_ids"]])
                    t0 = time.perf_counter_ns(); separate_batch(rt, k, separate_ring, stream)
                    sep_ms.append((time.perf_counter_ns() - t0) / 1e6 / k)
                    prefill(rt, [int(x) for x in prompt["prompt_ids"]])
                    t0 = time.perf_counter_ns(); parent.launch(stream)
                    par_ms.append((time.perf_counter_ns() - t0) / 1e6 / k)
                ss, ps = percentiles(sep_ms), percentiles(par_ms)
                speedup = float(ss["p50"] / ps["p50"])
                rec.update({
                    "status": "measured", "separate_ids": separate_ids,
                    "parent_ids": parent_ids,
                    "identical": separate_ids == parent_ids,
                    "separate_per_token_ms": ss,
                    "parent_per_token_ms": ps,
                    "speedup_p50": speedup,
                    "tok_s_parent_queued": 1000.0 / float(ps["p50"]),
                    "cudart": parent.lib_name,
                    "gates": {"exact_ids": separate_ids == parent_ids,
                              "speedup_ge_1_02": speedup >= 1.02},
                })
            except Exception as exc:
                rec.update(status="unsupported_or_failed",
                           error={"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()})
            payload["epochs"][str(k)] = rec

        measured = [x for x in payload["epochs"].values() if x.get("status") == "measured"]
        passing = [x for x in measured if all(x.get("gates", {}).values())]
        if passing:
            best = max(passing, key=lambda x: x["speedup_p50"])
            payload["best"] = {"k": best["k"], "speedup_p50": best["speedup_p50"],
                               "per_token_ms": best["parent_per_token_ms"]["p50"],
                               "tok_s_queued": best["tok_s_parent_queued"]}
            payload["status"] = "pass"
        elif measured:
            payload["status"] = "gate_failed"
        else:
            payload["status"] = "technical_blocked"
        payload["completed_utc"] = utc_now()
    except Exception as exc:
        payload.update(status="technical_failure", completed_utc=utc_now(),
                       error={"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()})
    finally:
        for p in parents:
            try: p.close()
            except Exception: pass
        for restore in reversed(restores):
            try: restore()
            except Exception: pass
        if bundle is not None: bundle.close()
    write_json(OUT, payload)
    print(json.dumps({"status": payload.get("status"), "best": payload.get("best"),
                      "output": str(OUT)}, indent=2))
    return 0 if payload.get("status") in {"pass", "gate_failed", "technical_blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
