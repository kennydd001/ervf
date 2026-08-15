from __future__ import annotations

import ctypes as C
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.streamq5_moe import run_het_next_cap0x_existing_runner_diagnostic as diag
from scripts.streamq5_moe import run_st2_mini_host_usm_q5 as st2


RUN = ROOT / "reports/runs/streamq5_moe/het_next_cap0x_r2_intel_usm_sentinel"
INTEL_RESULT = RUN / "intel_usm_sentinel.json"
NVIDIA_RESULT = RUN / "nvidia_d7.json"
NVIDIA_REPORT = RUN / "nvidia_d7.md"
RESULT = RUN / "cap0x_r2_result.json"
PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_CAP0X_R2_INTEL_USM_SENTINEL_DIAGNOSTIC_PREREGISTRATION_2026-08-13.md"
LAUNCHES = 1000
WORDS = 1024


KERNEL = r"""
__kernel void q5_ergv_host_usm(
    __global const uint* input,
    __global const int* unused_indices,
    __global const float* unused_x,
    __global uint* output,
    int rows,
    int cols,
    int batch_records)
{
    uint i = (uint)get_global_id(0);
    if (i < 1024U) {
        uint x = input[i];
        output[i] = ((x ^ 0x9e3779b9U) * 1664525U) + 1013904223U;
    }
}
"""


def input_words() -> np.ndarray:
    values = np.empty(WORDS, dtype=np.uint32)
    state = 0xC0A0_80B1
    for i in range(WORDS):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        values[i] = state ^ ((i * 0x45D9F3B) & 0xFFFFFFFF)
    return values


def expected(values: np.ndarray) -> np.ndarray:
    x = values.astype(np.uint64)
    return ((((x ^ np.uint64(0x9E3779B9)) * np.uint64(1664525)) + np.uint64(1013904223)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_intel() -> int:
    if INTEL_RESULT.exists():
        raise FileExistsError(INTEL_RESULT)
    original_kernel, original_bytes = st2.KERNEL_SOURCE, st2.ALLOC_BYTES
    st2.KERNEL_SOURCE, st2.ALLOC_BYTES = KERNEL, WORDS * 4
    cl = None
    payload: dict[str, object] = {}
    error = None
    try:
        cl = st2.OpenCL()
        capability = cl.setup()
        source = input_words()
        oracle = expected(source)
        C.memmove(C.c_void_p(cl.usm_ptr), C.c_void_p(source.ctypes.data), source.nbytes)
        dummy_indices = cl.buffer(st2.CL_MEM_READ_ONLY, 4)
        dummy_x = cl.buffer(st2.CL_MEM_READ_ONLY, 4)
        output_mem = cl.buffer(st2.CL_MEM_WRITE_ONLY, oracle.nbytes)
        cl.bind(dummy_indices, dummy_x, output_mem)
        cl.set_int_arg(4, WORDS)
        cl.set_int_arg(5, 1)
        cl.set_int_arg(6, 1)
        global_size = (C.c_size_t * 1)(WORDS)
        local_size = (C.c_size_t * 1)(256)
        submit = time.perf_counter_ns()
        for _ in range(LAUNCHES):
            st2.check(cl.lib.clEnqueueNDRangeKernel(cl.queue, cl.kernel, 1, None, global_size, local_size, 0, None, None), "clEnqueueNDRangeKernel")
        st2.check(cl.lib.clFinish(cl.queue), "clFinish")
        complete = time.perf_counter_ns()
        observed = np.empty_like(oracle)
        cl.read(output_mem, observed)
        diff = np.flatnonzero(observed != oracle)
        payload = {
            "status": "intel_host_usm_sentinel_exact" if diff.size == 0 else "intel_host_usm_sentinel_mismatch",
            "error": None,
            "capability": capability,
            "launches": LAUNCHES,
            "words": WORDS,
            "submit_qpc_ns": submit,
            "complete_qpc_ns": complete,
            "input_sha256": sha_bytes(source.tobytes()),
            "expected_sha256": sha_bytes(oracle.tobytes()),
            "observed_sha256": sha_bytes(observed.tobytes()),
            "correctness": {"bitwise_equal": bool(diff.size == 0), "different_bits": int(diff.size)},
            "host_usm_input_api": "clHostMemAllocINTEL + clSetKernelArgMemPointerINTEL",
            "explicit_input_copy_api_calls": 0,
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        payload = {"status": "intel_host_usm_sentinel_failure", "error": error, "correctness": {"bitwise_equal": False, "different_bits": -1}}
    finally:
        if cl is not None:
            cl.close()
        st2.KERNEL_SOURCE, st2.ALLOC_BYTES = original_kernel, original_bytes
    INTEL_RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def configure() -> None:
    diag.RUN = RUN
    diag.PREREG = PREREG
    diag.RESULT = RESULT
    diag.INTEL_RESULT = INTEL_RESULT
    diag.NVIDIA_RESULT = NVIDIA_RESULT
    diag.NVIDIA_REPORT = NVIDIA_REPORT
    diag.__file__ = str(Path(__file__).resolve())
    diag.child_command = lambda role: [sys.executable, str(Path(__file__).resolve()), "--role", role]


def main() -> int:
    configure()
    role = "coordinator"
    if "--role" in sys.argv:
        role = sys.argv[sys.argv.index("--role") + 1]
    if role == "intel":
        return run_intel()
    if role == "nvidia":
        return diag.run_nvidia()
    original_run = diag.subprocess.run
    diag.subprocess.run = lambda *args, **kwargs: SimpleNamespace(stdout="poll_disabled_in_cap0x_r2")
    try:
        return diag.run_coordinator()
    finally:
        diag.subprocess.run = original_run


if __name__ == "__main__":
    raise SystemExit(main())
