from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
R = ROOT / "reports/streamq5_moe"
PREREG = R / "P12_32G_4K_ENDURANCE_PREREGISTRATION.md"
P7_RUNNER = ROOT / "scripts/streamq5_moe/run_p7c_ervf_end_to_end.py"
P6_LOCK = R / "p6a_end_to_end_input_lock.json"
P7_TEST = R / "p7c_ervf_end_to_end_test.json"
OUTPUT = R / "p12_32g_4k_endurance.json"
LIMIT_BYTES = 32 * 2**30
TOTAL_TOKENS = 10_000
CONTEXT = 4096


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def apply_job_limit():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job: raise ctypes.WinError(ctypes.get_last_error())
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x00000100
    info.ProcessMemoryLimit = LIMIT_BYTES
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    return job


def load_runtime_class():
    source = P7_RUNNER.read_text(encoding="utf-8")
    old = 'exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": __file__, "ERVF_SOURCE": ERVF_SOURCE})'
    new = 'exec(compile(source, str(source_path), "exec"), globals())'
    if old not in source: raise RuntimeError("P7 import transform target missing")
    namespace = {"__name__": "p12_runtime", "__file__": str(P7_RUNNER)}
    exec(compile(source.replace(old, new), str(P7_RUNNER), "exec"), namespace)
    return namespace["Runtime"]


def memory(process):
    value = process.memory_info()
    return {
        "rss": int(value.rss), "peak_wset": int(value.peak_wset),
        "pagefile": int(value.pagefile), "peak_pagefile": int(value.peak_pagefile),
        "private": int(value.private), "num_page_faults": int(value.num_page_faults),
    }


def gpu_telemetry():
    try:
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm",
            "--format=csv,noheader,nounits"], text=True, timeout=5).strip().split(",")
        return {"temperature_c": float(raw[0]), "power_w": float(raw[1]), "sm_clock_mhz": float(raw[2])}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def timing_stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def main():
    job = apply_job_limit()
    process = psutil.Process(os.getpid())
    before = memory(process)
    Runtime = load_runtime_class()
    lock = json.loads(P6_LOCK.read_text(encoding="utf-8"))
    try:
        runtime = Runtime(lock)
    except Exception as exc:
        failure = {
            "kind": "streamq5_moe_p12_32g_allocation_failure",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "job_process_memory_limit_bytes": LIMIT_BYTES,
            "memory_before": before, "memory_at_failure": memory(process),
            "error": f"{type(exc).__name__}: {exc}",
            "preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
            "overall_pass": False,
        }
        (R / "p12_32g_allocation_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2), flush=True)
        return
    activation_ms = runtime.activate_domain("general")
    after_load = memory(process)
    pagefile_system_start = psutil.swap_memory().used
    prompt = json.loads(P7_TEST.read_text(encoding="utf-8"))["rollout"]["prompt_ids"]
    latencies, misses, predictions = [], [], []
    telemetry = []
    segment_position = 0; prompt_index = 0; token = int(prompt[0])
    kv_4k = None
    started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat()
    for index in range(TOTAL_TOKENS):
        result = runtime.decode(token, segment_position)
        latencies.append(result["wall_ms"]); misses.append(result["misses"]); predictions.append(result["prediction"])
        segment_position += 1
        if prompt_index + 1 < len(prompt):
            prompt_index += 1; token = int(prompt[prompt_index])
        else:
            token = int(result["prediction"])
        if index == CONTEXT - 1:
            kv_4k = runtime.kv_digest(CONTEXT)
        if segment_position == CONTEXT and index + 1 < TOTAL_TOKENS:
            runtime.reset_context(); segment_position = 0; prompt_index = 0; token = int(prompt[0])
        if (index + 1) % 250 == 0:
            point = {"tokens": index + 1, "elapsed_seconds": time.perf_counter() - started, "memory": memory(process), "gpu": gpu_telemetry(), "recent": timing_stats(latencies[-250:])}
            telemetry.append(point); print(json.dumps(point), flush=True)
    elapsed = time.perf_counter() - started
    final_memory = memory(process); pagefile_system_end = psutil.swap_memory().used
    first = timing_stats(latencies[16:1016]); last = timing_stats(latencies[-1000:]); full = timing_stats(latencies)
    thermal_ratios = {"mean": last["mean"] / first["mean"], "p95": last["p95"] / first["p95"]}
    gates = {
        "job_limit_applied": bool(job), "peak_commit_le_32g": final_memory["peak_pagefile"] <= LIMIT_BYTES,
        "tokens_10000": len(latencies) == TOTAL_TOKENS,
        "kv_4k_complete": bool(kv_4k and kv_4k["context"] == CONTEXT and kv_4k["nonzero"] > 0),
        "kv_write_count_exact": runtime.kv_layer_position_writes == TOTAL_TOKENS * 48,
        "mean_le_100ms": full["mean"] <= 100.0, "p95_le_150ms": full["p95"] <= 150.0,
        "p99_lt_110ms": full["p99"] < 110.0, "tps_ge_10": TOTAL_TOKENS / elapsed >= 10.0,
        "last_mean_le_110pct_first": thermal_ratios["mean"] <= 1.10,
        "last_p95_le_110pct_first": thermal_ratios["p95"] <= 1.10,
        "system_pagefile_growth_le_256mib": pagefile_system_end - pagefile_system_start <= 256 * 2**20,
    }
    result = {
        "kind": "streamq5_moe_p12_32g_4k_endurance", "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "preregistration_sha256": sha256(PREREG),
        "script_sha256": sha256(Path(__file__)), "p7_runner_sha256": sha256(P7_RUNNER),
        "job_process_memory_limit_bytes": LIMIT_BYTES, "activation_ms": activation_ms,
        "memory_before": before, "memory_after_load": after_load, "memory_final": final_memory,
        "system_pagefile_used_start": pagefile_system_start, "system_pagefile_used_end": pagefile_system_end,
        "tokens": TOTAL_TOKENS, "elapsed_seconds": elapsed, "tokens_per_second": TOTAL_TOKENS / elapsed,
        "wall_ms": latencies, "misses": misses, "predictions_sha256": hashlib.sha256(np.asarray(predictions, dtype=np.int32).tobytes()).hexdigest(),
        "timing": full, "first_1000_after_cold16": first, "last_1000": last, "thermal_ratios": thermal_ratios,
        "kv_4k": kv_4k, "kv_layer_position_writes": runtime.kv_layer_position_writes,
        "telemetry": telemetry, "gates": gates, "overall_pass": all(gates.values()),
        "claim_boundary": "Single local 10K-token endurance run with context resets at the physical 4096-token boundary; not a 60-minute or multi-seed reliability claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"tokens": TOTAL_TOKENS, "elapsed_seconds": elapsed, "tokens_per_second": result["tokens_per_second"], "timing": full, "thermal_ratios": thermal_ratios, "memory_final": final_memory, "kv_4k": kv_4k, "gates": gates, "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
