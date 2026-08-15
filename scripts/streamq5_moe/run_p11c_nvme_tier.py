from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
BANK_DIR = ROOT / "reports/runs/streamq5_moe/p1d_q5_bank"
PREREG = R / "P11C_NVME_TIER_PREREGISTRATION.md"
P7C = R / "p7c_ervf_end_to_end_test.json"
OUTPUT = R / "p11c_nvme_tier.json"
EXPERT_BYTES = 3_035_136
LAYERS, EXPERTS = 48, 128
SEED = 111208


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "min": float(x.min()), "max": float(x.max())}


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_RANDOM_ACCESS = 0x10000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
MEM_COMMIT, MEM_RESERVE, MEM_RELEASE, PAGE_READWRITE = 0x1000, 0x2000, 0x8000, 0x04

kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
kernel32.CreateFileW.restype = wintypes.HANDLE
kernel32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
kernel32.SetFilePointerEx.restype = wintypes.BOOL
kernel32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
kernel32.ReadFile.restype = wintypes.BOOL
kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
kernel32.VirtualAlloc.restype = ctypes.c_void_p
kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
kernel32.VirtualFree.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


class DirectReader:
    def __init__(self, random_access=True):
        flag = FILE_FLAG_RANDOM_ACCESS if random_access else FILE_FLAG_SEQUENTIAL_SCAN
        self.handles = []
        for layer in range(LAYERS):
            path = str(BANK_DIR / f"layer_{layer:02d}.q5bin")
            handle = kernel32.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | flag, None)
            if handle == INVALID_HANDLE_VALUE:
                raise ctypes.WinError(ctypes.get_last_error())
            self.handles.append(handle)
        self.buffer = kernel32.VirtualAlloc(None, EXPERT_BYTES, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        if not self.buffer: raise ctypes.WinError(ctypes.get_last_error())

    def read(self, layer, expert):
        offset = ctypes.c_longlong(expert * EXPERT_BYTES)
        if not kernel32.SetFilePointerEx(self.handles[layer], offset, None, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        read = wintypes.DWORD()
        begin = time.perf_counter_ns()
        ok = kernel32.ReadFile(self.handles[layer], self.buffer, EXPERT_BYTES, ctypes.byref(read), None)
        elapsed = (time.perf_counter_ns() - begin) / 1e6
        if not ok: raise ctypes.WinError(ctypes.get_last_error())
        if read.value != EXPERT_BYTES: raise IOError(f"short direct read {read.value}")
        return elapsed

    def sample(self, size=64):
        return ctypes.string_at(self.buffer, size)

    def close(self):
        if self.buffer: kernel32.VirtualFree(self.buffer, 0, MEM_RELEASE); self.buffer = None
        for handle in self.handles: kernel32.CloseHandle(handle)
        self.handles = []


def main():
    rng = random.Random(SEED)
    order = [(layer, expert) for layer in range(LAYERS) for expert in rng.sample(range(EXPERTS), 8)]
    random_reader = DirectReader(True)
    for layer, expert in order[:16]: random_reader.read(layer, expert)
    random_ms = []; integrity_failures = 0
    for index, (layer, expert) in enumerate(order):
        elapsed = random_reader.read(layer, expert); random_ms.append(elapsed)
        if index % 32 == 0:
            with (BANK_DIR / f"layer_{layer:02d}.q5bin").open("rb") as handle:
                handle.seek(expert * EXPERT_BYTES)
                expected = handle.read(64)
            integrity_failures += int(random_reader.sample(64) != expected)
            print(json.dumps({"random_reads": index + 1, "last_ms": elapsed}), flush=True)
    random_reader.close()

    sequential = DirectReader(False)
    sequential_begin = time.perf_counter()
    sequential_read_ms = []
    for layer in range(LAYERS):
        for expert in range(EXPERTS): sequential_read_ms.append(sequential.read(layer, expert))
        print(json.dumps({"sequential_layer": layer}), flush=True)
    sequential_seconds = time.perf_counter() - sequential_begin
    sequential.close()
    total_bytes = LAYERS * EXPERTS * EXPERT_BYTES

    p7c = json.loads(P7C.read_text(encoding="utf-8"))
    wall = np.asarray(p7c["quality"]["aggregate"]["wall_ms"], dtype=np.float64)
    misses = np.asarray(p7c["quality"]["aggregate"]["misses"], dtype=np.float64)
    random_stat = stats(random_ms)
    projected_mean_read = wall + misses * random_stat["mean"]
    projected_p95_read = wall + misses * random_stat["p95"]
    projection = {
        "using_random_mean": stats(projected_mean_read),
        "using_random_p95_per_record": stats(projected_p95_read),
        "tokens_per_second_using_mean": 1000.0 / float(projected_mean_read.mean()),
    }
    gates = {
        "integrity": integrity_failures == 0,
        "random_p95_le_2ms": random_stat["p95"] <= 2.0,
        "projected_mean_le_100ms": projection["using_random_mean"]["mean"] <= 100.0,
        "projected_p95_le_150ms": projection["using_random_p95_per_record"]["p95"] <= 150.0,
        "projected_tps_ge_10": projection["tokens_per_second_using_mean"] >= 10.0,
    }
    result = {
        "kind": "streamq5_moe_p11c_nvme_tier", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
        "seed": SEED, "direct_io": True, "record_bytes": EXPERT_BYTES, "random_reads": len(random_ms),
        "random_read_ms": random_ms, "random_stats": random_stat,
        "sequential_records": len(sequential_read_ms), "sequential_seconds": sequential_seconds,
        "sequential_GBps": total_bytes / sequential_seconds / 1e9,
        "sequential_read_stats": stats(sequential_read_ms), "integrity_failures": integrity_failures,
        "p7c_test_projection": projection, "gates": gates, "overall_pass": all(gates.values()),
        "claim_boundary": "Direct-I/O capacity projection; it does not include queue contention with model compute or SSD endurance and cannot imply a speedup over pinned RAM.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"random_stats": random_stat, "sequential_GBps": result["sequential_GBps"], "projection": projection, "gates": gates, "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
