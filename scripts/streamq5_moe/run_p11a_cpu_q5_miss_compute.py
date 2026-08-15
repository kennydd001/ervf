from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7a_kernel_roofline import load_q5
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE, comparison


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P11A_CPU_Q5_MISS_COMPUTE_PREREGISTRATION.md"
SOURCE = Path(__file__).with_name("p11a_cpu_q5.cpp")
OUTPUT = R / "p11a_cpu_q5_miss_compute.json"
P7B = R / "p7b_ervf_kernel.json"
P10B = R / "p10b_domain_cache_physical.json"
BANK = ROOT / "reports/runs/streamq5_moe/p1d_q5_bank/layer_00.q5bin"
RUN_DIR = ROOT / "reports/runs/streamq5_moe/p11a_cpu_q5"
CPU_OUTPUT = RUN_DIR / "cpu_outputs.f32bin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wsl(path: Path) -> str:
    resolved = path.resolve(); drive = resolved.drive[0].lower()
    return f"/mnt/{drive}{resolved.as_posix().split(':', 1)[1]}"


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    command = f"g++ -O3 -march=native -fopenmp -ffp-contract=off -std=c++20 '{wsl(SOURCE)}' -o /tmp/p11a_cpu_q5"
    subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-lc", command], check=True)
    command = f"/tmp/p11a_cpu_q5 '{wsl(BANK)}' '{wsl(CPU_OUTPUT)}'"
    measured = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-lc", command],
        check=True, capture_output=True, text=True,
    )
    cpu = json.loads(measured.stdout)

    q5_memory, q5 = load_q5()
    module = cp.RawModule(
        code=CUDA_SOURCE + ERVF_SOURCE, options=("--std=c++11",),
        name_expressions=("q5_gate_up_ervf16", "swiglu_n", "q5_down_ervf16"),
    )
    kernels = {name: module.get_function(name) for name in ("q5_gate_up_ervf16", "swiglu_n", "q5_down_ervf16")}
    stream = cp.cuda.Stream(non_blocking=True)
    x = cp.asarray(np.asarray([((column * 17) % 257 - 128) / 64.0 for column in range(2048)], dtype=np.float32))
    slots = cp.asarray(np.arange(8, dtype=np.int32)); positions = cp.asarray(np.arange(8, dtype=np.int32))
    gate = cp.empty(8 * 768, dtype=cp.float32); up = cp.empty_like(gate); down = cp.empty(8 * 2048, dtype=cp.float32)
    kernels["q5_gate_up_ervf16"]((768,), (256,), (x, q5, slots, positions, gate, up), stream=stream)
    kernels["swiglu_n"]((24,), (256,), (gate, up, positions), stream=stream)
    kernels["q5_down_ervf16"]((1024,), (256,), (gate, q5, slots, positions, down), stream=stream)
    stream.synchronize()
    gpu_outputs = np.concatenate((cp.asnumpy(gate), cp.asnumpy(up), cp.asnumpy(down)))
    cpu_outputs = np.fromfile(CPU_OUTPUT, dtype="<f4")
    correctness = comparison(cpu_outputs, gpu_outputs)

    p7b = json.loads(P7B.read_text(encoding="utf-8"))
    p10b = json.loads(P10B.read_text(encoding="utf-8"))
    universal = p10b["results"]["universal"]
    event_ms_per_copy_mean = universal["event_ms_stats"]["mean"] * universal["tokens"] / universal["total_copies"]
    event_ms_per_copy_p95_proxy = universal["event_ms_stats"]["p95"] / max(1.0, universal["copy_stats"]["p95"])
    gpu_q5_layer_p50 = p7b["test"]["q5"]["ervf"]["stats"]["p50"] / 48.0
    gpu_q5_layer_p95 = p7b["test"]["q5"]["ervf"]["stats"]["p95"] / 48.0
    gpu_all_cold_mean = 8 * event_ms_per_copy_mean + gpu_q5_layer_p50
    gpu_all_cold_p95_proxy = 8 * event_ms_per_copy_p95_proxy + gpu_q5_layer_p95
    gates = {
        "bitwise_equal": correctness["bitwise_equal"], "finite": cpu["finite"],
        "cpu_p50_le_95pct_gpu_mean": cpu["test_p50_ms"] <= 0.95 * gpu_all_cold_mean,
        "cpu_p95_le_95pct_gpu_p95_proxy": cpu["test_p95_ms"] <= 0.95 * gpu_all_cold_p95_proxy,
    }
    result = {
        "kind": "streamq5_moe_p11a_cpu_q5_miss_compute", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
        "cpp_source_sha256": sha256(SOURCE), "bank_sha256": sha256(BANK),
        "cpu": cpu, "correctness": correctness,
        "gpu_all_cold_reference": {
            "event_ms_per_copy_mean": event_ms_per_copy_mean,
            "event_ms_per_copy_p95_proxy": event_ms_per_copy_p95_proxy,
            "cached_q5_layer_p50_ms": gpu_q5_layer_p50, "cached_q5_layer_p95_ms": gpu_q5_layer_p95,
            "all_cold_mean_ms": gpu_all_cold_mean, "all_cold_p95_proxy_ms": gpu_all_cold_p95_proxy,
        },
        "gates": gates, "overall_pass": all(gates.values()),
        "claim_boundary": "Exact local OpenMP CPU implementation versus a measured per-copy GPU-path proxy; not a universal CPU-kernel impossibility proof.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"cpu": {"threads": cpu["selected_threads"], "p50_ms": cpu["test_p50_ms"], "p95_ms": cpu["test_p95_ms"]}, "gpu_reference": result["gpu_all_cold_reference"], "correctness": correctness, "gates": gates, "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
