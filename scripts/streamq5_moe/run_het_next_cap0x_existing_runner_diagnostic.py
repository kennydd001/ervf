from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "reports/runs/streamq5_moe/het_next_cap0x_existing_runner_diagnostic"
PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_CAP0X_EXISTING_RUNNER_CONCURRENCY_DIAGNOSTIC_PREREGISTRATION_2026-08-13.md"
RESULT = RUN / "cap0x_result.json"
INTEL_RESULT = RUN / "intel_st2_w8.json"
NVIDIA_RESULT = RUN / "nvidia_d7.json"
NVIDIA_REPORT = RUN / "nvidia_d7.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temp = path.with_suffix(path.suffix + ".inprogress")
    if path.exists() or temp.exists():
        raise FileExistsError(path)
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run_intel() -> int:
    from scripts.streamq5_moe import run_st2_mini_ergv_w8 as target

    target.OUTPUT = INTEL_RESULT
    target.main()
    return 0


def run_nvidia() -> int:
    from scripts.streamq5_moe import run_port80b_d7_staged_exact_q5_plane as target

    target.OUTPUT = NVIDIA_RESULT
    target.REPORT = NVIDIA_REPORT
    target.main()
    return 0


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def correctness_intel(value: dict[str, object]) -> bool:
    if value.get("error"):
        return False
    for key in ("correctness", "gates"):
        item = value.get(key)
        if isinstance(item, dict):
            if item.get("bitwise_equal") is False:
                return False
            if item.get("all_outputs_bit_exact") is False:
                return False
    text = json.dumps(value, sort_keys=True)
    return '"different_bits": 0' in text or '"bitwise_equal": true' in text or '"exact": true' in text


def correctness_nvidia(value: dict[str, object]) -> bool:
    correctness = value.get("correctness")
    return bool(
        not value.get("error")
        and isinstance(correctness, dict)
        and correctness.get("bitwise_equal") is True
        and value.get("unregister_failures") == []
    )


def child_command(role: str) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--role", role]


def run_coordinator() -> int:
    if RUN.exists():
        raise FileExistsError(f"refusing non-clean diagnostic directory: {RUN}")
    RUN.mkdir(parents=True)
    records: dict[str, dict[str, object]] = {}
    processes: dict[str, subprocess.Popen[str]] = {}
    stdout_files = {}
    stderr_files = {}
    monitor: list[dict[str, object]] = []
    error = None
    started = time.perf_counter_ns()
    try:
        for role in ("nvidia", "intel"):
            stdout_path = RUN / f"{role}.stdout.txt"
            stderr_path = RUN / f"{role}.stderr.txt"
            stdout_files[role] = stdout_path.open("x", encoding="utf-8")
            stderr_files[role] = stderr_path.open("x", encoding="utf-8")
            begin = time.perf_counter_ns()
            proc = subprocess.Popen(
                child_command(role),
                cwd=str(ROOT),
                stdout=stdout_files[role],
                stderr=stderr_files[role],
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            processes[role] = proc
            records[role] = {"pid": proc.pid, "start_qpc_ns": begin, "command": child_command(role)}

        deadline = time.monotonic() + 1800.0
        while True:
            now = time.perf_counter_ns()
            alive = {role: proc.poll() is None for role, proc in processes.items()}
            sample: dict[str, object] = {"qpc_ns": now, "alive": alive}
            if alive.get("nvidia"):
                try:
                    query = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=2,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    sample["nvidia_smi"] = query.stdout.strip()
                except Exception as exc:
                    sample["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
            monitor.append(sample)
            if not any(alive.values()):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("CAP0X children exceeded 1800 seconds")
            time.sleep(0.1)

        for role, proc in processes.items():
            records[role]["end_qpc_ns"] = time.perf_counter_ns()
            records[role]["exit_code"] = proc.returncode
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in processes.values():
            try:
                proc.wait(timeout=15)
            except Exception:
                if proc.poll() is None:
                    proc.kill()
    finally:
        for stream in list(stdout_files.values()) + list(stderr_files.values()):
            stream.close()

    for role, proc in processes.items():
        records.setdefault(role, {})["final_exit_code"] = proc.poll()
        records[role]["alive_after_wait"] = proc.poll() is None
        records[role]["stdout_sha256"] = sha256(RUN / f"{role}.stdout.txt")
        records[role]["stderr_sha256"] = sha256(RUN / f"{role}.stderr.txt")

    intel = load_json(INTEL_RESULT) if INTEL_RESULT.exists() else {}
    nvidia = load_json(NVIDIA_RESULT) if NVIDIA_RESULT.exists() else {}
    overlap = False
    if all(role in records and "end_qpc_ns" in records[role] for role in ("intel", "nvidia")):
        overlap = max(int(records["intel"]["start_qpc_ns"]), int(records["nvidia"]["start_qpc_ns"])) < min(
            int(records["intel"]["end_qpc_ns"]), int(records["nvidia"]["end_qpc_ns"])
        )
    gates = {
        "both_exit_zero": all(records.get(role, {}).get("final_exit_code") == 0 for role in ("intel", "nvidia")),
        "strict_process_interval_overlap": overlap,
        "intel_existing_runner_correctness_retained": correctness_intel(intel),
        "nvidia_existing_runner_correctness_retained": correctness_nvidia(nvidia),
        "no_surviving_child": all(records.get(role, {}).get("alive_after_wait") is False for role in ("intel", "nvidia")),
        "coordinator_error_absent": error is None,
    }
    result = {
        "kind": "het_next_cap0x_existing_runner_concurrency_diagnostic",
        "status": "exploratory_concurrency_diagnostic_positive" if all(gates.values()) else "exploratory_concurrency_diagnostic_negative",
        "positive": all(gates.values()),
        "started_utc": utc_now(),
        "wall_ms": (time.perf_counter_ns() - started) / 1e6,
        "bindings": {
            "preregistration_sha256": sha256(PREREG),
            "coordinator_sha256": sha256(Path(__file__)),
            "intel_runner_sha256": sha256(ROOT / "scripts/streamq5_moe/run_st2_mini_ergv_w8.py"),
            "nvidia_runner_sha256": sha256(ROOT / "scripts/streamq5_moe/run_port80b_d7_staged_exact_q5_plane.py"),
        },
        "processes": records,
        "monitor_samples": monitor,
        "intel_result_sha256": sha256(INTEL_RESULT) if INTEL_RESULT.exists() else None,
        "nvidia_result_sha256": sha256(NVIDIA_RESULT) if NVIDIA_RESULT.exists() else None,
        "gates": gates,
        "error": error,
        "claim_boundary": "Exploratory overlap of existing runner process lifetimes only; no proof of kernel overlap, same-process coexistence, hybrid speedup, model quality, deployment, or breakthrough.",
    }
    atomic_json(RESULT, result)
    print(json.dumps({"status": result["status"], "gates": gates, "processes": records, "error": error}, indent=2))
    return 0 if error is None else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("coordinator", "intel", "nvidia"), default="coordinator")
    args = parser.parse_args()
    if args.role == "intel":
        return run_intel()
    if args.role == "nvidia":
        return run_nvidia()
    return run_coordinator()


if __name__ == "__main__":
    raise SystemExit(main())
