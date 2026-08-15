#!/usr/bin/env python3
"""PH1 Intel R8: closed exact-venv runtime gate for a future R8A."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / "Scripts/python.exe"
PYVENV = VENV / "pyvenv.cfg"
LOCK = R / "het_next_l0_ph1_intel_execution_r8_lock.json"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8_PREREGISTRATION_2026-08-14.md"
PREFLIGHT = S / "preflight_het_next_l0_ph1_intel_execution_r8.py"
PREFLIGHT_VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r8p.py"
VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r8.py"
RUNTIME_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PSUTIL_FAILURE_AND_R8_RUNTIME_REPAIR_AUDIT_2026-08-14.md"
R7D1_FAILURE = R / "het_next_l0_ph1_intel_execution_r7d1_failed_attempts/attempt_7c45ba0bda09470eba7145ef75281ea3/failure.json"
R7A_VERIFICATION = R / "het_next_l0_ph1_intel_execution_r7a_independent_verification.json"
FAILED = R / "het_next_l0_ph1_intel_execution_r8_failed_attempts"
QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8_quarantine"
REVISION_OUT = R / "het_next_l0_ph1_intel_execution_r8"
VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r8_independent_verification.json"
PREFLIGHT_RESULT = R / "het_next_l0_ph1_intel_execution_r8_static_preflight.json"
PREFLIGHT_VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r8p_independent_verification.json"
ACK = "PH1_INTEL_EXECUTION_R8_AFTER_EXACT_VENV_R8P_AND_AUDIT_GO"

PYTHON_SHA = "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"
PYTHON_BYTES = 274424
PYTHON_VERSION = "3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]"
BASE_PREFIX = r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0"
PYVENV_SHA = "9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe"
PYVENV_BYTES = 477
PSUTIL_VERSION = "7.2.2"
NUMPY_VERSION = "2.2.6"
RUNTIME_FILES = {
    "pyvenv": (PYVENV, PYVENV_SHA, PYVENV_BYTES),
    "psutil_init": (VENV / "Lib/site-packages/psutil/__init__.py", "7b6a0675824eb1fa2ff0cb1eb36e358dc454703e51dfa4e9a0e6ccd26a159f0c", 92363),
    "psutil_windows": (VENV / "Lib/site-packages/psutil/_pswindows.py", "0bbd52dcb214735be4168d11a2ae192d5bc7265c8cf72c611179476479687f54", 36466),
    "psutil_native": (VENV / "Lib/site-packages/psutil/_psutil_windows.pyd", "0035450801bd7d938e9e146c5ec28e619cb5a5f4a18cdc53ac7e9734c7f94f78", 70656),
    "psutil_metadata": (VENV / "Lib/site-packages/psutil-7.2.2.dist-info/METADATA", "a263a40220d921d9cb963fc636d34f817aa2eb72c2696e3e3465d088cdb1976b", 22729),
    "psutil_record": (VENV / "Lib/site-packages/psutil-7.2.2.dist-info/RECORD", "55fd2f55e72c18fd0017a0a033af4661d0227e339c5d772a40a29375e6f740d7", 1875),
    "numpy_init": (VENV / "Lib/site-packages/numpy/__init__.py", "ad238e76e8c6fbd56a19e6c894864cf466bd2ed76004cac89e78c019fa625607", 23016),
    "numpy_metadata": (VENV / "Lib/site-packages/numpy-2.2.6.dist-info/METADATA", "229f3544b02805e0f6a12030e155d8a45fd3a4100b3291574175e6a76f20e1e1", 60844),
    "numpy_record": (VENV / "Lib/site-packages/numpy-2.2.6.dist-info/RECORD", "859c44e1afc26d39b7df8b6b05bee4aed41469d9888c0889710c8603e8520cdc", 108709),
}
LOCK_STATIC = {
    "python_sha256": PYTHON_SHA, "python_version": PYTHON_VERSION, "pyvenv_sha256": PYVENV_SHA,
    "psutil_version": PSUTIL_VERSION, "psutil_init_sha256": RUNTIME_FILES["psutil_init"][1],
    "psutil_native_sha256": RUNTIME_FILES["psutil_native"][1], "psutil_metadata_sha256": RUNTIME_FILES["psutil_metadata"][1],
    "psutil_record_sha256": RUNTIME_FILES["psutil_record"][1], "numpy_version": NUMPY_VERSION,
    "numpy_init_sha256": RUNTIME_FILES["numpy_init"][1], "numpy_metadata_sha256": RUNTIME_FILES["numpy_metadata"][1],
    "numpy_record_sha256": RUNTIME_FILES["numpy_record"][1],
    "preparation_digest": "f5a15db125c7a69357574111bd9549c36ae74b67af12205fc71a99a4c8962a49",
}
CHAIN = {
    "runner_sha256": Path(__file__),
    "verifier_sha256": VERIFIER,
    "preflight_sha256": PREFLIGHT,
    "preflight_verifier_sha256": PREFLIGHT_VERIFIER,
    "prereg_sha256": PREREG,
    "runtime_audit_sha256": RUNTIME_AUDIT,
    "r7d1_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7d1.py",
    "r7d1_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7d1.py",
    "r7d1_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PREREGISTRATION_2026-08-14.md",
    "r7d1_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7d1_lock.json",
    "r7d1_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7d1_failure_sha256": R7D1_FAILURE,
    "r7d_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7d.py",
    "r7d_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7d.py",
    "r7d_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D_PREREGISTRATION_2026-08-14.md",
    "r7d_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7d_lock.json",
    "r7d_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7c2_result_sha256": R / "het_next_l0_ph1_intel_execution_r7c2_static_preflight.json",
    "r7c2_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C2_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7a_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7a.py",
    "r7a_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7a.py",
    "r7a_authorization_result_sha256": R / "het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json",
    "r7p_result_sha256": R / "het_next_l0_ph1_intel_execution_r7p_static_preflight.json",
    "backend_sha256": S / "het_next_l0_ph1_intel_execution_r6_backend.py",
    "common_sha256": S / "het_next_l0_ph1_intel_execution_r6_common.py",
    "cpu_result_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/cpu_stage_freeze.json",
    "cpu_raw_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/cpu_stage_freeze.safetensors",
    "cpu_lut_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/bf16_silu_lut.bin",
    "cpu_manifest_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/manifest.json",
    "cpu_commit_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/commit.json",
    "cpu_handoff_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/handoff.json",
    "cpu_verification_sha256": R / "het_next_l0_ph1_cpu_freeze_r2_independent_verification.json",
    "cpu_verification_report_sha256": R / "HET_NEXT_L0_PH1_CPU_FREEZE_R2_INDEPENDENT_VERIFICATION_REPORT_2026-08-14.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_row(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def validate_runtime(observed: dict) -> bool:
    files = observed.get("runtime_files", {})
    return (
        set(observed) == {
            "python_executable", "python_sha256", "python_bytes", "python_version",
            "python_version_info", "implementation", "cache_tag", "platform", "prefix",
            "base_prefix", "isolated", "no_user_site", "dont_write_bytecode", "orig_flags",
            "psutil_version", "numpy_version", "runtime_files", "available", "rss", "peak_wset",
        }
        and observed["python_executable"].casefold() == str(VENV_PYTHON.resolve()).casefold()
        and observed["python_sha256"] == PYTHON_SHA
        and observed["python_bytes"] == PYTHON_BYTES
        and observed["python_version"] == PYTHON_VERSION
        and observed["python_version_info"] == [3, 12, 10]
        and observed["implementation"] == "cpython"
        and observed["cache_tag"] == "cpython-312"
        and observed["platform"] == "win32"
        and observed["prefix"].casefold() == str(VENV.resolve()).casefold()
        and observed["base_prefix"].casefold() == BASE_PREFIX.casefold()
        and observed["isolated"] == observed["no_user_site"] == observed["dont_write_bytecode"] == 1
        and observed["orig_flags"] == ["-I", "-B"]
        and observed["psutil_version"] == PSUTIL_VERSION
        and observed["numpy_version"] == NUMPY_VERSION
        and set(files) == set(RUNTIME_FILES)
        and all(files[name] == file_row(path) for name, (path, _, _) in RUNTIME_FILES.items())
        and all(files[name]["sha256"] == digest and files[name]["bytes"] == size for name, (_, digest, size) in RUNTIME_FILES.items())
        and all(isinstance(observed[name], int) and observed[name] > 0 for name in ("available", "rss", "peak_wset"))
    )


def collect_runtime() -> dict:
    executable = Path(sys.executable).resolve()
    python_only = {
        "python_executable": str(executable), "python_sha256": sha256(executable), "python_bytes": executable.stat().st_size,
        "python_version": sys.version, "python_version_info": list(sys.version_info[:3]), "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag, "platform": sys.platform, "prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()), "isolated": sys.flags.isolated,
        "no_user_site": sys.flags.no_user_site, "dont_write_bytecode": sys.flags.dont_write_bytecode,
        "orig_flags": list(sys.orig_argv[1:3]),
    }
    if not (
        python_only["python_executable"].casefold() == str(VENV_PYTHON.resolve()).casefold()
        and python_only["python_sha256"] == PYTHON_SHA and python_only["python_bytes"] == PYTHON_BYTES
        and python_only["python_version"] == PYTHON_VERSION and python_only["python_version_info"] == [3, 12, 10]
        and python_only["implementation"] == "cpython" and python_only["cache_tag"] == "cpython-312"
        and python_only["platform"] == "win32" and python_only["prefix"].casefold() == str(VENV.resolve()).casefold()
        and python_only["base_prefix"].casefold() == BASE_PREFIX.casefold()
        and python_only["isolated"] == python_only["no_user_site"] == python_only["dont_write_bytecode"] == 1
        and python_only["orig_flags"] == ["-I", "-B"]
        and all(path.is_file() and sha256(path) == digest and path.stat().st_size == size for path, digest, size in RUNTIME_FILES.values())
    ):
        raise RuntimeError("exact_python_runtime")
    psutil = importlib.import_module("psutil")
    numpy = importlib.import_module("numpy")
    native = importlib.import_module("psutil._psutil_windows")
    if Path(psutil.__file__).resolve() != RUNTIME_FILES["psutil_init"][0].resolve() or Path(native.__file__).resolve() != RUNTIME_FILES["psutil_native"][0].resolve() or Path(numpy.__file__).resolve() != RUNTIME_FILES["numpy_init"][0].resolve():
        raise RuntimeError("package_module_path")
    process = psutil.Process()
    evidence = {
        **python_only, "psutil_version": psutil.__version__, "numpy_version": numpy.__version__,
        "runtime_files": {name: file_row(path) for name, (path, _, _) in RUNTIME_FILES.items()},
        "available": psutil.virtual_memory().available, "rss": process.memory_info().rss, "peak_wset": process.memory_info().peak_wset,
    }
    if not validate_runtime(evidence):
        raise RuntimeError("exact_package_runtime")
    return evidence


def prior_failure_valid() -> bool:
    root = R7D1_FAILURE.parents[1]
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if files != [R7D1_FAILURE] or R7D1_FAILURE.stat().st_size != 931 or sha256(R7D1_FAILURE) != "88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0":
        return False
    row = json.loads(R7D1_FAILURE.read_text())
    return row.get("kind") == "ph1_intel_execution_r7c2_failure" and row.get("status") == "valid_negative_failure" and row.get("error") == "ModuleNotFoundError:No module named 'psutil'" and row.get("device_opened") is False and row.get("disposition") == "atomic_create_new_bounded_outer_failure"


def clean_now() -> bool:
    absent = (
        R / "het_next_l0_ph1_intel_execution_r7a", R / "het_next_l0_ph1_intel_execution_r7a_failed_attempts",
        R / "het_next_l0_ph1_intel_execution_r7a_quarantine", R7A_VERIFICATION,
        R / "het_next_l0_ph1_intel_execution_r7d", R / "het_next_l0_ph1_intel_execution_r7d_failed_attempts",
        R / "het_next_l0_ph1_intel_execution_r7d_quarantine", R / "het_next_l0_ph1_intel_execution_r7d_independent_verification.json",
        REVISION_OUT, FAILED, QUARANTINE, VERIFY_RESULT, PREFLIGHT_RESULT, PREFLIGHT_VERIFY_RESULT,
    )
    stale = [path for path in R.glob("*.inprogress*") if any(tag in path.name for tag in ("r7a", "r7d", "r8"))]
    return all(not path.exists() for path in absent) and stale == [] and prior_failure_valid()


def authorize() -> dict:
    runtime = collect_runtime()
    if not prior_failure_valid():
        raise RuntimeError("immutable_r7d1_failure")
    observed = {name: sha256(path) for name, path in CHAIN.items()}
    lock = json.loads(LOCK.read_text())
    static = LOCK_STATIC
    if not (
        set(lock) == {"kind", "execution_open", "audit_token", "physical_output", "physical_verifier", *static, *observed}
        and lock.get("kind") == "ph1_intel_execution_r8_lock" and lock.get("execution_open") is True
        and lock.get("audit_token") == ACK and lock.get("physical_output") == "het_next_l0_ph1_intel_execution_r7a"
        and lock.get("physical_verifier") == "verify_het_next_l0_ph1_intel_execution_r8.py"
        and all(lock.get(name) == value for name, value in static.items())
        and all(lock.get(name) == digest for name, digest in observed.items())
        and observed["runtime_audit_sha256"] == "a7fcef86b8cee812643593ad38e201798df798f63e2258001e4599135ed719b7"
        and observed["r7d1_audit_sha256"] == "86013a8bb2affd0cd65914a4910bc9dcbd2a3caaaec955b4aca61bfff016be1b"
        and observed["r7d1_failure_sha256"] == "88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0"
        and clean_now()
    ):
        raise RuntimeError("r8_authorization")
    # R7D1 is immutable and intentionally not imported or authorized here.
    r7d = importlib.import_module("run_het_next_l0_ph1_intel_execution_r7d")
    inherited = r7d.authorize()
    inherited["r8_authorization"] = {"lock_sha256": sha256(LOCK), "observed": observed, "runtime": runtime, "r7d1_failure_sha256": observed["r7d1_failure_sha256"], "audit_token": ACK}
    return inherited


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", required=True)
    args = parser.parse_args()
    if args.ack != ACK:
        return 3
    try:
        authorization = authorize()
    except Exception:
        return 3
    r7d = importlib.import_module("run_het_next_l0_ph1_intel_execution_r7d")
    r7d.lifecycle.OUTER_FAILED = FAILED
    r7d.lifecycle.OUTER_QUARANTINE = QUARANTINE
    r7d.lifecycle.REVISION_OUT = REVISION_OUT
    return r7d.lifecycle.outer_execute(authorization)


if __name__ == "__main__":
    raise SystemExit(main())
