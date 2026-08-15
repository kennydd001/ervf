#!/usr/bin/env python3
"""Standalone R8 runtime-extension and inherited numerical verifier."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
VENV = ROOT / ".venv"
PYTHON = VENV / "Scripts/python.exe"
OUT = R / "het_next_l0_ph1_intel_execution_r7a"
VERIFY = R / "het_next_l0_ph1_intel_execution_r8_independent_verification.json"
LOCK = R / "het_next_l0_ph1_intel_execution_r8_lock.json"
R7A_VERIFICATION = R / "het_next_l0_ph1_intel_execution_r7a_independent_verification.json"
FAILED = R / "het_next_l0_ph1_intel_execution_r8_failed_attempts"
QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8_quarantine"
REVISION_OUT = R / "het_next_l0_ph1_intel_execution_r8"
R7D1_FAILURE = R / "het_next_l0_ph1_intel_execution_r7d1_failed_attempts/attempt_7c45ba0bda09470eba7145ef75281ea3/failure.json"
ACK = "PH1_INTEL_EXECUTION_R8_AFTER_EXACT_VENV_R8P_AND_AUDIT_GO"
PYTHON_SHA = "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"
PYTHON_VERSION = "3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]"
BASE_PREFIX = r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0"
FILES = {
    "pyvenv": (VENV / "pyvenv.cfg", "9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe", 477),
    "psutil_init": (VENV / "Lib/site-packages/psutil/__init__.py", "7b6a0675824eb1fa2ff0cb1eb36e358dc454703e51dfa4e9a0e6ccd26a159f0c", 92363),
    "psutil_windows": (VENV / "Lib/site-packages/psutil/_pswindows.py", "0bbd52dcb214735be4168d11a2ae192d5bc7265c8cf72c611179476479687f54", 36466),
    "psutil_native": (VENV / "Lib/site-packages/psutil/_psutil_windows.pyd", "0035450801bd7d938e9e146c5ec28e619cb5a5f4a18cdc53ac7e9734c7f94f78", 70656),
    "psutil_metadata": (VENV / "Lib/site-packages/psutil-7.2.2.dist-info/METADATA", "a263a40220d921d9cb963fc636d34f817aa2eb72c2696e3e3465d088cdb1976b", 22729),
    "psutil_record": (VENV / "Lib/site-packages/psutil-7.2.2.dist-info/RECORD", "55fd2f55e72c18fd0017a0a033af4661d0227e339c5d772a40a29375e6f740d7", 1875),
    "numpy_init": (VENV / "Lib/site-packages/numpy/__init__.py", "ad238e76e8c6fbd56a19e6c894864cf466bd2ed76004cac89e78c019fa625607", 23016),
    "numpy_metadata": (VENV / "Lib/site-packages/numpy-2.2.6.dist-info/METADATA", "229f3544b02805e0f6a12030e155d8a45fd3a4100b3291574175e6a76f20e1e1", 60844),
    "numpy_record": (VENV / "Lib/site-packages/numpy-2.2.6.dist-info/RECORD", "859c44e1afc26d39b7df8b6b05bee4aed41469d9888c0889710c8603e8520cdc", 108709),
}
CHAIN = {
    "runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r8.py", "verifier_sha256": Path(__file__),
    "preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8.py", "preflight_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p.py",
    "prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8_PREREGISTRATION_2026-08-14.md", "runtime_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PSUTIL_FAILURE_AND_R8_RUNTIME_REPAIR_AUDIT_2026-08-14.md",
    "r7d1_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7d1.py", "r7d1_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7d1.py", "r7d1_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PREREGISTRATION_2026-08-14.md", "r7d1_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7d1_lock.json", "r7d1_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md", "r7d1_failure_sha256": R7D1_FAILURE,
    "r7d_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7d.py", "r7d_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7d.py", "r7d_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D_PREREGISTRATION_2026-08-14.md", "r7d_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7d_lock.json", "r7d_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7c2_result_sha256": R / "het_next_l0_ph1_intel_execution_r7c2_static_preflight.json", "r7c2_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C2_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7a_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7a.py", "r7a_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7a.py", "r7a_authorization_result_sha256": R / "het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json", "r7p_result_sha256": R / "het_next_l0_ph1_intel_execution_r7p_static_preflight.json", "backend_sha256": S / "het_next_l0_ph1_intel_execution_r6_backend.py", "common_sha256": S / "het_next_l0_ph1_intel_execution_r6_common.py",
    "cpu_result_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/cpu_stage_freeze.json", "cpu_raw_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/cpu_stage_freeze.safetensors", "cpu_lut_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/bf16_silu_lut.bin", "cpu_manifest_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/manifest.json", "cpu_commit_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/commit.json", "cpu_handoff_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/handoff.json", "cpu_verification_sha256": R / "het_next_l0_ph1_cpu_freeze_r2_independent_verification.json", "cpu_verification_report_sha256": R / "HET_NEXT_L0_PH1_CPU_FREEZE_R2_INDEPENDENT_VERIFICATION_REPORT_2026-08-14.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def file_row(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def collect_runtime() -> dict:
    executable = Path(sys.executable).resolve()
    if not (executable == PYTHON.resolve() and sha256(executable) == PYTHON_SHA and executable.stat().st_size == 274424 and sys.version == PYTHON_VERSION and list(sys.version_info[:3]) == [3, 12, 10] and sys.implementation.name == "cpython" and sys.implementation.cache_tag == "cpython-312" and sys.platform == "win32" and Path(sys.prefix).resolve() == VENV.resolve() and str(Path(sys.base_prefix).resolve()).casefold() == BASE_PREFIX.casefold() and sys.flags.isolated == sys.flags.no_user_site == sys.flags.dont_write_bytecode == 1 and list(sys.orig_argv[1:3]) == ["-I", "-B"] and all(path.is_file() and sha256(path) == digest and path.stat().st_size == size for path, digest, size in FILES.values())):
        raise RuntimeError("runtime")
    psutil = importlib.import_module("psutil"); numpy = importlib.import_module("numpy"); native = importlib.import_module("psutil._psutil_windows")
    if Path(psutil.__file__).resolve() != FILES["psutil_init"][0].resolve() or Path(numpy.__file__).resolve() != FILES["numpy_init"][0].resolve() or Path(native.__file__).resolve() != FILES["psutil_native"][0].resolve() or psutil.__version__ != "7.2.2" or numpy.__version__ != "2.2.6": raise RuntimeError("module")
    process = psutil.Process()
    return {"python_executable": str(executable), "python_sha256": sha256(executable), "python_bytes": executable.stat().st_size, "python_version": sys.version, "python_version_info": list(sys.version_info[:3]), "implementation": sys.implementation.name, "cache_tag": sys.implementation.cache_tag, "platform": sys.platform, "prefix": str(Path(sys.prefix).resolve()), "base_prefix": str(Path(sys.base_prefix).resolve()), "isolated": sys.flags.isolated, "no_user_site": sys.flags.no_user_site, "dont_write_bytecode": sys.flags.dont_write_bytecode, "orig_flags": list(sys.orig_argv[1:3]), "psutil_version": psutil.__version__, "numpy_version": numpy.__version__, "runtime_files": {name: file_row(path) for name, (path, _, _) in FILES.items()}, "available": psutil.virtual_memory().available, "rss": process.memory_info().rss, "peak_wset": process.memory_info().peak_wset}


def runtime_valid(value: dict) -> bool:
    try: current = collect_runtime()
    except Exception: return False
    telemetry = {"available", "rss", "peak_wset"}
    return set(value) == set(current) and all(value[name] == current[name] for name in current if name not in telemetry) and all(isinstance(value[name], int) and value[name] > 0 for name in telemetry)


def prior_failure_valid() -> bool:
    files = sorted(path for path in R7D1_FAILURE.parents[1].rglob("*") if path.is_file())
    if files != [R7D1_FAILURE] or R7D1_FAILURE.stat().st_size != 931 or sha256(R7D1_FAILURE) != "88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0": return False
    row = json.loads(R7D1_FAILURE.read_text())
    return row.get("device_opened") is False and row.get("error") == "ModuleNotFoundError:No module named 'psutil'"


def extension_valid(extension: dict, lock: dict, observed: dict) -> bool:
    static = {"python_sha256": PYTHON_SHA, "python_version": PYTHON_VERSION, "pyvenv_sha256": FILES["pyvenv"][1], "psutil_version": "7.2.2", "psutil_init_sha256": FILES["psutil_init"][1], "psutil_native_sha256": FILES["psutil_native"][1], "psutil_metadata_sha256": FILES["psutil_metadata"][1], "psutil_record_sha256": FILES["psutil_record"][1], "numpy_version": "2.2.6", "numpy_init_sha256": FILES["numpy_init"][1], "numpy_metadata_sha256": FILES["numpy_metadata"][1], "numpy_record_sha256": FILES["numpy_record"][1], "preparation_digest": "f5a15db125c7a69357574111bd9549c36ae74b67af12205fc71a99a4c8962a49"}
    return set(lock) == {"kind", "execution_open", "audit_token", "physical_output", "physical_verifier", *static, *observed} and lock.get("kind") == "ph1_intel_execution_r8_lock" and lock.get("execution_open") is True and lock.get("audit_token") == ACK and lock.get("physical_output") == "het_next_l0_ph1_intel_execution_r7a" and lock.get("physical_verifier") == "verify_het_next_l0_ph1_intel_execution_r8.py" and all(lock.get(name) == value for name, value in static.items()) and all(lock.get(name) == digest for name, digest in observed.items()) and set(extension) == {"lock_sha256", "observed", "runtime", "r7d1_failure_sha256", "audit_token"} and extension.get("lock_sha256") == sha256(LOCK) and extension.get("observed") == observed and runtime_valid(extension.get("runtime", {})) and extension.get("r7d1_failure_sha256") == observed.get("r7d1_failure_sha256") == "88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0" and extension.get("audit_token") == ACK and observed.get("runtime_audit_sha256") == "a7fcef86b8cee812643593ad38e201798df798f63e2258001e4599135ed719b7" and prior_failure_valid()


def atomic_create(path: Path, payload: bytes) -> None:
    if path.exists(): raise FileExistsError(path)
    temp = path.with_name(path.name + ".inprogress." + uuid.uuid4().hex)
    try:
        with temp.open("xb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def main() -> int:
    result_path, manifest_path, commit_path = (OUT / name for name in ("result.json", "manifest.json", "commit.json"))
    result_bytes = result_path.read_bytes(); result = json.loads(result_bytes); observed = {name: sha256(path) for name, path in CHAIN.items()}; lock = json.loads(LOCK.read_text()); extension = result.get("authorization", {}).get("r8_authorization", {})
    checks = {"r8_runtime_authorization": runtime_valid(collect_runtime()) and not R7A_VERIFICATION.exists() and extension_valid(extension, lock, observed)}
    if checks["r8_runtime_authorization"]:
        sys.path.insert(0, str(S)); inherited = importlib.import_module("verify_het_next_l0_ph1_intel_execution_r7d")
        r7d_extension = result.get("authorization", {}).get("r7d_authorization", {}); r7c2 = json.loads(inherited.R7C2_RESULT.read_text()); r7a = json.loads(inherited.R7A_RESULT.read_text()); r7p = json.loads(inherited.R7P_RESULT.read_text()); r7d_observed = {name: inherited.sha256(path) for name, path in inherited.CHAIN.items()}
        checks["inherited_r7d_authorization"] = inherited.extension_valid(r7d_extension, json.loads(inherited.LOCK.read_text()), r7d_observed, r7c2, r7a, r7p)
        if checks["inherited_r7d_authorization"]:
            numerical = importlib.import_module("verify_het_next_l0_ph1_intel_execution_r7a"); checks.update(numerical.verify_dict(result)); manifest = json.loads(manifest_path.read_text()); checks["bundle"] = numerical.verify_bundle_contract(result_bytes, manifest, json.loads(commit_path.read_text()), {path.name for path in OUT.iterdir()}, sum(path.stat().st_size for path in OUT.iterdir())); checks["r8_lifecycle_clean"] = not REVISION_OUT.exists() and not FAILED.exists() and not QUARANTINE.exists() and not R7A_VERIFICATION.exists() and not any(path for path in R.glob("*.inprogress*") if "r8" in path.name)
    output = {"kind": "ph1_intel_execution_r8_independent_verification", "checks": checks, "pass": all(checks.values()), "passed": sum(value is True for value in checks.values()), "total": len(checks), "claim": "one real expert/input Intel correctness component only"}
    atomic_create(VERIFY, (json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n").encode()); print(json.dumps(output, indent=2)); return 0 if output["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
