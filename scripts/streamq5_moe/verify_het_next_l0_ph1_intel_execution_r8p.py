#!/usr/bin/env python3
"""Independent R8P runtime/preparation verifier; no candidate imports/device."""
from __future__ import annotations

import base64
import copy
import csv
import hashlib
import importlib
import io
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
PYVENV = VENV / "pyvenv.cfg"
RESULT = R / "het_next_l0_ph1_intel_execution_r8_static_preflight.json"
OUTPUT = R / "het_next_l0_ph1_intel_execution_r8p_independent_verification.json"
LOCK = R / "het_next_l0_ph1_intel_execution_r8_lock.json"
CPU = R / "het_next_l0_ph1_cpu_freeze_r2"
FAILURE = R / "het_next_l0_ph1_intel_execution_r7d1_failed_attempts/attempt_7c45ba0bda09470eba7145ef75281ea3/failure.json"
ACK = "PH1_INTEL_EXECUTION_R8P_EXACT_VENV_CPU_PREPARATION_CLOSED"
PREPARATION_DIGEST = "f5a15db125c7a69357574111bd9549c36ae74b67af12205fc71a99a4c8962a49"
PYTHON_SHA = "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"
PYTHON_VERSION = "3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]"
BASE_PREFIX = r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0"
FILES = {
    "pyvenv": (PYVENV, "9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe", 477),
    "psutil_init": (VENV / "Lib/site-packages/psutil/__init__.py", "7b6a0675824eb1fa2ff0cb1eb36e358dc454703e51dfa4e9a0e6ccd26a159f0c", 92363),
    "psutil_windows": (VENV / "Lib/site-packages/psutil/_pswindows.py", "0bbd52dcb214735be4168d11a2ae192d5bc7265c8cf72c611179476479687f54", 36466),
    "psutil_native": (VENV / "Lib/site-packages/psutil/_psutil_windows.pyd", "0035450801bd7d938e9e146c5ec28e619cb5a5f4a18cdc53ac7e9734c7f94f78", 70656),
    "psutil_metadata": (VENV / "Lib/site-packages/psutil-7.2.2.dist-info/METADATA", "a263a40220d921d9cb963fc636d34f817aa2eb72c2696e3e3465d088cdb1976b", 22729),
    "psutil_record": (VENV / "Lib/site-packages/psutil-7.2.2.dist-info/RECORD", "55fd2f55e72c18fd0017a0a033af4661d0227e339c5d772a40a29375e6f740d7", 1875),
    "numpy_init": (VENV / "Lib/site-packages/numpy/__init__.py", "ad238e76e8c6fbd56a19e6c894864cf466bd2ed76004cac89e78c019fa625607", 23016),
    "numpy_metadata": (VENV / "Lib/site-packages/numpy-2.2.6.dist-info/METADATA", "229f3544b02805e0f6a12030e155d8a45fd3a4100b3291574175e6a76f20e1e1", 60844),
    "numpy_record": (VENV / "Lib/site-packages/numpy-2.2.6.dist-info/RECORD", "859c44e1afc26d39b7df8b6b05bee4aed41469d9888c0889710c8603e8520cdc", 108709),
}
CHECK_NAMES = {"hash_bindings", "closed_pending", "runtime_lock_contract", "exact_isolated_runtime", "start_ram_16gib", "full_wheel_records", "runtime_mutations", "immutable_r7d1_failure", "cpu_preparation_equivalence", "static_bootstrap_no_device", "clean_state", "result_absent"}
MUTATIONS = ("kind", "check_false", "runtime", "wheel", "record", "control", "stage", "preparation_digest", "device_claim", "failure")
LOCK_PATHS = {
    "runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r8.py", "verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8.py", "preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8.py", "preflight_verifier_sha256": Path(__file__), "prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8_PREREGISTRATION_2026-08-14.md", "runtime_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PSUTIL_FAILURE_AND_R8_RUNTIME_REPAIR_AUDIT_2026-08-14.md",
    "r7d1_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7d1.py", "r7d1_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7d1.py", "r7d1_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PREREGISTRATION_2026-08-14.md", "r7d1_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7d1_lock.json", "r7d1_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md", "r7d1_failure_sha256": FAILURE,
    "r7d_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7d.py", "r7d_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7d.py", "r7d_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D_PREREGISTRATION_2026-08-14.md", "r7d_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7d_lock.json", "r7d_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md", "r7c2_result_sha256": R / "het_next_l0_ph1_intel_execution_r7c2_static_preflight.json", "r7c2_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C2_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    "r7a_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7a.py", "r7a_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7a.py", "r7a_authorization_result_sha256": R / "het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json", "r7p_result_sha256": R / "het_next_l0_ph1_intel_execution_r7p_static_preflight.json", "backend_sha256": S / "het_next_l0_ph1_intel_execution_r6_backend.py", "common_sha256": S / "het_next_l0_ph1_intel_execution_r6_common.py",
    "cpu_result_sha256": CPU / "cpu_stage_freeze.json", "cpu_raw_sha256": CPU / "cpu_stage_freeze.safetensors", "cpu_lut_sha256": CPU / "bf16_silu_lut.bin", "cpu_manifest_sha256": CPU / "manifest.json", "cpu_commit_sha256": CPU / "commit.json", "cpu_handoff_sha256": CPU / "handoff.json", "cpu_verification_sha256": R / "het_next_l0_ph1_cpu_freeze_r2_independent_verification.json", "cpu_verification_report_sha256": R / "HET_NEXT_L0_PH1_CPU_FREEZE_R2_INDEPENDENT_VERIFICATION_REPORT_2026-08-14.md",
}
LOCK_STATIC = {"python_sha256": PYTHON_SHA, "python_version": PYTHON_VERSION, "pyvenv_sha256": FILES["pyvenv"][1], "psutil_version": "7.2.2", "psutil_init_sha256": FILES["psutil_init"][1], "psutil_native_sha256": FILES["psutil_native"][1], "psutil_metadata_sha256": FILES["psutil_metadata"][1], "psutil_record_sha256": FILES["psutil_record"][1], "numpy_version": "2.2.6", "numpy_init_sha256": FILES["numpy_init"][1], "numpy_metadata_sha256": FILES["numpy_metadata"][1], "numpy_record_sha256": FILES["numpy_record"][1], "preparation_digest": PREPARATION_DIGEST}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canon(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_create(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".inprogress." + uuid.uuid4().hex)
    try:
        with temp.open("xb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def file_row(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def collect_runtime() -> dict:
    executable = Path(sys.executable).resolve()
    if not (executable == PYTHON.resolve() and sha256(executable) == PYTHON_SHA and executable.stat().st_size == 274424 and sys.version == PYTHON_VERSION and list(sys.version_info[:3]) == [3, 12, 10] and sys.implementation.name == "cpython" and sys.implementation.cache_tag == "cpython-312" and sys.platform == "win32" and Path(sys.prefix).resolve() == VENV.resolve() and str(Path(sys.base_prefix).resolve()).casefold() == BASE_PREFIX.casefold() and sys.flags.isolated == sys.flags.no_user_site == sys.flags.dont_write_bytecode == 1 and list(sys.orig_argv[1:3]) == ["-I", "-B"]):
        raise RuntimeError("python_runtime")
    if not all(path.is_file() and sha256(path) == digest and path.stat().st_size == size for path, digest, size in FILES.values()):
        raise RuntimeError("runtime_files")
    psutil = importlib.import_module("psutil"); numpy = importlib.import_module("numpy"); native = importlib.import_module("psutil._psutil_windows")
    if Path(psutil.__file__).resolve() != FILES["psutil_init"][0].resolve() or Path(numpy.__file__).resolve() != FILES["numpy_init"][0].resolve() or Path(native.__file__).resolve() != FILES["psutil_native"][0].resolve() or psutil.__version__ != "7.2.2" or numpy.__version__ != "2.2.6":
        raise RuntimeError("module_identity")
    process = psutil.Process()
    return {"python_executable": str(executable), "python_sha256": sha256(executable), "python_bytes": executable.stat().st_size, "python_version": sys.version, "python_version_info": list(sys.version_info[:3]), "implementation": sys.implementation.name, "cache_tag": sys.implementation.cache_tag, "platform": sys.platform, "prefix": str(Path(sys.prefix).resolve()), "base_prefix": str(Path(sys.base_prefix).resolve()), "isolated": sys.flags.isolated, "no_user_site": sys.flags.no_user_site, "dont_write_bytecode": sys.flags.dont_write_bytecode, "orig_flags": list(sys.orig_argv[1:3]), "psutil_version": psutil.__version__, "numpy_version": numpy.__version__, "runtime_files": {name: file_row(path) for name, (path, _, _) in FILES.items()}, "available": psutil.virtual_memory().available, "rss": process.memory_info().rss, "peak_wset": process.memory_info().peak_wset}


def runtime_static_valid(value: dict) -> bool:
    current = collect_runtime()
    telemetry = {"available", "rss", "peak_wset"}
    return set(value) == set(current) and all(value[name] == current[name] for name in current if name not in telemetry) and all(isinstance(value[name], int) and value[name] > 0 for name in telemetry)


def wheel_record(path: Path) -> dict:
    site = path.parent.parent; rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8")))); checked = skipped = 0; unhashed = []
    for relative, encoded, declared_size in rows:
        normalized = relative.replace("\\", "/")
        if "/__pycache__/" in "/" + normalized or normalized.endswith((".pyc", ".pyo")):
            skipped += 1; continue
        target = (site / Path(*normalized.split("/"))).resolve()
        if not encoded:
            unhashed.append(relative); continue
        algorithm, text = encoded.split("=", 1); expected = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)).hex()
        if algorithm != "sha256" or not target.is_file() or target.stat().st_size != int(declared_size) or sha256(target) != expected:
            raise RuntimeError("record:" + relative)
        checked += 1
    if unhashed != [path.relative_to(site).as_posix()]: raise RuntimeError("record_schema")
    return {"record": str(path.resolve()), "rows": len(rows), "hashed_files_verified": checked, "cache_rows_excluded": skipped, "unhashed_rows": unhashed}


def independent_preparation() -> dict:
    sys.path.insert(0, str(S))
    import verify_het_next_l0_ph1_intel_execution_r7a as frozen
    records = {}; weights = {}
    for spec in frozen.SPECS:
        records[spec[0]], weights[spec[0]] = frozen.codec(frozen.rr(frozen.SHARD, spec[3][0], spec[3][1] - spec[3][0]), spec)
    input_bytes = frozen.rr(frozen.D2, 155138788, 4096); lut = (CPU / "bf16_silu_lut.bin").read_bytes(); iw = frozen.np.frombuffer(input_bytes, "<u2")
    gate = frozen.linear(weights["gate"], iw); up = frozen.linear(weights["up"], iw); silu = frozen.np.frombuffer(lut, "<u2")[gate]; activation = frozen.np.asarray([frozen.mul(int(a), int(b)) for a, b in zip(silu, up, strict=True)], frozen.np.uint16); down = frozen.linear(weights["down"], activation)
    stage = {"gate": gate.astype("<u2").tobytes(), "up": up.astype("<u2").tobytes(), "silu": silu.astype("<u2").tobytes(), "activation": activation.astype("<u2").tobytes(), "down": down.astype("<u2").tobytes()}
    return {
        "records": {spec[0]: {"bytes": len(records[spec[0]]), "sha256": sha_bytes(records[spec[0]]), "shape": list(spec[2])} for spec in frozen.SPECS},
        "input": {"bytes": len(input_bytes), "sha256": sha_bytes(input_bytes), "dtype": "BF16", "shape": [2048]},
        "lut": {"bytes": len(lut), "sha256": sha_bytes(lut), "dtype": "BF16", "shape": [65536]},
        "stages": {name: {"bytes": len(data), "sha256": sha_bytes(data), "dtype": "BF16", "shape": [len(data) // 2]} for name, data in stage.items()},
        "controls": frozen.rebuild_controls(records, lut),
        "cpu_evidence": {"result_sha256": sha256(CPU / "cpu_stage_freeze.json"), "raw_sha256": sha256(CPU / "cpu_stage_freeze.safetensors"), "manifest_sha256": sha256(CPU / "manifest.json"), "commit_sha256": sha256(CPU / "commit.json"), "verification_sha256": sha256(R / "het_next_l0_ph1_cpu_freeze_r2_independent_verification.json")},
    }


def valid_result(row: dict, independent: dict, live: dict, wheels: dict) -> bool:
    checks = row.get("checks", {})
    return set(row) == {"kind", "ack", "checks", "pass", "passed", "total", "no_compiler_device", "cpu_payload_read", "runtime", "wheel_records", "preparation", "preparation_digest", "rejected_runtime_mutations", "r7d1_failure_sha256"} and row.get("kind") == "ph1_intel_execution_r8p_static_preflight" and row.get("ack") == ACK and isinstance(checks, dict) and set(checks) == CHECK_NAMES and all(value is True for value in checks.values()) and row.get("pass") is True and row.get("passed") == row.get("total") == 12 and row.get("no_compiler_device") is True and row.get("cpu_payload_read") is True and runtime_static_valid(row.get("runtime", {})) and row.get("runtime", {}).get("available", 0) >= 16 * 2**30 and row.get("wheel_records") == wheels and row.get("preparation") == independent and row.get("preparation_digest") == PREPARATION_DIGEST == sha_bytes(canon(independent)) and row.get("rejected_runtime_mutations") == ["python_path", "python_hash", "isolation", "bytecode", "pyvenv", "psutil_native", "psutil_record", "numpy_version", "numpy_record", "ram"] and row.get("r7d1_failure_sha256") == sha256(FAILURE) == "88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0"


def lock_valid() -> bool:
    lock = json.loads(LOCK.read_text()); observed = {name: sha256(path) for name, path in LOCK_PATHS.items()}
    return set(lock) == {"kind", "execution_open", "audit_token", "physical_output", "physical_verifier", *LOCK_STATIC, *observed} and lock.get("kind") == "ph1_intel_execution_r8_lock" and lock.get("execution_open") is False and lock.get("audit_token") == "PENDING" and lock.get("physical_output") == "het_next_l0_ph1_intel_execution_r7a" and lock.get("physical_verifier") == "verify_het_next_l0_ph1_intel_execution_r8.py" and all(lock.get(name) == value for name, value in LOCK_STATIC.items()) and all(lock.get(name) == digest for name, digest in observed.items()) and observed["runtime_audit_sha256"] == "a7fcef86b8cee812643593ad38e201798df798f63e2258001e4599135ed719b7"


def main() -> int:
    raw = RESULT.read_bytes(); row = json.loads(raw); live = collect_runtime(); wheels = {"psutil": wheel_record(FILES["psutil_record"][0]), "numpy": wheel_record(FILES["numpy_record"][0])}; independent = independent_preparation(); baseline = valid_result(row, independent, live, wheels)
    mutations = {
        "kind": lambda x: x.__setitem__("kind", "wrong"), "check_false": lambda x: x["checks"].__setitem__("hash_bindings", False),
        "runtime": lambda x: x["runtime"].__setitem__("numpy_version", "2.4.4"), "wheel": lambda x: x["wheel_records"]["numpy"].__setitem__("hashed_files_verified", 0),
        "record": lambda x: x["preparation"]["records"]["gate"].__setitem__("sha256", "0" * 64), "control": lambda x: x["preparation"]["controls"].pop(),
        "stage": lambda x: x["preparation"]["stages"]["down"].__setitem__("sha256", "0" * 64), "preparation_digest": lambda x: x.__setitem__("preparation_digest", "0" * 64),
        "device_claim": lambda x: x.__setitem__("no_compiler_device", False), "failure": lambda x: x.__setitem__("r7d1_failure_sha256", "0" * 64),
    }
    rejected = []
    for name, mutate in mutations.items():
        candidate = copy.deepcopy(row); mutate(candidate)
        if not valid_result(candidate, independent, live, wheels): rejected.append(name)
    checks = {"result_schema": baseline, "independent_lock_chain": lock_valid(), "independent_runtime": runtime_static_valid(live), "full_wheel_records": wheels["psutil"]["rows"] == 28 and wheels["psutil"]["hashed_files_verified"] == 17 and wheels["psutil"]["cache_rows_excluded"] == 10 and wheels["numpy"]["rows"] == 1311 and wheels["numpy"]["hashed_files_verified"] == 899 and wheels["numpy"]["cache_rows_excluded"] == 411, "independent_cpu_preparation": sha_bytes(canon(independent)) == PREPARATION_DIGEST, "negative_mutations": tuple(rejected) == MUTATIONS, "result_sha256": sha_bytes(raw) == sha256(RESULT), "no_device_output": not (R / "het_next_l0_ph1_intel_execution_r7a").exists()}
    output = {"kind": "ph1_intel_execution_r8p_independent_verification", "checks": checks, "pass": all(checks.values()), "passed": sum(value is True for value in checks.values()), "total": len(checks), "result_sha256": sha256(RESULT), "preparation_digest": PREPARATION_DIGEST, "rejected_mutations": rejected, "no_compiler_device": True, "cpu_payload_read": True}
    atomic_create(OUTPUT, canon(output)); print(json.dumps(output, indent=2)); return 0 if output["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
