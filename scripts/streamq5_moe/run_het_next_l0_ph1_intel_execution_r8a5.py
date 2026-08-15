#!/usr/bin/env python3
"""R8A5 explicit-module physical wrapper; one authorized attempt, no ancestry chains."""
from __future__ import annotations

import ast
import ctypes as C
import hashlib
import json
import os
import sys
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))
import run_het_next_l0_ph1_intel_execution_r8a as historical_contract
import run_het_next_l0_ph1_intel_execution_r7a as physical

SCRIPT = Path(__file__).resolve()
VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r8a5.py"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A5_PREREGISTRATION_2026-08-14.md"
LOCK = R / "het_next_l0_ph1_intel_execution_r8a5_lock.json"
R8A4_RUNNER = S / "run_het_next_l0_ph1_intel_execution_r8a4.py"
R8A4_VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r8a4.py"
R8A4_PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A4_PREREGISTRATION_2026-08-14.md"
R8A4_LOCK = R / "het_next_l0_ph1_intel_execution_r8a4_lock.json"
R8A4_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A4_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
R8A4_DIAGNOSIS = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A4_PREDEVICE_BINDING_FAILURE_DIAGNOSIS_2026-08-14.md"
OUT = R / "het_next_l0_ph1_intel_execution_r8a5"
BACKEND_FAILED = R / "het_next_l0_ph1_intel_execution_r8a5_backend_failed_attempts"
BACKEND_QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8a5_backend_quarantine"
FAILED = R / "het_next_l0_ph1_intel_execution_r8a5_failed_attempts"
QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8a5_quarantine"
VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r8a5_independent_verification.json"
ACK = "PH1_INTEL_EXECUTION_R8A5_AFTER_R8P8_PASS_AND_EXPLICIT_BINDING_AUDIT_GO"
MAX_FAILURE_BYTES = 16 * 2**20
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / "Scripts/python.exe"
PYVENV = VENV / "pyvenv.cfg"
ALIAS = Path(r"C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe")
BASE_PREFIX = Path(r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0")
EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SCRIPT), "--ack", ACK]
EXPECTED_ARGV = [str(SCRIPT), "--ack", ACK]

HISTORICAL_PATH = S / "run_het_next_l0_ph1_intel_execution_r8a.py"
PHYSICAL_PATH = S / "run_het_next_l0_ph1_intel_execution_r7a.py"
PHYSICAL_VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r7a.py"
EXPECTED_MODULES = {
    "historical": (HISTORICAL_PATH, "552a7f08f83f2ba2ce3da29581029dfdd79e86fbb75faeb71356965073228f15"),
    "physical": (PHYSICAL_PATH, "01fa21266137335494de2d21adba11f45fe83ff95f660d90cef7acc389c1cb04"),
}
HISTORICAL_ATTRS = {
    "r8p8_pass", "r7d_contract", "exact_failure", "R7D1_FAILURE_ROOT", "R7D1_FAILURE",
    "R8P6_FAILURE_ROOT", "R8P6_FAILURE", "physical",
}
PHYSICAL_ATTRS = {"authorize", "execute_authorized", "verify_bundle", "base", "OUT", "FAILED", "QUAR"}

CHAIN = {
    "runner_sha256": SCRIPT,
    "verifier_sha256": VERIFIER,
    "prereg_sha256": PREREG,
    "r8a4_runner_sha256": R8A4_RUNNER,
    "r8a4_verifier_sha256": R8A4_VERIFIER,
    "r8a4_prereg_sha256": R8A4_PREREG,
    "r8a4_lock_sha256": R8A4_LOCK,
    "r8a4_audit_sha256": R8A4_AUDIT,
    "r8a4_diagnosis_sha256": R8A4_DIAGNOSIS,
    "historical_contract_sha256": HISTORICAL_PATH,
    "physical_runner_sha256": PHYSICAL_PATH,
    "physical_verifier_sha256": PHYSICAL_VERIFIER,
}

PRIOR_ABSENT = tuple(
    R / name
    for revision in ("r8a", "r8a1", "r8a2", "r8a3", "r8a4")
    for name in (
        f"het_next_l0_ph1_intel_execution_{revision}",
        f"het_next_l0_ph1_intel_execution_{revision}_backend_failed_attempts",
        f"het_next_l0_ph1_intel_execution_{revision}_backend_quarantine",
        f"het_next_l0_ph1_intel_execution_{revision}_failed_attempts",
        f"het_next_l0_ph1_intel_execution_{revision}_quarantine",
        f"het_next_l0_ph1_intel_execution_{revision}_independent_verification.json",
    )
)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

def same_path(a: object, b: object) -> bool:
    return isinstance(a, str) and isinstance(b, str) and a.casefold() == b.casefold()

def parse_commandline(raw: str) -> list[str]:
    kernel = C.WinDLL("kernel32", use_last_error=True)
    shell = C.WinDLL("shell32", use_last_error=True)
    parse = shell.CommandLineToArgvW
    parse.argtypes = (C.c_wchar_p, C.POINTER(C.c_int))
    parse.restype = C.POINTER(C.c_wchar_p)
    free = kernel.LocalFree
    free.argtypes = (C.c_void_p,)
    free.restype = C.c_void_p
    count = C.c_int()
    ptr = parse(raw, C.byref(count))
    if not ptr:
        raise C.WinError(C.get_last_error())
    try:
        return [ptr[index] for index in range(count.value)]
    finally:
        if free(C.cast(ptr, C.c_void_p)):
            raise C.WinError(C.get_last_error())

def invocation() -> dict:
    kernel = C.WinDLL("kernel32", use_last_error=True)
    get = kernel.GetCommandLineW
    get.argtypes = ()
    get.restype = C.c_wchar_p
    raw = get()
    return {
        "native_raw": raw, "native_argv": parse_commandline(raw), "orig_argv": list(sys.orig_argv),
        "argv": list(sys.argv), "sys_executable": sys.executable, "sys_prefix": sys.prefix,
        "base_executable": getattr(sys, "_base_executable", None), "base_prefix": sys.base_prefix,
        "isolated": sys.flags.isolated, "dont_write_bytecode": sys.dont_write_bytecode,
        "entry_name": __name__, "entry_spec_is_none": __spec__ is None, "entry_package": __package__,
        "entry_file": str(SCRIPT),
        "direct_entry": __name__ == "__main__" and __spec__ is None and __package__ in (None, "") and Path(__file__).resolve() == SCRIPT,
        "python_sha256": sha256(VENV_PYTHON), "pyvenv_sha256": sha256(PYVENV),
    }

def invocation_valid(row: dict) -> bool:
    keys = {"native_raw", "native_argv", "orig_argv", "argv", "sys_executable", "sys_prefix", "base_executable", "base_prefix", "isolated", "dont_write_bytecode", "entry_name", "entry_spec_is_none", "entry_package", "entry_file", "direct_entry", "python_sha256", "pyvenv_sha256"}
    try:
        raw_ok = parse_commandline(row["native_raw"]) == EXPECTED_NATIVE
    except Exception:
        return False
    direct = row.get("entry_name") == "__main__" and row.get("entry_spec_is_none") is True and row.get("entry_package") in (None, "") and same_path(row.get("entry_file"), str(SCRIPT))
    return set(row) == keys and raw_ok and row["native_argv"] == row["orig_argv"] == EXPECTED_NATIVE and row["argv"] == EXPECTED_ARGV and same_path(row["sys_executable"], str(VENV_PYTHON.resolve())) and same_path(row["sys_prefix"], str(VENV.resolve())) and same_path(row["base_executable"], str(ALIAS)) and same_path(row["base_prefix"], str(BASE_PREFIX)) and row["isolated"] == 1 and row["dont_write_bytecode"] is True and row["python_sha256"] == "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14" and row["pyvenv_sha256"] == "9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe" and row["direct_entry"] is True and direct

def ast_gate() -> bool:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    explicit = {"run_het_next_l0_ph1_intel_execution_r8a", "run_het_next_l0_ph1_intel_execution_r7a"}
    if not explicit <= imports:
        return False
    forbidden_imports = {f"run_het_next_l0_ph1_intel_execution_r8a{i}" for i in range(1, 5)}
    if imports & forbidden_imports:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts = []
            cursor = node
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            if isinstance(cursor, ast.Name) and cursor.id in {"prior", "ancestor", "frozen"} and len(parts) > 1:
                return False
    return True

def resolution_sentinel() -> tuple[bool, dict, dict]:
    modules = {"historical": historical_contract, "physical": physical}
    evidence = {}
    for name, module in modules.items():
        expected_path, expected_sha = EXPECTED_MODULES[name]
        actual_path = Path(module.__file__).resolve()
        evidence[name] = {"path": str(actual_path), "sha256": sha256(actual_path), "required_attrs": sorted(HISTORICAL_ATTRS if name == "historical" else PHYSICAL_ATTRS)}
        if actual_path != expected_path.resolve() or evidence[name]["sha256"] != expected_sha:
            return False, evidence, {}
        if not all(hasattr(module, attr) for attr in (HISTORICAL_ATTRS if name == "historical" else PHYSICAL_ATTRS)):
            return False, evidence, {}
    predicates = {
        "r8p8_pass": historical_contract.r8p8_pass(),
        "r7d_contract": historical_contract.r7d_contract(),
        "r7d1_failure": historical_contract.exact_failure(historical_contract.R7D1_FAILURE_ROOT, historical_contract.R7D1_FAILURE, "88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0", "ph1_intel_execution_r7c2_failure"),
        "r8p6_failure": historical_contract.exact_failure(historical_contract.R8P6_FAILURE_ROOT, historical_contract.R8P6_FAILURE, "03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372", "ph1_intel_execution_r8p6_failure"),
    }
    return all(predicates.values()), evidence, predicates

def clean_now() -> bool:
    current = (OUT, BACKEND_FAILED, BACKEND_QUARANTINE, FAILED, QUARANTINE, VERIFY_RESULT)
    return all(not path.exists() for path in (*current, *PRIOR_ABSENT)) and not list(R.glob("het_next_l0_ph1_intel_execution_r8a5*.inprogress*"))

def authorize(ident: dict) -> dict:
    if not invocation_valid(ident) or not clean_now() or not ast_gate():
        raise RuntimeError("identity_topology_or_ast")
    sentinel_ok, modules, predicates = resolution_sentinel()
    observed = {key: sha256(path) for key, path in CHAIN.items()}
    lock = json.loads(LOCK.read_text())
    exact_lock = set(lock) == {"kind", "execution_open", "audit_token", "one_attempt", *observed} and lock.get("kind") == "ph1_intel_execution_r8a5_lock" and lock.get("execution_open") is True and lock.get("audit_token") == ACK and lock.get("one_attempt") is True and all(lock.get(key) == value for key, value in observed.items())
    fixed = observed.get("r8a4_audit_sha256") == "4dc0c0a1f3e411f78f81ef667baf6e00e94f73204f0ed9b3d1794b0f300a7438" and observed.get("r8a4_diagnosis_sha256") == "d88ff5fd76e11757d7d53acf7279bc52897b9b74169414daee3ce18cd8bc6b21"
    if not (sentinel_ok and exact_lock and fixed):
        raise RuntimeError("authorization")
    inherited = physical.authorize()
    inherited["r8a5_authorization"] = {
        "kind": "ph1_intel_execution_r8a5_authorization", "lock_sha256": sha256(LOCK),
        "observed": observed, "audit_token": ACK, "invocation": ident,
        "explicit_modules": modules, "predicate_results": predicates,
        "ast_no_ancestry_chains": True, "one_attempt": True,
    }
    return inherited

def configure() -> None:
    physical.OUT = OUT
    physical.FAILED = BACKEND_FAILED
    physical.QUAR = BACKEND_QUARANTINE

def committed() -> dict | None:
    try:
        return physical.verify_bundle(OUT) if OUT.exists() else None
    except Exception:
        return None

def inherited_files() -> set[Path]:
    return {path.resolve() for path in BACKEND_FAILED.rglob("failure.json")} if BACKEND_FAILED.exists() else set()

def inherited_row(path: Path) -> dict:
    files = sorted(item for item in path.parent.rglob("*") if item.is_file())
    total = sum(item.stat().st_size for item in files)
    row = json.loads(path.read_text())
    normal = {"kind", "status", "error", "traceback", "device_opened", "backend_evidence", "secondary_resource_sample", "disposition"}
    oversize = {"kind", "status", "error", "device_opened", "oversized_temp_bytes", "oversized_temp_digest", "disposition"}
    valid = set(row) in (normal, oversize) and row.get("kind") == "ph1_intel_execution_r7a_failure" and row.get("status") == "valid_negative_failure" and isinstance(row.get("error"), str) and row.get("device_opened") is True and row.get("disposition") in ("attempt_archived_create_new", "oversized_temp_quarantined_not_retained_failure_bundle") and files == [path] and total <= MAX_FAILURE_BYTES
    return {"relative_path": str(path.relative_to(R)), "failure_sha256": sha256(path), "failure_bytes": path.stat().st_size, "bundle_files": [{"name": "failure.json", "bytes": path.stat().st_size, "sha256": sha256(path)}], "bundle_bytes": total, "bundle_file_count": 1, "kind": row.get("kind"), "status": row.get("status"), "disposition": row.get("disposition"), "device_opened": row.get("device_opened"), "valid": valid}

def _write_file(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())

def write_failure(row: dict) -> Path:
    row = dict(row)
    row["kind"] = "ph1_intel_execution_r8a5_failure"
    data = canonical(row)
    if len(data) > MAX_FAILURE_BYTES:
        data = canonical({"kind": "ph1_intel_execution_r8a5_failure", "status": "invalid_protocol", "terminal_type": "oversize_summary", "stage": row.get("stage"), "error": "wrapper_summary_oversize", "original_bytes": len(data), "original_sha256": hashlib.sha256(data).hexdigest(), "device_opened": bool(row.get("device_opened", False)), "disposition": "bounded_summary_only"})
    FAILED.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    temp = R / f"{FAILED.name}.{nonce}.inprogress"
    destination = FAILED / f"attempt_{nonce}"
    temp.mkdir()
    try:
        _write_file(temp / "failure.json", data)
        os.rename(temp, destination)
    except Exception:
        if temp.exists():
            QUARANTINE.mkdir(parents=True, exist_ok=True)
            os.rename(temp, QUARANTINE / f"failed_commit_{nonce}")
        raise
    return destination

def execute(auth: dict) -> int:
    before = inherited_files()
    try:
        code = physical.execute_authorized(auth)
    except Exception as exc:
        if committed() is not None:
            return 3
        write_failure({"status": "infrastructure_negative", "terminal_type": "early_outer_failure", "stage": "delegated_outer_boundary", "error": f"{type(exc).__name__}:{exc}"[:2048], "traceback": traceback.format_exc()[-32768:], "device_opened": False, "delegated_return": None, "inherited_failure_count": 0, "inherited": None, "disposition": "atomic_bounded_outer_failure"})
        return 3
    result = committed()
    if result is not None:
        return 0 if result.get("positive") is True else 3
    evidence = []
    for path in sorted(inherited_files() - before):
        try:
            evidence.append(inherited_row(path))
        except Exception as exc:
            evidence.append({"valid": False, "error": type(exc).__name__})
    correlated = code != 0 and len(evidence) == 1 and evidence[0].get("valid") is True and evidence[0].get("device_opened") is True
    write_failure({"status": "correlated_delegated_negative" if correlated else "invalid_protocol", "terminal_type": "delegated_failure" if correlated else "invalid_delegation", "stage": "delegated_return", "error": "delegated_execution_nonzero" if code else "success_without_commit", "device_opened": evidence[0].get("device_opened", False) if len(evidence) == 1 else False, "delegated_return": code, "inherited_failure_count": len(evidence), "inherited": evidence[0] if len(evidence) == 1 else evidence, "correlation_valid": correlated, "disposition": "atomic_bounded_correlated_summary" if correlated else "atomic_bounded_invalid_protocol_summary"})
    return 3

def main() -> int:
    if sys.argv != EXPECTED_ARGV:
        return 3
    ident = invocation()
    if not invocation_valid(ident):
        return 3
    if not clean_now():
        return 3
    try:
        auth = authorize(ident)
    except Exception as exc:
        write_failure({"status": "infrastructure_negative", "terminal_type": "early_outer_failure", "stage": "predevice_resolution_or_authorization", "error": f"{type(exc).__name__}:{exc}"[:2048], "traceback": traceback.format_exc()[-32768:], "device_opened": False, "delegated_return": None, "inherited_failure_count": 0, "inherited": None, "disposition": "atomic_bounded_outer_failure"})
        return 3
    configure()
    return execute(auth)

if __name__ == "__main__":
    raise SystemExit(main())
