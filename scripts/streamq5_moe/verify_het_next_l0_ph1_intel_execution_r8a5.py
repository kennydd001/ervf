#!/usr/bin/env python3
"""Independent R8A5 terminal verifier; no candidate-runner import."""
from __future__ import annotations

import copy
import ctypes as C
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))
import run_het_next_l0_ph1_intel_execution_r8a as historical_contract
import verify_het_next_l0_ph1_intel_execution_r8a2 as adjudication_contract

SELF = Path(__file__).resolve()
RUNNER = S / "run_het_next_l0_ph1_intel_execution_r8a5.py"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A5_PREREGISTRATION_2026-08-14.md"
LOCK = R / "het_next_l0_ph1_intel_execution_r8a5_lock.json"
R8A4_RUNNER = S / "run_het_next_l0_ph1_intel_execution_r8a4.py"
R8A4_VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r8a4.py"
R8A4_PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A4_PREREGISTRATION_2026-08-14.md"
R8A4_LOCK = R / "het_next_l0_ph1_intel_execution_r8a4_lock.json"
R8A4_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A4_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
R8A4_DIAGNOSIS = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A4_PREDEVICE_BINDING_FAILURE_DIAGNOSIS_2026-08-14.md"
HISTORICAL_RUNNER = S / "run_het_next_l0_ph1_intel_execution_r8a.py"
PHYSICAL_RUNNER = S / "run_het_next_l0_ph1_intel_execution_r7a.py"
PHYSICAL_VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r7a.py"
OUT = R / "het_next_l0_ph1_intel_execution_r8a5"
FAILED = R / "het_next_l0_ph1_intel_execution_r8a5_failed_attempts"
BACKEND_FAILED = R / "het_next_l0_ph1_intel_execution_r8a5_backend_failed_attempts"
QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8a5_quarantine"
BACKEND_QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8a5_backend_quarantine"
VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r8a5_independent_verification.json"
FAMILY_PARENT = R
FAMILY_PREFIX = "het_next_l0_ph1_intel_execution_r8a5"
ACK = "PH1_INTEL_EXECUTION_R8A5_AFTER_R8P8_PASS_AND_EXPLICIT_BINDING_AUDIT_GO"
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / "Scripts/python.exe"
PYVENV = VENV / "pyvenv.cfg"
ALIAS = Path(r"C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe")
BASE_PREFIX = Path(r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0")
RUN_NATIVE = [str(ALIAS), "-I", "-B", str(RUNNER), "--ack", ACK]
RUN_ARGV = [str(RUNNER), "--ack", ACK]
VERIFY_NATIVE = [str(ALIAS), "-I", "-B", str(SELF)]
VERIFY_ARGV = [str(SELF)]
CHAIN = {
    "runner_sha256": RUNNER, "verifier_sha256": SELF, "prereg_sha256": PREREG,
    "r8a4_runner_sha256": R8A4_RUNNER, "r8a4_verifier_sha256": R8A4_VERIFIER,
    "r8a4_prereg_sha256": R8A4_PREREG, "r8a4_lock_sha256": R8A4_LOCK,
    "r8a4_audit_sha256": R8A4_AUDIT, "r8a4_diagnosis_sha256": R8A4_DIAGNOSIS,
    "historical_contract_sha256": HISTORICAL_RUNNER, "physical_runner_sha256": PHYSICAL_RUNNER,
    "physical_verifier_sha256": PHYSICAL_VERIFIER,
}
IDENTITY_KEYS = {"native_raw", "native_argv", "orig_argv", "argv", "sys_executable", "sys_prefix", "base_executable", "base_prefix", "isolated", "dont_write_bytecode", "entry_name", "entry_spec_is_none", "entry_package", "entry_file", "direct_entry", "python_sha256", "pyvenv_sha256"}

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

def live_invocation() -> dict:
    kernel = C.WinDLL("kernel32", use_last_error=True)
    get = kernel.GetCommandLineW
    get.argtypes = ()
    get.restype = C.c_wchar_p
    raw = get()
    return {"native_raw": raw, "native_argv": parse_commandline(raw), "orig_argv": list(sys.orig_argv), "argv": list(sys.argv), "sys_executable": sys.executable, "sys_prefix": sys.prefix, "base_executable": getattr(sys, "_base_executable", None), "base_prefix": sys.base_prefix, "isolated": sys.flags.isolated, "dont_write_bytecode": sys.dont_write_bytecode, "entry_name": __name__, "entry_spec_is_none": __spec__ is None, "entry_package": __package__, "entry_file": str(SELF), "direct_entry": __name__ == "__main__" and __spec__ is None and __package__ in (None, "") and Path(__file__).resolve() == SELF, "python_sha256": sha256(VENV_PYTHON), "pyvenv_sha256": sha256(PYVENV)}

def identity_valid(row: dict, native: list[str], argv: list[str], script: Path) -> bool:
    try:
        raw_ok = parse_commandline(row["native_raw"]) == native
    except Exception:
        return False
    direct = row.get("entry_name") == "__main__" and row.get("entry_spec_is_none") is True and row.get("entry_package") in (None, "") and same_path(row.get("entry_file"), str(script))
    return set(row) == IDENTITY_KEYS and raw_ok and row["native_argv"] == row["orig_argv"] == native and row["argv"] == argv and same_path(row["sys_executable"], str(VENV_PYTHON.resolve())) and same_path(row["sys_prefix"], str(VENV.resolve())) and same_path(row["base_executable"], str(ALIAS)) and same_path(row["base_prefix"], str(BASE_PREFIX)) and row["isolated"] == 1 and row["dont_write_bytecode"] is True and row["python_sha256"] == "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14" and row["pyvenv_sha256"] == "9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe" and row["direct_entry"] is True and direct

def identity_mutations(row: dict, native: list[str], argv: list[str], script: Path) -> bool:
    cases = []
    for key, value in (("direct_entry", False), ("entry_name", "imported"), ("entry_spec_is_none", False), ("entry_package", "pkg"), ("entry_file", str(script) + ".wrong"), ("sys_executable", str(ALIAS)), ("isolated", 0), ("dont_write_bytecode", False)):
        changed = copy.deepcopy(row); changed[key] = value; cases.append(changed)
    for key in ("argv", "native_argv", "orig_argv"):
        changed = copy.deepcopy(row); changed[key] = [*changed[key], "extra"]; cases.append(changed)
    changed = copy.deepcopy(row); changed["native_raw"] = 'python -c "import x"'; cases.append(changed)
    return len(cases) == 12 and all(not identity_valid(case, native, argv, script) for case in cases)

def historical_valid() -> bool:
    return sha256(HISTORICAL_RUNNER) == "552a7f08f83f2ba2ce3da29581029dfdd79e86fbb75faeb71356965073228f15" and sha256(PHYSICAL_RUNNER) == "01fa21266137335494de2d21adba11f45fe83ff95f660d90cef7acc389c1cb04" and sha256(PHYSICAL_VERIFIER) == "18b64765469e38c5211d28afe586e0a559e97f6e2110f09f54c4f58d9c38dd88" and historical_contract.r8p8_pass() and historical_contract.r7d_contract() and historical_contract.exact_failure(historical_contract.R7D1_FAILURE_ROOT, historical_contract.R7D1_FAILURE, "88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0", "ph1_intel_execution_r7c2_failure") and historical_contract.exact_failure(historical_contract.R8P6_FAILURE_ROOT, historical_contract.R8P6_FAILURE, "03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372", "ph1_intel_execution_r8p6_failure")

def lock_contract() -> tuple[bool, dict]:
    observed = {key: sha256(path) for key, path in CHAIN.items()}
    lock = json.loads(LOCK.read_text())
    valid = set(lock) == {"kind", "execution_open", "audit_token", "one_attempt", *observed} and lock.get("kind") == "ph1_intel_execution_r8a5_lock" and lock.get("execution_open") is True and lock.get("audit_token") == ACK and lock.get("one_attempt") is True and all(lock.get(key) == value for key, value in observed.items()) and observed.get("r8a4_audit_sha256") == "4dc0c0a1f3e411f78f81ef667baf6e00e94f73204f0ed9b3d1794b0f300a7438" and observed.get("r8a4_diagnosis_sha256") == "d88ff5fd76e11757d7d53acf7279bc52897b9b74169414daee3ce18cd8bc6b21"
    return valid, observed

def extension(result: dict, observed: dict) -> bool:
    row = result.get("authorization", {}).get("r8a5_authorization", {})
    module_keys = {"path", "sha256", "required_attrs"}
    modules = row.get("explicit_modules", {})
    predicates = row.get("predicate_results", {})
    return set(row) == {"kind", "lock_sha256", "observed", "audit_token", "invocation", "explicit_modules", "predicate_results", "ast_no_ancestry_chains", "one_attempt"} and row.get("kind") == "ph1_intel_execution_r8a5_authorization" and row.get("lock_sha256") == sha256(LOCK) and row.get("observed") == observed and row.get("audit_token") == ACK and row.get("ast_no_ancestry_chains") is True and row.get("one_attempt") is True and set(modules) == {"historical", "physical"} and all(set(value) == module_keys for value in modules.values()) and same_path(modules["historical"]["path"], str(HISTORICAL_RUNNER.resolve())) and modules["historical"]["sha256"] == sha256(HISTORICAL_RUNNER) and same_path(modules["physical"]["path"], str(PHYSICAL_RUNNER.resolve())) and modules["physical"]["sha256"] == sha256(PHYSICAL_RUNNER) and predicates == {"r8p8_pass": True, "r7d_contract": True, "r7d1_failure": True, "r8p6_failure": True} and identity_valid(row.get("invocation", {}), RUN_NATIVE, RUN_ARGV, RUNNER) and identity_mutations(row["invocation"], RUN_NATIVE, RUN_ARGV, RUNNER)

def bundle() -> tuple[bool, dict | None]:
    try:
        result_path, manifest_path, commit_path = (OUT / name for name in ("result.json", "manifest.json", "commit.json"))
        result_bytes = result_path.read_bytes(); result = json.loads(result_bytes); manifest = json.loads(manifest_path.read_text()); commit = json.loads(commit_path.read_text())
        file_row = {"name": "result.json", "bytes": len(result_bytes), "sha256": hashlib.sha256(result_bytes).hexdigest()}
        valid = result.get("kind") == "ph1_intel_execution_r7a" and manifest == {"kind": "ph1_intel_execution_r7a_manifest", "files": [file_row]} and commit == {"kind": "ph1_intel_execution_r7a_commit", "manifest_sha256": hashlib.sha256(canonical(manifest)).hexdigest(), "result_sha256": file_row["sha256"]} and {path.name for path in OUT.iterdir()} == {"result.json", "manifest.json", "commit.json"} and sum(path.stat().st_size for path in OUT.iterdir()) <= 16 * 2**20
        return valid, result
    except Exception:
        return False, None

def exact_tree(root: Path) -> tuple[bool, Path | None]:
    if not root.is_dir():
        return False, None
    entries = sorted(root.rglob("*")); directories = [p for p in entries if p.is_dir()]; files = [p for p in entries if p.is_file()]
    direct_dirs = [p for p in root.iterdir() if p.is_dir()]; root_files = [p for p in root.iterdir() if p.is_file()]
    valid = len(entries) == 2 and len(directories) == len(direct_dirs) == 1 and not root_files and len(files) == 1 and files[0].parent == directories[0] and files[0].name == "failure.json" and not any("inprogress" in p.name.casefold() for p in entries)
    return valid, files[0] if valid else None

def reconstruct_backend(path: Path) -> dict:
    row = json.loads(path.read_text())
    normal = {"kind", "status", "error", "traceback", "device_opened", "backend_evidence", "secondary_resource_sample", "disposition"}
    oversize = {"kind", "status", "error", "device_opened", "oversized_temp_bytes", "oversized_temp_digest", "disposition"}
    valid = set(row) in (normal, oversize) and row.get("kind") == "ph1_intel_execution_r7a_failure" and row.get("status") == "valid_negative_failure" and isinstance(row.get("error"), str) and row.get("device_opened") is True and row.get("disposition") in ("attempt_archived_create_new", "oversized_temp_quarantined_not_retained_failure_bundle") and path.stat().st_size <= 16 * 2**20
    relative = str(path.relative_to(R)) if path.is_relative_to(R) else str(path)
    return {"relative_path": relative, "failure_sha256": sha256(path), "failure_bytes": path.stat().st_size, "bundle_files": [{"name": "failure.json", "bytes": path.stat().st_size, "sha256": sha256(path)}], "bundle_bytes": path.stat().st_size, "bundle_file_count": 1, "kind": row.get("kind"), "status": row.get("status"), "disposition": row.get("disposition"), "device_opened": row.get("device_opened"), "valid": valid}

def correlated(wrapper: dict, inherited: dict) -> bool:
    keys = {"kind", "status", "terminal_type", "stage", "error", "device_opened", "delegated_return", "inherited_failure_count", "inherited", "correlation_valid", "disposition"}
    return set(wrapper) == keys and wrapper.get("kind") == "ph1_intel_execution_r8a5_failure" and wrapper.get("status") == "correlated_delegated_negative" and wrapper.get("terminal_type") == "delegated_failure" and wrapper.get("stage") == "delegated_return" and wrapper.get("error") == "delegated_execution_nonzero" and isinstance(wrapper.get("delegated_return"), int) and not isinstance(wrapper.get("delegated_return"), bool) and wrapper["delegated_return"] != 0 and wrapper.get("device_opened") is True and wrapper.get("inherited_failure_count") == 1 and wrapper.get("correlation_valid") is True and wrapper.get("disposition") == "atomic_bounded_correlated_summary" and inherited.get("valid") is True and inherited.get("device_opened") is True and wrapper.get("inherited") == inherited

def failure_terminal() -> str:
    wrapper_ok, wrapper_path = exact_tree(FAILED)
    backend_ok, backend_path = exact_tree(BACKEND_FAILED)
    if not wrapper_ok or OUT.exists() or QUARANTINE.exists() or BACKEND_QUARANTINE.exists():
        return "invalid"
    try:
        wrapper = json.loads(wrapper_path.read_text())
    except Exception:
        return "invalid"
    if wrapper.get("terminal_type") == "early_outer_failure":
        exact = set(wrapper) == {"kind", "status", "terminal_type", "stage", "error", "traceback", "device_opened", "delegated_return", "inherited_failure_count", "inherited", "disposition"} and wrapper.get("kind") == "ph1_intel_execution_r8a5_failure" and wrapper.get("status") == "infrastructure_negative" and wrapper.get("stage") in ("predevice_resolution_or_authorization", "delegated_outer_boundary") and wrapper.get("device_opened") is False and wrapper.get("delegated_return") is None and wrapper.get("inherited_failure_count") == 0 and wrapper.get("inherited") is None and wrapper.get("disposition") == "atomic_bounded_outer_failure"
        return "early_invalid" if exact and not BACKEND_FAILED.exists() else "invalid"
    if not backend_ok:
        return "invalid"
    try:
        inherited = reconstruct_backend(backend_path)
    except Exception:
        return "invalid"
    return "correlated_device_negative" if correlated(wrapper, inherited) else "invalid"

def topology() -> bool:
    observed = {path.resolve() for path in FAMILY_PARENT.glob(FAMILY_PREFIX + "*")}
    allowed = {LOCK.resolve()} | {path.resolve() for path in (OUT, FAILED, BACKEND_FAILED, QUARANTINE, BACKEND_QUARANTINE, VERIFY_RESULT) if path.exists()}
    return observed == allowed and not list(FAMILY_PARENT.glob(FAMILY_PREFIX + "*.inprogress*"))

def production_terminal(committed: dict | None = None, numerical: dict | None = None, authorization: bool = True, bundle_ok: bool = True) -> str:
    if not topology():
        return "invalid"
    if OUT.exists():
        if any(path.exists() for path in (FAILED, BACKEND_FAILED, QUARANTINE, BACKEND_QUARANTINE)) or committed is None:
            return "invalid"
        return adjudication_contract.adjudicate_committed(committed, numerical or {}, authorization, bundle_ok)
    return failure_terminal()

def mutation_harness() -> dict:
    global FAMILY_PARENT, FAMILY_PREFIX, LOCK, OUT, FAILED, BACKEND_FAILED, QUARANTINE, BACKEND_QUARANTINE, VERIFY_RESULT
    saved = (FAMILY_PARENT, FAMILY_PREFIX, LOCK, OUT, FAILED, BACKEND_FAILED, QUARANTINE, BACKEND_QUARANTINE, VERIFY_RESULT)
    results = {}
    with tempfile.TemporaryDirectory(prefix="r8a5_terminal_") as directory:
        root = Path(directory); FAMILY_PARENT = root; FAMILY_PREFIX = "case"; LOCK = root / "case_lock.json"; OUT = root / "case_out"; FAILED = root / "case_failed"; BACKEND_FAILED = root / "case_backend"; QUARANTINE = root / "case_quarantine"; BACKEND_QUARANTINE = root / "case_backend_quarantine"; VERIFY_RESULT = root / "case_verify.json"; LOCK.write_text("{}")
        def clear() -> None:
            for path in (OUT, FAILED, BACKEND_FAILED, QUARANTINE, BACKEND_QUARANTINE, VERIFY_RESULT):
                if path.exists(): shutil.rmtree(path) if path.is_dir() else path.unlink()
            for path in root.glob("case*.inprogress*"):
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        def failure() -> tuple[Path, Path, dict, dict]:
            backend = BACKEND_FAILED / "attempt_backend"; backend.mkdir(parents=True)
            backend_row = {"kind": "ph1_intel_execution_r7a_failure", "status": "valid_negative_failure", "error": "device", "traceback": "x", "device_opened": True, "backend_evidence": None, "secondary_resource_sample": {"available": 1, "telemetry_error": None}, "disposition": "attempt_archived_create_new"}
            backend_path = backend / "failure.json"; backend_path.write_bytes(canonical(backend_row)); inherited = reconstruct_backend(backend_path)
            wrapper = FAILED / "attempt_wrapper"; wrapper.mkdir(parents=True)
            wrapper_row = {"kind": "ph1_intel_execution_r8a5_failure", "status": "correlated_delegated_negative", "terminal_type": "delegated_failure", "stage": "delegated_return", "error": "delegated_execution_nonzero", "device_opened": True, "delegated_return": 3, "inherited_failure_count": 1, "inherited": inherited, "correlation_valid": True, "disposition": "atomic_bounded_correlated_summary"}
            wrapper_path = wrapper / "failure.json"; wrapper_path.write_bytes(canonical(wrapper_row)); return backend_path, wrapper_path, backend_row, wrapper_row
        def record(name: str, expected: str, committed=None, numerical=None, authorization=True, bundle_ok=True) -> None:
            results[name] = production_terminal(committed, numerical, authorization, bundle_ok) == expected
        try:
            bp, wp, br, wr = failure(); record("baseline", "correlated_device_negative")
            clear(); bp, wp, br, wr = failure(); shutil.rmtree(FAILED); record("missing_wrapper", "invalid")
            clear(); bp, wp, br, wr = failure(); extra = FAILED / "attempt_extra"; extra.mkdir(); (extra / "failure.json").write_bytes(wp.read_bytes()); record("multiple_wrapper", "invalid")
            clear(); bp, wp, br, wr = failure(); (wp.parent / "extra.bin").write_bytes(b"x"); record("wrapper_extra", "invalid")
            clear(); bp, wp, br, wr = failure(); (FAILED / "orphan.bin").write_bytes(b"x"); record("wrapper_orphan", "invalid")
            clear(); bp, wp, br, wr = failure(); (wp.parent / "extra").mkdir(); record("wrapper_dir", "invalid")
            clear(); bp, wp, br, wr = failure(); (BACKEND_FAILED / "orphan.bin").write_bytes(b"x"); record("backend_root_extra", "invalid")
            clear(); bp, wp, br, wr = failure(); (bp.parent / "extra.bin").write_bytes(b"x"); record("backend_attempt_extra", "invalid")
            clear(); bp, wp, br, wr = failure(); (root / "case_temp.inprogress").write_bytes(b"x"); record("inprogress", "invalid")
            clear(); bp, wp, br, wr = failure(); QUARANTINE.mkdir(); record("quarantine", "invalid")
            clear(); bp, wp, br, wr = failure(); shutil.rmtree(BACKEND_FAILED); record("missing_backend", "invalid")
            clear(); bp, wp, br, wr = failure(); extra = BACKEND_FAILED / "attempt_extra"; extra.mkdir(); (extra / "failure.json").write_bytes(bp.read_bytes()); record("multiple_backend", "invalid")
            for name, key, value in (("wrong_kind", "kind", "wrong"), ("wrong_status", "status", "wrong"), ("wrong_disposition", "disposition", "wrong"), ("device_false", "device_opened", False), ("device_type", "device_opened", "true")):
                clear(); bp, wp, br, wr = failure(); br[key] = value; bp.write_bytes(canonical(br)); record(name, "invalid")
            for name, key, value in (("wrong_stage", "stage", "wrong"), ("wrong_correlation", "correlation_valid", False), ("wrong_hash", "inherited", None)):
                clear(); bp, wp, br, wr = failure(); wr[key] = ({**wr["inherited"], "failure_sha256": "0" * 64} if name == "wrong_hash" else value); wp.write_bytes(canonical(wr)); record(name, "invalid")
            clear(); bp, wp, br, wr = failure(); shutil.rmtree(FAILED); record("bare_nonzero", "invalid")
            clear(); record("success_no_commit", "invalid")
            clear(); bp, wp, br, wr = failure(); OUT.mkdir(); record("mixed_commit_failure", "invalid")
            clear(); FAILED.mkdir(); attempt = FAILED / "attempt_early"; attempt.mkdir(); early = {"kind": "ph1_intel_execution_r8a5_failure", "status": "infrastructure_negative", "terminal_type": "early_outer_failure", "stage": "predevice_resolution_or_authorization", "error": "x", "traceback": "x", "device_opened": False, "delegated_return": None, "inherited_failure_count": 0, "inherited": None, "disposition": "atomic_bounded_outer_failure"}; (attempt / "failure.json").write_bytes(canonical(early)); record("early", "early_invalid")
            gates = {key: True for key in adjudication_contract.GATES}; numerical = {key: True for key in adjudication_contract.NUMERICAL}; positive = {"positive": True, "status": "intel_execution_positive", "gates": gates}
            clear(); OUT.mkdir(); record("committed_positive", "positive", positive, numerical)
            clear(); OUT.mkdir(); negative = copy.deepcopy(positive); negative["positive"] = False; negative["status"] = "intel_execution_negative"; negative["gates"]["stages"] = False; changed = dict(numerical); changed["positive_schema"] = changed["runner_gates"] = changed["oracle_outputs"] = False; record("stages_negative", "allowed_device_negative", negative, changed)
            clear(); OUT.mkdir(); negative = copy.deepcopy(positive); negative["positive"] = False; negative["status"] = "intel_execution_negative"; negative["gates"]["counters"] = False; changed = dict(numerical); changed["positive_schema"] = changed["runner_gates"] = changed["counters"] = False; record("counters_negative", "allowed_device_negative", negative, changed)
            for name, gate in (("precheck_negative", "controls"), ("protocol_negative", "ledger_order"), ("lifecycle_negative", "release"), ("resource_negative", "resources")):
                clear(); OUT.mkdir(); negative = copy.deepcopy(positive); negative["positive"] = False; negative["status"] = "intel_execution_negative"; negative["gates"][gate] = False; changed = dict(numerical); changed["positive_schema"] = changed["runner_gates"] = False; record(name, "invalid", negative, changed)
        finally:
            FAMILY_PARENT, FAMILY_PREFIX, LOCK, OUT, FAILED, BACKEND_FAILED, QUARANTINE, BACKEND_QUARANTINE, VERIFY_RESULT = saved
    return results

def write_result(row: dict) -> None:
    if VERIFY_RESULT.exists():
        raise FileExistsError(VERIFY_RESULT)
    temp = R / f"{VERIFY_RESULT.name}.{uuid.uuid4().hex}.inprogress"
    try:
        with temp.open("xb") as handle:
            handle.write(canonical(row)); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, VERIFY_RESULT); temp.unlink()
    finally:
        if temp.exists(): temp.unlink()

def main() -> int:
    ident = live_invocation(); lock_ok, observed = lock_contract(); matrix = mutation_harness()
    checks = {"live_invocation": identity_valid(ident, VERIFY_NATIVE, VERIFY_ARGV, SELF), "live_invocation_mutations": identity_mutations(ident, VERIFY_NATIVE, VERIFY_ARGV, SELF), "lock": lock_ok, "historical": historical_valid(), "topology": topology(), "committed_adjudicator_mutations": adjudication_contract.adjudicator_mutations(), "production_matrix": len(matrix) == 31 and all(matrix.values())}
    bundle_ok, result = bundle(); state = "invalid"
    if bundle_ok and result is not None and not any(path.exists() for path in (FAILED, BACKEND_FAILED, QUARANTINE, BACKEND_QUARANTINE)):
        authorization = extension(result, observed); checks["authorization"] = authorization; numerical = {}
        if authorization:
            if sha256(PHYSICAL_VERIFIER) != "18b64765469e38c5211d28afe586e0a559e97f6e2110f09f54c4f58d9c38dd88":
                raise RuntimeError("numerical_verifier_hash")
            import verify_het_next_l0_ph1_intel_execution_r7a as numerical_verifier
            try: numerical = numerical_verifier.verify_dict(result)
            except Exception: numerical = {}
        state = production_terminal(result, numerical, authorization, bundle_ok); checks["terminal_contract"] = state in ("positive", "allowed_device_negative"); checks.update({"numerical:" + key: value for key, value in numerical.items()})
    else:
        state = production_terminal(); checks["terminal_contract"] = state == "correlated_device_negative"; checks["bundle_absent"] = not OUT.exists()
    valid = state in ("positive", "allowed_device_negative", "correlated_device_negative")
    passed = state == "positive" and all(checks.values())
    row = {"kind": "ph1_intel_execution_r8a5_independent_verification", "terminal_state": state, "terminal_valid": valid, "checks": checks, "mutation_matrix": matrix, "pass": passed, "passed": sum(value is True for value in checks.values()), "total": len(checks), "claim": "one real expert/input Intel correctness component only"}
    write_result(row); print(json.dumps(row, indent=2)); return 0 if passed else 3

if __name__ == "__main__":
    raise SystemExit(main())
