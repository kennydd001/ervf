#!/usr/bin/env python3
"""R8V1-R1 verifier-only repair: exact prior protocol-negative, positive bundle."""
from __future__ import annotations

import copy
import ctypes as C
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))
import verify_het_next_l0_ph1_intel_execution_r8v1 as frozen_verifier

SELF = Path(__file__).resolve()
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8V1R1_PREREGISTRATION_2026-08-14.md"
LOCK = R / "het_next_l0_ph1_intel_execution_r8v1r1_lock.json"
OUTPUT = R / "het_next_l0_ph1_intel_execution_r8v1r1_independent_verification.json"
OLD_PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8V1_PREREGISTRATION_2026-08-14.md"
OLD_LOCK = R / "het_next_l0_ph1_intel_execution_r8v1_lock.json"
OLD_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8V1_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
PRIOR_VERIFIER = R / "het_next_l0_ph1_intel_execution_r8a5_independent_verification.json"
ACK = "PENDING_INDEPENDENT_R8V1R1_SOURCE_AUDIT"
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / "Scripts/python.exe"
PYVENV = VENV / "pyvenv.cfg"
ALIAS = Path(r"C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe")
BASE_PREFIX = Path(r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0")
EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SELF), "--ack", ACK]
EXPECTED_ARGV = [str(SELF), "--ack", ACK]
OLD_AUDIT_SHA = "0863b759eaa5a2fa6eaca0a7c24d3dcec5dfe5454eee9c66ab2699a36719f587"
PRIOR_SHA = "d6b630658c59e1c6913ba099bb8d617fe1b451e14e31ee38b68d351fb9fde917"
PRIOR_CHECKS = {
    "authorization", "committed_adjudicator_mutations", "historical", "live_invocation", "live_invocation_mutations", "lock",
    "numerical:allocations", "numerical:args", "numerical:authorization", "numerical:compile_package", "numerical:controls", "numerical:counters", "numerical:extensions", "numerical:forbidden", "numerical:identity", "numerical:initialization", "numerical:launch_finish_read", "numerical:ledger_order", "numerical:oracle_outputs", "numerical:ownership", "numerical:positive_schema", "numerical:records_input_lut", "numerical:release", "numerical:resources", "numerical:runner_gates", "numerical:writes",
    "production_matrix", "terminal_contract", "topology",
}
PRIOR_FALSE = {"terminal_contract", "topology"}
PRIOR_MATRIX = {"backend_attempt_extra", "backend_root_extra", "bare_nonzero", "baseline", "committed_positive", "counters_negative", "device_false", "device_type", "early", "inprogress", "lifecycle_negative", "missing_backend", "missing_wrapper", "mixed_commit_failure", "multiple_backend", "multiple_wrapper", "precheck_negative", "protocol_negative", "quarantine", "resource_negative", "stages_negative", "success_no_commit", "wrapper_dir", "wrapper_extra", "wrapper_orphan", "wrong_correlation", "wrong_disposition", "wrong_hash", "wrong_kind", "wrong_stage", "wrong_status"}
OLD_NAMES = {OLD_PREREG.name, OLD_LOCK.name, OLD_AUDIT.name}
CURRENT_NAMES = {PREREG.name, LOCK.name}
EXPECTED_NAMES = frozen_verifier.R8A5_NAMES | OLD_NAMES | CURRENT_NAMES
CHAIN = {
    "verifier_sha256": SELF, "prereg_sha256": PREREG,
    "r8v1_verifier_sha256": Path(frozen_verifier.__file__), "r8v1_prereg_sha256": OLD_PREREG,
    "r8v1_lock_sha256": OLD_LOCK, "r8v1_audit_sha256": OLD_AUDIT,
    "r8a5_result_sha256": frozen_verifier.RESULT, "r8a5_manifest_sha256": frozen_verifier.MANIFEST,
    "r8a5_commit_sha256": frozen_verifier.COMMIT, "failed_verifier_sha256": PRIOR_VERIFIER,
    "r8a5_topology_diagnosis_sha256": frozen_verifier.TOPOLOGY_DIAGNOSIS,
    "r8a5_postrun_audit_sha256": frozen_verifier.POSTRUN_AUDIT,
    "r8a5_postrun_json_sha256": frozen_verifier.POSTRUN_JSON,
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

def same_path(a: object, b: object) -> bool:
    return isinstance(a, str) and isinstance(b, str) and a.casefold() == b.casefold()

def parse_commandline(raw: str) -> list[str]:
    kernel = C.WinDLL("kernel32", use_last_error=True); shell = C.WinDLL("shell32", use_last_error=True)
    parse = shell.CommandLineToArgvW; parse.argtypes = (C.c_wchar_p, C.POINTER(C.c_int)); parse.restype = C.POINTER(C.c_wchar_p)
    free = kernel.LocalFree; free.argtypes = (C.c_void_p,); free.restype = C.c_void_p
    count = C.c_int(); ptr = parse(raw, C.byref(count))
    if not ptr: raise C.WinError(C.get_last_error())
    try: return [ptr[index] for index in range(count.value)]
    finally:
        if free(C.cast(ptr, C.c_void_p)): raise C.WinError(C.get_last_error())

def invocation() -> dict:
    kernel = C.WinDLL("kernel32", use_last_error=True); get = kernel.GetCommandLineW; get.argtypes = (); get.restype = C.c_wchar_p; raw = get()
    return {"native_raw": raw, "native_argv": parse_commandline(raw), "orig_argv": list(sys.orig_argv), "argv": list(sys.argv), "sys_executable": sys.executable, "sys_prefix": sys.prefix, "base_executable": getattr(sys, "_base_executable", None), "base_prefix": sys.base_prefix, "isolated": sys.flags.isolated, "dont_write_bytecode": sys.dont_write_bytecode, "python_sha256": sha256(VENV_PYTHON), "pyvenv_sha256": sha256(PYVENV), "direct_entry": __name__ == "__main__" and __spec__ is None and __package__ in (None, "") and Path(__file__).resolve() == SELF}

def invocation_valid(row: dict) -> bool:
    keys = {"native_raw", "native_argv", "orig_argv", "argv", "sys_executable", "sys_prefix", "base_executable", "base_prefix", "isolated", "dont_write_bytecode", "python_sha256", "pyvenv_sha256", "direct_entry"}
    try: raw_ok = parse_commandline(row["native_raw"]) == EXPECTED_NATIVE
    except Exception: return False
    return set(row) == keys and raw_ok and row["native_argv"] == row["orig_argv"] == EXPECTED_NATIVE and row["argv"] == EXPECTED_ARGV and same_path(row["sys_executable"], str(VENV_PYTHON.resolve())) and same_path(row["sys_prefix"], str(VENV.resolve())) and same_path(row["base_executable"], str(ALIAS)) and same_path(row["base_prefix"], str(BASE_PREFIX)) and row["isolated"] == 1 and row["dont_write_bytecode"] is True and row["python_sha256"] == "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14" and row["pyvenv_sha256"] == "9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe" and row["direct_entry"] is True

def prior_verifier_contract(row: dict) -> bool:
    if sha256(PRIOR_VERIFIER) != PRIOR_SHA: return False
    checks = row.get("checks", {}); matrix = row.get("mutation_matrix", {})
    return set(row) == {"checks", "claim", "kind", "mutation_matrix", "pass", "passed", "terminal_state", "terminal_valid", "total"} and row.get("kind") == "ph1_intel_execution_r8a5_independent_verification" and row.get("claim") == "one real expert/input Intel correctness component only" and row.get("pass") is False and row.get("passed") == 27 and row.get("total") == 29 and row.get("terminal_state") == "invalid" and row.get("terminal_valid") is False and set(checks) == PRIOR_CHECKS and {key for key, value in checks.items() if value is False} == PRIOR_FALSE and all(isinstance(value, bool) for value in checks.values()) and all(value is True for key, value in checks.items() if key not in PRIOR_FALSE) and set(matrix) == PRIOR_MATRIX and len(matrix) == 31 and all(value is True for value in matrix.values())

def prior_verifier_mutations(row: dict) -> dict:
    cases = {}
    def add(name: str, changed: dict) -> None: cases[name] = not prior_verifier_contract(changed)
    for key, value in (("kind", "wrong"), ("claim", "wrong"), ("pass", True), ("passed", 28), ("total", 30), ("terminal_state", "positive"), ("terminal_valid", True)):
        changed = copy.deepcopy(row); changed[key] = value; add(key, changed)
    changed = copy.deepcopy(row); changed["model_forward"] = False; add("extra_model_field", changed)
    changed = copy.deepcopy(row); del changed["claim"]; add("missing_top_field", changed)
    changed = copy.deepcopy(row); del changed["checks"]["authorization"]; add("missing_check", changed)
    changed = copy.deepcopy(row); changed["checks"]["extra"] = True; add("extra_check", changed)
    changed = copy.deepcopy(row); changed["checks"]["authorization"] = False; add("third_false_check", changed)
    changed = copy.deepcopy(row); changed["checks"]["topology"] = True; add("missing_false_check", changed)
    changed = copy.deepcopy(row); changed["checks"]["authorization"] = 1; add("non_boolean_check", changed)
    changed = copy.deepcopy(row); del changed["mutation_matrix"]["baseline"]; add("missing_matrix", changed)
    changed = copy.deepcopy(row); changed["mutation_matrix"]["extra"] = True; add("extra_matrix", changed)
    changed = copy.deepcopy(row); changed["mutation_matrix"]["baseline"] = False; add("false_matrix", changed)
    return cases

def literal_names() -> set[str]:
    prefixes = ("het_next_l0_ph1_intel_execution_r8a5", "het_next_l0_ph1_intel_execution_r8v1")
    return {path.name for path in R.iterdir() if any(path.name.casefold().startswith(prefix.casefold()) for prefix in prefixes)}

def names_valid(names: set[str]) -> bool:
    return names == EXPECTED_NAMES and len({name.casefold() for name in names}) == len(names)

def topology() -> tuple[bool, dict]:
    literal = literal_names(); r8a5_glob = {path.name for path in R.glob("het_next_l0_ph1_intel_execution_r8a5*")}; r8v1_glob = {path.name for path in R.glob("het_next_l0_ph1_intel_execution_r8v1*")}
    expected_r8v1_glob = OLD_NAMES | CURRENT_NAMES
    invalid_roots = (R / "het_next_l0_ph1_intel_execution_r8v1r1_failed_attempts", R / "het_next_l0_ph1_intel_execution_r8v1r1_quarantine", OUTPUT)
    valid = names_valid(literal) and r8a5_glob == frozen_verifier.R8A5_NAMES and r8v1_glob == expected_r8v1_glob and all(not path.exists() for path in invalid_roots) and not any("inprogress" in name.casefold() for name in literal)
    return valid, {"expected": sorted(EXPECTED_NAMES), "literal": sorted(literal), "r8a5_glob": sorted(r8a5_glob), "r8v1_glob": sorted(r8v1_glob), "casefold_unique": len({name.casefold() for name in literal}) == len(literal)}

def topology_mutations() -> dict:
    cases = {}
    for name in EXPECTED_NAMES: cases[f"missing:{name}"] = EXPECTED_NAMES - {name}
    cases.update({"extra_upper": EXPECTED_NAMES | {"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8V1R1_EXTRA.MD"}, "extra_lower": EXPECTED_NAMES | {"het_next_l0_ph1_intel_execution_r8v1r1_extra.bin"}, "orphan": EXPECTED_NAMES | {"het_next_l0_ph1_intel_execution_r8v1_orphan"}, "temp": EXPECTED_NAMES | {"het_next_l0_ph1_intel_execution_r8v1r1.x.inprogress"}, "failure": EXPECTED_NAMES | {"het_next_l0_ph1_intel_execution_r8v1r1_failed_attempts"}, "quarantine": EXPECTED_NAMES | {"het_next_l0_ph1_intel_execution_r8v1r1_quarantine"}})
    victim = OLD_PREREG.name; cases["case_changed"] = (EXPECTED_NAMES - {victim}) | {victim.lower()}
    results = {name: not names_valid(value) for name, value in cases.items()}
    collision = [*EXPECTED_NAMES, victim.lower()]; results["casefold_collision"] = len({name.casefold() for name in collision}) != len(collision)
    return results

def lock_contract() -> tuple[bool, dict]:
    observed = {key: sha256(path) for key, path in CHAIN.items()}; lock = json.loads(LOCK.read_text())
    valid = set(lock) == {"kind", "execution_open", "audit_token", "one_attempt", *observed} and lock.get("kind") == "ph1_intel_execution_r8v1r1_lock" and lock.get("execution_open") is False and lock.get("audit_token") == ACK and lock.get("one_attempt") is True and all(lock.get(key) == value for key, value in observed.items()) and observed.get("r8v1_audit_sha256") == OLD_AUDIT_SHA
    return valid, observed

def verify() -> tuple[dict, dict]:
    lock_ok, observed = lock_contract(); topology_ok, topology_evidence = topology(); topology_cases = topology_mutations(); prior_row = json.loads(PRIOR_VERIFIER.read_text()); prior_cases = prior_verifier_mutations(prior_row)
    bundle_ok, result = frozen_verifier.bundle(); authorization_ok = bundle_ok and frozen_verifier.authorization_valid(result); direct = frozen_verifier.direct_physical(result) if bundle_ok else {}
    numerical = {}
    if authorization_ok and sha256(frozen_verifier.R7A_VERIFIER) == frozen_verifier.EXPECTED_HASHES[frozen_verifier.R7A_VERIFIER]:
        import verify_het_next_l0_ph1_intel_execution_r7a as numerical_verifier
        numerical = numerical_verifier.verify_dict(result)
    terminal_matrix = {}
    if sha256(frozen_verifier.R8A5_VERIFIER) == frozen_verifier.EXPECTED_HASHES[frozen_verifier.R8A5_VERIFIER]:
        import verify_het_next_l0_ph1_intel_execution_r8a5 as terminal_verifier
        terminal_matrix = terminal_verifier.mutation_harness()
    checks = {
        "lock_closed": lock_ok, "topology": topology_ok,
        "topology_mutations": len(topology_cases) == len(EXPECTED_NAMES) + 8 and all(topology_cases.values()),
        "prior_verifier_exact": prior_verifier_contract(prior_row),
        "prior_verifier_mutations": len(prior_cases) == 17 and all(prior_cases.values()),
        "bundle": bundle_ok, "authorization": authorization_ok,
        "direct_physical": len(direct) == 15 and all(direct.values()),
        "numerical_exact20": set(numerical) == frozen_verifier.NUMERICAL and len(numerical) == 20 and all(numerical.values()),
        "terminal_matrix31": len(terminal_matrix) == 31 and all(terminal_matrix.values()),
    }
    evidence = {"observed_hashes": observed, "topology": topology_evidence, "topology_mutations": topology_cases, "prior_verifier_mutations": prior_cases, "direct_physical": direct, "numerical": numerical, "terminal_matrix": terminal_matrix}
    return checks, evidence

def write_positive(row: dict) -> None:
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    temp = R / f"{OUTPUT.name}.{uuid.uuid4().hex}.inprogress"
    try:
        with temp.open("xb") as handle: handle.write(canonical(row)); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, OUTPUT); temp.unlink()
    finally:
        if temp.exists(): temp.unlink()

def main() -> int:
    ident = invocation()
    if sys.argv != EXPECTED_ARGV or not invocation_valid(ident): return 3
    lock_ok, _ = lock_contract(); lock = json.loads(LOCK.read_text())
    if not lock_ok or lock.get("execution_open") is not True: return 3
    checks, evidence = verify()
    if not all(checks.values()): return 3
    row = {"kind": "ph1_intel_execution_r8v1r1_independent_verification", "prior_verifier_outcome": "verifier_protocol_negative", "bundle_adjudication": "positive", "terminal_state": "positive", "terminal_valid": True, "checks": checks, "evidence": evidence, "pass": True, "passed": len(checks), "total": len(checks), "model_forward": False, "compiler_opened": False, "opencl_opened": False, "device_opened": False, "claim": "one real expert/input Intel correctness component only"}
    write_positive(row); print(json.dumps(row, indent=2)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
