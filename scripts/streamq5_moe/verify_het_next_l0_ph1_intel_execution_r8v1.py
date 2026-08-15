#!/usr/bin/env python3
"""R8V1 CPU-only verifier erratum for the immutable positive R8A5 bundle."""
from __future__ import annotations

import copy
import ctypes as C
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
SELF = Path(__file__).resolve()
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8V1_PREREGISTRATION_2026-08-14.md"
LOCK = R / "het_next_l0_ph1_intel_execution_r8v1_lock.json"
OUTPUT = R / "het_next_l0_ph1_intel_execution_r8v1_independent_verification.json"
R8A5_DIR = R / "het_next_l0_ph1_intel_execution_r8a5"
RESULT = R8A5_DIR / "result.json"
MANIFEST = R8A5_DIR / "manifest.json"
COMMIT = R8A5_DIR / "commit.json"
FAILED_VERIFIER = R / "het_next_l0_ph1_intel_execution_r8a5_independent_verification.json"
R8A5_LOCK = R / "het_next_l0_ph1_intel_execution_r8a5_lock.json"
R8A5_PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A5_PREREGISTRATION_2026-08-14.md"
R8A5_SOURCE_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A5_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
TOPOLOGY_DIAGNOSIS = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A5_VERIFIER_TOPOLOGY_FAILURE_DIAGNOSIS_2026-08-14.md"
POSTRUN_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A5_INDEPENDENT_POSTRUN_ADJUDICATION_AUDIT_2026-08-14.md"
POSTRUN_JSON = R / "het_next_l0_ph1_intel_execution_r8a5_postrun_independent_diagnosis.json"
R8A5_RUNNER = S / "run_het_next_l0_ph1_intel_execution_r8a5.py"
R8A5_VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r8a5.py"
R8A_RUNNER = S / "run_het_next_l0_ph1_intel_execution_r8a.py"
R7A_RUNNER = S / "run_het_next_l0_ph1_intel_execution_r7a.py"
R7A_VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r7a.py"
ACK = "PENDING_INDEPENDENT_R8V1_SOURCE_AUDIT"
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / "Scripts/python.exe"
PYVENV = VENV / "pyvenv.cfg"
ALIAS = Path(r"C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe")
BASE_PREFIX = Path(r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0")
EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SELF), "--ack", ACK]
EXPECTED_ARGV = [str(SELF), "--ack", ACK]
EXPECTED_HASHES = {
    RESULT: "9d1ac21f4fdd9657160e877f267369b5e831ff9f7a65e998f27895947c9cad50",
    MANIFEST: "2d13137f143ff183be3ffe89a3b85754cb2f35b52f92885580f49676e5fcfb7b",
    COMMIT: "07d9f03e8907a029d8bc31e40da6298de080b6bc0f0914769f8d52517b2dd965",
    FAILED_VERIFIER: "d6b630658c59e1c6913ba099bb8d617fe1b451e14e31ee38b68d351fb9fde917",
    TOPOLOGY_DIAGNOSIS: "e3be1fa3d05fe8a6437f0b3fbb047bc99ee83000c7c67dde725bfb9254715f1a",
    POSTRUN_AUDIT: "218ceb07f599bd7b7cad32c3da42373256f927c04b21f70a599c977411e4ae0b",
    POSTRUN_JSON: "01aba30e31db65d8b42c6dd047202391eb9a3da67fa434f944c8bbf1bf46978c",
    R8A5_LOCK: "13be47460512fe42a0a4dbe2995c2299e5bf02f75db85058b21296047cbe7979",
    R8A5_PREREG: "b7788e4185b29c8a6f194d0dbf96fc8a9e6b9bed78eba87e49d82b8d70c4b056",
    R8A5_SOURCE_AUDIT: "8a08de4833c90d59f87ea1a52c429e468df397b88abfca283c790fb0b42c29c6",
    R8A5_RUNNER: "1422fe70e2b0c33f19c1df969a40f7a7414b8a3734cc9914e3f687a5fcc25168",
    R8A5_VERIFIER: "75168d7502a141291f3b7459f779ae92439b7ffd32df667875fd73a365e62a66",
    R8A_RUNNER: "552a7f08f83f2ba2ce3da29581029dfdd79e86fbb75faeb71356965073228f15",
    R7A_RUNNER: "01fa21266137335494de2d21adba11f45fe83ff95f660d90cef7acc389c1cb04",
    R7A_VERIFIER: "18b64765469e38c5211d28afe586e0a559e97f6e2110f09f54c4f58d9c38dd88",
}
CHAIN = {"verifier_sha256": SELF, "prereg_sha256": PREREG, **{f"input_{index:02d}_sha256": path for index, path in enumerate(EXPECTED_HASHES)}}
R8A5_NAMES = {
    R8A5_DIR.name, FAILED_VERIFIER.name, R8A5_LOCK.name, R8A5_PREREG.name,
    R8A5_SOURCE_AUDIT.name, TOPOLOGY_DIAGNOSIS.name, POSTRUN_AUDIT.name, POSTRUN_JSON.name,
}
R8V1_NAMES = {PREREG.name, LOCK.name}
FAMILY_PREFIXES = ("het_next_l0_ph1_intel_execution_r8a5", "het_next_l0_ph1_intel_execution_r8v1")
GATES = {"allocations", "args", "compile_identity", "controls", "counters", "extensions", "finish_reads", "forbidden_static_and_runtime", "identity", "initialization", "launch", "ledger_order", "ownership", "release", "resource_samples", "resources", "stages", "writes"}
NUMERICAL = {"positive_schema", "oracle_outputs", "records_input_lut", "controls", "authorization", "compile_package", "identity", "ledger_order", "ownership", "allocations", "writes", "initialization", "args", "launch_finish_read", "release", "extensions", "counters", "resources", "forbidden", "runner_gates"}
STAGE_HASHES = {"gate": "e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867", "up": "f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08", "silu": "a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8", "activation": "762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f", "down": "142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc"}
EXPECTED_OPS = {"resource_sample": 12, "identity": 1, "context_create": 1, "queue_create": 1, "program_create_binary": 1, "kernel_create": 4, "host_usm_allocate": 14, "cpu_direct_write": 5, "initialize": 9, "set_pointer_arg": 18, "enqueue": 4, "finish": 1, "cpu_direct_read": 9, "release": 21, "cleanup": 1}
SAMPLE_STAGES = ["backend_entry", "pre_launch:0", "post_launch:0", "pre_launch:1", "post_launch:1", "pre_launch:2", "post_launch:2", "pre_launch:3", "post_launch:3", "pre_finish", "post_finish", "post_cleanup"]

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

def literal_family_names() -> set[str]:
    return {path.name for path in R.iterdir() if any(path.name.casefold().startswith(prefix.casefold()) for prefix in FAMILY_PREFIXES)}

def expected_family_names() -> set[str]:
    return R8A5_NAMES | R8V1_NAMES

def names_valid(names: set[str]) -> bool:
    expected = expected_family_names()
    return names == expected and len({name.casefold() for name in names}) == len(names)

def topology() -> tuple[bool, dict]:
    literal = literal_family_names(); globbed = {path.name for path in R.glob("het_next_l0_ph1_intel_execution_r8a5*")}
    expected = expected_family_names(); failures_absent = not any((R / name).exists() for name in ("het_next_l0_ph1_intel_execution_r8a5_failed_attempts", "het_next_l0_ph1_intel_execution_r8a5_backend_failed_attempts", "het_next_l0_ph1_intel_execution_r8a5_quarantine", "het_next_l0_ph1_intel_execution_r8a5_backend_quarantine"))
    temp_absent = not any("inprogress" in name.casefold() for name in literal)
    evidence = {"expected": sorted(expected), "literal": sorted(literal), "windows_glob_observation": sorted(globbed), "glob_expected_r8a5": sorted(R8A5_NAMES), "casefold_unique": len({name.casefold() for name in literal}) == len(literal), "failure_quarantine_absent": failures_absent, "temp_absent": temp_absent}
    return names_valid(literal) and globbed == R8A5_NAMES and failures_absent and temp_absent and not OUTPUT.exists(), evidence

def topology_mutations() -> dict:
    base = expected_family_names(); cases = {}
    for name in sorted(base): cases[f"missing:{name}"] = base - {name}
    cases.update({"extra_upper": base | {"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A5_ORPHAN.MD"}, "extra_lower": base | {"het_next_l0_ph1_intel_execution_r8a5_orphan.bin"}, "orphan": base | {"het_next_l0_ph1_intel_execution_r8v1_orphan"}, "temporary": base | {"het_next_l0_ph1_intel_execution_r8v1.x.inprogress"}, "failure": base | {"het_next_l0_ph1_intel_execution_r8a5_failed_attempts"}, "quarantine": base | {"het_next_l0_ph1_intel_execution_r8a5_quarantine"}})
    victim = R8A5_PREREG.name; cases["case_changed"] = (base - {victim}) | {victim.lower()}
    collision_list = [*base, victim.lower()]
    rejected = {name: not names_valid(value) for name, value in cases.items()}
    rejected["casefold_collision"] = len({name.casefold() for name in collision_list}) != len(collision_list)
    return rejected

def bundle() -> tuple[bool, dict]:
    if {path.name for path in R8A5_DIR.iterdir()} != {"result.json", "manifest.json", "commit.json"}: return False, {}
    if any(sha256(path) != digest for path, digest in EXPECTED_HASHES.items()): return False, {}
    result_bytes = RESULT.read_bytes(); result = json.loads(result_bytes); manifest = json.loads(MANIFEST.read_text()); commit = json.loads(COMMIT.read_text())
    file_row = {"name": "result.json", "bytes": len(result_bytes), "sha256": EXPECTED_HASHES[RESULT]}
    valid = manifest == {"kind": "ph1_intel_execution_r7a_manifest", "files": [file_row]} and commit == {"kind": "ph1_intel_execution_r7a_commit", "manifest_sha256": EXPECTED_HASHES[MANIFEST], "result_sha256": EXPECTED_HASHES[RESULT]} and result.get("kind") == "ph1_intel_execution_r7a"
    return valid, result

def authorization_valid(result: dict) -> bool:
    authorization = result.get("authorization", {}); extension = authorization.get("r8a5_authorization", {})
    r8a5_lock = json.loads(R8A5_LOCK.read_text()); observed = extension.get("observed", {})
    lock_hashes = {key: value for key, value in r8a5_lock.items() if key.endswith("_sha256")}
    modules = extension.get("explicit_modules", {}); predicates = extension.get("predicate_results", {})
    invocation_row = extension.get("invocation", {})
    expected_run_native = [str(ALIAS), "-I", "-B", str(R8A5_RUNNER), "--ack", "PH1_INTEL_EXECUTION_R8A5_AFTER_R8P8_PASS_AND_EXPLICIT_BINDING_AUDIT_GO"]
    try: native_ok = parse_commandline(invocation_row["native_raw"]) == expected_run_native
    except Exception: native_ok = False
    invocation_keys = {"native_raw", "native_argv", "orig_argv", "argv", "sys_executable", "sys_prefix", "base_executable", "base_prefix", "isolated", "dont_write_bytecode", "entry_name", "entry_spec_is_none", "entry_package", "entry_file", "direct_entry", "python_sha256", "pyvenv_sha256"}
    historical_attrs = ["R7D1_FAILURE", "R7D1_FAILURE_ROOT", "R8P6_FAILURE", "R8P6_FAILURE_ROOT", "exact_failure", "physical", "r7d_contract", "r8p8_pass"]
    physical_attrs = ["FAILED", "OUT", "QUAR", "authorize", "base", "execute_authorized", "verify_bundle"]
    return set(extension) == {"kind", "lock_sha256", "observed", "audit_token", "invocation", "explicit_modules", "predicate_results", "ast_no_ancestry_chains", "one_attempt"} and extension.get("kind") == "ph1_intel_execution_r8a5_authorization" and extension.get("lock_sha256") == EXPECTED_HASHES[R8A5_LOCK] and extension.get("audit_token") == "PH1_INTEL_EXECUTION_R8A5_AFTER_R8P8_PASS_AND_EXPLICIT_BINDING_AUDIT_GO" and extension.get("ast_no_ancestry_chains") is True and extension.get("one_attempt") is True and observed == lock_hashes and all(r8a5_lock.get(key) == value for key, value in observed.items()) and predicates == {"r8p8_pass": True, "r7d_contract": True, "r7d1_failure": True, "r8p6_failure": True} and set(modules) == {"historical", "physical"} and set(modules.get("historical", {})) == {"path", "sha256", "required_attrs"} and modules["historical"].get("required_attrs") == historical_attrs and same_path(modules["historical"].get("path"), str(R8A_RUNNER.resolve())) and modules["historical"].get("sha256") == EXPECTED_HASHES[R8A_RUNNER] and set(modules.get("physical", {})) == {"path", "sha256", "required_attrs"} and modules["physical"].get("required_attrs") == physical_attrs and same_path(modules["physical"].get("path"), str(R7A_RUNNER.resolve())) and modules["physical"].get("sha256") == EXPECTED_HASHES[R7A_RUNNER] and set(invocation_row) == invocation_keys and native_ok and invocation_row.get("native_argv") == invocation_row.get("orig_argv") == expected_run_native and invocation_row.get("argv") == [str(R8A5_RUNNER), "--ack", expected_run_native[-1]] and same_path(invocation_row.get("sys_executable"), str(VENV_PYTHON.resolve())) and same_path(invocation_row.get("sys_prefix"), str(VENV.resolve())) and same_path(invocation_row.get("base_executable"), str(ALIAS)) and same_path(invocation_row.get("base_prefix"), str(BASE_PREFIX)) and invocation_row.get("isolated") == 1 and invocation_row.get("dont_write_bytecode") is True and invocation_row.get("entry_name") == "__main__" and invocation_row.get("entry_spec_is_none") is True and invocation_row.get("entry_package") is None and same_path(invocation_row.get("entry_file"), str(R8A5_RUNNER)) and invocation_row.get("python_sha256") == "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14" and invocation_row.get("pyvenv_sha256") == "9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe" and invocation_row.get("direct_entry") is True

def direct_physical(result: dict) -> dict:
    evidence = result.get("evidence", {}); ledger = evidence.get("ledger", []); ownership = evidence.get("ownership_ledger", []); controls = result.get("controls", []); resources = result.get("resource", {}); gates = result.get("gates", {})
    operation_counts = Counter(row.get("op") for row in ledger); allocations = [row for row in ledger if row.get("op") == "host_usm_allocate"]; args = [row for row in ledger if row.get("op") == "set_pointer_arg"]; launches = [row for row in ledger if row.get("op") == "enqueue"]; finishes = [row for row in ledger if row.get("op") == "finish"]; reads = [row for row in ledger if row.get("op") == "cpu_direct_read"]; releases = [row for row in ledger if row.get("op") == "release"]; cleanups = [row for row in ledger if row.get("op") == "cleanup"]; samples = [row for row in ledger if row.get("op") == "resource_sample"]
    cleanup_ok = len(cleanups) == 1 and cleanups[0] == {"op": "cleanup", "cleanup_complete": True, "errors": [], "live_owned_resources": 0, "live_resource_names": [], "release_attempts": 21}
    identity = evidence.get("identity", {})
    forbidden = evidence.get("forbidden_calls", {})
    resource_ok = [row.get("stage") for row in samples] == SAMPLE_STAGES and all(row.get("telemetry_error") is None and isinstance(row.get("available"), int) and row["available"] >= 2 * 2**30 and isinstance(row.get("peak_wset"), int) and row["peak_wset"] <= 12 * 2**30 for row in samples) and resources.get("start_available", 0) >= 16 * 2**30 and resources.get("peak_retained_wset", 13 * 2**30) <= 12 * 2**30 and evidence.get("telemetry_errors") == []
    return {
        "positive_schema": result.get("positive") is True and result.get("status") == "intel_execution_positive" and result.get("claim") == "one real expert/input Intel correctness component only",
        "gates18": set(gates) == GATES and len(gates) == 18 and all(value is True for value in gates.values()),
        "identity": identity.get("name") == "Intel(R) Arc(TM) Pro 140T GPU (32GB)" and identity.get("vendor") == "Intel(R) Corporation" and identity.get("driver") == "32.0.101.8517" and identity.get("pci") == "0000:00:02.0",
        "ledger102": len(ledger) == 102 and operation_counts == Counter(EXPECTED_OPS),
        "ownership95": len(ownership) == 95 and all(row.get("attempted") is True and row.get("exception") is None and isinstance(row.get("returned"), int) for row in ownership),
        "alloc14": len(allocations) == 14 and len({row.get("pointer") for row in allocations}) == 14 and all(isinstance(row.get("pointer"), int) and row["pointer"] != 0 and row.get("alignment") == 4096 and row.get("type") == 0x4197 and row.get("base") == row.get("pointer") and row.get("queried_size") == row.get("bytes") for row in allocations),
        "args18": len(args) == 18 and all(row.get("pointer") in {allocation.get("pointer") for allocation in allocations} for row in args),
        "launch4": len(launches) == 4 and all(row.get("event_requested") is False for row in launches),
        "finish1_read9": len(finishes) == 1 and len(reads) == 9 and all(row.get("after_finish") is True for row in reads),
        "release21_cleanup_live0": len(releases) == 21 and [row.get("attempt_index") for row in releases] == list(range(21)) and all(row.get("attempted") is True and row.get("owned_before") is True and row.get("code") == 0 and row.get("exception") is None and row.get("owned_after") is False for row in releases) and cleanup_ok and evidence.get("cleanup_errors") == [],
        "controls22": len(controls) == 22 and all(row.get("pass") is True and all(value == 0 for value in row.get("predevice_counts", {}).values()) for row in controls),
        "extensions": evidence.get("extension_counts") == {"clHostMemAllocINTEL": 14, "clMemFreeINTEL": 14, "clSetKernelArgMemPointerINTEL": 18, "clGetMemAllocInfoINTEL": 42},
        "forbidden0": set(forbidden) == {"clCreateBuffer", "clEnqueueReadBuffer", "clEnqueueWriteBuffer", "clEnqueueCopyBuffer", "clEnqueueMigrateMemObjects", "clEnqueueMemAdviseINTEL"} and all(value == 0 for value in forbidden.values()),
        "resources": resource_ok,
        "stage_hashes": result.get("stage_hashes") == STAGE_HASHES,
    }

def lock_contract() -> tuple[bool, dict]:
    observed = {key: sha256(path) for key, path in CHAIN.items()}; lock = json.loads(LOCK.read_text())
    valid = set(lock) == {"kind", "execution_open", "audit_token", "one_attempt", *observed} and lock.get("kind") == "ph1_intel_execution_r8v1_lock" and lock.get("execution_open") is False and lock.get("audit_token") == ACK and lock.get("one_attempt") is True and all(lock.get(key) == value for key, value in observed.items())
    return valid, observed

def write_result(row: dict) -> None:
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    temp = R / f"{OUTPUT.name}.{uuid.uuid4().hex}.inprogress"
    try:
        with temp.open("xb") as handle: handle.write(canonical(row)); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, OUTPUT); temp.unlink()
    finally:
        if temp.exists(): temp.unlink()

def verify() -> tuple[dict, dict]:
    lock_ok, observed = lock_contract(); topology_ok, topology_evidence = topology(); mutations = topology_mutations(); bundle_ok, result = bundle()
    checks = {"lock_closed": lock_ok, "topology": topology_ok, "topology_mutations": len(mutations) == len(expected_family_names()) + 8 and all(mutations.values()), "bundle": bundle_ok, "authorization": bundle_ok and authorization_valid(result)}
    direct = direct_physical(result) if bundle_ok else {}; checks.update({"physical:" + key: value for key, value in direct.items()})
    numerical = {}
    if bundle_ok and checks["authorization"] and sha256(R7A_VERIFIER) == EXPECTED_HASHES[R7A_VERIFIER]:
        sys.path.insert(0, str(S)); import verify_het_next_l0_ph1_intel_execution_r7a as numerical_verifier
        numerical = numerical_verifier.verify_dict(result)
    checks["numerical_exact20"] = set(numerical) == NUMERICAL and len(numerical) == 20 and all(value is True for value in numerical.values())
    terminal_matrix = {}
    if sha256(R8A5_VERIFIER) == EXPECTED_HASHES[R8A5_VERIFIER]:
        sys.path.insert(0, str(S)); import verify_het_next_l0_ph1_intel_execution_r8a5 as terminal_verifier
        terminal_matrix = terminal_verifier.mutation_harness()
    failed_row = json.loads(FAILED_VERIFIER.read_text())
    checks["terminal_matrix31"] = len(terminal_matrix) == 31 and all(value is True for value in terminal_matrix.values()) and failed_row.get("mutation_matrix") == terminal_matrix and failed_row.get("checks", {}).get("committed_adjudicator_mutations") is True and failed_row.get("checks", {}).get("production_matrix") is True
    return checks, {"observed_hashes": observed, "topology": topology_evidence, "topology_mutations": mutations, "numerical": numerical, "direct_physical": direct, "terminal_matrix": terminal_matrix}

def main() -> int:
    ident = invocation()
    if sys.argv != EXPECTED_ARGV or not invocation_valid(ident): return 3
    lock_ok, _ = lock_contract()
    if not lock_ok or json.loads(LOCK.read_text()).get("execution_open") is not True: return 3
    checks, evidence = verify(); passed = all(checks.values())
    row = {"kind": "ph1_intel_execution_r8v1_independent_verification", "checks": checks, "evidence": evidence, "pass": passed, "passed": sum(value is True for value in checks.values()), "total": len(checks), "model_forward": False, "compiler_opened": False, "opencl_opened": False, "device_opened": False, "claim": "one real expert/input Intel correctness component only"}
    write_result(row); print(json.dumps(row, indent=2)); return 0 if passed else 3

if __name__ == "__main__":
    raise SystemExit(main())
