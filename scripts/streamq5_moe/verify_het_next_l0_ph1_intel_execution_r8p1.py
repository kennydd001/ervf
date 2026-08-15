#!/usr/bin/env python3
"""Independent R8P1 verifier: invocation, prior bundle and transaction topology."""
from __future__ import annotations

import copy
import ctypes as C
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))
import verify_het_next_l0_ph1_intel_execution_r8p as base

LOCK = R / "het_next_l0_ph1_intel_execution_r8p1_lock.json"
RESULT = R / "het_next_l0_ph1_intel_execution_r8p1_static_preflight.json"
MANIFEST = R / "het_next_l0_ph1_intel_execution_r8p1_static_preflight.manifest.json"
COMMIT = R / "het_next_l0_ph1_intel_execution_r8p1_static_preflight.commit.json"
OUTPUT = R / "het_next_l0_ph1_intel_execution_r8p1_independent_verification.json"
FAILED = R / "het_next_l0_ph1_intel_execution_r8p1_failed_attempts"
QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8p1_quarantine"
FAILURE_ROOT = R / "het_next_l0_ph1_intel_execution_r7d1_failed_attempts"
ATTEMPT = FAILURE_ROOT / "attempt_7c45ba0bda09470eba7145ef75281ea3"
FAILURE = ATTEMPT / "failure.json"
PREFLIGHT = S / "preflight_het_next_l0_ph1_intel_execution_r8p1.py"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_PREREGISTRATION_2026-08-14.md"
AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
ACK = "PH1_INTEL_EXECUTION_R8P1_EXACT_FULL_INVOCATION_CPU_PREPARATION_CLOSED"
PREFLIGHT_ORIG = [str(base.PYTHON.resolve()), "-I", "-B", str(PREFLIGHT.resolve()), "--ack", ACK]
PREFLIGHT_ARGV = [str(PREFLIGHT.resolve()), "--ack", ACK]
PREFLIGHT_COMMAND = subprocess.list2cmdline(PREFLIGHT_ORIG)
VERIFIER_ORIG = [str(base.PYTHON.resolve()), "-I", "-B", str(Path(__file__).resolve())]
VERIFIER_COMMAND = subprocess.list2cmdline(VERIFIER_ORIG)
CHECKS = {"exact_invocation", "hash_bindings", "closed_pending", "runtime_lock", "exact_runtime", "start_ram", "wheel_records", "runtime_mutations", "r7d1_failure", "cpu_preparation", "transaction_simulation", "static_no_device", "pre_run_topology", "base_clean"}
CHAIN = {
    "preflight_sha256": PREFLIGHT, "verifier_sha256": Path(__file__), "prereg_sha256": PREREG, "audit_sha256": AUDIT,
    "r8_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r8.py", "r8_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8.py", "r8_preflight_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p.py", "r8_physical_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8.py", "r8_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8_PREREGISTRATION_2026-08-14.md", "r8_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8_lock.json",
    "runtime_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PSUTIL_FAILURE_AND_R8_RUNTIME_REPAIR_AUDIT_2026-08-14.md", "r7d1_failure_sha256": FAILURE, "common_sha256": S / "het_next_l0_ph1_intel_execution_r6_common.py", "numerical_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7a.py", "cpu_result_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/cpu_stage_freeze.json", "cpu_raw_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/cpu_stage_freeze.safetensors",
}
LOCK_STATIC = {"python_sha256": base.PYTHON_SHA, "python_version": base.PYTHON_VERSION, "pyvenv_sha256": base.FILES["pyvenv"][1], "psutil_version": "7.2.2", "psutil_init_sha256": base.FILES["psutil_init"][1], "psutil_native_sha256": base.FILES["psutil_native"][1], "psutil_metadata_sha256": base.FILES["psutil_metadata"][1], "psutil_record_sha256": base.FILES["psutil_record"][1], "numpy_version": "2.2.6", "numpy_init_sha256": base.FILES["numpy_init"][1], "numpy_metadata_sha256": base.FILES["numpy_metadata"][1], "numpy_record_sha256": base.FILES["numpy_record"][1], "preparation_digest": base.PREPARATION_DIGEST}


def sha_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def sha256(path: Path) -> str: return sha_bytes(path.read_bytes())
def canon(value: object) -> bytes: return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def win32_commandline() -> tuple[str, list[str]]:
    kernel = C.WinDLL("kernel32", use_last_error=True); shell = C.WinDLL("shell32", use_last_error=True)
    get_command = kernel.GetCommandLineW; get_command.argtypes = (); get_command.restype = C.c_wchar_p
    parse = shell.CommandLineToArgvW; parse.argtypes = (C.c_wchar_p, C.POINTER(C.c_int)); parse.restype = C.POINTER(C.c_wchar_p)
    local_free = kernel.LocalFree; local_free.argtypes = (C.c_void_p,); local_free.restype = C.c_void_p
    raw = get_command(); count = C.c_int(); pointer = parse(raw, C.byref(count))
    if not pointer: raise C.WinError(C.get_last_error())
    try: parsed = [pointer[index] for index in range(count.value)]
    finally:
        if local_free(C.cast(pointer, C.c_void_p)): raise C.WinError(C.get_last_error())
    return raw, parsed


def verifier_invocation_valid() -> bool:
    raw, parsed = win32_commandline()
    return raw == VERIFIER_COMMAND and parsed == VERIFIER_ORIG and list(sys.orig_argv) == VERIFIER_ORIG and list(sys.argv) == [str(Path(__file__).resolve())] and __spec__ is None and (__package__ is None or __package__ == "")


def preflight_invocation_valid(row: dict) -> bool:
    return set(row) == {"win32_raw_commandline", "win32_parsed_argv", "orig_argv", "argv", "resolved_executable", "resolved_script", "entrypoint_absolute", "direct_script"} and row["win32_raw_commandline"] == PREFLIGHT_COMMAND and row["win32_parsed_argv"] == PREFLIGHT_ORIG and row["orig_argv"] == PREFLIGHT_ORIG and row["argv"] == PREFLIGHT_ARGV and row["resolved_executable"].casefold() == PREFLIGHT_ORIG[0].casefold() and row["resolved_script"].casefold() == PREFLIGHT_ORIG[3].casefold() and row["entrypoint_absolute"] is True and row["direct_script"] is True


def failure_bundle_valid(root: Path) -> bool:
    expected_attempt = root / ATTEMPT.name; expected_file = expected_attempt / "failure.json"
    if not root.is_dir() or sorted(path.name for path in root.iterdir()) != [ATTEMPT.name] or not expected_attempt.is_dir() or sorted(path.name for path in expected_attempt.iterdir()) != ["failure.json"]: return False
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if files != [expected_file] or expected_file.stat().st_size != 931 or sha256(expected_file) != "88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0": return False
    row = json.loads(expected_file.read_text())
    return set(row) == {"covered_stages", "device_opened", "disposition", "error", "kind", "stage", "status", "traceback"} and row["covered_stages"] == ["psutil_import", "start_ram", "payload", "post_payload_resource", "predevice", "device_execute", "serialize_commit"] and row["device_opened"] is False and row["disposition"] == "atomic_create_new_bounded_outer_failure" and row["error"] == "ModuleNotFoundError:No module named 'psutil'" and row["kind"] == "ph1_intel_execution_r7c2_failure" and row["stage"] == "r7a_outer_boundary" and row["status"] == "valid_negative_failure" and isinstance(row["traceback"], str) and "import psutil" in row["traceback"] and "ModuleNotFoundError: No module named 'psutil'" in row["traceback"]


def failure_bundle_mutations() -> dict:
    outcomes = {}
    with tempfile.TemporaryDirectory() as td:
        parent = Path(td)
        def fresh() -> Path:
            root = parent / ("case_" + uuid.uuid4().hex); attempt = root / ATTEMPT.name; attempt.mkdir(parents=True); shutil.copy2(FAILURE, attempt / "failure.json"); return root
        root = fresh(); outcomes["baseline"] = failure_bundle_valid(root)
        root = fresh(); (root / ATTEMPT.name / "failure.json").unlink(); outcomes["missing"] = not failure_bundle_valid(root)
        root = fresh(); (root / ATTEMPT.name / "extra.bin").write_bytes(b"x"); outcomes["extra"] = not failure_bundle_valid(root)
        root = fresh(); path = root / ATTEMPT.name / "failure.json"; path.write_bytes(path.read_bytes() + b"x"); outcomes["wrong_size_hash"] = not failure_bundle_valid(root)
        for name, field, value in (("wrong_boolean", "device_opened", 0), ("wrong_disposition", "disposition", "wrong"), ("wrong_stage", "stage", "wrong")):
            root = fresh(); path = root / ATTEMPT.name / "failure.json"; row = json.loads(path.read_text()); row[field] = value; path.write_bytes(canon(row)); outcomes[name] = not failure_bundle_valid(root)
    return outcomes


def atomic_create(path: Path, payload: bytes, *, link_fn=os.link, unlink_fn=Path.unlink) -> None:
    if path.exists(): raise FileExistsError(path)
    temp = path.with_name(path.name + ".inprogress." + uuid.uuid4().hex)
    try:
        with temp.open("xb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        link_fn(temp, path)
    finally:
        if temp.exists(): unlink_fn(temp)


def cleanup_temps(root: Path, stem: str) -> list[str]:
    rows = []
    for path in sorted(root.glob(stem + ".inprogress.*")): path.unlink(); rows.append(path.name)
    return rows


def transaction_mutations() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); target = root / "target.json"; atomic_create(target, b"one"); first = sha256(target); existing = False
        try: atomic_create(target, b"two")
        except FileExistsError: existing = sha256(target) == first
        def fail_link(_a, _b): raise OSError("link")
        hard = root / "hard.json"; hard_ok = False
        try: atomic_create(hard, b"x", link_fn=fail_link)
        except OSError: hard_ok = not hard.exists() and not list(root.glob(hard.name + ".inprogress.*"))
        def fail_unlink(_p): raise OSError("unlink")
        post = root / "post.json"; post_ok = False
        try: atomic_create(post, b"y", unlink_fn=fail_unlink)
        except OSError:
            before = list(root.glob(post.name + ".inprogress.*")); cleanup_temps(root, post.name); post_ok = post.read_bytes() == b"y" and len(before) == 1 and not list(root.glob(post.name + ".inprogress.*"))
        stale = root / ("stale.json.inprogress.fixed"); stale.write_bytes(b"s"); stale_ok = cleanup_temps(root, "stale.json") == [stale.name] and not stale.exists()
        verify = root / "verify.json"; atomic_create(verify, b"v"); repeated = False
        try: atomic_create(verify, b"z")
        except FileExistsError: repeated = verify.read_bytes() == b"v"
        return {"existing_destination": existing, "hardlink_failure": hard_ok, "postlink_cleanup_failure": post_ok, "stale_temp_cleanup": stale_ok, "repeated_verifier": repeated}


def bundle_valid() -> bool:
    if not all(path.is_file() for path in (RESULT, MANIFEST, COMMIT)): return False
    rb = RESULT.read_bytes(); mb = MANIFEST.read_bytes(); m = json.loads(mb); c = json.loads(COMMIT.read_text())
    return m == {"kind": "ph1_intel_execution_r8p1_manifest", "files": [{"name": RESULT.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]} and c == {"kind": "ph1_intel_execution_r8p1_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)}


def topology_valid() -> bool:
    absent = (R / "het_next_l0_ph1_intel_execution_r8", R / "het_next_l0_ph1_intel_execution_r8_failed_attempts", R / "het_next_l0_ph1_intel_execution_r8_quarantine", R / "het_next_l0_ph1_intel_execution_r8_independent_verification.json", R / "het_next_l0_ph1_intel_execution_r8_static_preflight.json", R / "het_next_l0_ph1_intel_execution_r8p_independent_verification.json", FAILED, QUARANTINE, OUTPUT)
    expected_family = {R / "het_next_l0_ph1_intel_execution_r8_lock.json", LOCK, RESULT, MANIFEST, COMMIT}
    family = {path for path in R.iterdir() if path.name.startswith("het_next_l0_ph1_intel_execution_r8")}
    return all(not path.exists() for path in absent) and all(path.is_file() for path in (RESULT, MANIFEST, COMMIT)) and not list(R.glob("het_next_l0_ph1_intel_execution_r8p1*.inprogress.*")) and not list(R.glob("het_next_l0_ph1_intel_execution_r8*.inprogress.*")) and family == expected_family


def lock_valid() -> bool:
    lock = json.loads(LOCK.read_text()); observed = {name: sha256(path) for name, path in CHAIN.items()}
    return set(lock) == {"kind", "execution_open", "audit_token", *LOCK_STATIC, *observed} and lock.get("kind") == "ph1_intel_execution_r8p1_lock" and lock.get("execution_open") is False and lock.get("audit_token") == "PENDING" and all(lock.get(name) == value for name, value in LOCK_STATIC.items()) and all(lock.get(name) == digest for name, digest in observed.items()) and observed["audit_sha256"] == "60518c8999e35e9a09e873c398c4cb18c15a9b5b6e6f0e4972c034f36b3e5a37"


def pre_topology_valid(row: dict) -> bool:
    absent = (
        R / "het_next_l0_ph1_intel_execution_r8", R / "het_next_l0_ph1_intel_execution_r8_failed_attempts", R / "het_next_l0_ph1_intel_execution_r8_quarantine",
        R / "het_next_l0_ph1_intel_execution_r8_independent_verification.json", R / "het_next_l0_ph1_intel_execution_r8_static_preflight.json", R / "het_next_l0_ph1_intel_execution_r8p_independent_verification.json",
        RESULT, MANIFEST, COMMIT, OUTPUT, FAILED, QUARANTINE,
    )
    expected_exact = {str(path): False for path in absent}; expected_family = sorted((str(R / "het_next_l0_ph1_intel_execution_r8_lock.json"), str(LOCK)))
    return row == {"exact_exists": expected_exact, "r8p1_temps": [], "r8_family_temps": [], "r8_family_paths": expected_family}


def result_valid(row: dict, preparation: dict, wheels: dict) -> bool:
    checks = row.get("checks", {})
    transaction_names = {"clean_bundle", "repeat_rejected_preserved", "stale_seen_cleaned", "hardlink_failure_clean", "postlink_cleanup_failure_detected_recovered", "verifier_repeat_rejected", "partial_quarantined"}
    return set(row) == {"kind", "ack", "invocation", "pre_run_topology", "checks", "pass", "passed", "total", "runtime", "wheel_records", "preparation", "preparation_digest", "transaction_simulation", "runtime_mutations_rejected", "no_compiler_device", "cpu_payload_read"} and row.get("kind") == "ph1_intel_execution_r8p1_static_preflight" and row.get("ack") == ACK and preflight_invocation_valid(row.get("invocation", {})) and pre_topology_valid(row.get("pre_run_topology", {})) and isinstance(checks, dict) and set(checks) == CHECKS and all(value is True for value in checks.values()) and row.get("pass") is True and row.get("passed") == row.get("total") == 14 and base.runtime_static_valid(row.get("runtime", {})) and row.get("runtime", {}).get("available", 0) >= 16 * 2**30 and row.get("wheel_records") == wheels and row.get("preparation") == preparation and row.get("preparation_digest") == base.PREPARATION_DIGEST == sha_bytes(canon(preparation)) and isinstance(row.get("transaction_simulation"), dict) and set(row["transaction_simulation"]) == transaction_names and all(row["transaction_simulation"].values()) and row.get("runtime_mutations_rejected") == ["python_path", "python_hash", "isolation", "bytecode", "pyvenv", "psutil_native", "psutil_record", "numpy_version", "numpy_record", "ram"] and row.get("no_compiler_device") is True and row.get("cpu_payload_read") is True


def main() -> int:
    if not verifier_invocation_valid(): return 3
    raw = RESULT.read_bytes(); row = json.loads(raw); live = base.collect_runtime(); wheels = {"psutil": base.wheel_record(base.FILES["psutil_record"][0]), "numpy": base.wheel_record(base.FILES["numpy_record"][0])}; preparation = base.independent_preparation(); failures = failure_bundle_mutations(); transactions = transaction_mutations()
    baseline = result_valid(row, preparation, wheels); rejected = []
    cases = {"kind": lambda x: x.__setitem__("kind", "wrong"), "check": lambda x: x["checks"].__setitem__("exact_invocation", False), "invocation": lambda x: x["invocation"].__setitem__("win32_raw_commandline", "wrong"), "control": lambda x: x["preparation"]["controls"].pop(), "digest": lambda x: x.__setitem__("preparation_digest", "0" * 64), "transaction": lambda x: x["transaction_simulation"].__setitem__("hardlink_failure_clean", False)}
    for name, mutate in cases.items():
        candidate = copy.deepcopy(row); mutate(candidate)
        if not result_valid(candidate, preparation, wheels): rejected.append(name)
    checks = {"exact_verifier_invocation": True, "lock_chain": lock_valid(), "bundle_commit_binds_result_sha": bundle_valid(), "full_topology_clean": topology_valid(), "r7d1_bundle_exact": failure_bundle_valid(FAILURE_ROOT), "r7d1_bundle_mutations": len(failures) == 7 and all(failures.values()), "runtime_exact": base.runtime_static_valid(live), "wheel_records": wheels["psutil"]["hashed_files_verified"] == 17 and wheels["numpy"]["hashed_files_verified"] == 899, "independent_preparation": sha_bytes(canon(preparation)) == base.PREPARATION_DIGEST, "result_schema": baseline, "result_mutations": rejected == ["kind", "check", "invocation", "control", "digest", "transaction"], "transaction_mutations": len(transactions) == 5 and all(transactions.values())}
    output = {"kind": "ph1_intel_execution_r8p1_independent_verification", "checks": checks, "pass": all(checks.values()), "passed": sum(value is True for value in checks.values()), "total": len(checks), "result_sha256": sha_bytes(raw), "manifest_sha256": sha256(MANIFEST), "commit_sha256": sha256(COMMIT), "failure_bundle_mutations": failures, "transaction_mutations": transactions, "no_compiler_device": True, "cpu_payload_read": True}
    atomic_create(OUTPUT, canon(output)); print(json.dumps(output, indent=2)); return 0 if output["pass"] else 3


if __name__ == "__main__": raise SystemExit(main())
