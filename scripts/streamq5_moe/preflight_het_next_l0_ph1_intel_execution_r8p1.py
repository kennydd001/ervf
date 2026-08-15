#!/usr/bin/env python3
"""R8P1 closed runtime/CPU preflight; exact invocation and transaction repair."""
from __future__ import annotations

import argparse
import ast
import copy
import ctypes as C
import hashlib
import json
import os
import sys
import tempfile
import uuid
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))
import preflight_het_next_l0_ph1_intel_execution_r8 as base

LOCK = R / "het_next_l0_ph1_intel_execution_r8p1_lock.json"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_PREREGISTRATION_2026-08-14.md"
VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r8p1.py"
RESULT = R / "het_next_l0_ph1_intel_execution_r8p1_static_preflight.json"
MANIFEST = R / "het_next_l0_ph1_intel_execution_r8p1_static_preflight.manifest.json"
COMMIT = R / "het_next_l0_ph1_intel_execution_r8p1_static_preflight.commit.json"
VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r8p1_independent_verification.json"
FAILED = R / "het_next_l0_ph1_intel_execution_r8p1_failed_attempts"
QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8p1_quarantine"
AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
ACK = "PH1_INTEL_EXECUTION_R8P1_EXACT_FULL_INVOCATION_CPU_PREPARATION_CLOSED"
EXPECTED_ORIG = [str(base.runner.VENV_PYTHON.resolve()), "-I", "-B", str(Path(__file__).resolve()), "--ack", ACK]
EXPECTED_ARGV = [str(Path(__file__).resolve()), "--ack", ACK]
EXPECTED_COMMANDLINE = subprocess.list2cmdline(EXPECTED_ORIG)
BASE_R8_PATHS = (
    base.runner.REVISION_OUT, base.runner.FAILED, base.runner.QUARANTINE, base.runner.VERIFY_RESULT,
    base.runner.PREFLIGHT_RESULT, base.runner.PREFLIGHT_VERIFY_RESULT,
)
CORE = (RESULT, MANIFEST, COMMIT)
CHAIN = {
    "preflight_sha256": Path(__file__), "verifier_sha256": VERIFIER, "prereg_sha256": PREREG, "audit_sha256": AUDIT,
    "r8_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r8.py", "r8_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8.py",
    "r8_preflight_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p.py", "r8_physical_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8.py",
    "r8_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8_PREREGISTRATION_2026-08-14.md", "r8_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8_lock.json",
    "runtime_audit_sha256": base.runner.RUNTIME_AUDIT, "r7d1_failure_sha256": base.runner.R7D1_FAILURE,
    "common_sha256": S / "het_next_l0_ph1_intel_execution_r6_common.py", "numerical_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7a.py",
    "cpu_result_sha256": base.CPU_RESULT, "cpu_raw_sha256": base.CPU_RAW,
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canon(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def invocation_evidence() -> dict:
    kernel = C.WinDLL("kernel32", use_last_error=True); shell = C.WinDLL("shell32", use_last_error=True)
    get_command = kernel.GetCommandLineW; get_command.argtypes = (); get_command.restype = C.c_wchar_p
    parse = shell.CommandLineToArgvW; parse.argtypes = (C.c_wchar_p, C.POINTER(C.c_int)); parse.restype = C.POINTER(C.c_wchar_p)
    local_free = kernel.LocalFree; local_free.argtypes = (C.c_void_p,); local_free.restype = C.c_void_p
    raw_command = get_command(); count = C.c_int(); pointer = parse(raw_command, C.byref(count))
    if not pointer: raise C.WinError(C.get_last_error())
    try: win32_argv = [pointer[index] for index in range(count.value)]
    finally:
        if local_free(C.cast(pointer, C.c_void_p)): raise C.WinError(C.get_last_error())
    raw_orig = list(sys.orig_argv); raw_argv = list(sys.argv)
    return {"win32_raw_commandline": raw_command, "win32_parsed_argv": win32_argv, "orig_argv": raw_orig, "argv": raw_argv, "resolved_executable": str(Path(raw_orig[0]).resolve()) if raw_orig else None, "resolved_script": str(Path(raw_orig[3]).resolve()) if len(raw_orig) > 3 and raw_orig[3] not in {"-c", "-m"} else None, "entrypoint_absolute": len(raw_orig) == 6 and Path(raw_orig[3]).is_absolute(), "direct_script": __spec__ is None and (__package__ is None or __package__ == "")}


def invocation_valid(row: dict) -> bool:
    return set(row) == {"win32_raw_commandline", "win32_parsed_argv", "orig_argv", "argv", "resolved_executable", "resolved_script", "entrypoint_absolute", "direct_script"} and row["win32_raw_commandline"] == EXPECTED_COMMANDLINE and row["win32_parsed_argv"] == EXPECTED_ORIG and row["orig_argv"] == EXPECTED_ORIG and row["argv"] == EXPECTED_ARGV and row["resolved_executable"].casefold() == EXPECTED_ORIG[0].casefold() and row["resolved_script"].casefold() == EXPECTED_ORIG[3].casefold() and row["entrypoint_absolute"] is True and row["direct_script"] is True


def invocation_mutations(row: dict) -> list[str]:
    cases = {
        "c_trampoline": lambda x: x["orig_argv"].__setitem__(3, "-c"), "wrong_script": lambda x: x.__setitem__("resolved_script", "C:/wrong.py"),
        "swapped_flags": lambda x: x["orig_argv"].__setitem__(slice(1, 3), ["-B", "-I"]), "extra_interpreter_flag": lambda x: x["orig_argv"].insert(3, "-s"),
        "wrong_ack": lambda x: x["argv"].__setitem__(2, "WRONG"), "extra_application_arg": lambda x: x["argv"].append("extra"),
        "raw_commandline": lambda x: x.__setitem__("win32_raw_commandline", x["win32_raw_commandline"] + " --extra"), "win32_parsed": lambda x: x["win32_parsed_argv"].append("extra"),
    }
    rejected = []
    for name, mutation in cases.items():
        candidate = copy.deepcopy(row); mutation(candidate)
        if not invocation_valid(candidate): rejected.append(name)
    return rejected


def topology_snapshot() -> dict:
    exact = {str(path): path.exists() for path in (*BASE_R8_PATHS, *CORE, VERIFY_RESULT, FAILED, QUARANTINE)}
    globs = sorted(str(path) for path in R.glob("het_next_l0_ph1_intel_execution_r8p1*.inprogress.*"))
    base_globs = sorted(str(path) for path in R.glob("het_next_l0_ph1_intel_execution_r8*.inprogress.*"))
    family = sorted(str(path) for path in R.iterdir() if path.name.startswith("het_next_l0_ph1_intel_execution_r8"))
    return {"exact_exists": exact, "r8p1_temps": globs, "r8_family_temps": base_globs, "r8_family_paths": family}


def topology_clean(row: dict) -> bool:
    expected_keys = {str(path) for path in (*BASE_R8_PATHS, *CORE, VERIFY_RESULT, FAILED, QUARANTINE)}
    expected_family = sorted((str(R / "het_next_l0_ph1_intel_execution_r8_lock.json"), str(LOCK)))
    return set(row) == {"exact_exists", "r8p1_temps", "r8_family_temps", "r8_family_paths"} and set(row["exact_exists"]) == expected_keys and all(value is False for value in row["exact_exists"].values()) and row["r8p1_temps"] == row["r8_family_temps"] == [] and row["r8_family_paths"] == expected_family


def atomic_create(path: Path, payload: bytes, *, link_fn=os.link, unlink_fn=Path.unlink) -> None:
    if path.exists(): raise FileExistsError(path)
    temp = path.with_name(path.name + ".inprogress." + uuid.uuid4().hex)
    try:
        with temp.open("xb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        link_fn(temp, path)
    finally:
        if temp.exists(): unlink_fn(temp)


def cleanup_temps(root: Path, stem: str) -> list[str]:
    removed = []
    for path in sorted(root.glob(stem + ".inprogress.*")):
        path.unlink(); removed.append(path.name)
    return removed


def quarantine_core(core: tuple[Path, ...], root: Path) -> list[dict]:
    rows = []; destination = root / ("attempt_" + uuid.uuid4().hex); destination.mkdir(parents=True)
    for path in core:
        if path.exists():
            target = destination / path.name; os.replace(path, target); rows.append({"source": path.name, "target": str(target), "sha256": sha256(target)})
    return rows


def verify_bundle(result: Path, manifest: Path, commit: Path) -> bool:
    if not all(path.is_file() for path in (result, manifest, commit)): return False
    rb = result.read_bytes(); mb = manifest.read_bytes(); m = json.loads(mb); c = json.loads(commit.read_text())
    return m == {"kind": "ph1_intel_execution_r8p1_manifest", "files": [{"name": result.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]} and c == {"kind": "ph1_intel_execution_r8p1_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)}


def publish_bundle(output: dict, result: Path, manifest: Path, commit: Path, quarantine_root: Path = QUARANTINE) -> None:
    if any(path.exists() for path in (result, manifest, commit)):
        raise FileExistsError("bundle_target_exists")
    rb = canon(output); mb = canon({"kind": "ph1_intel_execution_r8p1_manifest", "files": [{"name": result.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]}); cb = canon({"kind": "ph1_intel_execution_r8p1_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)})
    try:
        atomic_create(result, rb); atomic_create(manifest, mb); atomic_create(commit, cb)
        if not verify_bundle(result, manifest, commit): raise RuntimeError("bundle")
    except Exception:
        quarantine_core((result, manifest, commit), quarantine_root); raise


def transaction_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); result = root / "result.json"; manifest = root / "manifest.json"; commit = root / "commit.json"; payload = {"fixture": True}
        publish_bundle(payload, result, manifest, commit, root / "quarantine"); original = tuple(sha256(path) for path in (result, manifest, commit)); repeat = False
        try: publish_bundle(payload, result, manifest, commit, root / "quarantine")
        except FileExistsError: repeat = tuple(sha256(path) for path in (result, manifest, commit)) == original
        stale = result.with_name(result.name + ".inprogress.stale"); stale.write_bytes(b"stale"); stale_seen = bool(list(root.glob(result.name + ".inprogress.*"))); removed = cleanup_temps(root, result.name); stale_clean = removed == [stale.name] and not stale.exists()
        hard_target = root / "hard.json"
        def fail_link(_source, _target): raise OSError("injected_link")
        hard_fail = False
        try: atomic_create(hard_target, b"x", link_fn=fail_link)
        except OSError: hard_fail = not hard_target.exists() and not list(root.glob(hard_target.name + ".inprogress.*"))
        post_target = root / "post.json"
        def fail_unlink(_path): raise OSError("injected_unlink")
        post_fail = False
        try: atomic_create(post_target, b"y", unlink_fn=fail_unlink)
        except OSError:
            leftovers = list(root.glob(post_target.name + ".inprogress.*")); cleanup_temps(root, post_target.name); post_fail = post_target.read_bytes() == b"y" and len(leftovers) == 1 and not list(root.glob(post_target.name + ".inprogress.*"))
        verifier_target = root / "verify.json"; atomic_create(verifier_target, b"v"); verifier_repeat = False
        try: atomic_create(verifier_target, b"changed")
        except FileExistsError: verifier_repeat = verifier_target.read_bytes() == b"v"
        partial = root / "partial.json"; atomic_create(partial, b"p"); quarantined = quarantine_core((partial,), root / "failed"); partial_ok = len(quarantined) == 1 and not partial.exists() and Path(quarantined[0]["target"]).read_bytes() == b"p"
        return {"clean_bundle": verify_bundle(result, manifest, commit), "repeat_rejected_preserved": repeat, "stale_seen_cleaned": stale_seen and stale_clean, "hardlink_failure_clean": hard_fail, "postlink_cleanup_failure_detected_recovered": post_fail, "verifier_repeat_rejected": verifier_repeat, "partial_quarantined": partial_ok}


def static_contract() -> bool:
    tree = ast.parse(Path(__file__).read_text()); imports = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}; calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    return not ({"pyopencl", "cupy", "torch", "safetensors", "transformers"} & imports) and not ({"CDLL", "LoadLibrary"} & calls)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--ack", required=True); args = parser.parse_args()
    if args.ack != ACK: return 3
    invocation = invocation_evidence()
    if not invocation_valid(invocation): raise RuntimeError("exact_invocation")
    pre_topology = topology_snapshot()
    if not topology_clean(pre_topology): raise RuntimeError("pre_run_topology")
    runtime = base.runner.collect_runtime()
    if runtime["available"] < 16 * 2**30: raise RuntimeError("start_ram")
    wheel_records = {"psutil": base.verify_wheel_record(base.runner.RUNTIME_FILES["psutil_record"][0]), "numpy": base.verify_wheel_record(base.runner.RUNTIME_FILES["numpy_record"][0])}
    preparation = base.preparation_summary(); runtime_ok, runtime_rejected = base.runtime_mutations(runtime); transaction = transaction_simulation(); lock = json.loads(LOCK.read_text()); observed = {name: sha256(path) for name, path in CHAIN.items()}
    checks = {"exact_invocation": invocation_valid(invocation) and invocation_mutations(invocation) == ["c_trampoline", "wrong_script", "swapped_flags", "extra_interpreter_flag", "wrong_ack", "extra_application_arg", "raw_commandline", "win32_parsed"], "hash_bindings": all(lock.get(name) == digest for name, digest in observed.items()), "closed_pending": lock.get("kind") == "ph1_intel_execution_r8p1_lock" and lock.get("execution_open") is False and lock.get("audit_token") == "PENDING", "runtime_lock": all(lock.get(name) == value for name, value in base.runner.LOCK_STATIC.items()), "exact_runtime": base.runner.validate_runtime(runtime), "start_ram": runtime["available"] >= 16 * 2**30, "wheel_records": wheel_records["psutil"]["hashed_files_verified"] == 17 and wheel_records["numpy"]["hashed_files_verified"] == 899, "runtime_mutations": runtime_ok and len(runtime_rejected) == 10, "r7d1_failure": base.runner.prior_failure_valid(), "cpu_preparation": base.validate_preparation(preparation), "transaction_simulation": all(transaction.values()), "static_no_device": static_contract(), "pre_run_topology": topology_clean(pre_topology), "base_clean": base.runner.clean_now()}
    output = {"kind": "ph1_intel_execution_r8p1_static_preflight", "ack": ACK, "invocation": invocation, "pre_run_topology": pre_topology, "checks": checks, "pass": all(checks.values()), "passed": sum(value is True for value in checks.values()), "total": len(checks), "runtime": runtime, "wheel_records": wheel_records, "preparation": preparation, "preparation_digest": base.sha_bytes(base.canon(preparation)), "transaction_simulation": transaction, "runtime_mutations_rejected": runtime_rejected, "no_compiler_device": True, "cpu_payload_read": True}
    publish_bundle(output, RESULT, MANIFEST, COMMIT); print(json.dumps(output, indent=2)); return 0 if output["pass"] else 3


if __name__ == "__main__": raise SystemExit(main())
