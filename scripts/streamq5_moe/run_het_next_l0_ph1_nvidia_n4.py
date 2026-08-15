#!/usr/bin/env python3
"""Closed standalone PH1 NVIDIA N4 compile/physical runner."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import het_next_l0_ph1_nvidia_n4_common as common
from het_next_l0_ph1_nvidia_n4_backend import CompilerFailure, DriverBackend, DriverFailure, CUDA_SOURCE, OPTIONS, compile_one_program, file_sha
from het_next_l0_ph1_nvidia_n4_transaction import atomic_failure, canonical, clean_or_quarantine, publish_bundle, verify_bundle

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
LOCK = REPORTS / "het_next_l0_ph1_nvidia_n4_source_lock.json"
VERIFY = ROOT / "scripts/streamq5_moe/verify_het_next_l0_ph1_nvidia_n4.py"
COMPILE_OUT = REPORTS / "het_next_l0_ph1_nvidia_n4_compile"
PHYSICAL_OUT = REPORTS / "het_next_l0_ph1_nvidia_n4_physical"
COMPILE_FAILURE = REPORTS / "het_next_l0_ph1_nvidia_n4_compile_failures"
PHYSICAL_FAILURE = REPORTS / "het_next_l0_ph1_nvidia_n4_physical_failures"
QUARANTINE = REPORTS / "het_next_l0_ph1_nvidia_n4_quarantine"
COMPILE_KIND = "het_next_l0_ph1_nvidia_n4_compile"
PHYSICAL_KIND = "het_next_l0_ph1_nvidia_n4_physical"
ACK = {"compile": "ACK-PH1-NVIDIA-N4-COMPILE-ONE-PROGRAM", "physical": "ACK-PH1-NVIDIA-N4-PHYSICAL-ONE-ATTEMPT"}


def validate_authorization(phase: str, acknowledgement: str):
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("kind") != "het_next_l0_ph1_nvidia_n4_source_lock" or lock.get("revision") != "N4":
        raise RuntimeError("lock_kind")
    observed = {}
    for name, item in lock.get("bindings", {}).items():
        path = Path(item["path"]); path = path if path.is_absolute() else ROOT / path
        observed[name] = common.file_sha(path)
        if observed[name] != item["sha256"]:
            raise RuntimeError("hash_drift:" + name)
    if lock.get(phase + "_open") is not True or lock.get(phase + "_token") != ACK[phase] or acknowledgement != ACK[phase]:
        raise PermissionError("execution_closed")
    if phase == "compile" and any(lock.get(name) is True for name in ("capability_open", "physical_open")):
        raise RuntimeError("phase_separation")
    if phase == "physical":
        required = lock.get("compiled_artifacts", {})
        if set(required) != {"result.json", "manifest.json", "commit.json", "source.cu", "ptx.bin", "cubin.bin", "build.log", "cuobjdump.txt", "nvdisasm.txt"}:
            raise RuntimeError("compiled_artifact_lock")
        for name, expected in required.items():
            path = COMPILE_OUT / name
            if common.file_sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise RuntimeError("compiled_artifact_drift:" + name)
    return lock, observed


def independent(bundle: Path, mode: str):
    command = [sys.executable, "-I", "-B", str(VERIFY.resolve()), "--candidate-bundle", str(bundle.resolve()), "--mode", mode, "--no-write"]
    completed = subprocess.run(command, cwd=str(ROOT), stdin=subprocess.DEVNULL, capture_output=True, timeout=7200, check=False)
    if completed.returncode != 0:
        return False
    try:
        row = json.loads(completed.stdout)
    except Exception:
        return False
    return row.get("pass") is True and row.get("mode") == mode and row.get("candidate_bundle") == str(bundle.resolve())


def disassemble(cubin: bytes, lock):
    tools = lock.get("compiler_tools", {})
    if set(tools) != {"cuobjdump", "nvdisasm"}:
        raise RuntimeError("compiler_tools_pending")
    outputs = {}
    with tempfile.TemporaryDirectory(prefix="ph1_nvidia_n4_cubin_") as directory:
        cubin_path = Path(directory) / "program.cubin"; cubin_path.write_bytes(cubin)
        for name, arguments in (("cuobjdump", ("--dump-sass",)), ("nvdisasm", tuple())):
            tool = Path(tools[name]["path"])
            if common.file_sha(tool) != tools[name]["sha256"]:
                raise RuntimeError("tool_drift:" + name)
            command = [str(tool), *arguments, str(cubin_path)]
            completed = subprocess.run(command, cwd=directory, stdin=subprocess.DEVNULL, capture_output=True, timeout=120, check=False)
            if completed.returncode != 0 or completed.stderr:
                raise RuntimeError(f"{name}_failure:{completed.returncode}:{completed.stderr[:256]!r}")
            outputs[name + ".txt"] = completed.stdout
    return outputs


def compile_phase(lock, observed):
    state = clean_or_quarantine(COMPILE_OUT, COMPILE_FAILURE, QUARANTINE, COMPILE_KIND)
    if state == "already_complete":
        print(json.dumps({"status": state, "path": str(COMPILE_OUT)})); return 0
    source = CUDA_SOURCE.read_bytes()
    try:
        evidence = compile_one_program(source); artifacts = evidence.pop("artifacts")
        ptx, cubin, log = artifacts["ptx"].pop("data"), artifacts["cubin"].pop("data"), artifacts["log"].pop("data")
        disassembly = disassemble(cubin, lock)
        result = {"kind": COMPILE_KIND, "status": "compile_positive", "positive": True, "authorization": {"lock_sha256": common.file_sha(LOCK), "observed": observed}, "compiler": evidence, "artifact_manifest": {"source.cu": {"bytes": len(source), "sha256": common.sha(source)}, "ptx.bin": {"bytes": len(ptx), "sha256": common.sha(ptx), "label": "PTX_from_sm_120_targeted_compile"}, "cubin.bin": {"bytes": len(cubin), "sha256": common.sha(cubin)}, "build.log": {"bytes": len(log), "sha256": common.sha(log)}, **{name: {"bytes": len(data), "sha256": common.sha(data)} for name, data in disassembly.items()}}, "cudart_loaded": False, "runtime_version": "not_applicable_driver_api_only", "claim": "compile_provenance_only_no_device_result"}
        extras = {"source.cu": source, "ptx.bin": ptx, "cubin.bin": cubin, "build.log": log, **disassembly}
        publish_bundle(COMPILE_OUT, result, COMPILE_KIND, lambda path: independent(path, "compile"), extras, QUARANTINE)
    except Exception as exc:
        partial = exc.evidence if isinstance(exc, CompilerFailure) else {}
        path = atomic_failure(COMPILE_FAILURE, {"kind": COMPILE_KIND + "_failure", "stage": "compile", "error_type": type(exc).__name__, "error": str(exc), "device_opened": False, "compiler_opened": True, "partial": partial, "cleanup": partial.get("ledger", []), "traceback": traceback.format_exc(limit=20), "disposition": "bounded_create_new"})
        print(json.dumps({"status": "compile_failure", "failure": str(path)})); return 3
    print(json.dumps({"status": "compile_positive", "path": str(COMPILE_OUT)})); return 0


def _counter_positive(data: bytes, count: int):
    return len(data) == count * 4 and all(value == 1 for value in common.struct.unpack("<" + "I" * count, data))


def physical_phase(lock, observed):
    state = clean_or_quarantine(PHYSICAL_OUT, PHYSICAL_FAILURE, QUARANTINE, PHYSICAL_KIND)
    if state == "already_complete":
        print(json.dumps({"status": state, "path": str(PHYSICAL_OUT)})); return 0
    resources = []; evidence = None
    try:
        resources.append(common.host_sample("process_start"))
        if resources[0]["available"] < 16 * 2**30: raise RuntimeError("start_ram")
        resources.append(common.host_sample("post_authorization")); package = common.prepare_package(); resources.append(common.host_sample("post_cpu_package")); resources.append(common.host_sample("post_controls")); resources.append(common.host_sample("pre_cuda_init"))
        backend = DriverBackend(); evidence = backend.run(package["records"], package["input"], package["lut"], (COMPILE_OUT / "cubin.bin").read_bytes(), resources)
        outputs = {name: bytes.fromhex(value) for name, value in evidence["outputs"].items()}; stage_equal = {name: outputs.get(name) == value and common.sha(value) == common.STAGE_SHA[name] for name, value in package["oracle"].items()}
        counter_equal = {"gate_counters": _counter_positive(outputs.get("gate_counters", b""), 512), "up_counters": _counter_positive(outputs.get("up_counters", b""), 512), "activation_counters": _counter_positive(outputs.get("activation_counters", b""), 512), "down_counters": _counter_positive(outputs.get("down_counters", b""), 2048)}
        finite = all(not any(((word >> 7) & 255) == 255 for word in common.struct.unpack("<" + "H" * (len(outputs[name]) // 2), outputs[name])) for name in common.STAGE_SHA)
        device_samples = evidence["resources"]; provisional = {"kind": PHYSICAL_KIND, "status": "provisional", "evidence": evidence}; canonical(provisional); device_samples.append(common.host_sample("post_serialization"))
        resource_ok = len(device_samples) == 14 and [row["stage"] for row in device_samples] == list(common.RESOURCE_STAGES) and all(row["telemetry_error"] is None and row["available"] >= (16 * 2**30 if index == 0 else 2 * 2**30) and row["peak_wset"] <= 12 * 2**30 for index, row in enumerate(device_samples)) and all(row["device_query_state"] == ("attempted" if 5 <= index <= 11 else "not_attempted") for index, row in enumerate(device_samples)) and device_samples[11]["device_free_bytes"] >= device_samples[6]["device_free_bytes"] - 64 * 2**20 and device_samples[12].get("driver_context_calls_after_primary_release") == 0
        ledger = evidence["ledger"]
        operation_codes = all(row.get("code") == 0 for row in ledger if "code" in row)
        schedule_ok = (sum(row["op"] == "memset" for row in ledger) == 9 and
                       sum(row["op"] == "h2d" for row in ledger) == 5 and
                       sum(row["op"] == "launch" for row in ledger) == 4 and
                       sum(row["op"] == "d2h" for row in ledger) == 9 and
                       sum(row["op"] == "stream_synchronize" for row in ledger) == 1 and
                       sum(row["op"] == "meminfo" for row in ledger) == 7)
        runtime_surface = evidence.get("runtime_modules", {})
        protocol_gates = {
            "finite": finite,
            "controls": len(package["controls"]) == 22 and all(row["pass"] for row in package["controls"]),
            "resources": resource_ok,
            "cleanup": evidence["cleanup_errors"] == [] and evidence["live_owned_resources"] == 0 and evidence["primary_released"] is True,
            "operation_codes": operation_codes,
            "schedule": schedule_ok,
            "abi": len(evidence.get("abi", {})) == 30 and all(set(row) == {"argtypes", "restype"} for row in evidence.get("abi", {}).values()),
            "runtime_surface": runtime_surface.get("forbidden") == [] and runtime_surface.get("nvcuda_exact_count") == 1,
        }
        if not all(protocol_gates.values()):
            failed = sorted(name for name, value in protocol_gates.items() if not value)
            raise RuntimeError("invalid_protocol_gates:" + ",".join(failed))
        numerical_gates = {"stages_exact": all(stage_equal.values()), "counters_exact": all(counter_equal.values())}
        gates = {**protocol_gates, **numerical_gates}
        positive = all(numerical_gates.values()); status = "nvidia_physical_positive" if positive else "nvidia_device_numerical_negative"
        result = {"kind": PHYSICAL_KIND, "status": status, "positive": positive, "terminal_valid": True, "authorization": {"lock_sha256": common.file_sha(LOCK), "observed": observed, "compile_commit_sha256": common.file_sha(COMPILE_OUT / "commit.json")}, "cpu_package": {"prior": package["prior"], "record_sha256": {name: common.sha(data) for name, data in package["records"].items()}, "input_sha256": common.sha(package["input"]), "lut_sha256": common.sha(package["lut"]), "oracle_sha256": {name: common.sha(data) for name, data in package["oracle"].items()}}, "controls": package["controls"], "evidence": evidence, "stage_exact": stage_equal, "counter_exact": counter_equal, "gates": gates, "claim": "one real expert50/input NVIDIA correctness component only"}
        publish_bundle(PHYSICAL_OUT, result, PHYSICAL_KIND, lambda path: independent(path, "physical"), quarantine=QUARANTINE)
    except Exception as exc:
        partial = exc.evidence if isinstance(exc, DriverFailure) else (evidence if isinstance(evidence, dict) else {})
        path = atomic_failure(PHYSICAL_FAILURE, {"kind": PHYSICAL_KIND + "_failure", "stage": "physical", "error_type": type(exc).__name__, "error": str(exc), "device_opened": bool(partial.get("loaded_driver")), "partial": partial, "resources": partial.get("resources", resources), "cleanup_errors": partial.get("cleanup_errors", []), "live_owned_resources": partial.get("live_owned_resources"), "traceback": traceback.format_exc(limit=20), "disposition": "bounded_create_new_after_cleanup"})
        print(json.dumps({"status": "physical_failure", "failure": str(path)})); return 3
    print(json.dumps({"status": status, "path": str(PHYSICAL_OUT), "positive": positive})); return 0 if positive else 3


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("compile", "physical"), required=True); parser.add_argument("--ack", required=True); args = parser.parse_args()
    # A wrong acknowledgement is mutation-free.  Once the exact token is given,
    # every authorization/start failure receives bounded structured evidence.
    if args.ack != ACK[args.phase]:
        raise PermissionError("acknowledgement")
    # Closed, drifted, or otherwise invalid authorization is mutation-free.
    lock, observed = validate_authorization(args.phase, args.ack)
    return compile_phase(lock, observed) if args.phase == "compile" else physical_phase(lock, observed)


if __name__ == "__main__":
    raise SystemExit(main())

