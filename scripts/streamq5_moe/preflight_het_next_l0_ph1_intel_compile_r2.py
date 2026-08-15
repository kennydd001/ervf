#!/usr/bin/env python3
"""Static PH1-R2 source/compile contract preflight; no OpenCL, compiler or payload."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
REPORTS = ROOT / "reports/streamq5_moe"
BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r2_backend.py"
RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r2.py"
SOURCE_MODULE = SCRIPTS / "het_next_l0_ph1_intel_compile_r2_source.py"
PREREG = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2_PREREGISTRATION_2026-08-14.md"
DESIGN = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2_SOURCE_REVISION_2026-08-14.md"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2_lock.json"
R1_PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r1.py"
R1_BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r1_backend.py"
R1B_FAILURE = REPORTS / "het_next_l0_ph1_intel_compile_r1b_failed_attempts/attempt_failure_06df3c72c9c44379a04d39b43d301b53/failure.json"
OUT = REPORTS / "het_next_l0_ph1_intel_compile_r2"
RESULT = REPORTS / "het_next_l0_ph1_intel_compile_r2_static_preflight.json"
SOURCE_SHA = "f1b3ccdae6d202ed210810e3cd419f726ea89ffa8fba0c84df5c2bfca3a84d21"
FAILURE_SHA = "62107b4cee0809fd744bacfe5d6890c7e09ec9002b0b029a6e84c98359f95fbb"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_constant(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("constant:" + name)


def derive_source() -> tuple[str, str]:
    r0 = SCRIPTS / "het_next_l0_ph1_intel_backend.py"
    original = extract_constant(r0, "SRC")
    old = "#pragma OPENCL EXTENSION cl_intel_required_sub_group_size : enable\n#pragma OPENCL EXTENSION cl_khr_int64 : enable\n"
    r1 = original.replace(old, "#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n")
    r2 = r1.replace("#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n", "", 1).replace("ulong half =", "ulong halfway =", 1).replace("remainder > half || (remainder == half", "remainder > halfway || (remainder == halfway", 1)
    return r1, r2


def source_contract(source: str, exact_hash: bool = True) -> bool:
    gates = [
        "cl_intel_required_subgroup_size : enable" not in source,
        "cl_intel_required_sub_group_size" not in source,
        "cl_khr_int64" not in source,
        source.count("cl_intel_subgroups : enable") == 1,
        source.count("intel_reqd_sub_group_size(8)") == 3,
        "ulong half" not in source,
        source.count("halfway") == 3,
        "remainder > halfway || (remainder == halfway && (quotient & 1UL))" in source,
        all(source.count("void " + name + "(") == 1 for name in ("gate_linear", "up_linear", "activation", "down_linear")),
        all(token in source for token in ("float partial[32]", "float partial[8]", "for (int distance = 16", "for (int distance = 4")),
    ]
    if exact_hash:
        gates.extend((len(source.encode()) == 7852, hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA))
    return all(gates)


def mutations_rejected(source: str) -> bool:
    mutations = (
        source.replace("ulong halfway =", "ulong half =", 1),
        source.replace("#define CODE_BYTES", "#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n#define CODE_BYTES", 1),
        source.replace("remainder > halfway ||", "remainder >= halfway ||", 1),
        source.replace("void gate_linear(", "void gate_linear_broken(", 1),
        source.replace("for (int distance = 16", "for (int distance = 15", 1),
    )
    return all(not source_contract(mutated, exact_hash=False) for mutated in mutations)


def callgraph_contract() -> bool:
    r1_tree = ast.parse(R1_BACKEND.read_text(encoding="utf-8"))
    r2_tree = ast.parse(BACKEND.read_text(encoding="utf-8"))
    calls = {node.func.attr for node in ast.walk(r1_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("cl")}
    calls |= {node.value for node in ast.walk(r1_tree) if isinstance(node, ast.Constant) and node.value in {"clReleaseProgram", "clReleaseContext"}}
    allowed = {"clGetPlatformIDs", "clGetDeviceIDs", "clGetDeviceInfo", "clCreateContext", "clCreateProgramWithSource", "clBuildProgram", "clGetProgramBuildInfo", "clGetProgramInfo", "clReleaseProgram", "clReleaseContext"}
    forbidden_imports = {"torch", "cupy", "safetensors", "transformers", "mmap", "numpy"}
    imports = set()
    for tree in (r1_tree, r2_tree, ast.parse(RUNNER.read_text(encoding="utf-8"))):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    return calls == allowed and not (imports & forbidden_imports)


def failure_contract() -> bool:
    failure = json.loads(R1B_FAILURE.read_text(encoding="utf-8"))
    evidence = failure.get("backend_evidence", {})
    ledger = evidence.get("ledger", [])
    releases = [row for row in ledger if row.get("op") == "release"]
    cleanup = ledger[-1] if ledger else {}
    log = bytes.fromhex(evidence.get("build_log_hex", ""))
    return (
        sha(R1B_FAILURE) == FAILURE_SHA
        and failure.get("kind") == "het_next_l0_ph1_intel_compile_r1b_failure"
        and evidence.get("error") == "RuntimeError:program_build:-11"
        and evidence.get("identity", {}).get("driver") == "32.0.101.8517"
        and evidence.get("identity", {}).get("pci") == "0000:00:02.0"
        and evidence.get("identity", {}).get("name") == "Intel(R) Arc(TM) Pro 140T GPU (32GB)"
        and hashlib.sha256(log).hexdigest() == "91383f7935630334a5e0d250c01951645a1ce50c4dfe81aaef7f881529d2df2e"
        and b"ulong half" in log
        and b"unknown OpenCL extension 'cl_intel_required_subgroup_size'" in log
        and releases == [{"code": 0, "name": "program", "op": "release"}, {"code": 0, "name": "context", "op": "release"}]
        and cleanup.get("cleanup_complete") is True
        and cleanup.get("live_owned_resources") == 0
        and evidence.get("payload_read") is False
        and all(evidence.get(key) == 0 for key in ("queues_created", "kernels_created", "events_created", "memory_objects_created", "allocations", "kernels_launched"))
    )


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    r1, r2 = derive_source()
    observed = {"backend_sha256": sha(BACKEND), "runner_sha256": sha(RUNNER), "preflight_sha256": sha(Path(__file__)), "prereg_sha256": sha(PREREG), "design_sha256": sha(DESIGN), "source_module_sha256": sha(SOURCE_MODULE), "r1_backend_sha256": sha(R1_BACKEND), "r1b_failure_sha256": sha(R1B_FAILURE), "source_sha256": hashlib.sha256(r2.encode()).hexdigest()}
    tests = {
        "closed_self_bound_lock": lock.get("execution_open") is False and lock.get("audit_token") == "PENDING" and all(lock.get(key) == value for key, value in observed.items()),
        "exact_r1_to_r2_derivation": hashlib.sha256(r1.encode()).hexdigest() == "06be3a9ba863d5f01d4025dc8d5e5679cdcc9827c13db5663c149227f3254528" and source_contract(r2),
        "source_mutations_rejected": mutations_rejected(r2),
        "bf16_emulator_contract_retained": "emulate_multiply_bf16_exact" in R1_PREFLIGHT.read_text(encoding="utf-8") and "multiply_bf16_exact" in r2,
        "compile_only_callgraph_no_payload": callgraph_contract(),
        "nonempty_binary_gate_retained": all(token in R1_BACKEND.read_text(encoding="utf-8") for token in ("CL_PROGRAM_NUM_DEVICES", "empty_program_binary", "binary_query_read_length_mismatch", "declared_binary_bytes")),
        "r1b_negative_and_clean_lifecycle": failure_contract(),
        "output_and_result_absent": not OUT.exists() and not RESULT.exists(),
    }
    result = {"kind": "het_next_l0_ph1_intel_compile_r2_static_preflight", "tests": tests, "pass": all(tests.values()), "passed": sum(tests.values()), "total": len(tests), "compiler_calls": 0, "device_calls": 0, "payload_reads": 0}
    if RESULT.exists():
        raise FileExistsError(RESULT)
    with RESULT.open("xb") as handle:
        handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
