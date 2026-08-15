#!/usr/bin/env python3
"""Executable static PH1-R1 audit; it never imports or opens an OpenCL library."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
REPORTS = ROOT / "reports/streamq5_moe"
BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r1_backend.py"
RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r1.py"
R0_BACKEND = SCRIPTS / "het_next_l0_ph1_intel_backend.py"
PREREG = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R1_PREREGISTRATION_2026-08-14.md"
AUDIT = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R0_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1_lock.json"
OUT = REPORTS / "het_next_l0_ph1_intel_compile_r1"
PREFLIGHT_RESULT = REPORTS / "het_next_l0_ph1_intel_compile_r1_static_preflight.json"
SOURCE_SHA256 = "06be3a9ba863d5f01d4025dc8d5e5679cdcc9827c13db5663c149227f3254528"
AUDIT_SHA256 = "ad1151b2a0a907e99ab0a99a6ac1b426587a14549fc4282821966f912544a841"
ALLOWED_OPENCL = {
    "clBuildProgram",
    "clCreateContext",
    "clCreateProgramWithSource",
    "clGetDeviceIDs",
    "clGetDeviceInfo",
    "clGetPlatformIDs",
    "clGetProgramBuildInfo",
    "clGetProgramInfo",
    "clReleaseContext",
    "clReleaseProgram",
}
FORBIDDEN_OPENCL_PREFIXES = (
    "clCreateCommandQueue",
    "clCreateKernel",
    "clCreateBuffer",
    "clCreateUserEvent",
    "clEnqueue",
    "clFinish",
    "clFlush",
    "clHostMemAllocINTEL",
    "clDeviceMemAllocINTEL",
    "clSharedMemAllocINTEL",
    "clMemFreeINTEL",
    "clSetKernelArg",
    "clGetExtensionFunctionAddress",
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def extract_r0_source() -> str:
    tree = ast.parse(R0_BACKEND.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SRC" for t in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError("r0_src")


def reconstruct_source() -> str:
    original = extract_r0_source()
    wrong = "#pragma OPENCL EXTENSION cl_intel_required_sub_group_size : enable\n#pragma OPENCL EXTENSION cl_khr_int64 : enable\n"
    if original.count(wrong) != 1:
        raise RuntimeError("r0_pragma_contract")
    return original.replace(wrong, "#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n")


def function_source(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise RuntimeError("unbalanced_source")


def emulate_multiply_bf16_exact(a: int, b: int) -> tuple[int, int]:
    """Independent Python execution of the integer operations frozen in the OpenCL routine."""
    sign = (a ^ b) & 0x8000
    ae, be = (a >> 7) & 255, (b >> 7) & 255
    af, bf = a & 127, b & 127
    if ae == 255 or be == 255:
        return 0xFFFF, 0
    if (ae == 0 and af == 0) or (be == 0 and bf == 0):
        return sign, 1
    an, bn = (af if ae == 0 else 128 + af), (bf if be == 0 else 128 + bf)
    ax, bx = (-133 if ae == 0 else ae - 134), (-133 if be == 0 else be - 134)
    number, exponent = an * bn, ax + bx
    highest = number.bit_length() - 1
    top = highest + exponent

    def rse(value: int, shift: int) -> int:
        if shift <= 0:
            return value << -shift
        if shift >= 64:
            return 0
        quotient = value >> shift
        remainder = value & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        return quotient + int(remainder > half or (remainder == half and (quotient & 1)))

    if top > 127:
        return 0xFFFF, 0
    if top >= -126:
        shift = highest - 7
        significand = rse(number, shift)
        if significand == 256:
            significand, shift = 128, shift + 1
        unbiased = exponent + shift + 7
        if unbiased > 127:
            return 0xFFFF, 0
        return sign | ((unbiased + 127) << 7) | (significand & 127), 1
    fraction = rse(number, -133 - exponent)
    if fraction == 0:
        return sign, 1
    if fraction >= 128:
        return sign | 0x0080, 1
    return sign | fraction, 1


def validate_source(source: str, expected_hash: bool = True) -> bool:
    gates = [
        source.count("#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable") == 1,
        "cl_intel_required_sub_group_size" not in source,
        "cl_khr_int64" not in source,
        all(source.count("void " + name + "(") == 1 for name in ("gate_linear", "up_linear", "activation", "down_linear")),
        all(token in source for token in ("intel_reqd_sub_group_size(8)", "float partial[32]", "float partial[8]", "for (int distance = 16", "for (int distance = 4")),
        source.count("inline ushort multiply_bf16_exact") == 1,
        "remainder > half || (remainder == half && (quotient & 1UL))" in source,
        not any(flag in source for flag in ("fast-relaxed-math", "finite-math-only", "unsafe-math", "mad-enable", "ftz")),
    ]
    if expected_hash:
        gates.extend((len(source.encode()) == 7909, sha_bytes(source.encode()) == SOURCE_SHA256))
    return all(gates)


def source_and_mutation_test() -> bool:
    source = reconstruct_source()
    vectors = {
        (0x0000, 0x3F80): (0x0000, 1),
        (0x8000, 0x3F80): (0x8000, 1),
        (0x0000, 0xBF80): (0x8000, 1),
        (0x8000, 0xBF80): (0x0000, 1),
        (0x3F80, 0x3F80): (0x3F80, 1),
        (0xBF80, 0x3F80): (0xBF80, 1),
        (0x3F00, 0x4000): (0x3F80, 1),
        (0x0001, 0x3F80): (0x0001, 1),
        (0x0001, 0x3F00): (0x0000, 1),
        (0x0003, 0x3F00): (0x0002, 1),
        (0x7F7F, 0x0001): (0x3CFF, 1),
    }
    if not validate_source(source) or any(emulate_multiply_bf16_exact(a, b) != expected for (a, b), expected in vectors.items()):
        return False
    routine = function_source(source, "inline ushort multiply_bf16_exact")
    if sha_bytes(routine.encode()) == sha_bytes(b""):
        return False
    mutations = (
        source.replace("cl_intel_required_subgroup_size", "cl_intel_required_sub_group_size", 1),
        source.replace("#define CODE_BYTES", "#pragma OPENCL EXTENSION cl_khr_int64 : enable\n#define CODE_BYTES", 1),
        source.replace("void gate_linear(", "void gate_linear_broken(", 1),
        source.replace("for (int distance = 16", "for (int distance = 15", 1),
        source.replace("remainder > half ||", "remainder >= half ||", 1),
    )
    return all(not validate_source(mutated, expected_hash=False) for mutated in mutations)


def ast_callgraph_and_no_payload() -> bool:
    backend_text = BACKEND.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(backend_text)
    direct_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("cl")
    }
    indirect_release_calls = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in {"clReleaseProgram", "clReleaseContext"}
    }
    calls = direct_calls | indirect_release_calls
    if calls != ALLOWED_OPENCL or any(token in backend_text for token in FORBIDDEN_OPENCL_PREFIXES):
        return False
    forbidden_imports = {"torch", "safetensors", "cupy", "transformers", "numpy", "mmap"}
    forbidden_literals = (".safetensors", "model-", "D2R3", "Q5", "safe_open", "from_pretrained")
    for path, text in ((BACKEND, backend_text), (RUNNER, runner_text)):
        parsed = ast.parse(text, filename=str(path))
        imports = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        if imports & forbidden_imports or any(literal in text for literal in forbidden_literals):
            return False
    return True


def transaction_simulation() -> bool:
    specification = importlib.util.spec_from_file_location("ph1_r1_runner_for_preflight", RUNNER)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="ph1_r1_static_") as temporary:
        root = Path(temporary)
        out = root / "out"
        failed = root / "failed"
        quarantine = root / "quarantine"
        stale = root / "out.deadbeef.inprogress"
        stale.mkdir()
        (stale / "partial").write_bytes(b"x")
        try:
            module.recover_before_device(root, out, failed, quarantine)
            return False
        except module.RecoveryAbort:
            pass
        if stale.exists() or len(list(quarantine.glob("stale_temp_*"))) != 1:
            return False
        attempt = root / "out.fresh.inprogress"
        attempt.mkdir()
        dummy_binary = b"R1-nonempty-binary"
        compiled = {
            "source": reconstruct_source(),
            "binary_hex": dummy_binary.hex(),
            "build_log_hex": b"static-only".hex(),
            "binary_nonempty": True,
            "queried_program_devices": 1,
            "declared_binary_bytes": len(dummy_binary),
            "read_binary_bytes": len(dummy_binary),
            "binary_sha256": sha_bytes(dummy_binary),
            "cleanup_errors": [],
            "payload_read": False,
            "queues_created": 0,
            "kernels_created": 0,
            "events_created": 0,
            "memory_objects_created": 0,
            "allocations": 0,
            "kernels_launched": 0,
        }
        module.build_positive_bundle(attempt, {"started_utc": "static", "test": True}, compiled)
        module.durable_move(attempt, out)
        if not module.verify_bundle(out)["result"]["positive"]:
            return False
        if not module.recover_before_device(root, out, failed, quarantine)["already_complete"]:
            return False
        (out / "result.json").write_bytes(b"corrupt")
        try:
            module.recover_before_device(root, out, failed, quarantine)
            return False
        except module.RecoveryAbort:
            pass
        if out.exists() or len(list(quarantine.glob("corrupt_final_*"))) != 1:
            return False
        partial = root / "attempt.inprogress"
        partial.mkdir()
        module.immutable_failure(failed, "attempt_failure", {"error": "injected"}, partial)
        rows = list(failed.glob("attempt_failure_*"))
        return len(rows) == 1 and (rows[0] / "failure.json").is_file() and not partial.exists()


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    source = reconstruct_source()
    hashes = {
        "backend_sha256": file_sha(BACKEND),
        "runner_sha256": file_sha(RUNNER),
        "preflight_sha256": file_sha(Path(__file__)),
        "prereg_sha256": file_sha(PREREG),
        "source_sha256": sha_bytes(source.encode()),
        "prior_audit_sha256": file_sha(AUDIT),
    }
    tests = {
        "closed_lock": lock.get("execution_open") is False and lock.get("audit_token") == "PENDING",
        "self_and_source_hashes": all(lock.get(key) == value for key, value in hashes.items()) and hashes["prior_audit_sha256"] == AUDIT_SHA256,
        "canonical_source_and_mutations": source_and_mutation_test(),
        "compile_only_ast_callgraph_no_payload": ast_callgraph_and_no_payload(),
        "production_transaction_simulation": transaction_simulation(),
        "binary_gate_source_contract": all(token in BACKEND.read_text(encoding="utf-8") for token in ("CL_PROGRAM_NUM_DEVICES", "empty_program_binary", "binary_query_read_length_mismatch", "declared_binary_bytes")),
        "physical_output_absent": not OUT.exists(),
        "preflight_result_absent": not PREFLIGHT_RESULT.exists(),
    }
    result = {
        "kind": "het_next_l0_ph1_intel_compile_r1_static_preflight",
        "tests": tests,
        "pass": all(tests.values()),
        "passed": sum(tests.values()),
        "total": len(tests),
        "device_calls": 0,
        "compiler_calls": 0,
        "payload_reads": 0,
    }
    if PREFLIGHT_RESULT.exists():
        raise FileExistsError(PREFLIGHT_RESULT)
    data = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
    with PREFLIGHT_RESULT.open("xb") as handle:
        handle.write(data)
        handle.flush()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
