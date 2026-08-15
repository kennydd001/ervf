#!/usr/bin/env python3
"""Corrected R2P static preflight with executable emulator and transaction tests."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
REPORTS = ROOT / "reports/streamq5_moe"
PREFLIGHT = Path(__file__)
REVISION = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2P_PREFLIGHT_REVISION_2026-08-14.md"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2p_lock.json"
SOURCE_MODULE = SCRIPTS / "het_next_l0_ph1_intel_compile_r2_source.py"
BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r2_backend.py"
RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r2.py"
R2_PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r2.py"
R2_PREREG = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2_PREREGISTRATION_2026-08-14.md"
R2_DESIGN = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2_SOURCE_REVISION_2026-08-14.md"
R2_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2_lock.json"
R1_BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r1_backend.py"
R1B_FAILURE = REPORTS / "het_next_l0_ph1_intel_compile_r1b_failed_attempts/attempt_failure_06df3c72c9c44379a04d39b43d301b53/failure.json"
OUT = REPORTS / "het_next_l0_ph1_intel_compile_r2p"
RESULT = REPORTS / "het_next_l0_ph1_intel_compile_r2p_static_preflight.json"
SOURCE_SHA = "f1b3ccdae6d202ed210810e3cd419f726ea89ffa8fba0c84df5c2bfca3a84d21"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_constant(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("constant:" + name)


def derive_r2() -> str:
    original = extract_constant(SCRIPTS / "het_next_l0_ph1_intel_backend.py", "SRC")
    r1 = original.replace("#pragma OPENCL EXTENSION cl_intel_required_sub_group_size : enable\n#pragma OPENCL EXTENSION cl_khr_int64 : enable\n", "#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n")
    return r1.replace("#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n", "", 1).replace("ulong half =", "ulong halfway =", 1).replace("remainder > half || (remainder == half", "remainder > halfway || (remainder == halfway", 1)


def emulate(a: int, b: int) -> tuple[int, int]:
    sign = (a ^ b) & 0x8000
    ae, be, af, bf = (a >> 7) & 255, (b >> 7) & 255, a & 127, b & 127
    if ae == 255 or be == 255:
        return 0xFFFF, 0
    if (ae == 0 and af == 0) or (be == 0 and bf == 0):
        return sign, 1
    an, bn = (af if ae == 0 else 128 + af), (bf if be == 0 else 128 + bf)
    exponent = (-133 if ae == 0 else ae - 134) + (-133 if be == 0 else be - 134)
    number = an * bn
    highest = number.bit_length() - 1

    def round_even(value: int, shift: int) -> int:
        if shift <= 0:
            return value << -shift
        if shift >= 64:
            return 0
        quotient = value >> shift
        remainder = value & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        return quotient + int(remainder > halfway or (remainder == halfway and (quotient & 1)))

    if highest + exponent > 127:
        return 0xFFFF, 0
    if highest + exponent >= -126:
        shift = highest - 7
        significand = round_even(number, shift)
        if significand == 256:
            significand, shift = 128, shift + 1
        unbiased = exponent + shift + 7
        if unbiased > 127:
            return 0xFFFF, 0
        return sign | ((unbiased + 127) << 7) | (significand & 127), 1
    fraction = round_even(number, -133 - exponent)
    if fraction == 0:
        return sign, 1
    if fraction >= 128:
        return sign | 0x0080, 1
    return sign | fraction, 1


def source_contract(source: str, exact: bool = True) -> bool:
    gates = [
        re.search(r"\bulong\s+half\b", source) is None,
        len(re.findall(r"\bhalfway\b", source)) == 3,
        "cl_intel_required_subgroup_size : enable" not in source,
        source.count("cl_intel_subgroups : enable") == 1,
        source.count("intel_reqd_sub_group_size(8)") == 3,
        "remainder > halfway || (remainder == halfway && (quotient & 1UL))" in source,
        all(source.count("void " + name + "(") == 1 for name in ("gate_linear", "up_linear", "activation", "down_linear")),
        all(token in source for token in ("float partial[32]", "float partial[8]", "for (int distance = 16", "for (int distance = 4")),
    ]
    if exact:
        gates.extend((len(source.encode()) == 7852, hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA))
    return all(gates)


def emulator_and_mutations() -> bool:
    vectors = {
        (0x0000, 0x3F80): (0x0000, 1), (0x8000, 0x3F80): (0x8000, 1),
        (0x0000, 0xBF80): (0x8000, 1), (0x8000, 0xBF80): (0x0000, 1),
        (0x3F80, 0x3F80): (0x3F80, 1), (0xBF80, 0x3F80): (0xBF80, 1),
        (0x3F00, 0x4000): (0x3F80, 1), (0x0001, 0x3F80): (0x0001, 1),
        (0x0001, 0x3F00): (0x0000, 1), (0x0003, 0x3F00): (0x0002, 1),
        (0x7F7F, 0x0001): (0x3CFF, 1),
    }
    if any(emulate(a, b) != expected for (a, b), expected in vectors.items()):
        return False
    source = derive_r2()
    mutations = (
        source.replace("ulong halfway =", "ulong half =", 1),
        source.replace("#define CODE_BYTES", "#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n#define CODE_BYTES", 1),
        source.replace("remainder > halfway ||", "remainder >= halfway ||", 1),
        source.replace("void gate_linear(", "void gate_linear_broken(", 1),
        source.replace("for (int distance = 16", "for (int distance = 15", 1),
    )
    return source_contract(source) and all(not source_contract(mutated, exact=False) for mutated in mutations)


def ast_contract() -> bool:
    trees = [ast.parse(path.read_text(encoding="utf-8"), filename=str(path)) for path in (BACKEND, RUNNER, SOURCE_MODULE)]
    forbidden_imports = {"torch", "cupy", "safetensors", "transformers", "mmap", "numpy"}
    imports = set()
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    runner_functions = {node.name for node in trees[1].body if isinstance(node, ast.FunctionDef)}
    return not (imports & forbidden_imports) and {"configure_base", "verify_bundle", "authorization", "build", "main"} <= runner_functions and all(token in R1_BACKEND.read_text(encoding="utf-8") for token in ("CL_PROGRAM_NUM_DEVICES", "empty_program_binary", "binary_query_read_length_mismatch"))


def transaction_simulation() -> bool:
    spec = importlib.util.spec_from_file_location("ph1_r2_runner_static", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="ph1_r2p_") as temporary:
        root = Path(temporary)
        out, failed, quarantine = root / "out", root / "failed", root / "quarantine"
        module.OUT, module.FAILED, module.QUARANTINE = out, failed, quarantine
        module.configure_base()
        module.base.verify_bundle = module.verify_bundle
        stale = root / "out.dead.inprogress"
        stale.mkdir()
        (stale / "partial").write_bytes(b"x")
        try:
            module.base.recover()
            return False
        except RuntimeError:
            pass
        if stale.exists() or len(list(quarantine.glob("stale_temp_*"))) != 1:
            return False
        attempt = root / "out.fresh.inprogress"
        attempt.mkdir()
        binary = b"R2P-nonempty"
        compiled = {"source": derive_r2(), "binary_hex": binary.hex(), "build_log_hex": b"static".hex(), "binary_nonempty": True, "queried_program_devices": 1, "declared_binary_bytes": len(binary), "read_binary_bytes": len(binary), "binary_sha256": hashlib.sha256(binary).hexdigest(), "cleanup_errors": [], "payload_read": False, "queues_created": 0, "kernels_created": 0, "events_created": 0, "memory_objects_created": 0, "allocations": 0, "kernels_launched": 0}
        module.build(attempt, {"static": True}, compiled)
        module.base.durable_move(attempt, out)
        if not module.verify_bundle(out)["result"]["positive"] or not module.base.recover()["already_complete"]:
            return False
        (out / "result.json").write_bytes(b"corrupt")
        try:
            module.base.recover()
            return False
        except RuntimeError:
            pass
        if out.exists() or len(list(quarantine.glob("corrupt_final_*"))) != 1:
            return False
        partial = root / "failed.inprogress"
        partial.mkdir()
        module.base.archive(failed, "attempt_failure", {"kind": "injected"}, partial)
        rows = list(failed.glob("attempt_failure_*"))
        return len(rows) == 1 and (rows[0] / "failure.json").is_file() and not partial.exists()


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    observed = {"preflight_sha256": sha(PREFLIGHT), "revision_sha256": sha(REVISION), "source_module_sha256": sha(SOURCE_MODULE), "backend_sha256": sha(BACKEND), "runner_sha256": sha(RUNNER), "r2_preflight_sha256": sha(R2_PREFLIGHT), "r2_prereg_sha256": sha(R2_PREREG), "r2_design_sha256": sha(R2_DESIGN), "r2_closed_lock_sha256": sha(R2_LOCK), "r1b_failure_sha256": sha(R1B_FAILURE), "source_sha256": hashlib.sha256(derive_r2().encode()).hexdigest()}
    tests = {
        "closed_and_self_bound": lock.get("execution_open") is False and lock.get("audit_token") == "PENDING" and all(lock.get(key) == value for key, value in observed.items()),
        "lexical_source_contract": source_contract(derive_r2()),
        "executed_emulator_and_mutations": emulator_and_mutations(),
        "ast_compile_transaction_contract": ast_contract(),
        "actual_r2_transaction_simulation": transaction_simulation(),
        "physical_output_absent": not OUT.exists(),
        "preflight_result_absent": not RESULT.exists(),
    }
    result = {"kind": "het_next_l0_ph1_intel_compile_r2p_static_preflight", "tests": tests, "pass": all(tests.values()), "passed": sum(tests.values()), "total": len(tests), "compiler_calls": 0, "device_calls": 0, "payload_reads": 0}
    if RESULT.exists():
        raise FileExistsError(RESULT)
    with RESULT.open("xb") as handle:
        handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
