#!/usr/bin/env python3
"""Closed R8P: exact venv plus CPU preparation equivalence; no compiler/device."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import csv
import hashlib
import io
import json
import os
import struct
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))
import run_het_next_l0_ph1_intel_execution_r8 as runner

LOCK = R / "het_next_l0_ph1_intel_execution_r8_lock.json"
RESULT = R / "het_next_l0_ph1_intel_execution_r8_static_preflight.json"
VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r8p_independent_verification.json"
CPU_DIR = R / "het_next_l0_ph1_cpu_freeze_r2"
CPU_RESULT = CPU_DIR / "cpu_stage_freeze.json"
CPU_RAW = CPU_DIR / "cpu_stage_freeze.safetensors"
ACK = "PH1_INTEL_EXECUTION_R8P_EXACT_VENV_CPU_PREPARATION_CLOSED"
EXPECTED_STAGE = {
    "gate": "e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867",
    "up": "f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08",
    "silu": "a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8",
    "activation": "762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f",
    "down": "142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc",
}
EXPECTED_PREPARATION_DIGEST = "f5a15db125c7a69357574111bd9549c36ae74b67af12205fc71a99a4c8962a49"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canon(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_create(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".inprogress." + uuid.uuid4().hex)
    try:
        with temp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def verify_wheel_record(record_path: Path) -> dict:
    site = record_path.parent.parent
    rows = list(csv.reader(io.StringIO(record_path.read_text(encoding="utf-8"))))
    checked = 0
    skipped_cache = 0
    unhashed = []
    for relative, encoded, declared_size in rows:
        normalized = relative.replace("\\", "/")
        if "/__pycache__/" in "/" + normalized or normalized.endswith((".pyc", ".pyo")):
            skipped_cache += 1
            continue
        path = (site / Path(*normalized.split("/"))).resolve()
        if not encoded:
            unhashed.append(relative)
            continue
        algorithm, digest_text = encoded.split("=", 1)
        expected = base64.urlsafe_b64decode(digest_text + "=" * (-len(digest_text) % 4)).hex()
        if algorithm != "sha256" or not path.is_file() or path.stat().st_size != int(declared_size) or sha256(path) != expected:
            raise RuntimeError("wheel_record:" + relative)
        checked += 1
    if unhashed != [record_path.relative_to(site).as_posix()] or checked == 0:
        raise RuntimeError("wheel_record_schema")
    return {"record": str(record_path.resolve()), "rows": len(rows), "hashed_files_verified": checked, "cache_rows_excluded": skipped_cache, "unhashed_rows": unhashed}


def expected_controls() -> list[dict]:
    zero = {name: 0 for name in ("opencl_load", "context", "program", "kernel", "allocation", "launch")}
    rows = []
    for record in ("gate", "up", "down"):
        for control, observed in (
            ("truncation", "size"), ("wrong_projection", "identity"), ("stale_crc", "crc"),
            ("code_mutation", "canonical_digest"), ("scale_mutation", "canonical_digest"),
            ("field31", "field31"), ("wrong_input", "input_digest"),
        ):
            rows.append({"record": record, "control": control, "expected": observed, "observed": observed, "pass": True, "predevice_counts": dict(zero)})
    rows.append({"record": "global", "control": "wrong_lut_digest", "expected": "lut_digest", "observed": "lut_digest", "pass": True, "presented_sha256": "232ab9aa614031d930e40adca3b5424edfd0ca29fa5f32225796305001d8f2a8", "expected_sha256": "a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed", "predevice_counts": dict(zero)})
    return rows


def read_safetensors(path: Path) -> dict[str, dict]:
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + length])
    base = 8 + length
    result = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        data = raw[base + start:base + end]
        result[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "bytes": len(data), "sha256": sha_bytes(data), "data": data}
    return result


def preparation_summary() -> dict:
    # Imports occur only after exact runtime/RAM/RECORD gates in main.
    import het_next_l0_ph1_intel_execution_r6_common as common
    import verify_het_next_l0_ph1_intel_execution_r7a as independent

    records, input_bytes, lut, controls = common.package()
    independent_records = {}
    weights = {}
    for spec in independent.SPECS:
        source = independent.rr(independent.SHARD, spec[3][0], spec[3][1] - spec[3][0])
        independent_records[spec[0]], weights[spec[0]] = independent.codec(source, spec)
    independent_controls = independent.rebuild_controls(independent_records, lut)
    iw = independent.np.frombuffer(input_bytes, "<u2")
    gate = independent.linear(weights["gate"], iw)
    up = independent.linear(weights["up"], iw)
    silu = independent.np.frombuffer(lut, "<u2")[gate]
    activation = independent.np.asarray([independent.mul(int(a), int(b)) for a, b in zip(silu, up, strict=True)], independent.np.uint16)
    down = independent.linear(weights["down"], activation)
    stage_data = {"gate": gate.astype("<u2").tobytes(), "up": up.astype("<u2").tobytes(), "silu": silu.astype("<u2").tobytes(), "activation": activation.astype("<u2").tobytes(), "down": down.astype("<u2").tobytes()}
    raw = read_safetensors(CPU_RAW)
    cpu = json.loads(CPU_RESULT.read_text())
    expected_raw = {"gate": "cpu_q5_gate", "up": "cpu_q5_up", "silu": "cpu_q5_silu", "activation": "cpu_q5_activation", "down": "cpu_q5_down"}
    if not (
        records == independent_records and controls == independent_controls == expected_controls()
        and all(raw[expected_raw[name]]["data"] == data and sha_bytes(data) == EXPECTED_STAGE[name] for name, data in stage_data.items())
        and raw["natural_input"]["data"] == input_bytes
        and cpu.get("environment", {}).get("numpy") == "2.2.6"
        and cpu.get("environment", {}).get("python") == "3.12.10"
        and cpu.get("stage_hashes", {}).get("natural_input") == common.INPUT[2]
        and all(cpu.get("stage_hashes", {}).get("cpu_q5_" + name) == digest for name, digest in EXPECTED_STAGE.items())
    ):
        raise RuntimeError("cpu_preparation_equivalence")
    summary = {
        "records": {spec[0]: {"bytes": len(records[spec[0]]), "sha256": sha_bytes(records[spec[0]]), "shape": list(spec[2])} for spec in common.SPECS},
        "input": {"bytes": len(input_bytes), "sha256": sha_bytes(input_bytes), "dtype": "BF16", "shape": [2048]},
        "lut": {"bytes": len(lut), "sha256": sha_bytes(lut), "dtype": "BF16", "shape": [65536]},
        "stages": {name: {"bytes": len(data), "sha256": sha_bytes(data), "dtype": "BF16", "shape": [len(data) // 2]} for name, data in stage_data.items()},
        "controls": controls,
        "cpu_evidence": {"result_sha256": sha256(CPU_RESULT), "raw_sha256": sha256(CPU_RAW), "manifest_sha256": sha256(CPU_DIR / "manifest.json"), "commit_sha256": sha256(CPU_DIR / "commit.json"), "verification_sha256": sha256(R / "het_next_l0_ph1_cpu_freeze_r2_independent_verification.json")},
    }
    return summary


def expected_preparation() -> dict:
    expected_records = {
        "gate": {"bytes": 675840, "sha256": "e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9", "shape": [512, 2048]},
        "up": {"bytes": 675840, "sha256": "6da7025af27de06c4f6011ddfc82672263b6f0593b2dcacf77705a443f44fbfb", "shape": [512, 2048]},
        "down": {"bytes": 675840, "sha256": "bd1a8ef9ae689fefebf73408f3985c96a0725670dc0b0f7f46268a5a89d12157", "shape": [2048, 512]},
    }
    expected_stages = {name: {"bytes": 4096 if name == "down" else 1024, "sha256": digest, "dtype": "BF16", "shape": [2048 if name == "down" else 512]} for name, digest in EXPECTED_STAGE.items()}
    return {
        "records": expected_records,
        "input": {"bytes": 4096, "sha256": "5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f", "dtype": "BF16", "shape": [2048]},
        "lut": {"bytes": 131072, "sha256": "a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed", "dtype": "BF16", "shape": [65536]},
        "stages": expected_stages,
        "controls": expected_controls(),
        "cpu_evidence": {
            "result_sha256": "520b19d320cf88c71c5c972d0cb3b7ad8b5e29ed152f155909daba9e1d442090",
            "raw_sha256": "c2fbc4d6c3c400ecb0ac7af36b36c88a1c8122d3066cb123430f934bd750d6a8",
            "manifest_sha256": "63f6c842f377fb18738d6016b133c7529803581d0cd661739c0ffd648a82ac54",
            "commit_sha256": "f3677e9610bea03649fec172b97c0c314f2f2e4c0d40bf9d864df0ec88a44f06",
            "verification_sha256": "1c7f2772fb637485020be00f74b6f9295a18ec3d7d10af0587ea350e8756cbc8",
        },
    }


def validate_preparation(summary: dict) -> bool:
    return summary == expected_preparation() and sha_bytes(canon(summary)) == EXPECTED_PREPARATION_DIGEST


def runtime_mutations(actual: dict) -> tuple[bool, list[str]]:
    mutations = {
        "python_path": lambda x: x.__setitem__("python_executable", "C:/wrong/python.exe"),
        "python_hash": lambda x: x.__setitem__("python_sha256", "0" * 64),
        "isolation": lambda x: x.__setitem__("isolated", 0),
        "bytecode": lambda x: x.__setitem__("dont_write_bytecode", 0),
        "pyvenv": lambda x: x["runtime_files"]["pyvenv"].__setitem__("sha256", "0" * 64),
        "psutil_native": lambda x: x["runtime_files"]["psutil_native"].__setitem__("sha256", "0" * 64),
        "psutil_record": lambda x: x["runtime_files"]["psutil_record"].__setitem__("bytes", 0),
        "numpy_version": lambda x: x.__setitem__("numpy_version", "2.4.4"),
        "numpy_record": lambda x: x["runtime_files"]["numpy_record"].__setitem__("path", "C:/wrong/RECORD"),
        "ram": lambda x: x.__setitem__("available", 0),
    }
    rejected = []
    for name, mutate in mutations.items():
        candidate = copy.deepcopy(actual)
        mutate(candidate)
        if not runner.validate_runtime(candidate):
            rejected.append(name)
    return runner.validate_runtime(actual) and set(rejected) == set(mutations), rejected


def static_contract() -> bool:
    runner_tree = ast.parse(Path(runner.__file__).read_text())
    authorize = next(node for node in runner_tree.body if isinstance(node, ast.FunctionDef) and node.name == "authorize")
    first = authorize.body[0]
    bootstrap = isinstance(first, ast.Assign) and isinstance(first.value, ast.Call) and isinstance(first.value.func, ast.Name) and first.value.func.id == "collect_runtime"
    r7d1_calls = [node for node in ast.walk(authorize) if isinstance(node, ast.Call) and any("r7d1" in getattr(part, "id", "").lower() for part in ast.walk(node.func))]
    current = ast.parse(Path(__file__).read_text())
    imports = {alias.name.split(".")[0] for node in current.body if isinstance(node, ast.Import) for alias in node.names}
    calls = {node.func.id for node in ast.walk(current) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    return bootstrap and not r7d1_calls and not ({"pyopencl", "cupy", "torch", "safetensors", "transformers"} & imports) and not ({"WinDLL", "CDLL", "LoadLibrary"} & calls)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", required=True)
    args = parser.parse_args()
    if args.ack != ACK:
        return 3
    runtime = runner.collect_runtime()
    if runtime["available"] < 16 * 2**30:
        raise RuntimeError("start_ram")
    mutations_ok, rejected = runtime_mutations(runtime)
    wheel_records = {
        "psutil": verify_wheel_record(runner.RUNTIME_FILES["psutil_record"][0]),
        "numpy": verify_wheel_record(runner.RUNTIME_FILES["numpy_record"][0]),
    }
    preparation = preparation_summary()
    lock = json.loads(LOCK.read_text())
    observed = {name: sha256(path) for name, path in runner.CHAIN.items()}
    checks = {
        "hash_bindings": all(lock.get(name) == digest for name, digest in observed.items()),
        "closed_pending": lock.get("kind") == "ph1_intel_execution_r8_lock" and lock.get("execution_open") is False and lock.get("audit_token") == "PENDING",
        "runtime_lock_contract": all(lock.get(name) == value for name, value in runner.LOCK_STATIC.items()),
        "exact_isolated_runtime": runner.validate_runtime(runtime),
        "start_ram_16gib": runtime["available"] >= 16 * 2**30,
        "full_wheel_records": wheel_records["psutil"] == {"record": str(runner.RUNTIME_FILES["psutil_record"][0].resolve()), "rows": 28, "hashed_files_verified": 17, "cache_rows_excluded": 10, "unhashed_rows": ["psutil-7.2.2.dist-info/RECORD"]} and wheel_records["numpy"] == {"record": str(runner.RUNTIME_FILES["numpy_record"][0].resolve()), "rows": 1311, "hashed_files_verified": 899, "cache_rows_excluded": 411, "unhashed_rows": ["numpy-2.2.6.dist-info/RECORD"]},
        "runtime_mutations": mutations_ok and len(rejected) == 10,
        "immutable_r7d1_failure": runner.prior_failure_valid(),
        "cpu_preparation_equivalence": validate_preparation(preparation),
        "static_bootstrap_no_device": static_contract(),
        "clean_state": runner.clean_now(),
        "result_absent": not RESULT.exists() and not VERIFY_RESULT.exists() and not any(R.glob(RESULT.name + ".inprogress.*")),
    }
    output = {
        "kind": "ph1_intel_execution_r8p_static_preflight", "ack": ACK, "checks": checks,
        "pass": all(checks.values()), "passed": sum(value is True for value in checks.values()), "total": len(checks),
        "no_compiler_device": True, "cpu_payload_read": True, "runtime": runtime, "wheel_records": wheel_records,
        "preparation": preparation, "preparation_digest": sha_bytes(canon(preparation)),
        "rejected_runtime_mutations": rejected, "r7d1_failure_sha256": sha256(runner.R7D1_FAILURE),
    }
    atomic_create(RESULT, canon(output))
    print(json.dumps(output, indent=2))
    return 0 if output["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
