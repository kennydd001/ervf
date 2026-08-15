from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import het_next_l0_ph0r3_common as common
import diagnose_het_next_l0_ph0x_r4_cuda_compile_staging as r4


RUN = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r5_nvidia_only_real_projection"
RESULT = RUN / "ph0x_r5_result.json"
PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_L0_PH0X_R5_NVIDIA_ONLY_REAL_PROJECTION_PREREGISTRATION_2026-08-13.md"
R3_RESULT = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r3_exploratory_real_projection/ph0x_r3_result.json"
R4_RESULT = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r4_cuda_compile_staging_diagnostic.json"

EXPECTED = {
    Path(common.__file__).resolve(): "899659ada099bd2efe1b95809a169b1f73b887ef31e1eb357d9f55f233121a46",
    Path(r4.__file__).resolve(): "59fd8890485df17b122416c9e6e5953d7909c57478261cab151370407efd45bc",
    R3_RESULT: "e5fea8e2609f11dd294733645c9a4ecb08892c9d2070de33baacbd1a74b0df7c",
    R4_RESULT: "43da909a23d13ba16090d26fac64d255898e988ddcbe28fb21c384b00f8eb77d",
}
EXPECTED_RECORD_SHA = "e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9"
EXPECTED_INPUT_SHA = "5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f"
EXPECTED_CPU_SHA = "e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867"
EXPECTED_CUDA_SOURCE_SHA = "3ede786f3e71b76ee74f2591bde4cbb317a94f05e84bfd3ef5d64c22f6ce8435"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def predevice_gate() -> tuple[dict[str, str], dict[str, object]]:
    observed = {str(path): common.file_digest(path) for path in EXPECTED}
    if any(observed[str(path)] != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("dependency_or_prior_evidence_hash_drift")
    if sha(r4.CUDA_SOURCE.encode()) != EXPECTED_CUDA_SOURCE_SHA:
        raise RuntimeError("cuda_source_hash_drift")
    prior = json.loads(R3_RESULT.read_text(encoding="utf-8"))
    cmp = prior.get("intel_comparison", {})
    ledger = prior.get("intel", {}).get("ledger", [])
    if not (
        prior.get("status") == "exploratory_single_real_projection_failure"
        and prior.get("error", "").startswith("ValueError: memoryview assignment")
        and len(prior.get("controls", [])) == 9
        and all(row.get("pass") is True for row in prior["controls"])
        and prior.get("cpu_output_sha256") == EXPECTED_CPU_SHA
        and cmp.get("different_words") == 0
        and cmp.get("output_sha256") == EXPECTED_CPU_SHA
        and cmp.get("counters_all_one") is True
        and cmp.get("sentinel_overwritten") is True
        and ledger
        and ledger[-1].get("cleanup_complete") is True
        and prior.get("intel", {}).get("identity", {}).get("pci") == "0000:00:02.0"
    ):
        raise RuntimeError("r3_intel_evidence_gate")
    diag = json.loads(R4_RESULT.read_text(encoding="utf-8"))
    if not (
        diag.get("diagnostic_pass") is True
        and diag.get("kernel_launched") is False
        and diag.get("h2d_calls") == 0
        and diag.get("d2h_calls") == 0
        and diag.get("cuda_source_sha256") == EXPECTED_CUDA_SOURCE_SHA
        and diag.get("staging", {}).get("matches_record") is True
        and diag.get("cleanup", {}).get("errors") == []
    ):
        raise RuntimeError("r4_diagnostic_gate")
    return observed, prior


def run_nvidia(record: bytes, input_bytes: bytes) -> dict[str, object]:
    import cupy as cp

    count = cp.cuda.runtime.getDeviceCount()
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    pci = cp.cuda.runtime.deviceGetPCIBusId(0)
    if isinstance(pci, bytes):
        pci = pci.decode()
    if count != 1 or name != "NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU" or pci != "0000:01:00.0":
        raise RuntimeError(f"nvidia_identity:{count}:{name}:{pci}")

    ledger: list[dict[str, object]] = []
    cleanup_errors: list[str] = []
    pinned: list[tuple[str, object, int]] = []
    device: list[tuple[str, int, int]] = []
    stream_ptr = 0
    module = function = external_stream = None
    output = counters = b""
    sizes = (
        ("record", common.RECORD_BYTES),
        ("input", common.INPUT_BYTES),
        ("output", common.ROWS * 2),
        ("counters", common.COUNTER_BYTES),
    )
    try:
        module = cp.RawModule(
            code=r4.CUDA_SOURCE,
            backend="nvrtc",
            options=("--std=c++17", "--fmad=true", "--prec-div=true", "--prec-sqrt=true", "--ftz=false"),
            name_expressions=("ph0",),
        )
        function = module.get_function("ph0")
        ledger.append({"op": "compile", "source_sha256": sha(r4.CUDA_SOURCE.encode()), "success": True})
        stream_ptr = int(cp.cuda.runtime.streamCreateWithFlags(1))
        external_stream = cp.cuda.ExternalStream(stream_ptr, device_id=0)
        ledger.append({"op": "stream_create", "pointer": stream_ptr, "code": 0})
        for label, size in sizes:
            host = cp.cuda.alloc_pinned_memory(size)
            dev_ptr = int(cp.cuda.runtime.malloc(size))
            pinned.append((label, host, size))
            device.append((label, dev_ptr, size))
            ledger.append(
                {
                    "op": "allocate",
                    "name": label,
                    "bytes": size,
                    "pinned_pointer": int(host.ptr),
                    "device_pointer": dev_ptr,
                }
            )
        ctypes.memmove(int(pinned[0][1].ptr), record, len(record))
        ctypes.memmove(int(pinned[1][1].ptr), input_bytes, len(input_bytes))
        if ctypes.string_at(int(pinned[0][1].ptr), len(record)) != record:
            raise RuntimeError("pinned_record_staging_mismatch")
        if ctypes.string_at(int(pinned[1][1].ptr), len(input_bytes)) != input_bytes:
            raise RuntimeError("pinned_input_staging_mismatch")
        cp.cuda.runtime.memsetAsync(device[2][1], 0xFF, common.ROWS * 2, stream_ptr)
        cp.cuda.runtime.memsetAsync(device[3][1], 0, common.COUNTER_BYTES, stream_ptr)
        ledger.extend(
            (
                {"op": "memset", "target": "output", "bytes": common.ROWS * 2},
                {"op": "memset", "target": "counters", "bytes": common.COUNTER_BYTES},
            )
        )
        cp.cuda.runtime.memcpyAsync(device[0][1], int(pinned[0][1].ptr), common.RECORD_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream_ptr)
        cp.cuda.runtime.memcpyAsync(device[1][1], int(pinned[1][1].ptr), common.INPUT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream_ptr)
        ledger.extend(
            (
                {"op": "H2D", "target": "record", "bytes": common.RECORD_BYTES},
                {"op": "H2D", "target": "input", "bytes": common.INPUT_BYTES},
            )
        )
        function(
            (16,),
            (256,),
            tuple(np.uint64(pointer) for _, pointer, _ in device),
            stream=external_stream,
        )
        ledger.append({"op": "kernel", "grid": [16], "block": [256]})
        cp.cuda.runtime.memcpyAsync(int(pinned[2][1].ptr), device[2][1], common.ROWS * 2, cp.cuda.runtime.memcpyDeviceToHost, stream_ptr)
        cp.cuda.runtime.memcpyAsync(int(pinned[3][1].ptr), device[3][1], common.COUNTER_BYTES, cp.cuda.runtime.memcpyDeviceToHost, stream_ptr)
        ledger.extend(
            (
                {"op": "D2H", "target": "output", "bytes": common.ROWS * 2},
                {"op": "D2H", "target": "counters", "bytes": common.COUNTER_BYTES},
            )
        )
        cp.cuda.runtime.streamSynchronize(stream_ptr)
        ledger.append({"op": "synchronize", "code": 0})
        output = ctypes.string_at(int(pinned[2][1].ptr), common.ROWS * 2)
        counters = ctypes.string_at(int(pinned[3][1].ptr), common.COUNTER_BYTES)
    finally:
        if stream_ptr:
            try:
                cp.cuda.runtime.streamSynchronize(stream_ptr)
            except Exception as exc:
                cleanup_errors.append(f"sync:{exc}")
        for label, pointer, _ in reversed(device):
            try:
                cp.cuda.runtime.free(pointer)
                ledger.append({"release": f"device_{label}", "code": 0})
            except Exception as exc:
                cleanup_errors.append(f"device_{label}:{exc}")
        for label, host, _ in reversed(pinned):
            try:
                host.mem.free()
                ledger.append({"release": f"pinned_{label}", "code": 0})
            except Exception as exc:
                cleanup_errors.append(f"pinned_{label}:{exc}")
        function = module = external_stream = None
        gc.collect()
        if stream_ptr:
            try:
                cp.cuda.runtime.streamDestroy(stream_ptr)
                ledger.append({"release": "stream", "code": 0})
            except Exception as exc:
                cleanup_errors.append(f"stream:{exc}")
        ledger.append({"cleanup_complete": not cleanup_errors, "errors": cleanup_errors})
    return {
        "identity": {
            "count": count,
            "name": name,
            "pci": pci,
            "driver": cp.cuda.runtime.driverGetVersion(),
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
        },
        "output_hex": output.hex(),
        "counters_hex": counters.hex(),
        "ledger": ledger,
    }


def main() -> int:
    if RUN.exists():
        raise FileExistsError(RUN)
    bindings, prior = predevice_gate()
    RUN.mkdir(parents=True)
    result: dict[str, object] = {
        "kind": "het_next_l0_ph0x_r5_nvidia_only_real_projection",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "prereg_sha256": common.file_digest(PREREG),
            "runner_sha256": common.file_digest(Path(__file__)),
            "dependencies": bindings,
            "cuda_source_sha256": EXPECTED_CUDA_SOURCE_SHA,
        },
        "intel_reexecuted": False,
    }
    error = None
    try:
        source = common.read_exact(common.SHARD, common.SOURCE_OFFSET, common.SOURCE_BYTES)
        input_bytes = common.read_exact(common.D2, common.INPUT_OFFSET, common.INPUT_BYTES)
        if sha(input_bytes) != EXPECTED_INPUT_SHA:
            raise RuntimeError("input_hash_drift")
        record, evidence = common.build_record(source)
        if evidence["record_sha256"] != EXPECTED_RECORD_SHA:
            raise RuntimeError("record_hash_drift")
        common.safe_check(record, input_bytes)
        oracle = common.cpu_oracle(record, input_bytes)
        if sha(oracle.tobytes()) != EXPECTED_CPU_SHA or oracle.tobytes().hex() != prior.get("cpu_output_hex"):
            raise RuntimeError("cpu_oracle_drift")
        nvidia = run_nvidia(record, input_bytes)
        output = np.frombuffer(bytes.fromhex(nvidia["output_hex"]), "<u2")
        counters = np.frombuffer(bytes.fromhex(nvidia["counters_hex"]), "<u4")
        comparison = {
            "words": int(output.size),
            "different_words": int(np.count_nonzero(output != oracle)) if output.size == oracle.size else -1,
            "output_sha256": sha(output.tobytes()),
            "counters_all_one": bool(counters.size == common.ROWS and np.all(counters == 1)),
            "sentinel_overwritten": bool(output.size == common.ROWS and np.all(output != 0xFFFF)),
        }
        gates = {
            "nvidia_exact": comparison["different_words"] == 0 and comparison["output_sha256"] == EXPECTED_CPU_SHA,
            "nvidia_counters": comparison["counters_all_one"],
            "nvidia_sentinel": comparison["sentinel_overwritten"],
            "nvidia_identity": nvidia["identity"]["count"] == 1 and nvidia["identity"]["name"] == "NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU" and nvidia["identity"]["pci"] == "0000:01:00.0",
            "nvidia_cleanup": bool(nvidia["ledger"][-1]["cleanup_complete"]),
            "distinct_from_bound_intel": prior["intel"]["identity"]["pci"] != nvidia["identity"]["pci"],
        }
        result.update(
            {
                "record_evidence": evidence,
                "cpu_output_sha256": EXPECTED_CPU_SHA,
                "bound_intel_comparison": prior["intel_comparison"],
                "nvidia": nvidia,
                "nvidia_comparison": comparison,
                "gates": gates,
                "positive": all(gates.values()),
            }
        )
        result["status"] = "exploratory_nvidia_completion_positive" if result["positive"] else "exploratory_nvidia_completion_negative"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result.update({"positive": False, "status": "exploratory_nvidia_completion_failure", "error": error, "traceback": traceback.format_exc()})
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    result["claim_boundary"] = "NVIDIA completion for one real projection/input, bound to prior Intel evidence; no full expert/layer/model/performance/concurrency/deployment/novelty/breakthrough claim."
    common.write_atomic_new(RESULT, common.canonical(result))
    print(json.dumps({"status": result["status"], "positive": result["positive"], "gates": result.get("gates"), "error": error}, indent=2))
    return 0 if result.get("positive") is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
