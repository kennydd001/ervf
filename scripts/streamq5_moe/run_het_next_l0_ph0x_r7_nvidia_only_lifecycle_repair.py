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
import run_het_next_l0_ph0x_r5_nvidia_only_real_projection as r5
import run_het_next_l0_ph0x_r6_nvidia_only_ledger_repair as r6


RUN = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r7_nvidia_only_lifecycle_repair"
RESULT = RUN / "ph0x_r7_result.json"
PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_L0_PH0X_R7_NVIDIA_ONLY_LIFECYCLE_REPAIR_PREREGISTRATION_2026-08-13.md"
R6_PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_L0_PH0X_R6_NVIDIA_ONLY_LEDGER_REPAIR_PREREGISTRATION_2026-08-13.md"
EXPECTED_R6_RUNNER_SHA = "a1369c314a4e1367fa4ce3584555a7dc4db30ed9480cbdff289aa18af8417bdf"
EXPECTED_R6_PREREG_SHA = "7e5c0ad01797120c66ce140f32207ed3460821aa3a0f4acbd6aff8f5a8231732"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_nvidia(record: bytes, input_bytes: bytes) -> dict[str, object]:
    ledger: list[dict[str, object]] = []
    evidence: dict[str, object] = {
        "identity": {},
        "output_hex": "",
        "counters_hex": "",
        "ledger": ledger,
        "module_disposal": "cupy_raii_after_reference_drop; unload_not_independently_observed",
    }
    cleanup_errors: list[str] = []
    pinned: list[tuple[str, object, int]] = []
    device: list[tuple[str, int, int]] = []
    stream_ptr = 0
    primary_sync_complete = False
    module = function = external_stream = None
    active_exception: BaseException | None = None
    cp = None
    try:
        import cupy as cp_module

        cp = cp_module
        count = cp.cuda.runtime.getDeviceCount()
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        pci = cp.cuda.runtime.deviceGetPCIBusId(0)
        if isinstance(pci, bytes):
            pci = pci.decode()
        evidence["identity"] = {"count": count, "name": name, "pci": pci, "driver": cp.cuda.runtime.driverGetVersion(), "runtime": cp.cuda.runtime.runtimeGetVersion()}
        if count != 1 or name != "NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU" or pci != "0000:01:00.0":
            raise RuntimeError(f"nvidia_identity:{count}:{name}:{pci}")
        module = cp.RawModule(code=r5.r4.CUDA_SOURCE, backend="nvrtc", options=("--std=c++17", "--fmad=true", "--prec-div=true", "--prec-sqrt=true", "--ftz=false"), name_expressions=("ph0",))
        function = module.get_function("ph0")
        ledger.append({"op": "compile", "source_sha256": sha(r5.r4.CUDA_SOURCE.encode()), "success": True})
        stream_ptr = int(cp.cuda.runtime.streamCreateWithFlags(1))
        external_stream = cp.cuda.ExternalStream(stream_ptr, device_id=0)
        ledger.append({"op": "stream_create", "pointer": stream_ptr, "code": 0})
        sizes = (("record", common.RECORD_BYTES), ("input", common.INPUT_BYTES), ("output", common.ROWS * 2), ("counters", common.COUNTER_BYTES))
        for label, size in sizes:
            host = cp.cuda.alloc_pinned_memory(size)
            pinned.append((label, host, size))
            dev_ptr = int(cp.cuda.runtime.malloc(size))
            device.append((label, dev_ptr, size))
            ledger.append({"op": "allocate", "name": label, "bytes": size, "pinned_pointer": int(host.ptr), "device_pointer": dev_ptr})
        ctypes.memmove(int(pinned[0][1].ptr), record, len(record))
        ctypes.memmove(int(pinned[1][1].ptr), input_bytes, len(input_bytes))
        if ctypes.string_at(int(pinned[0][1].ptr), len(record)) != record or ctypes.string_at(int(pinned[1][1].ptr), len(input_bytes)) != input_bytes:
            raise RuntimeError("pinned_staging_mismatch")
        cp.cuda.runtime.memsetAsync(device[2][1], 0xFF, common.ROWS * 2, stream_ptr)
        cp.cuda.runtime.memsetAsync(device[3][1], 0, common.COUNTER_BYTES, stream_ptr)
        ledger.extend(({"op": "memset", "target": "output", "bytes": common.ROWS * 2}, {"op": "memset", "target": "counters", "bytes": common.COUNTER_BYTES}))
        cp.cuda.runtime.memcpyAsync(device[0][1], int(pinned[0][1].ptr), common.RECORD_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream_ptr)
        cp.cuda.runtime.memcpyAsync(device[1][1], int(pinned[1][1].ptr), common.INPUT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream_ptr)
        ledger.extend(({"op": "H2D", "target": "record", "bytes": common.RECORD_BYTES}, {"op": "H2D", "target": "input", "bytes": common.INPUT_BYTES}))
        function((16,), (256,), tuple(np.uint64(pointer) for _, pointer, _ in device), stream=external_stream)
        ledger.append({"op": "kernel", "grid": [16], "block": [256]})
        cp.cuda.runtime.memcpyAsync(int(pinned[2][1].ptr), device[2][1], common.ROWS * 2, cp.cuda.runtime.memcpyDeviceToHost, stream_ptr)
        cp.cuda.runtime.memcpyAsync(int(pinned[3][1].ptr), device[3][1], common.COUNTER_BYTES, cp.cuda.runtime.memcpyDeviceToHost, stream_ptr)
        ledger.extend(({"op": "D2H", "target": "output", "bytes": common.ROWS * 2}, {"op": "D2H", "target": "counters", "bytes": common.COUNTER_BYTES}))
        cp.cuda.runtime.streamSynchronize(stream_ptr)
        primary_sync_complete = True
        ledger.append({"op": "synchronize", "code": 0})
        evidence["output_hex"] = ctypes.string_at(int(pinned[2][1].ptr), common.ROWS * 2).hex()
        evidence["counters_hex"] = ctypes.string_at(int(pinned[3][1].ptr), common.COUNTER_BYTES).hex()
    except BaseException as exc:
        active_exception = exc
    finally:
        if cp is not None and stream_ptr and not primary_sync_complete:
            try:
                cp.cuda.runtime.streamSynchronize(stream_ptr)
                ledger.append({"op": "failure_cleanup_sync", "code": 0})
            except Exception as exc:
                cleanup_errors.append(f"sync:{exc}")
                ledger.append({"op": "failure_cleanup_sync", "code": -1, "error": str(exc)})
        if cp is not None:
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
        if cp is not None and stream_ptr:
            try:
                cp.cuda.runtime.streamDestroy(stream_ptr)
                ledger.append({"release": "stream", "code": 0})
            except Exception as exc:
                cleanup_errors.append(f"stream:{exc}")
        ledger.append({"cleanup_complete": not cleanup_errors, "errors": cleanup_errors})
    if active_exception is None and cleanup_errors:
        active_exception = RuntimeError("cleanup_errors:" + "|".join(cleanup_errors))
    if active_exception is not None:
        evidence["failure"] = {"type": type(active_exception).__name__, "message": str(active_exception)}
        raise r6.NvidiaRunFailure(f"{type(active_exception).__name__}: {active_exception}", evidence) from active_exception
    return evidence


def main() -> int:
    if RUN.exists():
        raise FileExistsError(RUN)
    if common.file_digest(Path(r6.__file__).resolve()) != EXPECTED_R6_RUNNER_SHA or common.file_digest(R6_PREREG) != EXPECTED_R6_PREREG_SHA:
        raise RuntimeError("r6_hash_drift")
    if common.file_digest(Path(r5.__file__).resolve()) != r6.EXPECTED_R5_RUNNER_SHA or common.file_digest(r6.R5_PREREG) != r6.EXPECTED_R5_PREREG_SHA:
        raise RuntimeError("r5_hash_drift")
    bindings, prior = r5.predevice_gate()
    RUN.mkdir(parents=True)
    result: dict[str, object] = {
        "kind": "het_next_l0_ph0x_r7_nvidia_only_lifecycle_repair",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {"prereg_sha256": common.file_digest(PREREG), "runner_sha256": common.file_digest(Path(__file__)), "r6_runner_sha256": EXPECTED_R6_RUNNER_SHA, "r6_prereg_sha256": EXPECTED_R6_PREREG_SHA, "dependencies": bindings},
        "intel_reexecuted": False,
    }
    error = None
    try:
        source = common.read_exact(common.SHARD, common.SOURCE_OFFSET, common.SOURCE_BYTES)
        input_bytes = common.read_exact(common.D2, common.INPUT_OFFSET, common.INPUT_BYTES)
        if sha(input_bytes) != r5.EXPECTED_INPUT_SHA:
            raise RuntimeError("input_hash_drift")
        record, evidence = common.build_record(source)
        if evidence["record_sha256"] != r5.EXPECTED_RECORD_SHA:
            raise RuntimeError("record_hash_drift")
        common.safe_check(record, input_bytes)
        oracle = common.cpu_oracle(record, input_bytes)
        if sha(oracle.tobytes()) != r5.EXPECTED_CPU_SHA or oracle.tobytes().hex() != prior.get("cpu_output_hex"):
            raise RuntimeError("cpu_oracle_drift")
        nvidia = run_nvidia(record, input_bytes)
        result["nvidia"] = nvidia
        ledger_validation = r6.validate_success_ledger(nvidia["ledger"])
        output = np.frombuffer(bytes.fromhex(nvidia["output_hex"]), "<u2")
        counters = np.frombuffer(bytes.fromhex(nvidia["counters_hex"]), "<u4")
        comparison = {"words": int(output.size), "different_words": int(np.count_nonzero(output != oracle)) if output.size == oracle.size else -1, "output_sha256": sha(output.tobytes()), "counters_all_one": bool(counters.size == common.ROWS and np.all(counters == 1)), "sentinel_overwritten": bool(output.size == common.ROWS and np.all(output != 0xFFFF))}
        gates = {"nvidia_exact": comparison["different_words"] == 0 and comparison["output_sha256"] == r5.EXPECTED_CPU_SHA, "nvidia_counters": comparison["counters_all_one"], "nvidia_sentinel": comparison["sentinel_overwritten"], "nvidia_identity": nvidia["identity"].get("count") == 1 and nvidia["identity"].get("name") == "NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU" and nvidia["identity"].get("pci") == "0000:01:00.0", "nvidia_ledger_exact": ledger_validation["ordered_exact"], "distinct_from_bound_intel": prior["intel"]["identity"]["pci"] != nvidia["identity"]["pci"]}
        result.update({"record_evidence": evidence, "cpu_output_sha256": r5.EXPECTED_CPU_SHA, "bound_intel_comparison": prior["intel_comparison"], "nvidia_comparison": comparison, "ledger_validation": ledger_validation, "gates": gates, "positive": all(gates.values())})
        result["status"] = "exploratory_nvidia_completion_positive" if result["positive"] else "exploratory_nvidia_completion_negative"
    except r6.NvidiaRunFailure as exc:
        error = str(exc)
        result.update({"nvidia_failure_evidence": exc.evidence, "positive": False, "status": "exploratory_nvidia_completion_failure", "error": error, "traceback": traceback.format_exc()})
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
