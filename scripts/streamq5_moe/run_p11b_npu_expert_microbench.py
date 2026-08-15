from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset14 as ops


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
PREREG = R / "P11B_NPU_EXPERT_MICROBENCH_PREREGISTRATION.md"
P11A = R / "p11a_cpu_q5_miss_compute.json"
OUTPUT = R / "p11b_npu_expert_microbench.json"
SEED = 110812


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "min": float(x.min()), "max": float(x.max())}


def build_model(experts: int):
    rng = np.random.default_rng(SEED + experts)
    x = ops.parameter([1, 2048], np.float16, name="x")
    outputs = []
    scale = np.float16(1 / np.sqrt(2048))
    for expert in range(experts):
        gate_w = ops.constant((rng.standard_normal((2048, 768), dtype=np.float32) * scale).astype(np.float16))
        up_w = ops.constant((rng.standard_normal((2048, 768), dtype=np.float32) * scale).astype(np.float16))
        down_w = ops.constant((rng.standard_normal((768, 2048), dtype=np.float32) * np.float16(1 / np.sqrt(768))).astype(np.float16))
        gate = ops.matmul(x, gate_w, False, False)
        up = ops.matmul(x, up_w, False, False)
        hidden = ops.multiply(ops.multiply(gate, ops.sigmoid(gate)), up)
        outputs.append(ops.matmul(hidden, down_w, False, False))
    value = outputs[0]
    for item in outputs[1:]: value = ops.add(value, item)
    return ov.Model([value], [x], f"swiglu_experts_{experts}")


def bench(core, model, device, x):
    started = time.perf_counter()
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
    compile_ms = (time.perf_counter() - started) * 1000
    request = compiled.create_infer_request()
    started = time.perf_counter(); output = request.infer({"x": x})[compiled.output(0)].copy()
    first_ms = (time.perf_counter() - started) * 1000
    for _ in range(5): request.infer({"x": x})
    values = []
    for _ in range(100):
        begin = time.perf_counter_ns(); output = request.infer({"x": x})[compiled.output(0)].copy()
        values.append((time.perf_counter_ns() - begin) / 1e6)
    return {"compile_ms": compile_ms, "first_infer_ms": first_ms, "latency_ms": stats(values), "output": output}


def main():
    core = ov.Core()
    available = list(core.available_devices)
    names = {device: core.get_property(device, "FULL_DEVICE_NAME") for device in available}
    devices = [device for device in ("NPU", "CPU", "GPU.0", "GPU.1") if device in available]
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((1, 2048), dtype=np.float32).astype(np.float16)
    runs = {}
    errors = {}
    outputs = {}
    for experts in (1, 8):
        runs[str(experts)] = {}; outputs[str(experts)] = {}
        model = build_model(experts)
        for device in devices:
            try:
                result = bench(core, model, device, x)
                outputs[str(experts)][device] = result.pop("output")
                runs[str(experts)][device] = result
                print(json.dumps({"experts": experts, "device": device, **result}), flush=True)
            except Exception as exc:
                errors[f"{experts}:{device}"] = f"{type(exc).__name__}: {exc}"
                print(json.dumps({"experts": experts, "device": device, "error": errors[f'{experts}:{device}']}), flush=True)
    correctness = {}
    for experts in (1, 8):
        if "NPU" in outputs[str(experts)] and "CPU" in outputs[str(experts)]:
            observed = outputs[str(experts)]["NPU"].astype(np.float32)
            reference = outputs[str(experts)]["CPU"].astype(np.float32)
            delta = np.abs(observed - reference)
            relative = delta / np.maximum(np.abs(reference), 1e-3)
            correctness[str(experts)] = {"max_abs": float(delta.max()), "max_rel": float(relative.max()), "mean_abs": float(delta.mean()), "finite": bool(np.isfinite(observed).all())}
    gpu_floor = json.loads(P11A.read_text(encoding="utf-8"))["gpu_all_cold_reference"]
    npu8 = runs.get("8", {}).get("NPU")
    check = correctness.get("8")
    gates = {
        "npu_compiled": npu8 is not None,
        "finite": bool(check and check["finite"]),
        "max_abs_le_0_05": bool(check and check["max_abs"] <= 0.05),
        "max_rel_le_0_02": bool(check and check["max_rel"] <= 0.02),
        "p50_le_95pct_gpu_floor": bool(npu8 and npu8["latency_ms"]["p50"] <= 0.95 * gpu_floor["all_cold_mean_ms"]),
        "p95_le_95pct_gpu_floor": bool(npu8 and npu8["latency_ms"]["p95"] <= 0.95 * gpu_floor["all_cold_p95_proxy_ms"]),
    }
    result = {
        "kind": "streamq5_moe_p11b_npu_expert_microbench", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "openvino_version": ov.get_version(), "preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
        "available_devices": available, "device_names": names, "runs": runs, "errors": errors,
        "correctness_vs_cpu": correctness, "gpu_all_cold_reference": gpu_floor, "gates": gates,
        "overall_pass": all(gates.values()),
        "claim_boundary": "Resident FP16 best-case microbenchmark; no packed-Q5 NPU decoder or model-quality claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"npu8": npu8, "correctness8": check, "gates": gates, "overall_pass": result["overall_pass"], "errors": errors}, indent=2), flush=True)


if __name__ == "__main__":
    main()
