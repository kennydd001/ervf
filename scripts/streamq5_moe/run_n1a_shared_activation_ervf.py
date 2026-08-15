from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p3a_integrated_expert import LAYERS
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7a_kernel_roofline import load_q5, load_q8
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE, comparison, measure


R = ROOT / "reports/streamq5_moe"
PREREG = R / "N1A_SHARED_ACTIVATION_ERVF_PREREGISTRATION.md"
OUTPUT = R / "n1a_shared_activation_ervf.json"
SEED = 120819


STAGED_SOURCE = r'''
extern "C" __global__ void q8_ervf16_staged(
    const float* x, const unsigned char* bank, long long base,
    long long code_bytes, int rows, int cols, float* output) {
    extern __shared__ float staged[];
    for (int col = (int)threadIdx.x; col < cols; col += 256) staged[col] = x[col];
    __syncthreads();
    const int group = (int)threadIdx.x >> 4;
    const int lane = (int)threadIdx.x & 15;
    const int row = (int)blockIdx.x * 16 + group;
    if (row >= rows) return;
    const signed char* codes = (const signed char*)(bank + base);
    const unsigned short* scales = (const unsigned short*)(bank + base + code_bytes);
    float value = q8_ervf_row<16>(staged, codes, scales, row, cols, lane);
    if (lane == 0) output[row] = round_bf16(value);
}

extern "C" __global__ void q5_gate_up_ervf16_staged(
    const float* x, const unsigned char* cache, const int* slots,
    const int* positions, float* gate, float* up) {
    __shared__ float staged[2048];
    for (int col = (int)threadIdx.x; col < 2048; col += 256) staged[col] = x[col];
    __syncthreads();
    const int group = (int)threadIdx.x >> 4;
    const int lane = (int)threadIdx.x & 15;
    const int global_row = (int)blockIdx.x * 16 + group;
    if (global_row >= 8 * 1536) return;
    const int expert = global_row / 1536;
    const int local = global_row - expert * 1536;
    const int projection = local >= 768;
    const int row = local - projection * 768;
    const int output_expert = positions[expert];
    const long long base = (long long)slots[expert] * 3035136LL + (long long)projection * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float value = q5_ervf_row<16>(staged, packed, scales, row, 2048, lane);
    if (lane == 0) {
        if (projection) up[output_expert * 768 + row] = round_bf16(value);
        else gate[output_expert * 768 + row] = round_bf16(value);
    }
}

extern "C" __global__ void q5_down_ervf16_staged(
    const float* activation, const unsigned char* cache, const int* slots,
    const int* positions, float* down) {
    __shared__ float staged[768];
    const int first_row = (int)blockIdx.x * 16;
    const int expert = first_row / 2048;
    const int output_expert = positions[expert];
    for (int col = (int)threadIdx.x; col < 768; col += 256)
        staged[col] = activation[output_expert * 768 + col];
    __syncthreads();
    const int group = (int)threadIdx.x >> 4;
    const int lane = (int)threadIdx.x & 15;
    const int global_row = first_row + group;
    if (global_row >= 8 * 2048) return;
    const int row = global_row - expert * 2048;
    const long long base = (long long)slots[expert] * 3035136LL + 2LL * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float value = q5_ervf_row<16>(staged, packed, scales, row, 768, lane);
    if (lane == 0) down[output_expert * 2048 + row] = round_bf16(value);
}
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value: dict) -> dict:
    return value["stats"] | {"iterations": len(value["event_ms"])}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest, q8_pin, q8_host, q8_mem, q8, q8_records, q8_sha = load_q8()
    q5_mem, q5 = load_q5()
    names = (
        "q8_ervf16", "q5_gate_up_ervf16", "q5_down_ervf16",
        "q8_ervf16_staged", "q5_gate_up_ervf16_staged", "q5_down_ervf16_staged",
    )
    module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE + STAGED_SOURCE,
                          options=("--std=c++11",), name_expressions=names)
    k = {name: module.get_function(name) for name in names}
    stream = cp.cuda.Stream(non_blocking=True)
    rng = np.random.default_rng(SEED)
    x = cp.asarray(rng.standard_normal(4096, dtype=np.float32))
    positions = cp.asarray(np.arange(8, dtype=np.int32))
    slots = [cp.asarray(np.arange(layer * 8, layer * 8 + 8, dtype=np.int32)) for layer in range(LAYERS)]
    q8_rows = sum(record["rows"] for _, record in q8_records)
    q8_output = cp.empty(q8_rows, dtype=cp.float32)
    gate = cp.empty(8 * 768, dtype=cp.float32); up = cp.empty_like(gate); down = cp.empty(8 * 2048, dtype=cp.float32)

    def q8_plane(staged: bool):
        cursor = 0
        for base, record in q8_records:
            name = "q8_ervf16_staged" if staged else "q8_ervf16"
            k[name](((record["rows"] + 15) // 16,), (256,),
                    (x, q8, np.int64(base), np.int64(record["code_bytes"]), np.int32(record["rows"]), np.int32(record["cols"]), q8_output[cursor:]),
                    shared_mem=record["cols"] * 4 if staged else 0, stream=stream)
            cursor += record["rows"]

    def q5_layer(layer: int, staged: bool):
        suffix = "_staged" if staged else ""
        k[f"q5_gate_up_ervf16{suffix}"]((768,), (256,), (x, q5, slots[layer], positions, gate, up), stream=stream)
        k[f"q5_down_ervf16{suffix}"]((1024,), (256,), (gate, q5, slots[layer], positions, down), stream=stream)

    def q5_plane(staged: bool):
        for layer in range(LAYERS): q5_layer(layer, staged)

    def capture_q8(staged: bool) -> np.ndarray:
        q8_plane(staged); stream.synchronize(); return cp.asnumpy(q8_output)

    def capture_q5(staged: bool) -> np.ndarray:
        out = np.empty((LAYERS, 8 * (768 + 768 + 2048)), dtype=np.float32)
        for layer in range(LAYERS):
            q5_layer(layer, staged); stream.synchronize()
            out[layer] = np.concatenate((cp.asnumpy(gate), cp.asnumpy(up), cp.asnumpy(down)))
        return out

    correctness = {
        "q8": comparison(capture_q8(True), capture_q8(False)),
        "q5": comparison(capture_q5(True), capture_q5(False)),
    }
    validation = {}
    test = {}
    for name, launch in (("q8", q8_plane), ("q5", q5_plane)):
        baseline = measure(stream, lambda f=launch: f(False), 5, 30)
        staged = measure(stream, lambda f=launch: f(True), 5, 30)
        ratio = staged["stats"]["p50"] / baseline["stats"]["p50"]
        opened = correctness[name]["bitwise_equal"] and ratio <= 0.97
        validation[name] = {"baseline": compact(baseline), "staged": compact(staged), "p50_ratio": ratio, "test_opened": opened}
        if opened:
            base_test = measure(stream, lambda f=launch: f(False), 10, 120)
            staged_test = measure(stream, lambda f=launch: f(True), 10, 120)
            p50_ratio = staged_test["stats"]["p50"] / base_test["stats"]["p50"]
            p95_ratio = staged_test["stats"]["p95"] / base_test["stats"]["p95"]
            test[name] = {"baseline": compact(base_test), "staged": compact(staged_test),
                          "p50_ratio": p50_ratio, "p95_ratio": p95_ratio,
                          "pass": p50_ratio <= 0.92 and p95_ratio <= 0.95}
        else:
            test[name] = None
    component_pass = {name: bool(test[name] and test[name]["pass"]) for name in ("q8", "q5")}
    result = {
        "kind": "streamq5_moe_n1a_shared_activation_ervf", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
                   "q8_manifest_sha256": sha256(R / "p6a_exact_runtime_bank_result.json"), "q8_aggregate_sha256": q8_sha},
        "seed": SEED, "correctness": correctness, "validation": validation, "test": test,
        "component_pass": component_pass, "overall_pass": all(component_pass.values()),
        "claim_boundary": "Isolated Q8/Q5 projection planes; no end-to-end decoder claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"correctness": correctness, "validation": validation, "test": test,
                      "component_pass": component_pass, "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
