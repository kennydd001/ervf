from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p3a_integrated_expert import EXPERT_BYTES, LAYERS
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7a_kernel_roofline import load_q5, load_q8


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P7B_ERVF_KERNEL_PREREGISTRATION.md"
OUTPUT = R / "p7b_ervf_kernel.json"
SEED = 270813
WIDTHS = (8, 16, 32)


ERVF_SOURCE = r'''
template<int WIDTH>
__device__ __forceinline__ float q8_ervf_row(
    const float* x, const signed char* codes, const unsigned short* scales,
    int row, int cols, int lane) {
    const int VIRTUAL = 256 / WIDTH;
    float partial[VIRTUAL];
    int groups = cols >> 7;
    #pragma unroll
    for (int virtual_index = 0; virtual_index < VIRTUAL; ++virtual_index) {
        int tid = lane + WIDTH * virtual_index;
        float sum = 0.0f;
        for (int col = tid; col < cols; col += 256) {
            float scale = bf16_to_float(scales[row * groups + (col >> 7)]);
            float weight = round_bf16(((float)codes[(long long)row * cols + col]) * scale);
            sum += weight * x[col];
        }
        partial[virtual_index] = sum;
    }
    #pragma unroll
    for (int stride = 128; stride >= WIDTH; stride >>= 1) {
        #pragma unroll
        for (int index = 0; index < stride / WIDTH; ++index) partial[index] += partial[index + stride / WIDTH];
    }
    float value = partial[0];
    #pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffU, value, offset, WIDTH);
    return value;
}

template<int WIDTH>
__device__ __forceinline__ float q5_ervf_row(
    const float* x, const unsigned char* packed, const unsigned short* scales,
    int row, int cols, int lane) {
    const int VIRTUAL = 256 / WIDTH;
    float partial[VIRTUAL];
    int packs = cols >> 3;
    int groups = cols >> 7;
    #pragma unroll
    for (int virtual_index = 0; virtual_index < VIRTUAL; ++virtual_index) {
        int tid = lane + WIDTH * virtual_index;
        float sum = 0.0f;
        for (int pack = tid; pack < packs; pack += 256) {
            const unsigned char* source = packed + ((long long)row * packs + pack) * 5LL;
            unsigned long long word = ((unsigned long long)source[0])
                | ((unsigned long long)source[1] << 8)
                | ((unsigned long long)source[2] << 16)
                | ((unsigned long long)source[3] << 24)
                | ((unsigned long long)source[4] << 32);
            int column = pack << 3;
            float scale = bf16_to_float(scales[row * groups + (column >> 7)]);
            #pragma unroll
            for (int item = 0; item < 8; ++item) {
                int code = ((word >> (item * 5)) & 31ULL) - 15;
                float weight = round_bf16(((float)code) * scale);
                sum += weight * x[column + item];
            }
        }
        partial[virtual_index] = sum;
    }
    #pragma unroll
    for (int stride = 128; stride >= WIDTH; stride >>= 1) {
        #pragma unroll
        for (int index = 0; index < stride / WIDTH; ++index) partial[index] += partial[index + stride / WIDTH];
    }
    float value = partial[0];
    #pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffU, value, offset, WIDTH);
    return value;
}

#define DEFINE_Q8(WIDTH) \
extern "C" __global__ void q8_ervf##WIDTH( \
    const float* x, const unsigned char* bank, long long base, long long code_bytes, \
    int rows, int cols, float* output) { \
    const int GROUPS = 256 / WIDTH; \
    int group = (int)threadIdx.x / WIDTH; \
    int lane = (int)threadIdx.x & (WIDTH - 1); \
    int row = (int)blockIdx.x * GROUPS + group; \
    if (row >= rows) return; \
    const signed char* codes = (const signed char*)(bank + base); \
    const unsigned short* scales = (const unsigned short*)(bank + base + code_bytes); \
    float value = q8_ervf_row<WIDTH>(x, codes, scales, row, cols, lane); \
    if (lane == 0) output[row] = round_bf16(value); \
}

#define DEFINE_Q5_GATE(WIDTH) \
extern "C" __global__ void q5_gate_up_ervf##WIDTH( \
    const float* x, const unsigned char* cache, const int* slots, const int* positions, \
    float* gate, float* up) { \
    const int GROUPS = 256 / WIDTH; \
    int group = (int)threadIdx.x / WIDTH; \
    int lane = (int)threadIdx.x & (WIDTH - 1); \
    int global_row = (int)blockIdx.x * GROUPS + group; \
    if (global_row >= 8 * 1536) return; \
    int expert = global_row / 1536; \
    int local = global_row - expert * 1536; \
    int projection = local >= 768; \
    int row = local - projection * 768; \
    int output_expert = positions[expert]; \
    long long base = (long long)slots[expert] * 3035136LL + (long long)projection * 1011712LL; \
    const unsigned char* packed = cache + base + 64; \
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040); \
    float value = q5_ervf_row<WIDTH>(x, packed, scales, row, 2048, lane); \
    if (lane == 0) { if (projection) up[output_expert * 768 + row] = round_bf16(value); else gate[output_expert * 768 + row] = round_bf16(value); } \
}

#define DEFINE_Q5_DOWN(WIDTH) \
extern "C" __global__ void q5_down_ervf##WIDTH( \
    const float* activation, const unsigned char* cache, const int* slots, \
    const int* positions, float* down) { \
    const int GROUPS = 256 / WIDTH; \
    int group = (int)threadIdx.x / WIDTH; \
    int lane = (int)threadIdx.x & (WIDTH - 1); \
    int global_row = (int)blockIdx.x * GROUPS + group; \
    if (global_row >= 8 * 2048) return; \
    int expert = global_row / 2048; \
    int row = global_row - expert * 2048; \
    int output_expert = positions[expert]; \
    long long base = (long long)slots[expert] * 3035136LL + 2LL * 1011712LL; \
    const unsigned char* packed = cache + base + 64; \
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040); \
    float value = q5_ervf_row<WIDTH>(activation + output_expert * 768, packed, scales, row, 768, lane); \
    if (lane == 0) down[output_expert * 2048 + row] = round_bf16(value); \
}

DEFINE_Q8(8)
DEFINE_Q8(16)
DEFINE_Q8(32)
DEFINE_Q5_GATE(8)
DEFINE_Q5_GATE(16)
DEFINE_Q5_GATE(32)
DEFINE_Q5_DOWN(8)
DEFINE_Q5_DOWN(16)
DEFINE_Q5_DOWN(32)
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "min": float(x.min()), "max": float(x.max())}


def measure(stream, launch, warmups, iterations):
    for _ in range(warmups): launch()
    stream.synchronize()
    values = []
    for _ in range(iterations):
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record(stream); launch(); end.record(stream); end.synchronize()
        values.append(float(cp.cuda.get_elapsed_time(begin, end)))
    return {"event_ms": values, "stats": stats(values)}


def comparison(observed: np.ndarray, expected: np.ndarray) -> dict:
    bits_equal = bool(np.array_equal(observed.view(np.uint32), expected.view(np.uint32)))
    delta = observed.astype(np.float64) - expected.astype(np.float64)
    return {"bitwise_equal": bits_equal, "elements": int(expected.size), "different": int(np.count_nonzero(observed.view(np.uint32) != expected.view(np.uint32))), "max_abs": float(np.abs(delta).max(initial=0.0)), "finite": bool(np.isfinite(observed).all())}


def main():
    started = time.perf_counter()
    manifest, q8_pin, q8_host, q8_mem, q8, q8_records, q8_sha = load_q8()
    q5_mem, q5 = load_q5()
    names = ["q8_gemv", "q5_gate_up_n", "q5_down_n"]
    for width in WIDTHS: names += [f"q8_ervf{width}", f"q5_gate_up_ervf{width}", f"q5_down_ervf{width}"]
    module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE, options=("--std=c++11",), name_expressions=tuple(names))
    k = {name: module.get_function(name) for name in names}
    stream = cp.cuda.Stream(non_blocking=True)
    rng = np.random.default_rng(SEED)
    x = cp.asarray(rng.standard_normal(4096, dtype=np.float32))
    positions = cp.asarray(np.arange(8, dtype=np.int32))
    slots = [cp.asarray(np.arange(layer * 8, layer * 8 + 8, dtype=np.int32)) for layer in range(LAYERS)]
    q8_total_rows = sum(record["rows"] for _, record in q8_records)
    q8_outputs = {"baseline": cp.empty(q8_total_rows, dtype=cp.float32)}
    q8_outputs.update({str(width): cp.empty(q8_total_rows, dtype=cp.float32) for width in WIDTHS})
    q5_shapes = (LAYERS, 8 * (768 + 768 + 2048))
    q5_host_outputs = {"baseline": np.empty(q5_shapes, dtype=np.float32)}
    q5_host_outputs.update({str(width): np.empty(q5_shapes, dtype=np.float32) for width in WIDTHS})
    gate = cp.empty(8 * 768, dtype=cp.float32); up = cp.empty_like(gate); down = cp.empty(8 * 2048, dtype=cp.float32)

    def fill_q8(kind):
        cursor = 0
        for base, record in q8_records:
            out = q8_outputs[kind][cursor:]
            if kind == "baseline":
                k["q8_gemv"]((record["rows"],), (256,), (x, q8, np.int64(base), np.int64(record["code_bytes"]), np.int32(record["rows"]), np.int32(record["cols"]), out), stream=stream)
            else:
                width = int(kind); groups = 256 // width
                grid = (record["rows"] + groups - 1) // groups
                k[f"q8_ervf{width}"]((grid,), (256,), (x, q8, np.int64(base), np.int64(record["code_bytes"]), np.int32(record["rows"]), np.int32(record["cols"]), out), stream=stream)
            cursor += record["rows"]

    def fill_q5(kind, capture=False):
        width = None if kind == "baseline" else int(kind)
        for layer in range(LAYERS):
            if width is None:
                k["q5_gate_up_n"]((8 * 1536,), (256,), (x, q5, slots[layer], positions, gate, up), stream=stream)
                k["q5_down_n"]((8 * 2048,), (256,), (gate, q5, slots[layer], positions, down), stream=stream)
            else:
                groups = 256 // width
                k[f"q5_gate_up_ervf{width}"](((8 * 1536 + groups - 1) // groups,), (256,), (x, q5, slots[layer], positions, gate, up), stream=stream)
                k[f"q5_down_ervf{width}"](((8 * 2048 + groups - 1) // groups,), (256,), (gate, q5, slots[layer], positions, down), stream=stream)
            if capture:
                stream.synchronize()
                q5_host_outputs[kind][layer] = np.concatenate((cp.asnumpy(gate), cp.asnumpy(up), cp.asnumpy(down)))

    fill_q8("baseline"); stream.synchronize()
    for width in WIDTHS: fill_q8(str(width))
    stream.synchronize()
    q8_reference = cp.asnumpy(q8_outputs["baseline"])
    correctness = {"q8": {str(width): comparison(cp.asnumpy(q8_outputs[str(width)]), q8_reference) for width in WIDTHS}}
    fill_q5("baseline", capture=True)
    correctness["q5"] = {}
    for width in WIDTHS:
        fill_q5(str(width), capture=True)
        correctness["q5"][str(width)] = comparison(q5_host_outputs[str(width)], q5_host_outputs["baseline"])

    validation = {"q8": {}, "q5": {}}
    validation["q8"]["baseline"] = measure(stream, lambda: fill_q8("baseline"), 5, 30)
    validation["q5"]["baseline"] = measure(stream, lambda: fill_q5("baseline"), 5, 30)
    for width in WIDTHS:
        validation["q8"][str(width)] = measure(stream, lambda w=width: fill_q8(str(w)), 5, 30)
        validation["q5"][str(width)] = measure(stream, lambda w=width: fill_q5(str(w)), 5, 30)
    selected = {}
    for bank in ("q8", "q5"):
        eligible = [width for width in WIDTHS if correctness[bank][str(width)]["bitwise_equal"] and correctness[bank][str(width)]["finite"]]
        if not eligible: raise RuntimeError(f"no bit-exact {bank} ERVF variant")
        selected[bank] = min(eligible, key=lambda width: validation[bank][str(width)]["stats"]["p50"])

    test = {}
    for bank, fill in (("q8", fill_q8), ("q5", fill_q5)):
        baseline = measure(stream, lambda f=fill: f("baseline"), 10, 120)
        width = selected[bank]
        variant = measure(stream, lambda f=fill, w=width: f(str(w)), 10, 120)
        p50_ratio = variant["stats"]["p50"] / baseline["stats"]["p50"]
        p95_ratio = variant["stats"]["p95"] / baseline["stats"]["p95"]
        test[bank] = {"selected_width": width, "baseline": baseline, "ervf": variant, "p50_ratio": p50_ratio, "p95_ratio": p95_ratio, "speedup_p50": 1.0 / p50_ratio, "speedup_p95": 1.0 / p95_ratio, "pass": bool(p50_ratio <= 0.90 and p95_ratio <= 0.95)}
    result = {
        "kind": "streamq5_moe_p7b_ervf_kernel", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
        "seed": SEED, "correctness": correctness, "validation": validation,
        "selected": selected, "test": test, "overall_pass": bool(test["q8"]["pass"] and test["q5"]["pass"]),
        "inputs": {"q8_manifest_sha256": sha256(R / "p6a_exact_runtime_bank_result.json"), "q8_pinned_aggregate_sha256": q8_sha, "q5_slots": LAYERS * 8, "q5_record_bytes": EXPERT_BYTES},
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Bit-exact isolated local projection-plane result; end-to-end P6B integration remains required.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "selected": selected, "test": {bank: {key: value for key, value in data.items() if key not in ("baseline", "ervf")} | {"baseline_p50": data["baseline"]["stats"]["p50"], "ervf_p50": data["ervf"]["stats"]["p50"]} for bank, data in test.items()}, "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
