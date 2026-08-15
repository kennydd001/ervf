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
PREREG = R / "N1C_GENERALIZED_EXACT_REDUCTION_AUTOTUNER_PREREGISTRATION.md"
OUTPUT = R / "n1c_generalized_exact_reduction_autotuner.json"
SEED = 120831
WIDTHS = (4, 8, 16, 32, 64)
VALIDATION_WARMUPS = 3
VALIDATION_ROUNDS = 15
TEST_WARMUPS = 10
TEST_ROUNDS = 120


ERVF_SOURCE = r'''
template<int WIDTH>
__device__ __forceinline__ float q8_ervf_row_n1c(
    const float* x, const signed char* codes, const unsigned short* scales,
    int row, int cols, int lane, unsigned mask) {
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
        for (int index = 0; index < stride / WIDTH; ++index)
            partial[index] += partial[index + stride / WIDTH];
    }
    float value = partial[0];
    #pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(mask, value, offset, WIDTH);
    return value;
}

template<int WIDTH>
__device__ __forceinline__ float q5_ervf_row_n1c(
    const float* x, const unsigned char* packed, const unsigned short* scales,
    int row, int cols, int lane, unsigned mask) {
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
        for (int index = 0; index < stride / WIDTH; ++index)
            partial[index] += partial[index + stride / WIDTH];
    }
    float value = partial[0];
    #pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(mask, value, offset, WIDTH);
    return value;
}

__device__ __forceinline__ float q8_pre64_n1c(
    const float* x, const signed char* codes, const unsigned short* scales,
    int row, int cols, int lane) {
    float partial[4];
    int groups = cols >> 7;
    #pragma unroll
    for (int virtual_index = 0; virtual_index < 4; ++virtual_index) {
        int tid = lane + 64 * virtual_index;
        float sum = 0.0f;
        for (int col = tid; col < cols; col += 256) {
            float scale = bf16_to_float(scales[row * groups + (col >> 7)]);
            float weight = round_bf16(((float)codes[(long long)row * cols + col]) * scale);
            sum += weight * x[col];
        }
        partial[virtual_index] = sum;
    }
    partial[0] += partial[2];
    partial[1] += partial[3];
    partial[0] += partial[1];
    return partial[0];
}

__device__ __forceinline__ float q5_pre64_n1c(
    const float* x, const unsigned char* packed, const unsigned short* scales,
    int row, int cols, int lane) {
    float partial[4];
    int packs = cols >> 3;
    int groups = cols >> 7;
    #pragma unroll
    for (int virtual_index = 0; virtual_index < 4; ++virtual_index) {
        int tid = lane + 64 * virtual_index;
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
    partial[0] += partial[2];
    partial[1] += partial[3];
    partial[0] += partial[1];
    return partial[0];
}

#define DEFINE_Q8_N1C(WIDTH) \
extern "C" __global__ void q8_n1c_##WIDTH( \
    const float* x, const unsigned char* bank, long long base, long long code_bytes, \
    int rows, int cols, float* output) { \
    const int GROUPS = 256 / WIDTH; \
    int group = (int)threadIdx.x / WIDTH; \
    int lane = (int)threadIdx.x & (WIDTH - 1); \
    int row = (int)blockIdx.x * GROUPS + group; \
    unsigned mask = __ballot_sync(0xffffffffU, row < rows); \
    if (row >= rows) return; \
    const signed char* codes = (const signed char*)(bank + base); \
    const unsigned short* scales = (const unsigned short*)(bank + base + code_bytes); \
    float value = q8_ervf_row_n1c<WIDTH>(x, codes, scales, row, cols, lane, mask); \
    if (lane == 0) output[row] = round_bf16(value); \
}

#define DEFINE_Q5_GATE_N1C(WIDTH) \
extern "C" __global__ void q5_gate_up_n1c_##WIDTH( \
    const float* x, const unsigned char* cache, const int* slots, const int* positions, \
    float* gate, float* up) { \
    const int GROUPS = 256 / WIDTH; \
    int group = (int)threadIdx.x / WIDTH; \
    int lane = (int)threadIdx.x & (WIDTH - 1); \
    int global_row = (int)blockIdx.x * GROUPS + group; \
    unsigned mask = __ballot_sync(0xffffffffU, global_row < 8 * 1536); \
    if (global_row >= 8 * 1536) return; \
    int expert = global_row / 1536; \
    int local = global_row - expert * 1536; \
    int projection = local >= 768; \
    int row = local - projection * 768; \
    int output_expert = positions[expert]; \
    long long base = (long long)slots[expert] * 3035136LL + (long long)projection * 1011712LL; \
    const unsigned char* packed = cache + base + 64; \
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040); \
    float value = q5_ervf_row_n1c<WIDTH>(x, packed, scales, row, 2048, lane, mask); \
    if (lane == 0) { \
        if (projection) up[output_expert * 768 + row] = round_bf16(value); \
        else gate[output_expert * 768 + row] = round_bf16(value); \
    } \
}

#define DEFINE_Q5_DOWN_N1C(WIDTH) \
extern "C" __global__ void q5_down_n1c_##WIDTH( \
    const float* activation, const unsigned char* cache, const int* slots, \
    const int* positions, float* down) { \
    const int GROUPS = 256 / WIDTH; \
    int group = (int)threadIdx.x / WIDTH; \
    int lane = (int)threadIdx.x & (WIDTH - 1); \
    int global_row = (int)blockIdx.x * GROUPS + group; \
    unsigned mask = __ballot_sync(0xffffffffU, global_row < 8 * 2048); \
    if (global_row >= 8 * 2048) return; \
    int expert = global_row / 2048; \
    int row = global_row - expert * 2048; \
    int output_expert = positions[expert]; \
    long long base = (long long)slots[expert] * 3035136LL + 2LL * 1011712LL; \
    const unsigned char* packed = cache + base + 64; \
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040); \
    float value = q5_ervf_row_n1c<WIDTH>(activation + output_expert * 768, packed, scales, row, 768, lane, mask); \
    if (lane == 0) down[output_expert * 2048 + row] = round_bf16(value); \
}

DEFINE_Q8_N1C(4)
DEFINE_Q8_N1C(8)
DEFINE_Q8_N1C(16)
DEFINE_Q8_N1C(32)
DEFINE_Q5_GATE_N1C(4)
DEFINE_Q5_GATE_N1C(8)
DEFINE_Q5_GATE_N1C(16)
DEFINE_Q5_GATE_N1C(32)
DEFINE_Q5_DOWN_N1C(4)
DEFINE_Q5_DOWN_N1C(8)
DEFINE_Q5_DOWN_N1C(16)
DEFINE_Q5_DOWN_N1C(32)

extern "C" __global__ void q8_n1c_64(
    const float* x, const unsigned char* bank, long long base, long long code_bytes,
    int rows, int cols, float* output) {
    int group = (int)threadIdx.x >> 6;
    int lane = (int)threadIdx.x & 63;
    int row = (int)blockIdx.x * 4 + group;
    bool valid = row < rows;
    float value = 0.0f;
    if (valid) {
        const signed char* codes = (const signed char*)(bank + base);
        const unsigned short* scales = (const unsigned short*)(bank + base + code_bytes);
        value = q8_pre64_n1c(x, codes, scales, row, cols, lane);
    }
    __shared__ float scratch[256];
    scratch[threadIdx.x] = value;
    __syncthreads();
    if (lane < 32) value += scratch[threadIdx.x + 32];
    if (lane < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    if (valid && lane == 0) output[row] = round_bf16(value);
}

extern "C" __global__ void q5_gate_up_n1c_64(
    const float* x, const unsigned char* cache, const int* slots, const int* positions,
    float* gate, float* up) {
    int group = (int)threadIdx.x >> 6;
    int lane = (int)threadIdx.x & 63;
    int global_row = (int)blockIdx.x * 4 + group;
    int expert = global_row / 1536;
    int local = global_row - expert * 1536;
    int projection = local >= 768;
    int row = local - projection * 768;
    int output_expert = positions[expert];
    long long base = (long long)slots[expert] * 3035136LL + (long long)projection * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float value = q5_pre64_n1c(x, packed, scales, row, 2048, lane);
    __shared__ float scratch[256];
    scratch[threadIdx.x] = value;
    __syncthreads();
    if (lane < 32) value += scratch[threadIdx.x + 32];
    if (lane < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    if (lane == 0) {
        if (projection) up[output_expert * 768 + row] = round_bf16(value);
        else gate[output_expert * 768 + row] = round_bf16(value);
    }
}

extern "C" __global__ void q5_down_n1c_64(
    const float* activation, const unsigned char* cache, const int* slots,
    const int* positions, float* down) {
    int group = (int)threadIdx.x >> 6;
    int lane = (int)threadIdx.x & 63;
    int global_row = (int)blockIdx.x * 4 + group;
    int expert = global_row / 2048;
    int row = global_row - expert * 2048;
    int output_expert = positions[expert];
    long long base = (long long)slots[expert] * 3035136LL + 2LL * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float value = q5_pre64_n1c(activation + output_expert * 768, packed, scales, row, 768, lane);
    __shared__ float scratch[256];
    scratch[threadIdx.x] = value;
    __syncthreads();
    if (lane < 32) value += scratch[threadIdx.x + 32];
    if (lane < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    if (lane == 0) down[output_expert * 2048 + row] = round_bf16(value);
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def compact(values: list[float]) -> dict[str, float | int]:
    return stats(values) | {"iterations": len(values)}


def timed_launch(stream: cp.cuda.Stream, launch) -> float:
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    begin.record(stream)
    launch()
    end.record(stream)
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(begin, end))


def balanced_measure(stream: cp.cuda.Stream, launches: dict[int, object]) -> dict[str, dict]:
    for width in WIDTHS:
        for _ in range(VALIDATION_WARMUPS):
            launches[width]()
    stream.synchronize()
    measured = {width: [] for width in WIDTHS}
    for round_index in range(VALIDATION_ROUNDS):
        rotation = round_index % len(WIDTHS)
        order = list(WIDTHS[rotation:] + WIDTHS[:rotation])
        if round_index & 1:
            order.reverse()
        for width in order:
            measured[width].append(timed_launch(stream, launches[width]))
    return {str(width): compact(measured[width]) for width in WIDTHS}


def paired_measure(stream: cp.cuda.Stream, baseline, candidate) -> dict[str, dict]:
    for _ in range(TEST_WARMUPS):
        baseline()
        candidate()
    stream.synchronize()
    measured = {"baseline": [], "candidate": []}
    for round_index in range(TEST_ROUNDS):
        order = (("baseline", baseline), ("candidate", candidate))
        if round_index & 1:
            order = tuple(reversed(order))
        for name, launch in order:
            measured[name].append(timed_launch(stream, launch))
    return {name: compact(values) for name, values in measured.items()}


def comparison(observed: np.ndarray, expected: np.ndarray) -> dict:
    observed_bits = observed.view(np.uint32)
    expected_bits = expected.view(np.uint32)
    delta = observed.astype(np.float64) - expected.astype(np.float64)
    return {
        "bitwise_equal": bool(np.array_equal(observed_bits, expected_bits)),
        "elements": int(expected.size),
        "different": int(np.count_nonzero(observed_bits != expected_bits)),
        "max_abs": float(np.abs(delta).max(initial=0.0)),
        "finite": bool(np.isfinite(observed).all()),
    }


def choose_width(results: dict[str, dict], eligible: list[int]) -> int:
    best_p50 = min(results[str(width)]["p50"] for width in eligible)
    equivalent = [width for width in eligible if results[str(width)]["p50"] <= best_p50 * 1.005]
    return 16 if 16 in equivalent else min(equivalent)


def main() -> None:
    started = time.perf_counter()
    _, q8_pin, q8_host, q8_mem, q8, q8_records, q8_sha = load_q8()
    q5_mem, q5 = load_q5()
    names = ["q8_gemv", "q5_gate_up_n", "q5_down_n"]
    for width in WIDTHS:
        names.extend((f"q8_n1c_{width}", f"q5_gate_up_n1c_{width}", f"q5_down_n1c_{width}"))
    module = cp.RawModule(
        code=CUDA_SOURCE + ERVF_SOURCE,
        options=("--std=c++11",),
        name_expressions=tuple(names),
    )
    kernels = {name: module.get_function(name) for name in names}
    kernel_resources = {
        family: {
            str(width): {
                key: int(value)
                for key, value in kernels[f"{prefix}{width}"].attributes.items()
                if isinstance(value, (int, np.integer))
            }
            for width in WIDTHS
        }
        for family, prefix in (
            ("q8", "q8_n1c_"),
            ("q5_gate_up", "q5_gate_up_n1c_"),
            ("q5_down", "q5_down_n1c_"),
        )
    }
    stream = cp.cuda.Stream(non_blocking=True)
    rng = np.random.default_rng(SEED)
    x = cp.asarray(rng.standard_normal(4096, dtype=np.float32))
    positions = cp.asarray(np.arange(8, dtype=np.int32))
    slots = [cp.asarray(np.arange(layer * 8, layer * 8 + 8, dtype=np.int32)) for layer in range(LAYERS)]
    gate = cp.empty(8 * 768, dtype=cp.float32)
    up = cp.empty_like(gate)
    down = cp.empty(8 * 2048, dtype=cp.float32)
    q8_total_rows = sum(record[1]["rows"] for record in q8_records)
    q8_out = cp.empty(q8_total_rows, dtype=cp.float32)
    by_name: dict[str, list] = {}
    for item in q8_records:
        by_name.setdefault(item[1]["name"], []).append(item)

    def launch_q8_record(base, record, width: int | None, output) -> None:
        if width is None:
            kernels["q8_gemv"](
                (record["rows"],), (256,),
                (x, q8, np.int64(base), np.int64(record["code_bytes"]),
                 np.int32(record["rows"]), np.int32(record["cols"]), output),
                stream=stream,
            )
        else:
            groups = 256 // width
            grid = (record["rows"] + groups - 1) // groups
            kernels[f"q8_n1c_{width}"](
                (grid,), (256,),
                (x, q8, np.int64(base), np.int64(record["code_bytes"]),
                 np.int32(record["rows"]), np.int32(record["cols"]), output),
                stream=stream,
            )

    def fill_q8(widths: int | None | dict[str, int]) -> None:
        cursor = 0
        for base, record in q8_records:
            width = widths if not isinstance(widths, dict) else widths[record["name"]]
            launch_q8_record(base, record, width, q8_out[cursor:])
            cursor += record["rows"]

    def fill_q8_name(name: str, width: int) -> None:
        cursor = 0
        for base, record in by_name[name]:
            launch_q8_record(base, record, width, q8_out[cursor:])
            cursor += record["rows"]

    def q5_gate_up_layer(width: int | None, layer: int) -> None:
        if width is None:
            kernels["q5_gate_up_n"](
                (8 * 1536,), (256,), (x, q5, slots[layer], positions, gate, up), stream=stream
            )
        else:
            groups = 256 // width
            kernels[f"q5_gate_up_n1c_{width}"](
                ((8 * 1536 + groups - 1) // groups,), (256,),
                (x, q5, slots[layer], positions, gate, up), stream=stream,
            )

    def q5_down_layer(width: int | None, layer: int) -> None:
        if width is None:
            kernels["q5_down_n"](
                (8 * 2048,), (256,), (gate, q5, slots[layer], positions, down), stream=stream
            )
        else:
            groups = 256 // width
            kernels[f"q5_down_n1c_{width}"](
                ((8 * 2048 + groups - 1) // groups,), (256,),
                (gate, q5, slots[layer], positions, down), stream=stream,
            )

    def q5_gate_up_plane(width: int) -> None:
        for layer in range(LAYERS):
            q5_gate_up_layer(width, layer)

    def q5_down_plane(width: int) -> None:
        for layer in range(LAYERS):
            q5_down_layer(width, layer)

    def fill_q5(widths: int | None | dict[str, int]) -> None:
        gate_width = widths if not isinstance(widths, dict) else widths["gate_up"]
        down_width = widths if not isinstance(widths, dict) else widths["down"]
        for layer in range(LAYERS):
            q5_gate_up_layer(gate_width, layer)
            q5_down_layer(down_width, layer)

    def capture_q8(widths: int | None | dict[str, int]) -> np.ndarray:
        fill_q8(widths)
        stream.synchronize()
        return cp.asnumpy(q8_out)

    def capture_q5(widths: int | None | dict[str, int]) -> np.ndarray:
        captured = np.empty((LAYERS, 8 * (768 + 768 + 2048)), dtype=np.float32)
        gate_width = widths if not isinstance(widths, dict) else widths["gate_up"]
        down_width = widths if not isinstance(widths, dict) else widths["down"]
        for layer in range(LAYERS):
            q5_gate_up_layer(gate_width, layer)
            q5_down_layer(down_width, layer)
            stream.synchronize()
            captured[layer] = np.concatenate((cp.asnumpy(gate), cp.asnumpy(up), cp.asnumpy(down)))
        return captured

    q8_reference = capture_q8(None)
    q5_reference = capture_q5(None)
    correctness = {"q8": {}, "q5": {}}
    for width in WIDTHS:
        correctness["q8"][str(width)] = comparison(capture_q8(width), q8_reference)
        correctness["q5"][str(width)] = comparison(capture_q5(width), q5_reference)

    eligible_q8 = [width for width in WIDTHS if correctness["q8"][str(width)]["bitwise_equal"] and correctness["q8"][str(width)]["finite"]]
    eligible_q5 = [width for width in WIDTHS if correctness["q5"][str(width)]["bitwise_equal"] and correctness["q5"][str(width)]["finite"]]
    if not eligible_q8 or not eligible_q5:
        raise RuntimeError("no exact candidate for one or more banks")

    validation = {"q8": {}, "q5": {"gate_up": {}, "down": {}}}
    selected_q8 = {}
    for name in sorted(by_name):
        launches = {width: (lambda n=name, w=width: fill_q8_name(n, w)) for width in eligible_q8}
        # balanced_measure expects every preregistered width; ineligible widths have
        # already failed the hard gate, so substitute width 16 only for scheduling.
        launches.update({width: launches.get(width, lambda n=name: fill_q8_name(n, 16)) for width in WIDTHS})
        validation["q8"][name] = balanced_measure(stream, launches)
        selected_q8[name] = choose_width(validation["q8"][name], eligible_q8)

    for part, plane in (("gate_up", q5_gate_up_plane), ("down", q5_down_plane)):
        launches = {width: (lambda p=plane, w=width: p(w)) for width in eligible_q5}
        launches.update({width: launches.get(width, lambda p=plane: p(16)) for width in WIDTHS})
        validation["q5"][part] = balanced_measure(stream, launches)
    selected_q5 = {
        part: choose_width(validation["q5"][part], eligible_q5)
        for part in ("gate_up", "down")
    }

    graph_correctness = {
        "q8": comparison(capture_q8(selected_q8), capture_q8(16)),
        "q5": comparison(capture_q5(selected_q5), capture_q5(16)),
    }
    test = {}
    for bank, baseline, candidate in (
        ("q8", lambda: fill_q8(16), lambda: fill_q8(selected_q8)),
        ("q5", lambda: fill_q5(16), lambda: fill_q5(selected_q5)),
    ):
        paired = paired_measure(stream, baseline, candidate)
        p50_ratio = paired["candidate"]["p50"] / paired["baseline"]["p50"]
        p95_ratio = paired["candidate"]["p95"] / paired["baseline"]["p95"]
        exact = graph_correctness[bank]["bitwise_equal"] and graph_correctness[bank]["finite"]
        test[bank] = paired | {
            "p50_ratio": p50_ratio,
            "p95_ratio": p95_ratio,
            "speedup_p50": 1.0 / p50_ratio,
            "speedup_p95": 1.0 / p95_ratio,
            "pass": bool(exact and p50_ratio <= 0.97 and p95_ratio <= 0.97),
        }

    result = {
        "kind": "streamq5_moe_n1c_generalized_exact_reduction_autotuner",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG),
        "script_sha256": sha256(Path(__file__)),
        "seed": SEED,
        "widths": list(WIDTHS),
        "correctness": correctness,
        "validation": validation,
        "selected": {"q8": selected_q8, "q5": selected_q5},
        "kernel_resources": kernel_resources,
        "graph_correctness": graph_correctness,
        "test": test,
        "overall_pass": bool(test["q8"]["pass"] and test["q5"]["pass"]),
        "inputs": {
            "q8_manifest_sha256": sha256(R / "p6a_exact_runtime_bank_result.json"),
            "q8_pinned_aggregate_sha256": q8_sha,
            "q5_slots": LAYERS * 8,
            "q5_record_bytes": EXPERT_BYTES,
        },
        "protocol": {
            "validation_warmups": VALIDATION_WARMUPS,
            "validation_rounds_per_width": VALIDATION_ROUNDS,
            "test_warmups_per_variant": TEST_WARMUPS,
            "paired_test_rounds_per_variant": TEST_ROUNDS,
            "tie_band": 0.005,
        },
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Bit-exact local physical projection-plane autotuning on one GPU; no end-to-end or novelty claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "selected": result["selected"],
        "graph_correctness": graph_correctness,
        "ratios": {bank: {"p50": data["p50_ratio"], "p95": data["p95_ratio"], "pass": data["pass"]} for bank, data in test.items()},
        "overall_pass": result["overall_pass"],
        "wall_seconds": result["wall_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
