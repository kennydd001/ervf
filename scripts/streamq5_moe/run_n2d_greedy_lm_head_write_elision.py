from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
PREREG = R / "N2D_GREEDY_LM_HEAD_WRITE_ELISION_PREREGISTRATION.md"
MANIFEST = R / "p6a_exact_runtime_bank_result.json"
COMPILE_OUTPUT = R / "n2d_greedy_lm_head_compile.json"
VALIDATION_OUTPUT = R / "n2d_greedy_lm_head_validation.json"
TEST_OUTPUT = R / "n2d_greedy_lm_head_test.json"

ROWS, COLS, WIDTH = 151_936, 2_048, 16
BLOCK_CANDIDATES = ROWS // WIDTH
VALIDATION_SEED, TEST_SEED = 120_823, 120_824
VALIDATION_CYCLES, TEST_CYCLES = 48, 96
WARMUPS = 15


CUDA_SOURCE = r'''
__device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}
__device__ __forceinline__ unsigned short float_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned int lsb = (bits >> 16) & 1U;
    return (unsigned short)((bits + 0x7FFFU + lsb) >> 16);
}
__device__ __forceinline__ float round_bf16(float value) {
    return bf16_to_float(float_to_bf16(value));
}
template<int WIDTH>
__device__ __forceinline__ float q8_ervf_row_n2d(
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
        for (int index = 0; index < stride / WIDTH; ++index)
            partial[index] += partial[index + stride / WIDTH];
    }
    float value = partial[0];
    #pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffU, value, offset, WIDTH);
    return value;
}
extern "C" __global__ void q8_ervf16_full(
    const float* x, const unsigned char* bank, long long code_bytes,
    int rows, int cols, float* logits) {
    int group = (int)threadIdx.x >> 4;
    int lane = (int)threadIdx.x & 15;
    int row = (int)blockIdx.x * 16 + group;
    if (row >= rows) return;
    const signed char* codes = (const signed char*)bank;
    const unsigned short* scales = (const unsigned short*)(bank + code_bytes);
    float value = q8_ervf_row_n2d<16>(x, codes, scales, row, cols, lane);
    if (lane == 0) logits[row] = round_bf16(value);
}
extern "C" __global__ void logits_stats_current(
    const float* logits, int target, float* values, int* argmax_out) {
    int tid = (int)threadIdx.x;
    float local_max = -3.402823466e+38F;
    int local_index = 0;
    for (int i = tid; i < 151936; i += 256) {
        float value = logits[i];
        if (value > local_max || (value == local_max && i < local_index)) {
            local_max = value; local_index = i;
        }
    }
    __shared__ float maxima[256];
    __shared__ int indices[256];
    maxima[tid] = local_max; indices[tid] = local_index; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = maxima[tid + stride]; int other_index = indices[tid + stride];
            if (other > maxima[tid] || (other == maxima[tid] && other_index < indices[tid])) {
                maxima[tid] = other; indices[tid] = other_index;
            }
        }
        __syncthreads();
    }
    float maximum = maxima[0];
    float sum = 0.0f;
    for (int i = tid; i < 151936; i += 256) sum += expf(logits[i] - maximum);
    maxima[tid] = sum; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) maxima[tid] += maxima[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        values[0] = maximum + logf(maxima[0]);
        values[1] = target >= 0 ? logits[target] : 0.0f;
        argmax_out[0] = indices[0];
    }
}
extern "C" __global__ void argmax_only_full(
    const float* logits, int rows, float* maximum_out, int* argmax_out) {
    int tid = (int)threadIdx.x;
    float local_max = -3.402823466e+38F;
    int local_index = 2147483647;
    for (int i = tid; i < rows; i += 256) {
        float value = logits[i];
        if (value > local_max || (value == local_max && i < local_index)) {
            local_max = value; local_index = i;
        }
    }
    __shared__ float maxima[256];
    __shared__ int indices[256];
    maxima[tid] = local_max; indices[tid] = local_index; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = maxima[tid + stride]; int other_index = indices[tid + stride];
            if (other > maxima[tid] || (other == maxima[tid] && other_index < indices[tid])) {
                maxima[tid] = other; indices[tid] = other_index;
            }
        }
        __syncthreads();
    }
    if (tid == 0) { maximum_out[0] = maxima[0]; argmax_out[0] = indices[0]; }
}
extern "C" __global__ void q8_ervf16_block_argmax(
    const float* x, const unsigned char* bank, long long code_bytes,
    int rows, int cols, float* candidate_values, int* candidate_indices) {
    int group = (int)threadIdx.x >> 4;
    int lane = (int)threadIdx.x & 15;
    int row = (int)blockIdx.x * 16 + group;
    const signed char* codes = (const signed char*)bank;
    const unsigned short* scales = (const unsigned short*)(bank + code_bytes);
    float value = -3.402823466e+38F;
    if (row < rows) value = round_bf16(q8_ervf_row_n2d<16>(x, codes, scales, row, cols, lane));
    __shared__ float maxima[16];
    __shared__ int indices[16];
    if (lane == 0) { maxima[group] = value; indices[group] = row < rows ? row : 2147483647; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float best = maxima[0]; int best_index = indices[0];
        #pragma unroll
        for (int item = 1; item < 16; ++item) {
            float other = maxima[item]; int other_index = indices[item];
            if (other > best || (other == best && other_index < best_index)) {
                best = other; best_index = other_index;
            }
        }
        candidate_values[blockIdx.x] = best;
        candidate_indices[blockIdx.x] = best_index;
    }
}
extern "C" __global__ void reduce_block_candidates(
    const float* candidate_values, const int* candidate_indices, int count,
    float* maximum_out, int* argmax_out) {
    int tid = (int)threadIdx.x;
    float local_max = -3.402823466e+38F;
    int local_index = 2147483647;
    for (int i = tid; i < count; i += 256) {
        float value = candidate_values[i]; int index = candidate_indices[i];
        if (value > local_max || (value == local_max && index < local_index)) {
            local_max = value; local_index = index;
        }
    }
    __shared__ float maxima[256];
    __shared__ int indices[256];
    maxima[tid] = local_max; indices[tid] = local_index; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = maxima[tid + stride]; int other_index = indices[tid + stride];
            if (other > maxima[tid] || (other == maxima[tid] && other_index < indices[tid])) {
                maxima[tid] = other; indices[tid] = other_index;
            }
        }
        __syncthreads();
    }
    if (tid == 0) { maximum_out[0] = maxima[0]; argmax_out[0] = indices[0]; }
}
'''

KERNEL_NAMES = (
    "q8_ervf16_full",
    "logits_stats_current",
    "argmax_only_full",
    "q8_ervf16_block_argmax",
    "reduce_block_candidates",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16_round(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    bits = result.view(np.uint32)
    bits += np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    bits &= np.uint32(0xFFFF0000)
    return result


def stats(values: list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "min": float(x.min()),
        "max": float(x.max()),
        "samples": int(x.size),
    }


def compile_kernels():
    module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++11",), name_expressions=KERNEL_NAMES)
    return module, {name: module.get_function(name) for name in KERNEL_NAMES}


def head_record() -> tuple[Path, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record = next(row for row in manifest["records"] if row["name"] == "head")
    path = ROOT / record["artifact"]
    if record["rows"] != ROWS or record["cols"] != COLS or record["bytes"] != 316_026_880:
        raise ValueError("physical head shape/byte contract mismatch")
    if sha256(path) != record["artifact_sha256"]:
        raise ValueError("physical head artifact hash mismatch")
    return path, record


def compile_only() -> None:
    if COMPILE_OUTPUT.exists():
        raise FileExistsError(COMPILE_OUTPUT)
    started = time.perf_counter()
    module, kernels = compile_kernels()
    elapsed = time.perf_counter() - started
    result = {
        "kind": "streamq5_moe_n2d_compile_check",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "compile_pass_timing_sealed",
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "script_sha256": sha256(Path(__file__)),
            "manifest_sha256": sha256(MANIFEST),
        },
        "compiled_kernels": sorted(kernels),
        "compile_wall_seconds": elapsed,
        "gpu_kernel_launched": False,
        "head_loaded": False,
        "timing_partition_opened": False,
        "claim_boundary": "NVRTC compile check only; no head load, kernel launch, correctness or timing.",
    }
    del kernels, module
    COMPILE_OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def load_resident_head(path: Path, record: dict):
    pinned = cp.cuda.alloc_pinned_memory(record["bytes"])
    host = np.frombuffer(pinned, dtype=np.uint8, count=record["bytes"])
    with path.open("rb", buffering=8 * 2**20) as handle:
        if handle.readinto(host) != record["bytes"]:
            raise RuntimeError("short physical head read")
    memory = cp.cuda.alloc(record["bytes"])
    device = cp.ndarray((record["bytes"],), dtype=cp.uint8, memptr=memory)
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.runtime.memcpyAsync(memory.ptr, pinned.ptr, record["bytes"], cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
    stream.synchronize()
    return pinned, host, memory, device


def make_inputs(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return bf16_round(rng.standard_normal((16, COLS), dtype=np.float32))


def evaluate(phase: str) -> dict:
    output = VALIDATION_OUTPUT if phase == "validation" else TEST_OUTPUT
    if output.exists():
        raise FileExistsError(output)
    compile_result = json.loads(COMPILE_OUTPUT.read_text(encoding="utf-8"))
    if compile_result["status"] != "compile_pass_timing_sealed":
        raise RuntimeError("successful sealed compile result required")
    if compile_result["inputs"]["script_sha256"] != sha256(Path(__file__)):
        raise ValueError("script changed after compile check")
    if compile_result["inputs"]["preregistration_sha256"] != sha256(PREREG):
        raise ValueError("preregistration changed after compile check")
    if phase == "test":
        validation = json.loads(VALIDATION_OUTPUT.read_text(encoding="utf-8"))
        if not validation.get("test_authorized"):
            raise RuntimeError("N2D test remains sealed")

    seed = VALIDATION_SEED if phase == "validation" else TEST_SEED
    cycles = VALIDATION_CYCLES if phase == "validation" else TEST_CYCLES
    host_inputs = make_inputs(seed)
    input_hash = hashlib.sha256(host_inputs.tobytes()).hexdigest()
    path, record = head_record()
    cp.cuda.runtime.deviceSynchronize()
    free_before, total_vram = cp.cuda.runtime.memGetInfo()
    pinned, pinned_array, head_memory, head = load_resident_head(path, record)
    module, kernels = compile_kernels()
    stream = cp.cuda.Stream(non_blocking=True)
    inputs = cp.asarray(host_inputs)
    zero = cp.zeros(COLS, dtype=cp.float32)
    logits = cp.empty(ROWS, dtype=cp.float32)
    stats_values = cp.empty(2, dtype=cp.float32)
    full_max = cp.empty(1, dtype=cp.float32)
    fused_max = cp.empty(1, dtype=cp.float32)
    arg_a = cp.empty(1, dtype=cp.int32)
    arg_b = cp.empty(1, dtype=cp.int32)
    arg_c = cp.empty(1, dtype=cp.int32)
    candidate_values = cp.empty(BLOCK_CANDIDATES, dtype=cp.float32)
    candidate_indices = cp.empty(BLOCK_CANDIDATES, dtype=cp.int32)

    def launch_head(x, output_logits):
        kernels["q8_ervf16_full"](
            (BLOCK_CANDIDATES,), (256,),
            (x, head, np.int64(record["code_bytes"]), np.int32(ROWS), np.int32(COLS), output_logits),
            stream=stream,
        )

    def launch_a(x):
        launch_head(x, logits)
        kernels["logits_stats_current"]((1,), (256,), (logits, np.int32(-1), stats_values, arg_a), stream=stream)

    def launch_b(x):
        launch_head(x, logits)
        kernels["argmax_only_full"]((1,), (256,), (logits, np.int32(ROWS), full_max, arg_b), stream=stream)

    def launch_c(x):
        kernels["q8_ervf16_block_argmax"](
            (BLOCK_CANDIDATES,), (256,),
            (x, head, np.int64(record["code_bytes"]), np.int32(ROWS), np.int32(COLS), candidate_values, candidate_indices),
            stream=stream,
        )
        kernels["reduce_block_candidates"](
            (1,), (256,),
            (candidate_values, candidate_indices, np.int32(BLOCK_CANDIDATES), fused_max, arg_c),
            stream=stream,
        )

    correctness = []
    correctness_inputs = [zero] + [inputs[index] for index in range(16)]
    for index, x in enumerate(correctness_inputs):
        launch_a(x); stream.synchronize()
        host_logits = cp.asnumpy(logits)
        a_index = int(cp.asnumpy(arg_a)[0])
        launch_b(x); launch_c(x); stream.synchronize()
        b_index = int(cp.asnumpy(arg_b)[0]); c_index = int(cp.asnumpy(arg_c)[0])
        b_value = cp.asnumpy(full_max)[0]; c_value = cp.asnumpy(fused_max)[0]
        exact_index = int(np.argmax(host_logits)); exact_value = host_logits[exact_index]
        correctness.append({
            "input": "all_zero_tie" if index == 0 else index - 1,
            "numpy_argmax": exact_index,
            "path_a_argmax": a_index,
            "path_b_argmax": b_index,
            "path_c_argmax": c_index,
            "selected_value_u32": int(np.asarray(exact_value, dtype=np.float32).view(np.uint32)),
            "path_b_value_u32": int(np.asarray(b_value, dtype=np.float32).view(np.uint32)),
            "path_c_value_u32": int(np.asarray(c_value, dtype=np.float32).view(np.uint32)),
            "all_indices_exact": a_index == b_index == c_index == exact_index,
            "all_values_exact": bool(
                int(np.asarray(exact_value).view(np.uint32))
                == int(np.asarray(b_value).view(np.uint32))
                == int(np.asarray(c_value).view(np.uint32))
            ),
            "finite": bool(np.isfinite(exact_value) and np.isfinite(b_value) and np.isfinite(c_value)),
        })

    for _ in range(WARMUPS):
        for index, fn in enumerate((launch_a, launch_b, launch_c)):
            fn(inputs[index])
    stream.synchronize()

    timings = {"a_current": [], "b_full_argmax": [], "c_fused_candidates": []}
    paired = {"b_over_a": [], "c_over_a": []}

    def timed(fn, x) -> float:
        start = cp.cuda.Event(); end = cp.cuda.Event()
        start.record(stream); fn(x); end.record(stream); end.synchronize()
        return float(cp.cuda.get_elapsed_time(start, end))

    def abba(candidate_name, candidate_fn, x):
        a1 = timed(launch_a, x); candidate1 = timed(candidate_fn, x)
        candidate2 = timed(candidate_fn, x); a2 = timed(launch_a, x)
        timings["a_current"].extend((a1, a2))
        timings[candidate_name].extend((candidate1, candidate2))
        paired[f"{'b' if candidate_name.startswith('b_') else 'c'}_over_a"].append(
            ((candidate1 + candidate2) * 0.5) / ((a1 + a2) * 0.5)
        )

    for cycle in range(cycles):
        x = inputs[cycle % 16]
        pairs = (("b_full_argmax", launch_b), ("c_fused_candidates", launch_c))
        if cycle & 1:
            pairs = tuple(reversed(pairs))
        for name, fn in pairs:
            abba(name, fn, x)

    timing_stats = {name: stats(values) for name, values in timings.items()}
    paired_stats = {name: stats(values) for name, values in paired.items()}
    ratios = {
        "b_over_a_p50": timing_stats["b_full_argmax"]["p50"] / timing_stats["a_current"]["p50"],
        "b_over_a_p95": timing_stats["b_full_argmax"]["p95"] / timing_stats["a_current"]["p95"],
        "c_over_a_p50": timing_stats["c_fused_candidates"]["p50"] / timing_stats["a_current"]["p50"],
        "c_over_a_p95": timing_stats["c_fused_candidates"]["p95"] / timing_stats["a_current"]["p95"],
        "c_over_b_p50": timing_stats["c_fused_candidates"]["p50"] / timing_stats["b_full_argmax"]["p50"],
        "c_over_b_p95": timing_stats["c_fused_candidates"]["p95"] / timing_stats["b_full_argmax"]["p95"],
    }
    output_bytes = {
        "a_current": ROWS * 4 + 2 * 4 + 4,
        "b_full_argmax": ROWS * 4 + 4 + 4,
        "c_fused_candidates": BLOCK_CANDIDATES * 4 * 2 + 4 + 4,
    }
    output_bytes["c_bytes_saved_vs_a"] = output_bytes["a_current"] - output_bytes["c_fused_candidates"]
    output_bytes["c_fraction_eliminated_vs_a"] = output_bytes["c_bytes_saved_vs_a"] / output_bytes["a_current"]
    all_correct = all(row["all_indices_exact"] and row["all_values_exact"] and row["finite"] for row in correctness)
    byte_contract = output_bytes == {
        "a_current": 607_756,
        "b_full_argmax": 607_752,
        "c_fused_candidates": 75_976,
        "c_bytes_saved_vs_a": 531_780,
        "c_fraction_eliminated_vs_a": 531_780 / 607_756,
    }
    common_gates = {
        "all_17_inputs_exact": all_correct and len(correctness) == 17,
        "all_zero_tie_selects_index_zero": correctness[0]["numpy_argmax"] == correctness[0]["path_c_argmax"] == 0,
        "output_byte_contract_exact": byte_contract,
        "physical_head_resident": int(head.size) == record["bytes"],
    }
    if phase == "validation":
        performance_gates = {
            "c_over_a_p50_le_1_02": ratios["c_over_a_p50"] <= 1.02,
            "c_over_a_p95_le_1_05": ratios["c_over_a_p95"] <= 1.05,
        }
    else:
        performance_gates = {
            "c_over_a_p50_le_0_98": ratios["c_over_a_p50"] <= 0.98,
            "c_over_a_p95_le_1_00": ratios["c_over_a_p95"] <= 1.00,
            "c_over_b_p50_le_0_99": ratios["c_over_b_p50"] <= 0.99,
            "c_over_b_p95_le_1_00": ratios["c_over_b_p95"] <= 1.00,
        }
    gates = {**common_gates, **performance_gates}
    overall_pass = all(gates.values())
    result = {
        "kind": "streamq5_moe_n2d_greedy_lm_head_write_elision",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "status": f"{phase}_{'pass' if overall_pass else 'negative'}",
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "script_sha256": sha256(Path(__file__)),
            "manifest_sha256": sha256(MANIFEST),
            "compile_result_sha256": sha256(COMPILE_OUTPUT),
            "head_artifact": str(path.relative_to(ROOT)).replace("\\", "/"),
            "head_sha256": record["artifact_sha256"],
            "seed": seed,
            "input_sha256": input_hash,
        },
        "physical": {
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "total_vram_bytes": int(total_vram),
            "free_before_bytes": int(free_before),
            "head_resident_bytes": int(head.size),
            "head_rows": ROWS,
            "head_cols": COLS,
            "head_code_bytes": record["code_bytes"],
            "head_scale_bytes": record["scale_bytes"],
        },
        "protocol": {"warmups_per_path": WARMUPS, "abba_cycles": cycles, "candidate_rows": BLOCK_CANDIDATES},
        "correctness": correctness,
        "output_bytes": output_bytes,
        "timing_ms": timing_stats,
        "paired_abba_ratios": paired_stats,
        "ratios": ratios,
        "gates": gates,
        "overall_pass": overall_pass,
        "test_authorized": bool(overall_pass) if phase == "validation" else None,
        "claim_boundary": (
            "Resident physical Q8 LM-head greedy argmax component only. No CE/top-k/top-p/logit processors, "
            "full decoder, tokens/s, quality, or 80B claim."
        ),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "phase": phase,
        "status": result["status"],
        "timing_ms": timing_stats,
        "ratios": ratios,
        "output_bytes": output_bytes,
        "gates": gates,
        "test_authorized": result["test_authorized"],
    }, indent=2), flush=True)
    del pinned_array, pinned, head, head_memory, module, kernels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("compile", "validation", "test"), default="compile")
    args = parser.parse_args()
    if args.phase == "compile":
        compile_only()
    else:
        evaluate(args.phase)


if __name__ == "__main__":
    main()
