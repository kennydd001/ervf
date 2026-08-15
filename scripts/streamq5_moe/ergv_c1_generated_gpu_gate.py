from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from moe_lab.ergv_compiler import (  # noqa: E402
    build_exact_reduction_ir,
    generate_cuda_source,
    source_sha256,
)
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE  # noqa: E402


REPORTS = ROOT / "reports" / "streamq5_moe"
PREREG = REPORTS / "ERGV_C1_GENERATED_GPU_PREREGISTRATION.md"
OUTPUT = REPORTS / "ergv_c1_generated_gpu_gate.json"
P7_SOURCE_PATH = ROOT / "scripts" / "streamq5_moe" / "run_p7b_ervf_kernel.py"
COMPILER_PATH = ROOT / "src" / "moe_lab" / "ergv_compiler.py"
SEED = 120843
WIDTH = 16
EXPERTS = 8
INTERMEDIATE = 768
HIDDEN = 2048
RECORD_BYTES = 3_035_136
PROJECTION_BYTES = 1_011_712
PACKED_BYTES = 983_040
SCALE_VALUES = 12_288


CUDA_PRELUDE = r'''
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
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16_bits(values: np.ndarray) -> np.ndarray:
    floats = np.asarray(values, dtype=np.float32).copy()
    bits = floats.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def build_wrappers() -> str:
    # Kept separate from the generator: these wrappers only map the local P7
    # bank ABI onto the generated row reducers.  All arithmetic remains in the
    # generated helper.
    return r'''
extern "C" __global__ void q8_ergv_generated_w16(
    const float* x, const unsigned char* bank, long long base, long long code_bytes,
    int rows, int cols, float* output) {
    const int WIDTH = 16;
    const int GROUPS = 256 / WIDTH;
    int group = (int)threadIdx.x / WIDTH;
    int lane = (int)threadIdx.x & (WIDTH - 1);
    int row = (int)blockIdx.x * GROUPS + group;
    if (row >= rows) return;
    const signed char* codes = (const signed char*)(bank + base);
    const unsigned short* scales = (const unsigned short*)(bank + base + code_bytes);
    float value = ergv_q8_row_w16<>(x, codes, scales, row, cols, lane, 0xffffffffU);
    if (lane == 0) output[row] = round_bf16(value);
}
extern "C" __global__ void q5_gate_up_ergv_generated_w16(
    const float* x, const unsigned char* cache, const int* slots, const int* positions,
    float* gate, float* up) {
    const int WIDTH = 16;
    const int GROUPS = 256 / WIDTH;
    int group = (int)threadIdx.x / WIDTH;
    int lane = (int)threadIdx.x & (WIDTH - 1);
    int global_row = (int)blockIdx.x * GROUPS + group;
    if (global_row >= 8 * 1536) return;
    int expert = global_row / 1536;
    int local = global_row - expert * 1536;
    int projection = local >= 768;
    int row = local - projection * 768;
    int output_expert = positions[expert];
    long long base = (long long)slots[expert] * 3035136LL + (long long)projection * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float value = ergv_q5_row_w16<>(x, packed, scales, row, 2048, lane, 0xffffffffU);
    if (lane == 0) {
        if (projection) up[output_expert * 768 + row] = round_bf16(value);
        else gate[output_expert * 768 + row] = round_bf16(value);
    }
}
extern "C" __global__ void q5_down_ergv_generated_w16(
    const float* activation, const unsigned char* cache, const int* slots,
    const int* positions, float* down) {
    const int WIDTH = 16;
    const int GROUPS = 256 / WIDTH;
    int group = (int)threadIdx.x / WIDTH;
    int lane = (int)threadIdx.x & (WIDTH - 1);
    int global_row = (int)blockIdx.x * GROUPS + group;
    if (global_row >= 8 * 2048) return;
    int expert = global_row / 2048;
    int row = global_row - expert * 2048;
    int output_expert = positions[expert];
    long long base = (long long)slots[expert] * 3035136LL + 2LL * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float value = ergv_q5_row_w16<>(activation + output_expert * 768, packed, scales, row, 768, lane, 0xffffffffU);
    if (lane == 0) down[output_expert * 2048 + row] = round_bf16(value);
}
'''


def build_q8_bank(rows: int, columns: int, rng: np.random.Generator) -> np.ndarray:
    codes = rng.integers(-15, 17, size=rows * columns, dtype=np.int8)
    scales = bf16_bits(rng.uniform(0.001, 0.025, size=rows * (columns // 128)).astype(np.float32))
    return np.concatenate((codes.view(np.uint8), scales.view(np.uint8)))


def build_q5_cache(rng: np.random.Generator) -> np.ndarray:
    cache = np.zeros(EXPERTS * RECORD_BYTES, dtype=np.uint8)
    scale_payload = bf16_bits(
        rng.uniform(0.001, 0.025, size=EXPERTS * 3 * SCALE_VALUES).astype(np.float32)
    ).reshape(EXPERTS, 3, SCALE_VALUES)
    for expert in range(EXPERTS):
        for projection in range(3):
            base = expert * RECORD_BYTES + projection * PROJECTION_BYTES
            cache[base + 64 : base + 64 + PACKED_BYTES] = rng.integers(
                0, 256, size=PACKED_BYTES, dtype=np.uint8
            )
            scale_bytes = scale_payload[expert, projection].view(np.uint8)
            start = base + 64 + PACKED_BYTES
            cache[start : start + scale_bytes.size] = scale_bytes
    return cache


def input_families(length: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    alternating = np.empty(length, dtype=np.float32)
    alternating[0::2] = np.float32(16.0)
    alternating[1::2] = np.float32(-1.0 / 256.0)
    cancellation = np.resize(
        np.asarray([1.0e4, 1.0, -1.0e4, -1.0, 3.0, -3.0], dtype=np.float32),
        length,
    )
    return {
        "random": rng.normal(0.0, 0.5, size=length).astype(np.float32),
        "zero": np.zeros(length, dtype=np.float32),
        "alternating_scale": alternating,
        "cancellation": cancellation,
    }


def comparison(observed: np.ndarray, expected: np.ndarray) -> dict:
    observed_bits = np.ascontiguousarray(observed).view(np.uint32)
    expected_bits = np.ascontiguousarray(expected).view(np.uint32)
    return {
        "elements": int(expected.size),
        "different_bits": int(np.count_nonzero(observed_bits != expected_bits)),
        "bitwise_equal": bool(np.array_equal(observed_bits, expected_bits)),
        "finite_observed": bool(np.isfinite(observed).all()),
        "finite_expected": bool(np.isfinite(expected).all()),
        "max_abs": float(
            np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64)), initial=0.0)
        ),
    }


def main() -> None:
    wall_started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    q8_ir = build_exact_reduction_ir("q8", 2048)
    q5_ir = build_exact_reduction_ir("q5", 2048)
    generated = generate_cuda_source(((q8_ir, WIDTH), (q5_ir, WIDTH)))
    wrappers = build_wrappers()
    combined_source = CUDA_PRELUDE + ERVF_SOURCE + generated + wrappers
    names = (
        "q8_ervf16",
        "q5_gate_up_ervf16",
        "q5_down_ervf16",
        "q8_ergv_generated_w16",
        "q5_gate_up_ergv_generated_w16",
        "q5_down_ergv_generated_w16",
    )

    compile_started = time.perf_counter()
    module = cp.RawModule(
        code=combined_source,
        options=("--std=c++11",),
        name_expressions=names,
    )
    kernels = {name: module.get_function(name) for name in names}
    cp.cuda.runtime.deviceSynchronize()
    compile_seconds = time.perf_counter() - compile_started
    if compile_seconds > 120.0:
        raise RuntimeError(f"compile time exceeded 120 seconds: {compile_seconds:.3f}")

    stream = cp.cuda.Stream(non_blocking=True)
    comparisons: list[dict] = []

    q8_inputs = input_families(4096, rng)
    for rows, columns in ((137, 2048), (65, 4096)):
        bank = cp.asarray(build_q8_bank(rows, columns, rng))
        code_bytes = rows * columns
        manual = cp.empty(rows, dtype=cp.float32)
        candidate = cp.empty(rows, dtype=cp.float32)
        grid = ((rows + 15) // 16,)
        for family, host_values in q8_inputs.items():
            x = cp.asarray(host_values[:columns])
            arguments = (
                x,
                bank,
                np.int64(0),
                np.int64(code_bytes),
                np.int32(rows),
                np.int32(columns),
            )
            kernels["q8_ervf16"](grid, (256,), arguments + (manual,), stream=stream)
            kernels["q8_ergv_generated_w16"](
                grid, (256,), arguments + (candidate,), stream=stream
            )
            stream.synchronize()
            check = comparison(cp.asnumpy(candidate), cp.asnumpy(manual))
            comparisons.append(
                {"family": "q8", "shape": [rows, columns], "input": family, **check}
            )

    q5_cache = cp.asarray(build_q5_cache(rng))
    slots = cp.asarray(np.arange(EXPERTS, dtype=np.int32))
    positions = cp.asarray(np.asarray([5, 2, 7, 0, 6, 1, 4, 3], dtype=np.int32))
    gate_manual = cp.empty(EXPERTS * INTERMEDIATE, dtype=cp.float32)
    gate_candidate = cp.empty_like(gate_manual)
    up_manual = cp.empty_like(gate_manual)
    up_candidate = cp.empty_like(gate_manual)
    down_manual = cp.empty(EXPERTS * HIDDEN, dtype=cp.float32)
    down_candidate = cp.empty_like(down_manual)
    q5_inputs = input_families(HIDDEN, rng)
    gate_grid = ((EXPERTS * 1536 + 15) // 16,)
    down_grid = ((EXPERTS * 2048 + 15) // 16,)
    for family, host_values in q5_inputs.items():
        x = cp.asarray(host_values)
        activation_host = np.concatenate(
            [np.roll(host_values[:INTERMEDIATE], expert * 17) for expert in range(EXPERTS)]
        ).astype(np.float32)
        activation = cp.asarray(activation_host)
        kernels["q5_gate_up_ervf16"](
            gate_grid,
            (256,),
            (x, q5_cache, slots, positions, gate_manual, up_manual),
            stream=stream,
        )
        kernels["q5_gate_up_ergv_generated_w16"](
            gate_grid,
            (256,),
            (x, q5_cache, slots, positions, gate_candidate, up_candidate),
            stream=stream,
        )
        kernels["q5_down_ervf16"](
            down_grid,
            (256,),
            (activation, q5_cache, slots, positions, down_manual),
            stream=stream,
        )
        kernels["q5_down_ergv_generated_w16"](
            down_grid,
            (256,),
            (activation, q5_cache, slots, positions, down_candidate),
            stream=stream,
        )
        stream.synchronize()
        for projection, candidate, manual in (
            ("gate", gate_candidate, gate_manual),
            ("up", up_candidate, up_manual),
            ("down", down_candidate, down_manual),
        ):
            check = comparison(cp.asnumpy(candidate), cp.asnumpy(manual))
            comparisons.append(
                {"family": "q5", "projection": projection, "input": family, **check}
            )

    total_elements = sum(item["elements"] for item in comparisons)
    different = sum(item["different_bits"] for item in comparisons)
    all_finite = all(
        item["finite_observed"] and item["finite_expected"] for item in comparisons
    )
    overall_pass = bool(different == 0 and all_finite and compile_seconds <= 120.0)
    result = {
        "kind": "ergv_c1_generated_vs_manual_p7_gpu_gate",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": overall_pass,
        "compile_seconds": compile_seconds,
        "comparison_groups": len(comparisons),
        "elements_compared": total_elements,
        "different_bits": different,
        "all_finite": all_finite,
        "comparisons": comparisons,
        "source": {
            "preregistration_sha256": sha256(PREREG),
            "compiler_sha256": sha256(COMPILER_PATH),
            "runner_sha256": sha256(Path(__file__)),
            "manual_p7_source_sha256": sha256(P7_SOURCE_PATH),
            "generated_cuda_sha256": source_sha256(generated),
            "combined_cuda_sha256": source_sha256(combined_source),
            "generated_cuda_bytes": len(generated.encode("utf-8")),
        },
        "protocol": {
            "seed": SEED,
            "width": WIDTH,
            "q8_shapes": [[137, 2048], [65, 4096]],
            "q5_experts": EXPERTS,
            "inputs": list(q5_inputs),
            "performance_timing": False,
        },
        "device": {
            "id": int(cp.cuda.Device().id),
            "name": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        },
        "wall_seconds": time.perf_counter() - wall_started,
        "claim_boundary": (
            "Generated width-16 synthetic GPU equality versus manual P7 only; "
            "no performance, real-model, second-architecture, public-baseline, or novelty claim."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "overall_pass": overall_pass,
                "compile_seconds": compile_seconds,
                "comparison_groups": len(comparisons),
                "elements_compared": total_elements,
                "different_bits": different,
                "all_finite": all_finite,
                "wall_seconds": result["wall_seconds"],
            },
            indent=2,
        )
    )
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
