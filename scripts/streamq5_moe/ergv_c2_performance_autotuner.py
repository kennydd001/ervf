from __future__ import annotations

import argparse
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
    WIDTHS,
    build_exact_reduction_ir,
    generate_cuda_source,
    source_sha256,
)
from scripts.streamq5_moe.run_n1c_generalized_exact_reduction_autotuner import (  # noqa: E402
    ERVF_SOURCE as N1C_SOURCE,
)
from scripts.streamq5_moe.run_p3a_integrated_expert import (  # noqa: E402
    EXPERT_BYTES,
    LAYERS,
)
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE  # noqa: E402
from scripts.streamq5_moe.run_p7a_kernel_roofline import load_q5, load_q8  # noqa: E402
from scripts.streamq5_moe.run_p7b_ervf_kernel import (  # noqa: E402
    ERVF_SOURCE as P7_SOURCE,
)


REPORTS = ROOT / "reports" / "streamq5_moe"
PREREG = REPORTS / "ERGV_C2_PERFORMANCE_AUTOTUNER_PREREGISTRATION.md"
COMPILE_OUTPUT = REPORTS / "ergv_c2_compile.json"
RUN_OUTPUT = REPORTS / "ergv_c2_performance_autotuner.json"
COMPILER_PATH = ROOT / "src" / "moe_lab" / "ergv_compiler.py"
P7_PATH = ROOT / "scripts" / "streamq5_moe" / "run_p7b_ervf_kernel.py"
N1C_PATH = ROOT / "scripts" / "streamq5_moe" / "run_n1c_generalized_exact_reduction_autotuner.py"
SEED = 120844
VALIDATION_WARMUPS = 3
VALIDATION_ROUNDS = 15
TEST_WARMUPS = 10
TEST_ROUNDS = 120
TIE_BAND = 0.005
N1C_Q8 = {"head": 16, "k": 64, "o": 16, "q": 16, "router": 64, "v": 64}
N1C_Q5 = {"gate_up": 8, "down": 8}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordinary_q8_wrapper(width: int) -> str:
    return f'''
extern "C" __global__ void q8_ergv_c2_w{width}(
    const float* x, const unsigned char* bank, long long base, long long code_bytes,
    int rows, int cols, float* output) {{
    const int WIDTH = {width};
    const int GROUPS = 256 / WIDTH;
    int group = (int)threadIdx.x / WIDTH;
    int lane = (int)threadIdx.x & (WIDTH - 1);
    int row = (int)blockIdx.x * GROUPS + group;
    unsigned mask = __ballot_sync(0xffffffffU, row < rows);
    if (row >= rows) return;
    const signed char* codes = (const signed char*)(bank + base);
    const unsigned short* scales = (const unsigned short*)(bank + base + code_bytes);
    float value = ergv_q8_row_w{width}<>(x, codes, scales, row, cols, lane, mask);
    if (lane == 0) output[row] = round_bf16(value);
}}
'''


def _ordinary_q5_gate_wrapper(width: int) -> str:
    return f'''
extern "C" __global__ void q5_gate_up_ergv_c2_w{width}(
    const float* x, const unsigned char* cache, const int* slots, const int* positions,
    float* gate, float* up) {{
    const int WIDTH = {width};
    const int GROUPS = 256 / WIDTH;
    int group = (int)threadIdx.x / WIDTH;
    int lane = (int)threadIdx.x & (WIDTH - 1);
    int global_row = (int)blockIdx.x * GROUPS + group;
    unsigned mask = __ballot_sync(0xffffffffU, global_row < 8 * 1536);
    if (global_row >= 8 * 1536) return;
    int expert = global_row / 1536;
    int local = global_row - expert * 1536;
    int projection = local >= 768;
    int row = local - projection * 768;
    int output_expert = positions[expert];
    long long base = (long long)slots[expert] * 3035136LL + (long long)projection * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float value = ergv_q5_row_w{width}<>(x, packed, scales, row, 2048, lane, mask);
    if (lane == 0) {{
        if (projection) up[output_expert * 768 + row] = round_bf16(value);
        else gate[output_expert * 768 + row] = round_bf16(value);
    }}
}}
'''


def _ordinary_q5_down_wrapper(width: int) -> str:
    return f'''
extern "C" __global__ void q5_down_ergv_c2_w{width}(
    const float* activation, const unsigned char* cache, const int* slots,
    const int* positions, float* down) {{
    const int WIDTH = {width};
    const int GROUPS = 256 / WIDTH;
    int group = (int)threadIdx.x / WIDTH;
    int lane = (int)threadIdx.x & (WIDTH - 1);
    int global_row = (int)blockIdx.x * GROUPS + group;
    unsigned mask = __ballot_sync(0xffffffffU, global_row < 8 * 2048);
    if (global_row >= 8 * 2048) return;
    int expert = global_row / 2048;
    int row = global_row - expert * 2048;
    int output_expert = positions[expert];
    long long base = (long long)slots[expert] * 3035136LL + 2LL * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float value = ergv_q5_row_w{width}<>(activation + output_expert * 768, packed, scales, row, 768, lane, mask);
    if (lane == 0) down[output_expert * 2048 + row] = round_bf16(value);
}}
'''


def _width64_wrappers() -> str:
    return r'''
extern "C" __global__ void q8_ergv_c2_w64(
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
        value = ergv_q8_row_w64_pre64<>(x, codes, scales, row, cols, lane, 0xffffffffU);
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

extern "C" __global__ void q5_gate_up_ergv_c2_w64(
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
    float value = ergv_q5_row_w64_pre64<>(x, packed, scales, row, 2048, lane, 0xffffffffU);
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

extern "C" __global__ void q5_down_ergv_c2_w64(
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
    float value = ergv_q5_row_w64_pre64<>(activation + output_expert * 768, packed, scales, row, 768, lane, 0xffffffffU);
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


def generated_source() -> str:
    q8_ir = build_exact_reduction_ir("q8", 2048)
    q5_ir = build_exact_reduction_ir("q5", 2048)
    helpers = generate_cuda_source(
        [(q8_ir, width) for width in WIDTHS]
        + [(q5_ir, width) for width in WIDTHS]
    )
    wrappers = []
    for width in WIDTHS:
        if width == 64:
            continue
        wrappers.extend(
            (
                _ordinary_q8_wrapper(width),
                _ordinary_q5_gate_wrapper(width),
                _ordinary_q5_down_wrapper(width),
            )
        )
    wrappers.append(_width64_wrappers())
    return helpers + "\n".join(wrappers)


def kernel_names() -> tuple[str, ...]:
    names = [
        "q8_gemv",
        "q5_gate_up_n",
        "q5_down_n",
        "q8_ervf16",
        "q5_gate_up_ervf16",
        "q5_down_ervf16",
    ]
    for width in WIDTHS:
        names.extend(
            (
                f"q8_n1c_{width}",
                f"q5_gate_up_n1c_{width}",
                f"q5_down_n1c_{width}",
                f"q8_ergv_c2_w{width}",
                f"q5_gate_up_ergv_c2_w{width}",
                f"q5_down_ergv_c2_w{width}",
            )
        )
    return tuple(names)


def compile_module():
    generated = generated_source()
    combined = CUDA_SOURCE + P7_SOURCE + N1C_SOURCE + generated
    names = kernel_names()
    started = time.perf_counter()
    module = cp.RawModule(
        code=combined,
        options=("--std=c++11",),
        name_expressions=names,
    )
    kernels = {name: module.get_function(name) for name in names}
    cp.cuda.runtime.deviceSynchronize()
    compile_seconds = time.perf_counter() - started
    resources = {
        name: {
            key: int(value)
            for key, value in kernel.attributes.items()
            if isinstance(value, (int, np.integer))
        }
        for name, kernel in kernels.items()
    }
    return module, kernels, generated, combined, compile_seconds, resources


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def measured(values: list[float]) -> dict:
    return {"event_ms": values, "stats": stats(values), "iterations": len(values)}


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
    samples = {width: [] for width in WIDTHS}
    for round_index in range(VALIDATION_ROUNDS):
        rotation = round_index % len(WIDTHS)
        order = list(WIDTHS[rotation:] + WIDTHS[:rotation])
        if round_index & 1:
            order.reverse()
        for width in order:
            samples[width].append(timed_launch(stream, launches[width]))
    return {str(width): measured(samples[width]) for width in WIDTHS}


def paired_measure(stream: cp.cuda.Stream, reference, candidate) -> dict[str, dict]:
    for _ in range(TEST_WARMUPS):
        reference()
        candidate()
    stream.synchronize()
    samples = {"reference": [], "candidate": []}
    for round_index in range(TEST_ROUNDS):
        order = (("reference", reference), ("candidate", candidate))
        if round_index & 1:
            order = tuple(reversed(order))
        for name, launch in order:
            samples[name].append(timed_launch(stream, launch))
    return {name: measured(values) for name, values in samples.items()}


def compare(observed: np.ndarray, expected: np.ndarray) -> dict:
    observed_bits = np.ascontiguousarray(observed).view(np.uint32)
    expected_bits = np.ascontiguousarray(expected).view(np.uint32)
    delta = observed.astype(np.float64) - expected.astype(np.float64)
    return {
        "bitwise_equal": bool(np.array_equal(observed_bits, expected_bits)),
        "elements": int(expected.size),
        "different": int(np.count_nonzero(observed_bits != expected_bits)),
        "max_abs": float(np.abs(delta).max(initial=0.0)),
        "finite": bool(np.isfinite(observed).all() and np.isfinite(expected).all()),
    }


def choose_width(results: dict[str, dict], eligible: list[int]) -> int:
    best = min(results[str(width)]["stats"]["p50"] for width in eligible)
    equivalent = [
        width
        for width in eligible
        if results[str(width)]["stats"]["p50"] <= best * (1.0 + TIE_BAND)
    ]
    return 16 if 16 in equivalent else min(equivalent)


def provenance(generated: str, combined: str) -> dict:
    return {
        "preregistration_sha256": sha256(PREREG),
        "compiler_sha256": sha256(COMPILER_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "manual_p7_source_sha256": sha256(P7_PATH),
        "manual_n1c_source_sha256": sha256(N1C_PATH),
        "generated_cuda_sha256": source_sha256(generated),
        "combined_cuda_sha256": source_sha256(combined),
        "generated_cuda_bytes": len(generated.encode("utf-8")),
    }


def compile_phase() -> None:
    started = time.perf_counter()
    module, kernels, generated, combined, compile_seconds, resources = compile_module()
    result = {
        "kind": "ergv_c2_compile_only",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": bool(len(kernels) == len(kernel_names())),
        "compile_seconds": compile_seconds,
        "kernel_count": len(kernels),
        "kernel_resources": resources,
        "source": provenance(generated, combined),
        "device": {
            "id": int(cp.cuda.Device().id),
            "name": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        },
        "physical_banks_loaded": False,
        "kernels_executed": False,
        "timing_executed": False,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Compile-only engineering gate; no kernel output or performance result.",
    }
    COMPILE_OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(COMPILE_OUTPUT),
                "overall_pass": result["overall_pass"],
                "compile_seconds": compile_seconds,
                "kernel_count": len(kernels),
                "generated_cuda_bytes": result["source"]["generated_cuda_bytes"],
                "physical_banks_loaded": False,
                "timing_executed": False,
            },
            indent=2,
        )
    )
    del module, kernels


def run_phase() -> None:
    compile_lock = json.loads(COMPILE_OUTPUT.read_text(encoding="utf-8"))
    if not compile_lock.get("overall_pass"):
        raise RuntimeError("C2 compile pass required")
    started = time.perf_counter()
    module, kernels, generated, combined, compile_seconds, resources = compile_module()
    current_provenance = provenance(generated, combined)
    for key in ("preregistration_sha256", "compiler_sha256", "runner_sha256", "generated_cuda_sha256", "combined_cuda_sha256"):
        if current_provenance[key] != compile_lock["source"][key]:
            raise RuntimeError(f"compile-lock provenance mismatch: {key}")

    _, q8_pin, q8_host, q8_mem, q8, q8_records, q8_sha = load_q8()
    q5_mem, q5 = load_q5()
    stream = cp.cuda.Stream(non_blocking=True)
    rng = np.random.default_rng(SEED)
    x = cp.asarray(rng.standard_normal(4096, dtype=np.float32))
    positions = cp.asarray(np.arange(8, dtype=np.int32))
    slots = [
        cp.asarray(np.arange(layer * 8, layer * 8 + 8, dtype=np.int32))
        for layer in range(LAYERS)
    ]
    gate = cp.empty(8 * 768, dtype=cp.float32)
    up = cp.empty_like(gate)
    down = cp.empty(8 * 2048, dtype=cp.float32)
    q8_total_rows = sum(record["rows"] for _, record in q8_records)
    q8_out = cp.empty(q8_total_rows, dtype=cp.float32)
    by_name: dict[str, list] = {}
    for item in q8_records:
        by_name.setdefault(item[1]["name"], []).append(item)

    def launch_q8_record(backend: str, width: int | None, base, record, output) -> None:
        args = (
            x,
            q8,
            np.int64(base),
            np.int64(record["code_bytes"]),
            np.int32(record["rows"]),
            np.int32(record["cols"]),
            output,
        )
        if backend == "p6":
            kernels["q8_gemv"]((record["rows"],), (256,), args, stream=stream)
            return
        assert width is not None
        groups = 256 // width
        grid = ((record["rows"] + groups - 1) // groups,)
        name = {
            "p7": "q8_ervf16",
            "n1c": f"q8_n1c_{width}",
            "generated": f"q8_ergv_c2_w{width}",
        }[backend]
        kernels[name](grid, (256,), args, stream=stream)

    def fill_q8(backend: str, widths: int | None | dict[str, int]) -> None:
        cursor = 0
        for base, record in q8_records:
            width = widths if not isinstance(widths, dict) else widths[record["name"]]
            launch_q8_record(backend, width, base, record, q8_out[cursor:])
            cursor += record["rows"]

    def fill_q8_name(name: str, backend: str, width: int) -> None:
        cursor = 0
        for base, record in by_name[name]:
            launch_q8_record(backend, width, base, record, q8_out[cursor:])
            cursor += record["rows"]

    def q5_gate_layer(backend: str, width: int | None, layer: int) -> None:
        if backend == "p6":
            name, grid = "q5_gate_up_n", (8 * 1536,)
        else:
            assert width is not None
            groups = 256 // width
            grid = ((8 * 1536 + groups - 1) // groups,)
            name = {
                "p7": "q5_gate_up_ervf16",
                "n1c": f"q5_gate_up_n1c_{width}",
                "generated": f"q5_gate_up_ergv_c2_w{width}",
            }[backend]
        kernels[name](grid, (256,), (x, q5, slots[layer], positions, gate, up), stream=stream)

    def q5_down_layer(backend: str, width: int | None, layer: int) -> None:
        if backend == "p6":
            name, grid = "q5_down_n", (8 * 2048,)
        else:
            assert width is not None
            groups = 256 // width
            grid = ((8 * 2048 + groups - 1) // groups,)
            name = {
                "p7": "q5_down_ervf16",
                "n1c": f"q5_down_n1c_{width}",
                "generated": f"q5_down_ergv_c2_w{width}",
            }[backend]
        kernels[name](grid, (256,), (gate, q5, slots[layer], positions, down), stream=stream)

    def q5_gate_plane(backend: str, width: int) -> None:
        for layer in range(LAYERS):
            q5_gate_layer(backend, width, layer)

    def q5_down_plane(backend: str, width: int) -> None:
        for layer in range(LAYERS):
            q5_down_layer(backend, width, layer)

    def fill_q5(backend: str, widths: int | None | dict[str, int]) -> None:
        gate_width = widths if not isinstance(widths, dict) else widths["gate_up"]
        down_width = widths if not isinstance(widths, dict) else widths["down"]
        for layer in range(LAYERS):
            q5_gate_layer(backend, gate_width, layer)
            q5_down_layer(backend, down_width, layer)

    def capture_q8(backend: str, widths: int | None | dict[str, int]) -> np.ndarray:
        fill_q8(backend, widths)
        stream.synchronize()
        return cp.asnumpy(q8_out)

    def capture_q5(backend: str, widths: int | None | dict[str, int]) -> np.ndarray:
        captured = np.empty((LAYERS, 8 * (768 + 768 + 2048)), dtype=np.float32)
        gate_width = widths if not isinstance(widths, dict) else widths["gate_up"]
        down_width = widths if not isinstance(widths, dict) else widths["down"]
        for layer in range(LAYERS):
            q5_gate_layer(backend, gate_width, layer)
            q5_down_layer(backend, down_width, layer)
            stream.synchronize()
            captured[layer] = np.concatenate(
                (cp.asnumpy(gate), cp.asnumpy(up), cp.asnumpy(down))
            )
        return captured

    q8_reference = capture_q8("p6", None)
    q5_reference = capture_q5("p6", None)
    correctness = {
        "generated": {"q8": {}, "q5": {}},
        "manual_p7_reproduction": {},
        "manual_n1c_reproduction": {},
    }
    for width in WIDTHS:
        correctness["generated"]["q8"][str(width)] = compare(
            capture_q8("generated", width), q8_reference
        )
        correctness["generated"]["q5"][str(width)] = compare(
            capture_q5("generated", width), q5_reference
        )
    correctness["manual_p7_reproduction"] = {
        "q8": compare(capture_q8("generated", 16), capture_q8("p7", 16)),
        "q5": compare(capture_q5("generated", 16), capture_q5("p7", 16)),
    }
    correctness["manual_n1c_reproduction"] = {
        "q8": compare(capture_q8("generated", N1C_Q8), capture_q8("n1c", N1C_Q8)),
        "q5": compare(capture_q5("generated", N1C_Q5), capture_q5("n1c", N1C_Q5)),
    }
    exact_all = all(
        item["bitwise_equal"] and item["finite"]
        for section in correctness["generated"].values()
        for item in section.values()
    ) and all(
        item["bitwise_equal"] and item["finite"]
        for name in ("manual_p7_reproduction", "manual_n1c_reproduction")
        for item in correctness[name].values()
    )
    if not exact_all:
        raise RuntimeError("C2 correctness gate failed before timing")

    validation = {"q8": {}, "q5": {"gate_up": {}, "down": {}}}
    selected_q8: dict[str, int] = {}
    for name in sorted(by_name):
        launches = {
            width: (lambda n=name, w=width: fill_q8_name(n, "generated", w))
            for width in WIDTHS
        }
        validation["q8"][name] = balanced_measure(stream, launches)
        selected_q8[name] = choose_width(validation["q8"][name], list(WIDTHS))
    for part, plane in (
        ("gate_up", q5_gate_plane),
        ("down", q5_down_plane),
    ):
        launches = {
            width: (lambda p=plane, w=width: p("generated", w))
            for width in WIDTHS
        }
        validation["q5"][part] = balanced_measure(stream, launches)
    selected_q5 = {
        part: choose_width(validation["q5"][part], list(WIDTHS))
        for part in ("gate_up", "down")
    }

    frozen_correctness = {
        "q8": compare(
            capture_q8("generated", selected_q8), capture_q8("p7", 16)
        ),
        "q5": compare(
            capture_q5("generated", selected_q5), capture_q5("p7", 16)
        ),
    }
    if not all(item["bitwise_equal"] and item["finite"] for item in frozen_correctness.values()):
        raise RuntimeError("frozen generated graph failed exactness")

    tests = {"versus_manual_p7": {}, "versus_manual_n1c": {}}
    for bank, p7, manual_n1c, candidate in (
        (
            "q8",
            lambda: fill_q8("p7", 16),
            lambda: fill_q8("n1c", N1C_Q8),
            lambda: fill_q8("generated", selected_q8),
        ),
        (
            "q5",
            lambda: fill_q5("p7", 16),
            lambda: fill_q5("n1c", N1C_Q5),
            lambda: fill_q5("generated", selected_q5),
        ),
    ):
        for label, reference in (
            ("versus_manual_p7", p7),
            ("versus_manual_n1c", manual_n1c),
        ):
            paired = paired_measure(stream, reference, candidate)
            p50_ratio = paired["candidate"]["stats"]["p50"] / paired["reference"]["stats"]["p50"]
            p95_ratio = paired["candidate"]["stats"]["p95"] / paired["reference"]["stats"]["p95"]
            tests[label][bank] = paired | {
                "p50_ratio": p50_ratio,
                "p95_ratio": p95_ratio,
                "p50_speedup": 1.0 / p50_ratio,
                "p95_speedup": 1.0 / p95_ratio,
            }

    p7_ratios = tests["versus_manual_p7"]
    breakthrough_families = [
        bank
        for bank, item in p7_ratios.items()
        if item["p50_ratio"] <= 0.98 and item["p95_ratio"] <= 1.00
    ]
    no_bank_regression = all(
        item["p50_ratio"] <= 1.02 and item["p95_ratio"] <= 1.02
        for item in p7_ratios.values()
    )
    gates = {
        "all_generated_widths_exact_q8_q5": exact_all,
        "manual_p7_width16_reproduced": all(
            item["bitwise_equal"] and item["finite"]
            for item in correctness["manual_p7_reproduction"].values()
        ),
        "at_least_one_family_p50_le_0_98_and_p95_le_1_00": bool(breakthrough_families),
        "no_family_regression_over_1_02": no_bank_regression,
    }
    result = {
        "kind": "ergv_c2_generated_physical_bank_performance_autotuner",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(gates.values()),
        "source": current_provenance | {"compile_lock_sha256": sha256(COMPILE_OUTPUT)},
        "compile_seconds_this_process": compile_seconds,
        "correctness": correctness,
        "frozen_correctness": frozen_correctness,
        "validation": validation,
        "selected": {"q8": selected_q8, "q5": selected_q5},
        "manual_n1c_frozen": {"q8": N1C_Q8, "q5": N1C_Q5},
        "tests": tests,
        "breakthrough_families_vs_p7": breakthrough_families,
        "gates": gates,
        "kernel_resources": resources,
        "inputs": {
            "q8_manifest_sha256": sha256(REPORTS / "p6a_exact_runtime_bank_result.json"),
            "q8_pinned_aggregate_sha256": q8_sha,
            "q5_slots": LAYERS * 8,
            "q5_record_bytes": EXPERT_BYTES,
            "seed": SEED,
        },
        "protocol": {
            "widths": list(WIDTHS),
            "validation_warmups": VALIDATION_WARMUPS,
            "validation_rounds_per_width": VALIDATION_ROUNDS,
            "test_warmups_per_pair": TEST_WARMUPS,
            "paired_test_rounds": TEST_ROUNDS,
            "tie_band": TIE_BAND,
        },
        "device": {
            "id": int(cp.cuda.Device().id),
            "name": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        },
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Generated local physical projection-plane autotuning on one GPU; "
            "no end-to-end, second-model, second-architecture, public-baseline, or novelty claim."
        ),
    }
    RUN_OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(RUN_OUTPUT),
                "overall_pass": result["overall_pass"],
                "selected": result["selected"],
                "breakthrough_families_vs_p7": breakthrough_families,
                "p7_ratios": {
                    bank: {"p50": item["p50_ratio"], "p95": item["p95_ratio"]}
                    for bank, item in p7_ratios.items()
                },
                "n1c_parity_ratios": {
                    bank: {"p50": item["p50_ratio"], "p95": item["p95_ratio"]}
                    for bank, item in tests["versus_manual_n1c"].items()
                },
                "gates": gates,
                "wall_seconds": result["wall_seconds"],
            },
            indent=2,
        )
    )
    del module, kernels, q8_mem, q5_mem, q8_pin, q8_host
    if not result["overall_pass"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("compile", "run"), default="compile")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "compile":
        compile_phase()
    else:
        run_phase()
