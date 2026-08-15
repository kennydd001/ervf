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
from scripts.streamq5_moe.run_p3a_integrated_expert import LAYERS
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7a_kernel_roofline import load_q5, load_q8
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE, WIDTHS, comparison, measure


R = ROOT / "reports/streamq5_moe"
P8A_PREREG = R / "P8A_PROJECTION_ADAPTIVE_ERVF_PREREGISTRATION.md"
P8B_PREREG = R / "P8B_SCALE_BROADCAST_ERVF_PREREGISTRATION.md"
P8A_OUTPUT = R / "p8a_projection_adaptive_ervf.json"
P8B_OUTPUT = R / "p8b_scale_broadcast_ervf.json"
SEED = 280812


BROADCAST_SOURCE = r'''
template<int WIDTH>
__device__ __forceinline__ float q5_ervf_broadcast_row(
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
            unsigned int scale_bits = lane == 0
                ? (unsigned int)scales[row * groups + (column >> 7)] : 0U;
            scale_bits = __shfl_sync(0xffffffffU, scale_bits, 0, WIDTH);
            float scale = bf16_to_float((unsigned short)scale_bits);
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
        value += __shfl_down_sync(0xffffffffU, value, offset, WIDTH);
    return value;
}

extern "C" __global__ void q5_gate_up_broadcast16(
    const float* x, const unsigned char* cache, const int* slots,
    const int* positions, float* gate, float* up) {
    const int group = (int)threadIdx.x / 16;
    const int lane = (int)threadIdx.x & 15;
    const int global_row = (int)blockIdx.x * 16 + group;
    if (global_row >= 8 * 1536) return;
    const int expert = global_row / 1536;
    const int local = global_row - expert * 1536;
    const int projection = local >= 768;
    const int row = local - projection * 768;
    const int output_expert = positions[expert];
    const long long base = (long long)slots[expert] * 3035136LL
        + (long long)projection * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float value = q5_ervf_broadcast_row<16>(x, packed, scales, row, 2048, lane);
    if (lane == 0) {
        if (projection) up[output_expert * 768 + row] = round_bf16(value);
        else gate[output_expert * 768 + row] = round_bf16(value);
    }
}

extern "C" __global__ void q5_down_broadcast16(
    const float* activation, const unsigned char* cache, const int* slots,
    const int* positions, float* down) {
    const int group = (int)threadIdx.x / 16;
    const int lane = (int)threadIdx.x & 15;
    const int global_row = (int)blockIdx.x * 16 + group;
    if (global_row >= 8 * 2048) return;
    const int expert = global_row / 2048;
    const int row = global_row - expert * 2048;
    const int output_expert = positions[expert];
    const long long base = (long long)slots[expert] * 3035136LL + 2LL * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float value = q5_ervf_broadcast_row<16>(activation + output_expert * 768,
        packed, scales, row, 768, lane);
    if (lane == 0) down[output_expert * 2048 + row] = round_bf16(value);
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact(measured: dict) -> dict:
    return measured["stats"] | {"iterations": len(measured["event_ms"])}


def main() -> None:
    started = time.perf_counter()
    _, q8_pin, q8_host, q8_mem, q8, q8_records, q8_sha = load_q8()
    q5_mem, q5 = load_q5()
    names = ["q5_gate_up_broadcast16", "q5_down_broadcast16"]
    for width in WIDTHS:
        names.extend((f"q8_ervf{width}", f"q5_gate_up_ervf{width}", f"q5_down_ervf{width}"))
    module = cp.RawModule(
        code=CUDA_SOURCE + ERVF_SOURCE + BROADCAST_SOURCE,
        options=("--std=c++11",), name_expressions=tuple(names)
    )
    kernels = {name: module.get_function(name) for name in names}
    stream = cp.cuda.Stream(non_blocking=True)
    rng = np.random.default_rng(SEED)
    x = cp.asarray(rng.standard_normal(4096, dtype=np.float32))
    positions = cp.asarray(np.arange(8, dtype=np.int32))
    slots = [cp.asarray(np.arange(layer * 8, layer * 8 + 8, dtype=np.int32)) for layer in range(LAYERS)]
    gate = cp.empty(8 * 768, dtype=cp.float32)
    up = cp.empty_like(gate)
    down = cp.empty(8 * 2048, dtype=cp.float32)
    q8_total_rows = sum(record["rows"] for _, record in q8_records)
    q8_out = cp.empty(q8_total_rows, dtype=cp.float32)

    def launch_q8_record(base, record, width, output):
        groups = 256 // width
        grid = (record["rows"] + groups - 1) // groups
        kernels[f"q8_ervf{width}"](
            (grid,), (256,),
            (x, q8, np.int64(base), np.int64(record["code_bytes"]),
             np.int32(record["rows"]), np.int32(record["cols"]), output),
            stream=stream,
        )

    by_name = {}
    for item in q8_records:
        by_name.setdefault(item[1]["name"], []).append(item)

    def fill_q8(widths):
        cursor = 0
        for base, record in q8_records:
            width = widths if isinstance(widths, int) else widths[record["name"]]
            launch_q8_record(base, record, width, q8_out[cursor:])
            cursor += record["rows"]

    def fill_q8_name(name, width):
        cursor = 0
        for base, record in by_name[name]:
            launch_q8_record(base, record, width, q8_out[cursor:])
            cursor += record["rows"]

    def q5_gate_up_layer(width, layer):
        groups = 256 // width
        kernels[f"q5_gate_up_ervf{width}"](
            ((8 * 1536 + groups - 1) // groups,), (256,),
            (x, q5, slots[layer], positions, gate, up), stream=stream,
        )

    def q5_down_layer(width, layer):
        groups = 256 // width
        kernels[f"q5_down_ervf{width}"](
            ((8 * 2048 + groups - 1) // groups,), (256,),
            (gate, q5, slots[layer], positions, down), stream=stream,
        )

    def q5_gate_up_plane(width):
        for layer in range(LAYERS):
            q5_gate_up_layer(width, layer)

    def q5_down_plane(width):
        for layer in range(LAYERS):
            q5_down_layer(width, layer)

    def fill_q5(widths, broadcast=False):
        for layer in range(LAYERS):
            if broadcast:
                kernels["q5_gate_up_broadcast16"]((768,), (256,),
                    (x, q5, slots[layer], positions, gate, up), stream=stream)
                kernels["q5_down_broadcast16"]((1024,), (256,),
                    (gate, q5, slots[layer], positions, down), stream=stream)
            else:
                gate_width = widths if isinstance(widths, int) else widths["gate_up"]
                down_width = widths if isinstance(widths, int) else widths["down"]
                groups = 256 // gate_width
                kernels[f"q5_gate_up_ervf{gate_width}"](
                    ((8 * 1536 + groups - 1) // groups,), (256,),
                    (x, q5, slots[layer], positions, gate, up), stream=stream)
                groups = 256 // down_width
                kernels[f"q5_down_ervf{down_width}"](
                    ((8 * 2048 + groups - 1) // groups,), (256,),
                    (gate, q5, slots[layer], positions, down), stream=stream)

    def capture_q5(widths, broadcast=False):
        captured = np.empty((LAYERS, 8 * (768 + 768 + 2048)), dtype=np.float32)
        for layer in range(LAYERS):
            if broadcast:
                kernels["q5_gate_up_broadcast16"]((768,), (256,),
                    (x, q5, slots[layer], positions, gate, up), stream=stream)
                kernels["q5_down_broadcast16"]((1024,), (256,),
                    (gate, q5, slots[layer], positions, down), stream=stream)
            else:
                gate_width = widths if isinstance(widths, int) else widths["gate_up"]
                down_width = widths if isinstance(widths, int) else widths["down"]
                q5_gate_up_layer(gate_width, layer)
                q5_down_layer(down_width, layer)
            stream.synchronize()
            captured[layer] = np.concatenate((cp.asnumpy(gate), cp.asnumpy(up), cp.asnumpy(down)))
        return captured

    validation_q8 = {}
    selected_q8 = {}
    for name in sorted(by_name):
        validation_q8[name] = {}
        for width in WIDTHS:
            measured = measure(stream, lambda n=name, w=width: fill_q8_name(n, w), 5, 30)
            validation_q8[name][str(width)] = compact(measured)
        selected_q8[name] = min(WIDTHS, key=lambda w: validation_q8[name][str(w)]["p50"])

    validation_q5 = {"gate_up": {}, "down": {}}
    for width in WIDTHS:
        validation_q5["gate_up"][str(width)] = compact(measure(stream, lambda w=width: q5_gate_up_plane(w), 5, 30))
        q5_gate_up_layer(width, 0)
        validation_q5["down"][str(width)] = compact(measure(stream, lambda w=width: q5_down_plane(w), 5, 30))
    selected_q5 = {
        part: min(WIDTHS, key=lambda w: validation_q5[part][str(w)]["p50"])
        for part in ("gate_up", "down")
    }

    fill_q8(16); stream.synchronize(); q8_reference = cp.asnumpy(q8_out)
    fill_q8(selected_q8); stream.synchronize(); q8_observed = cp.asnumpy(q8_out)
    q5_reference = capture_q5(16)
    q5_observed = capture_q5(selected_q5)

    p8a_test = {}
    for bank, baseline_fn, adaptive_fn in (
        ("q8", lambda: fill_q8(16), lambda: fill_q8(selected_q8)),
        ("q5", lambda: fill_q5(16), lambda: fill_q5(selected_q5)),
    ):
        baseline = measure(stream, baseline_fn, 10, 120)
        adaptive = measure(stream, adaptive_fn, 10, 120)
        p50_ratio = adaptive["stats"]["p50"] / baseline["stats"]["p50"]
        p95_ratio = adaptive["stats"]["p95"] / baseline["stats"]["p95"]
        p8a_test[bank] = {
            "baseline_ervf16": compact(baseline), "adaptive": compact(adaptive),
            "p50_ratio": p50_ratio, "p95_ratio": p95_ratio,
            "pass": bool(p50_ratio <= 0.97 and p95_ratio <= 0.97),
        }
    p8a_correctness = {
        "q8": comparison(q8_observed, q8_reference),
        "q5": comparison(q5_observed, q5_reference),
    }
    p8a = {
        "kind": "streamq5_moe_p8a_projection_adaptive_ervf",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(P8A_PREREG), "script_sha256": sha256(Path(__file__)),
        "seed": SEED, "validation": {"q8": validation_q8, "q5": validation_q5},
        "selected": {"q8": selected_q8, "q5": selected_q5},
        "correctness": p8a_correctness, "test": p8a_test,
        "overall_pass": bool(all(v["bitwise_equal"] for v in p8a_correctness.values()) and all(v["pass"] for v in p8a_test.values())),
        "claim_boundary": "Local isolated projection-plane result against ERVF-16.",
    }
    P8A_OUTPUT.write_text(json.dumps(p8a, indent=2), encoding="utf-8")

    q5_reference = capture_q5(16)
    q5_broadcast = capture_q5(16, broadcast=True)
    p8b_correctness = comparison(q5_broadcast, q5_reference)
    p8b_validation = {
        "baseline_ervf16": compact(measure(stream, lambda: fill_q5(16), 5, 30)),
        "broadcast": compact(measure(stream, lambda: fill_q5(16, broadcast=True), 5, 30)),
    }
    p8b_baseline = measure(stream, lambda: fill_q5(16), 10, 120)
    p8b_variant = measure(stream, lambda: fill_q5(16, broadcast=True), 10, 120)
    p8b_p50_ratio = p8b_variant["stats"]["p50"] / p8b_baseline["stats"]["p50"]
    p8b_p95_ratio = p8b_variant["stats"]["p95"] / p8b_baseline["stats"]["p95"]
    p8b = {
        "kind": "streamq5_moe_p8b_scale_broadcast_ervf",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(P8B_PREREG), "script_sha256": sha256(Path(__file__)),
        "seed": SEED, "correctness": p8b_correctness, "validation": p8b_validation,
        "test": {
            "baseline_ervf16": compact(p8b_baseline), "broadcast": compact(p8b_variant),
            "p50_ratio": p8b_p50_ratio, "p95_ratio": p8b_p95_ratio,
            "pass": bool(p8b_correctness["bitwise_equal"] and p8b_p50_ratio <= 0.97 and p8b_p95_ratio <= 0.97),
        },
        "overall_pass": bool(p8b_correctness["bitwise_equal"] and p8b_p50_ratio <= 0.97 and p8b_p95_ratio <= 0.97),
        "claim_boundary": "Local isolated Q5 projection-plane result against ERVF-16.",
        "wall_seconds_including_p8a": time.perf_counter() - started,
    }
    P8B_OUTPUT.write_text(json.dumps(p8b, indent=2), encoding="utf-8")
    print(json.dumps({
        "p8a": {"selected": p8a["selected"], "correctness": p8a_correctness,
                 "ratios": {k: {"p50": v["p50_ratio"], "p95": v["p95_ratio"], "pass": v["pass"]} for k, v in p8a_test.items()},
                 "overall_pass": p8a["overall_pass"]},
        "p8b": {"correctness": p8b_correctness, "p50_ratio": p8b_p50_ratio,
                 "p95_ratio": p8b_p95_ratio, "overall_pass": p8b["overall_pass"]},
        "outputs": [str(P8A_OUTPUT), str(P8B_OUTPUT)],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
