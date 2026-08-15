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
from scripts.streamq5_moe.run_p7a_kernel_roofline import load_q5
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE, comparison, stats


R = ROOT / "reports/streamq5_moe"
PREREG = R / "N1B_Q5_VECTORIZED_LOADS_PREREGISTRATION.md"
OUTPUT = R / "n1b_q5_vectorized_loads.json"
SEED = 120821
CANDIDATES = ("aligned64x2", "aligned32x2")


VECTOR_SOURCE = r'''
__device__ __forceinline__ unsigned long long q5_word_aligned64x2(
    const unsigned char* source) {
    unsigned long long address = (unsigned long long)source;
    int shift = (int)(address & 7ULL) * 8;
    const unsigned long long* aligned =
        (const unsigned long long*)(address & ~7ULL);
    unsigned long long word = aligned[0] >> shift;
    if (shift > 24) word |= aligned[1] << (64 - shift);
    return word & 0xffffffffffULL;
}

__device__ __forceinline__ unsigned long long q5_word_aligned32x2(
    const unsigned char* source) {
    unsigned long long address = (unsigned long long)source;
    int shift = (int)(address & 3U) * 8;
    const unsigned int* aligned = (const unsigned int*)(address & ~3ULL);
    unsigned long long window = (unsigned long long)aligned[0]
        | ((unsigned long long)aligned[1] << 32);
    return (window >> shift) & 0xffffffffffULL;
}

template<int MODE>
__device__ __forceinline__ float q5_ervf_row_vector(
    const float* x, const unsigned char* packed, const unsigned short* scales,
    int row, int cols, int lane) {
    const int WIDTH = 16;
    const int VIRTUAL = 16;
    float partial[VIRTUAL];
    int packs = cols >> 3;
    int groups = cols >> 7;
    #pragma unroll
    for (int virtual_index = 0; virtual_index < VIRTUAL; ++virtual_index) {
        int tid = lane + WIDTH * virtual_index;
        float sum = 0.0f;
        for (int pack = tid; pack < packs; pack += 256) {
            const unsigned char* source = packed
                + ((long long)row * packs + pack) * 5LL;
            unsigned long long word = MODE == 0
                ? q5_word_aligned64x2(source)
                : q5_word_aligned32x2(source);
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
        value += __shfl_down_sync(0xffffffffU, value, offset, WIDTH);
    return value;
}

#define DEFINE_VEC(NAME, MODE) \
extern "C" __global__ void q5_gate_up_##NAME( \
    const float* x, const unsigned char* cache, const int* slots, \
    const int* positions, float* gate, float* up) { \
    int group=(int)threadIdx.x/16; int lane=(int)threadIdx.x&15; \
    int global_row=(int)blockIdx.x*16+group; \
    if(global_row>=8*1536)return; int expert=global_row/1536; \
    int local=global_row-expert*1536; int projection=local>=768; \
    int row=local-projection*768; int oe=positions[expert]; \
    long long base=(long long)slots[expert]*3035136LL \
        +(long long)projection*1011712LL; \
    const unsigned char* packed=cache+base+64; \
    const unsigned short* scales=(const unsigned short*)(cache+base+64+983040); \
    float value=q5_ervf_row_vector<MODE>(x,packed,scales,row,2048,lane); \
    if(lane==0){if(projection)up[oe*768+row]=round_bf16(value); \
        else gate[oe*768+row]=round_bf16(value);} } \
extern "C" __global__ void q5_down_##NAME( \
    const float* activation, const unsigned char* cache, const int* slots, \
    const int* positions, float* down) { \
    int group=(int)threadIdx.x/16; int lane=(int)threadIdx.x&15; \
    int global_row=(int)blockIdx.x*16+group; \
    if(global_row>=8*2048)return; int expert=global_row/2048; \
    int row=global_row-expert*2048; int oe=positions[expert]; \
    long long base=(long long)slots[expert]*3035136LL+2LL*1011712LL; \
    const unsigned char* packed=cache+base+64; \
    const unsigned short* scales=(const unsigned short*)(cache+base+64+983040); \
    float value=q5_ervf_row_vector<MODE>(activation+oe*768,packed,scales,row,768,lane); \
    if(lane==0)down[oe*2048+row]=round_bf16(value); }

DEFINE_VEC(aligned64x2, 0)
DEFINE_VEC(aligned32x2, 1)
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_measure(stream, launches: dict[str, object], warmups: int, iterations: int) -> dict:
    names = list(launches)
    for _ in range(warmups):
        for name in names:
            launches[name]()
    stream.synchronize()
    values = {name: [] for name in names}
    for iteration in range(iterations):
        order = names if iteration % 2 == 0 else list(reversed(names))
        for name in order:
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record(stream)
            launches[name]()
            end.record(stream)
            end.synchronize()
            values[name].append(float(cp.cuda.get_elapsed_time(begin, end)))
    return {name: {"event_ms": rows, "stats": stats(rows)} for name, rows in values.items()}


def main() -> None:
    started = time.perf_counter()
    q5_mem, q5 = load_q5()
    names = ["q5_gate_up_ervf16", "q5_down_ervf16"]
    for candidate in CANDIDATES:
        names += [f"q5_gate_up_{candidate}", f"q5_down_{candidate}"]
    module = cp.RawModule(
        code=CUDA_SOURCE + ERVF_SOURCE + VECTOR_SOURCE,
        options=("--std=c++11",),
        name_expressions=tuple(names),
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
    host_outputs = {
        name: np.empty((LAYERS, 8 * (768 + 768 + 2048)), dtype=np.float32)
        for name in ("baseline",) + CANDIDATES
    }

    def plane(name: str, capture: bool = False) -> None:
        suffix = "ervf16" if name == "baseline" else name
        for layer in range(LAYERS):
            kernels[f"q5_gate_up_{suffix}"](
                (768,), (256,), (x, q5, slots[layer], positions, gate, up), stream=stream
            )
            kernels[f"q5_down_{suffix}"](
                (1024,), (256,), (gate, q5, slots[layer], positions, down), stream=stream
            )
            if capture:
                stream.synchronize()
                host_outputs[name][layer] = np.concatenate(
                    (cp.asnumpy(gate), cp.asnumpy(up), cp.asnumpy(down))
                )

    plane("baseline", capture=True)
    correctness = {}
    for candidate in CANDIDATES:
        plane(candidate, capture=True)
        correctness[candidate] = comparison(host_outputs[candidate], host_outputs["baseline"])

    launches = {name: (lambda n=name: plane(n)) for name in ("baseline",) + CANDIDATES}
    validation = paired_measure(stream, launches, warmups=5, iterations=30)
    correct = [name for name in CANDIDATES if correctness[name]["bitwise_equal"] and correctness[name]["finite"]]
    selected = min(correct, key=lambda name: validation[name]["stats"]["p50"]) if correct else None
    validation_ratio = (
        validation[selected]["stats"]["p50"] / validation["baseline"]["stats"]["p50"]
        if selected else None
    )
    test_opened = bool(selected and validation_ratio <= 0.98)
    test = None
    if test_opened:
        test_measurements = paired_measure(
            stream,
            {"baseline": launches["baseline"], selected: launches[selected]},
            warmups=10,
            iterations=120,
        )
        p50_ratio = test_measurements[selected]["stats"]["p50"] / test_measurements["baseline"]["stats"]["p50"]
        p95_ratio = test_measurements[selected]["stats"]["p95"] / test_measurements["baseline"]["stats"]["p95"]
        test = {
            "measurements": test_measurements,
            "p50_ratio": p50_ratio,
            "p95_ratio": p95_ratio,
            "speedup_p50": 1.0 / p50_ratio,
            "pass": bool(p50_ratio <= 0.97 and p95_ratio <= 1.00 and correctness[selected]["bitwise_equal"]),
        }

    props = cp.cuda.runtime.getDeviceProperties(0)
    result = {
        "kind": "streamq5_moe_n1b_q5_vectorized_loads",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG),
        "script_sha256": sha256(Path(__file__)),
        "seed": SEED,
        "device": {"name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"]},
        "inputs": {"q5_layers": LAYERS, "q5_experts_per_layer": 8, "physical_q5_bytes": int(q5.size)},
        "correctness": correctness,
        "validation": validation,
        "selected": selected,
        "validation_p50_ratio": validation_ratio,
        "test_opened": test_opened,
        "test": test,
        "overall_pass": bool(test and test["pass"]),
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Physical isolated resident Q5 projection-plane test of two source-level load forms; no end-to-end, quality, capacity, cross-GPU, or SOTA claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT), "selected": selected,
        "validation_p50_ratio": validation_ratio, "test_opened": test_opened,
        "test": None if test is None else {key: value for key, value in test.items() if key != "measurements"},
        "overall_pass": result["overall_pass"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
