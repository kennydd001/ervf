from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p3a_integrated_expert import (
    BANK_BYTES,
    BANK_RESULT_PATH,
    CACHE_BYTES,
    DOMAINS,
    EXPERT_BYTES,
    KV_BYTES,
    LAYERS,
    MIN_SCRATCH,
    P2A_VERIFY,
    P2C_VERIFY,
    STATIC_SLOTS,
    TRUNK_BYTES,
    bases,
    copy_expert,
    dynamic_slots,
    kernels as serial_kernels,
    launch_layer,
    load_routes,
    pin_bank,
    static_sets,
)


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P4A_CAUSAL_ASYNC_PREREGISTRATION.md"
INPUT_LOCK = R / "p4a_causal_async_input_lock.json"
EVALUATOR_LOCK = R / "p4a_causal_async_evaluator_lock.json"
P1D_VERIFY = R / "p1d_physical_bank_verification.json"
P3A_CAPTURE = R / "p3a_route_capture_result.json"
P3A_VALIDATION = R / "p3a_integrated_expert_validation.json"
P3A_TEST = R / "p3a_integrated_expert_test.json"


CUDA_SOURCE = r'''
__device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}
__device__ __forceinline__ float q5_dot(
    const float* x, const unsigned char* packed, const unsigned short* scales,
    int row, int cols, int tid) {
    int packs_per_row = cols >> 3;
    int groups_per_row = cols >> 7;
    float sum = 0.0f;
    for (int pack = tid; pack < packs_per_row; pack += blockDim.x) {
        const unsigned char* source = packed + ((long long)row * packs_per_row + pack) * 5;
        unsigned long long word = ((unsigned long long)source[0])
            | ((unsigned long long)source[1] << 8)
            | ((unsigned long long)source[2] << 16)
            | ((unsigned long long)source[3] << 24)
            | ((unsigned long long)source[4] << 32);
        int column = pack << 3;
        float scale = bf16_to_float(scales[row * groups_per_row + (column >> 7)]);
        #pragma unroll
        for (int item = 0; item < 8; ++item) {
            int code = ((word >> (item * 5)) & 31ULL) - 15;
            sum += ((float)code) * scale * x[column + item];
        }
    }
    return sum;
}
extern "C" __global__ void q5_gate_up_n(
    const float* x, const unsigned char* cache, const int* slots,
    const int* positions, float* gate, float* up) {
    int local_expert = (int)blockIdx.x / 1536;
    int local = (int)blockIdx.x - local_expert * 1536;
    int projection = local >= 768;
    int row = local - projection * 768;
    int output_expert = positions[local_expert];
    long long base = (long long)slots[local_expert] * 3035136LL
        + (long long)projection * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float sum = q5_dot(x, packed, scales, row, 2048, (int)threadIdx.x);
    __shared__ float reduction[256];
    reduction[threadIdx.x] = sum; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (projection) up[output_expert * 768 + row] = reduction[0];
        else gate[output_expert * 768 + row] = reduction[0];
    }
}
extern "C" __global__ void swiglu_n(
    float* gate, const float* up, const int* positions) {
    int local = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int local_expert = local / 768;
    int column = local - local_expert * 768;
    int output_index = positions[local_expert] * 768 + column;
    float value = gate[output_index];
    gate[output_index] = (value / (1.0f + expf(-value))) * up[output_index];
}
extern "C" __global__ void q5_down_n(
    const float* activation, const unsigned char* cache, const int* slots,
    const int* positions, float* down) {
    int local_expert = (int)blockIdx.x / 2048;
    int row = (int)blockIdx.x - local_expert * 2048;
    int output_expert = positions[local_expert];
    long long base = (long long)slots[local_expert] * 3035136LL + 2LL * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float sum = q5_dot(activation + output_expert * 768, packed, scales,
                       row, 768, (int)threadIdx.x);
    __shared__ float reduction[256];
    reduction[threadIdx.x] = sum; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down[output_expert * 2048 + row] = reduction[0];
}
extern "C" __global__ void reduce_experts(const float* down, float* state) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < 2048) {
        float sum = 0.0f;
        #pragma unroll
        for (int expert = 0; expert < 8; ++expert) sum += down[expert * 2048 + index];
        state[index] = sum * 0.125f;
    }
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def async_kernels():
    return (
        cp.RawKernel(CUDA_SOURCE, "q5_gate_up_n", options=("--std=c++11",)),
        cp.RawKernel(CUDA_SOURCE, "swiglu_n", options=("--std=c++11",)),
        cp.RawKernel(CUDA_SOURCE, "q5_down_n", options=("--std=c++11",)),
        cp.RawKernel(CUDA_SOURCE, "reduce_experts", options=("--std=c++11",)),
    )


def launch_group(kernel_set, stream, state, cache, slots, positions, count, gate, up, down):
    if count == 0:
        return
    gate_up, swiglu, down_kernel, _reduce = kernel_set
    gate_up((count * 1536,), (256,), (state, cache, slots, positions, gate, up), stream=stream)
    swiglu((count * 3,), (256,), (gate, up, positions), stream=stream)
    down_kernel((count * 2048,), (256,), (gate, cache, slots, positions, down), stream=stream)


def initialize_static(stream, pinned, cache, layer_bases, fixed_ids):
    for layer in range(LAYERS):
        for slot, expert in enumerate(fixed_ids[layer]):
            copy_expert(stream, pinned, cache, layer_bases, layer, expert, slot)
    stream.synchronize()


def plan_layer(route_ids, layer, fixed_ids, fixed_set, dynamic):
    slots = np.empty(8, dtype=np.int32)
    hit_positions: list[int] = []
    miss_positions: list[int] = []
    miss_copies: list[tuple[int, int]] = []
    lru = dynamic[layer]
    for position, raw in enumerate(route_ids):
        expert = int(raw)
        if expert in fixed_set[layer]:
            slot = fixed_ids[layer].index(expert)
            hit_positions.append(position)
        elif expert in lru:
            slot = lru[expert]
            lru.move_to_end(expert)
            hit_positions.append(position)
        else:
            if len(lru) < dynamic_slots(layer):
                slot = STATIC_SLOTS + len(lru)
            else:
                _evicted, slot = lru.popitem(last=False)
            lru[expert] = slot
            miss_positions.append(position)
            miss_copies.append((expert, slot))
        slots[position] = bases()[layer] + slot
    return slots, hit_positions, miss_positions, miss_copies


def run_domain_serial(domain, route, fixed_ids, begin, end, pinned, cache, cache_view,
                      layer_bases, stream, kernel_set, seed):
    initialize_static(stream, pinned, cache, layer_bases, fixed_ids)
    fixed_set = [frozenset(row) for row in fixed_ids]
    dynamic = [OrderedDict() for _ in range(LAYERS)]
    state = cp.empty(2048, dtype=cp.float32)
    gate = cp.empty(6144, dtype=cp.float32)
    up = cp.empty(6144, dtype=cp.float32)
    down = cp.empty(16384, dtype=cp.float32)
    slot_ids = cp.empty(8, dtype=cp.int32)
    rng = np.random.default_rng(seed)
    times = np.empty(end - begin, dtype=np.float64)
    misses = np.zeros(end - begin, dtype=np.int64)
    outputs = np.empty((end - begin, 2048), dtype=np.float32)
    for local, token in enumerate(range(begin, end)):
        state.set(rng.standard_normal(2048, dtype=np.float32), stream=stream)
        wall_begin = time.perf_counter_ns()
        for layer in range(LAYERS):
            slots, _hits, miss_positions, miss_copies = plan_layer(
                route[token, layer], layer, fixed_ids, fixed_set, dynamic
            )
            misses[local] += len(miss_positions)
            for expert, slot in miss_copies:
                copy_expert(stream, pinned, cache, layer_bases, layer, expert, slot)
            slot_ids.set(slots, stream=stream)
            launch_layer(kernel_set, stream, state, cache_view, slot_ids, gate, up, down)
        stream.synchronize()
        times[local] = (time.perf_counter_ns() - wall_begin) / 1e6
        outputs[local] = cp.asnumpy(state)
        if local % 64 == 0:
            print(json.dumps({"mode": "serial", "domain": domain, "token": token,
                              "misses": int(misses[local]), "ms": float(times[local])}), flush=True)
    return times, misses, outputs


def run_domain_async(domain, route, fixed_ids, begin, end, pinned, cache, cache_view,
                     layer_bases, compute_stream, copy_stream, kernel_set, seed):
    initialize_static(compute_stream, pinned, cache, layer_bases, fixed_ids)
    fixed_set = [frozenset(row) for row in fixed_ids]
    dynamic = [OrderedDict() for _ in range(LAYERS)]
    state = cp.empty(2048, dtype=cp.float32)
    gate = cp.empty(6144, dtype=cp.float32)
    up = cp.empty(6144, dtype=cp.float32)
    down = cp.empty(16384, dtype=cp.float32)
    hit_slots = cp.empty(8, dtype=cp.int32)
    hit_positions_device = cp.empty(8, dtype=cp.int32)
    miss_slots = cp.empty(8, dtype=cp.int32)
    miss_positions_device = cp.empty(8, dtype=cp.int32)
    ready = [cp.cuda.Event() for _ in range(LAYERS)]
    reduce = kernel_set[3]
    rng = np.random.default_rng(seed)
    times = np.empty(end - begin, dtype=np.float64)
    misses = np.zeros(end - begin, dtype=np.int64)
    outputs = np.empty((end - begin, 2048), dtype=np.float32)
    hidden_records = 0
    for local, token in enumerate(range(begin, end)):
        state.set(rng.standard_normal(2048, dtype=np.float32), stream=compute_stream)
        wall_begin = time.perf_counter_ns()
        for layer in range(LAYERS):
            slots, hit_positions, miss_positions, miss_copies = plan_layer(
                route[token, layer], layer, fixed_ids, fixed_set, dynamic
            )
            misses[local] += len(miss_positions)
            for expert, slot in miss_copies:
                copy_expert(copy_stream, pinned, cache, layer_bases, layer, expert, slot)
            if miss_copies:
                ready[layer].record(copy_stream)
            if hit_positions:
                hp = np.asarray(hit_positions, dtype=np.int32)
                hs = slots[hp]
                hit_slots[:len(hp)].set(hs, stream=compute_stream)
                hit_positions_device[:len(hp)].set(hp, stream=compute_stream)
                launch_group(kernel_set, compute_stream, state, cache_view,
                             hit_slots, hit_positions_device, len(hp), gate, up, down)
            if miss_positions:
                mp = np.asarray(miss_positions, dtype=np.int32)
                ms = slots[mp]
                miss_slots[:len(mp)].set(ms, stream=compute_stream)
                miss_positions_device[:len(mp)].set(mp, stream=compute_stream)
                compute_stream.wait_event(ready[layer])
                launch_group(kernel_set, compute_stream, state, cache_view,
                             miss_slots, miss_positions_device, len(mp), gate, up, down)
            reduce((8,), (256,), (down, state), stream=compute_stream)
        compute_stream.synchronize()
        copy_stream.synchronize()
        times[local] = (time.perf_counter_ns() - wall_begin) / 1e6
        outputs[local] = cp.asnumpy(state)
        hidden_records += int(misses[local])
        if local % 64 == 0:
            print(json.dumps({"mode": "causal_async", "domain": domain,
                              "token": token, "misses": int(misses[local]),
                              "ms": float(times[local])}), flush=True)
    return times, misses, outputs, hidden_records


def result_row(times, misses):
    return {
        "tokens": int(len(times)),
        "misses": misses.tolist(),
        "wall_ms": times.tolist(),
        "miss_stats": percentile(misses.astype(np.float64)),
        "wall_ms_stats": percentile(times),
    }


def run(phase: str) -> dict:
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    evaluator = json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"))
    provenance = {
        PREREG: lock["preregistration_sha256"],
        BANK_RESULT_PATH: lock["p1d_bank_result_sha256"],
        P1D_VERIFY: lock["p1d_verification_sha256"],
        P3A_CAPTURE: lock["p3a_route_capture_sha256"],
        P3A_VALIDATION: lock["p3a_validation_sha256"],
        P3A_TEST: lock["p3a_test_sha256"],
    }
    if any(sha256(path) != expected for path, expected in provenance.items()):
        raise ValueError("P4A provenance mismatch")
    if sha256(INPUT_LOCK) != evaluator["input_lock_sha256"]:
        raise ValueError("P4A input lock mismatch")
    if sha256(Path(__file__)) != evaluator["evaluator_sha256"]:
        raise ValueError("P4A evaluator mismatch")
    output = R / f"p4a_causal_async_{phase}.json"
    report = R / f"P4A_CAUSAL_ASYNC_{phase.upper()}.md"
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite P4A result")
    if phase == "test":
        validation_path = R / "p4a_causal_async_validation.json"
        if (not validation_path.exists()
                or json.loads(validation_path.read_text(encoding="utf-8"))["status"]
                != "p4a_validation_pass_test_authorized"):
            raise RuntimeError("P4A test is not authorized")

    routes, route_hashes = load_routes()
    fixed = static_sets(routes)
    bank = json.loads(BANK_RESULT_PATH.read_text(encoding="utf-8"))
    pinned, pinned_hashes, pin_ms = pin_bank(bank)
    layer_bases = bases()
    serial_set = serial_kernels()
    async_set = async_kernels()
    cp.get_default_memory_pool().free_all_blocks()
    free_before, total_vram = cp.cuda.runtime.memGetInfo()
    cache = cp.cuda.alloc(CACHE_BYTES)
    cache_view = cp.ndarray((CACHE_BYTES,), dtype=cp.uint8, memptr=cache)
    trunk = cp.cuda.alloc(TRUNK_BYTES)
    kv = cp.cuda.alloc(KV_BYTES)
    compute_stream = cp.cuda.Stream(non_blocking=True)
    copy_stream = cp.cuda.Stream(non_blocking=True)
    with compute_stream:
        cp.cuda.runtime.memsetAsync(cache.ptr, 0, CACHE_BYTES, compute_stream.ptr)
        cp.cuda.runtime.memsetAsync(trunk.ptr, 0, TRUNK_BYTES, compute_stream.ptr)
        cp.cuda.runtime.memsetAsync(kv.ptr, 0, KV_BYTES, compute_stream.ptr)
    compute_stream.synchronize()
    free_after, _ = cp.cuda.runtime.memGetInfo()

    begin, end = lock["partitions"][phase]
    selected_domains = ("general",) if phase == "smoke" else DOMAINS
    serial_rows = {}
    async_rows = {}
    correctness = {}
    all_serial_times = []
    all_async_times = []
    all_serial_misses = []
    all_async_misses = []
    total_async_copied_records = 0
    p3a = json.loads((P3A_VALIDATION if phase != "test" else P3A_TEST).read_text(encoding="utf-8"))
    for domain_index, domain in enumerate(selected_domains):
        seed = lock["initial_state_seed"] + DOMAINS.index(domain) * 100000 + begin
        st, sm, so = run_domain_serial(
            domain, routes[domain], fixed[domain], begin, end, pinned, cache,
            cache_view, layer_bases, compute_stream, serial_set, seed
        )
        at, am, ao, copied_records = run_domain_async(
            domain, routes[domain], fixed[domain], begin, end, pinned, cache,
            cache_view, layer_bases, compute_stream, copy_stream, async_set, seed
        )
        delta = ao - so
        correctness[domain] = {
            "exact": bool(np.array_equal(ao, so)),
            "max_abs": float(np.abs(delta).max()),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(so), 1e-30)),
            "finite": bool(np.isfinite(ao).all()) and bool(np.isfinite(so).all()),
            "state_values_compared": int(ao.size),
        }
        serial_rows[domain] = result_row(st, sm)
        async_rows[domain] = result_row(at, am)
        all_serial_times.append(st); all_async_times.append(at)
        all_serial_misses.append(sm); all_async_misses.append(am)
        total_async_copied_records += copied_records

    serial_times = np.concatenate(all_serial_times)
    async_times = np.concatenate(all_async_times)
    serial_misses = np.concatenate(all_serial_misses)
    async_misses = np.concatenate(all_async_misses)
    expected_misses = np.concatenate([
        np.asarray(p3a["per_domain"][domain]["misses"], dtype=np.int64)
        for domain in selected_domains
    ]) if phase != "smoke" else serial_misses.copy()
    serial_reference_mean = (
        float(np.concatenate([
            np.asarray(p3a["per_domain"][domain]["wall_ms"], dtype=np.float64)
            for domain in selected_domains
        ]).mean()) if phase != "smoke" else float(serial_times.mean())
    )
    aggregate = {
        "serial": {"tokens": int(serial_times.size), "wall_ms": percentile(serial_times),
                   "misses": percentile(serial_misses.astype(np.float64))},
        "causal_async": {"tokens": int(async_times.size), "wall_ms": percentile(async_times),
                         "misses": percentile(async_misses.astype(np.float64))},
        "speedup": float(serial_times.mean() / async_times.mean()),
        "mean_prediction_relative_error": float(abs(async_times.mean() - lock["prediction_ms"]["mean"]) / lock["prediction_ms"]["mean"]),
    }
    gates = {
        "full_bank_pinned_and_hashed": len(pinned) * bank["layer_bytes"] == BANK_BYTES
        if "layer_bytes" in bank else len(pinned) == LAYERS,
        "pinned_hashes_match": pinned_hashes == {
            str(layer): bank["manifests"][str(layer)]["artifact_sha256"]
            for layer in range(LAYERS)
        },
        "device_co_resident_and_scratch": free_after >= lock["gates"]["minimum_scratch_bytes"],
        "serial_misses_match_p3a": bool(np.array_equal(serial_misses, expected_misses)),
        "async_misses_match_serial": bool(np.array_equal(async_misses, serial_misses)),
        "async_copy_record_count_exact": total_async_copied_records == int(async_misses.sum()),
        "async_copy_bytes_exact": total_async_copied_records * EXPERT_BYTES == int(async_misses.sum()) * EXPERT_BYTES,
        "outputs_exact": all(row["exact"] for row in correctness.values()),
        "outputs_within_numeric_limit": all(
            row["max_abs"] <= lock["gates"]["max_abs_output_error"]
            and row["relative_l2"] <= lock["gates"]["relative_l2_output_error"]
            for row in correctness.values()
        ),
        "finite_outputs_and_timings": all(row["finite"] for row in correctness.values())
        and bool(np.isfinite(serial_times).all()) and bool(np.isfinite(async_times).all()),
        "aggregate_async_mean_le_20": aggregate["causal_async"]["wall_ms"]["mean"] <= lock["gates"]["aggregate_mean_ms_max"],
        "aggregate_async_p95_le_25": aggregate["causal_async"]["wall_ms"]["p95"] <= lock["gates"]["aggregate_p95_ms_max"],
        "all_domain_async_mean_le_22": all(row["wall_ms_stats"]["mean"] <= lock["gates"]["all_domain_mean_ms_max"] for row in async_rows.values()),
        "all_domain_async_p95_le_30": all(row["wall_ms_stats"]["p95"] <= lock["gates"]["all_domain_p95_ms_max"] for row in async_rows.values()),
        "serial_reproduces_p3a_within_15pct": abs(serial_times.mean() - serial_reference_mean) / serial_reference_mean <= lock["gates"]["serial_reproduction_relative_tolerance"],
    }
    gates = {name: bool(value) for name, value in gates.items()}
    passed = all(gates.values())
    if phase == "smoke":
        status = "p4a_smoke_pass" if all(correctness[d]["exact"] for d in correctness) else "p4a_smoke_fail"
    elif phase == "validation":
        status = "p4a_validation_pass_test_authorized" if passed else "p4a_validation_closed_test_unopened"
    else:
        status = "p4a_causal_async_pass" if passed else "p4a_causal_async_closed"
    result = {
        "kind": "streamq5_moe_p4a_causal_same_layer_async",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "status": status,
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "input_lock_sha256": sha256(INPUT_LOCK),
            "evaluator_lock_sha256": sha256(EVALUATOR_LOCK),
            "evaluator_sha256": sha256(Path(__file__)),
            "route_artifact_sha256": route_hashes,
        },
        "physical": {
            "pinned_bank_bytes": len(pinned) * 388497408,
            "pin_and_hash_ms": pin_ms,
            "cache_bytes": CACHE_BYTES,
            "trunk_reservation_bytes": TRUNK_BYTES,
            "kv_reservation_bytes": KV_BYTES,
            "free_before_bytes": int(free_before),
            "free_after_bytes": int(free_after),
            "total_vram_bytes": int(total_vram),
            "async_copied_records": int(total_async_copied_records),
            "async_copied_bytes": int(total_async_copied_records * EXPERT_BYTES),
        },
        "policy": {"static": 20, "dynamic_layers_0_7": 15,
                   "dynamic_layers_8_47": 14, "lookahead_layers": 0},
        "serial": serial_rows,
        "causal_async": async_rows,
        "correctness": correctness,
        "aggregate": aggregate,
        "prediction": {
            "external_mean_ms": lock["prediction_ms"]["mean"],
            "external_p95_ms": lock["prediction_ms"]["p95"],
            "within_15pct_mean_band": bool(aggregate["mean_prediction_relative_error"] <= lock["prediction_ms"]["mean_tolerance_fraction"]),
        },
        "gates": gates,
        "claim_boundary": (
            "Physical causal same-layer expert H2D overlap only, with trace-replayed "
            "routes and unweighted reduction; no future-route prefetch or full-model decode."
        ),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.write_text(
        f"# STREAMQ5-MoE P4A — causal async {phase}\n\n"
        f"Status: **{status}**. Serial mean/p95 "
        f"{aggregate['serial']['wall_ms']['mean']:.3f}/"
        f"{aggregate['serial']['wall_ms']['p95']:.3f} ms; async "
        f"{aggregate['causal_async']['wall_ms']['mean']:.3f}/"
        f"{aggregate['causal_async']['wall_ms']['p95']:.3f} ms.\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "validation", "test"), required=True)
    args = parser.parse_args()
    result = run(args.phase)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"],
                      "prediction": result["prediction"], "gates": result["gates"]}, indent=2))
    if result["status"].endswith("fail") or "closed" in result["status"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
