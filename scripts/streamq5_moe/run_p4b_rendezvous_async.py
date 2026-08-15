from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p3a_integrated_expert import (
    BANK_BYTES, BANK_RESULT_PATH, CACHE_BYTES, DOMAINS, EXPERT_BYTES, KV_BYTES,
    LAYERS, STATIC_SLOTS, TRUNK_BYTES, bases, copy_expert, dynamic_slots,
    kernels as serial_kernels, launch_layer, pin_bank,
)
from scripts.streamq5_moe.run_p4a_causal_async import (
    CUDA_SOURCE, initialize_static, percentile, result_row, sha256,
)


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P4B_RENDEZVOUS_ASYNC_PREREGISTRATION.md"
LOCK_PATH = R / "p4b_rendezvous_input_lock.json"
EVALUATOR_LOCK = R / "p4b_rendezvous_evaluator_lock.json"
P1D_VERIFY = R / "p1d_physical_bank_verification.json"
P2B_CAPTURE = R / "p2b_route_capture_result.json"
P2C_VALIDATION = R / "p2c_physical_h2d_validation.json"
P2C_TEST = R / "p2c_physical_h2d_test.json"


def load_routes(route_dir: Path):
    routes = {domain: [] for domain in DOMAINS}
    hashes = {}
    for layer in range(LAYERS):
        path = route_dir / f"layer_{layer:02d}.safetensors"
        hashes[str(layer)] = sha256(path)
        with safe_open(path, framework="numpy") as handle:
            for domain in DOMAINS:
                routes[domain].append(
                    handle.get_tensor(f"{domain}_router_ids").astype(np.int64)
                )
    return {domain: np.stack(rows, axis=1) for domain, rows in routes.items()}, hashes


def static_sets(routes):
    selected = {}
    for domain in DOMAINS:
        selected[domain] = []
        for layer in range(LAYERS):
            counts = np.bincount(routes[domain][:512, layer].reshape(-1), minlength=128)
            selected[domain].append(tuple(int(v) for v in np.lexsort((np.arange(128), -counts))[:STATIC_SLOTS]))
    return selected


def gate_up_kernel():
    return cp.RawKernel(CUDA_SOURCE, "q5_gate_up_n", options=("--std=c++11",))


def plan_layer(route_ids, layer, layer_bases, fixed_ids, fixed_set, dynamic):
    slots = np.empty(8, dtype=np.int32)
    hit_positions = []
    miss_positions = []
    miss_copies = []
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
        slots[position] = layer_bases[layer] + slot
    return slots, hit_positions, miss_positions, miss_copies


def run_domain_serial(domain, route, fixed_ids, begin, end, pinned, cache,
                      cache_view, layer_bases, stream, kernel_set, seed):
    initialize_static(stream, pinned, cache, layer_bases, fixed_ids)
    fixed_set = [frozenset(row) for row in fixed_ids]
    dynamic = [OrderedDict() for _ in range(LAYERS)]
    state = cp.empty(2048, dtype=cp.float32)
    gate = cp.empty(6144, dtype=cp.float32)
    up = cp.empty(6144, dtype=cp.float32)
    down = cp.empty(16384, dtype=cp.float32)
    slots_device = cp.empty(8, dtype=cp.int32)
    rng = np.random.default_rng(seed)
    times = np.empty(end - begin, dtype=np.float64)
    misses = np.zeros(end - begin, dtype=np.int64)
    outputs = np.empty((end - begin, 2048), dtype=np.float32)
    for local, token in enumerate(range(begin, end)):
        state.set(rng.standard_normal(2048, dtype=np.float32), stream=stream)
        wall_begin = time.perf_counter_ns()
        for layer in range(LAYERS):
            slots, _hits, miss_positions, miss_copies = plan_layer(
                route[token, layer], layer, layer_bases, fixed_ids, fixed_set, dynamic
            )
            misses[local] += len(miss_positions)
            for expert, slot in miss_copies:
                copy_expert(stream, pinned, cache, layer_bases, layer, expert, slot)
            slots_device.set(slots, stream=stream)
            launch_layer(kernel_set, stream, state, cache_view, slots_device, gate, up, down)
        stream.synchronize()
        times[local] = (time.perf_counter_ns() - wall_begin) / 1e6
        outputs[local] = cp.asnumpy(state)
        if local % 64 == 0:
            print(json.dumps({"mode": "serial", "domain": domain, "token": token,
                              "misses": int(misses[local]), "ms": float(times[local])}), flush=True)
    return times, misses, outputs


def run_domain_rendezvous(domain, route, fixed_ids, begin, end, pinned, cache,
                          cache_view, layer_bases, compute_stream, copy_stream,
                          serial_set, split_gate_up, seed):
    initialize_static(compute_stream, pinned, cache, layer_bases, fixed_ids)
    fixed_set = [frozenset(row) for row in fixed_ids]
    dynamic = [OrderedDict() for _ in range(LAYERS)]
    state = cp.empty(2048, dtype=cp.float32)
    gate = cp.empty(6144, dtype=cp.float32)
    up = cp.empty(6144, dtype=cp.float32)
    down = cp.empty(16384, dtype=cp.float32)
    metadata = cp.empty(40, dtype=cp.int32)
    full_slots_d = metadata[0:8]
    hit_slots_d = metadata[8:16]
    hit_positions_d = metadata[16:24]
    miss_slots_d = metadata[24:32]
    miss_positions_d = metadata[32:40]
    pinned_meta = cp.cuda.alloc_pinned_memory(LAYERS * 40 * 4)
    host_metadata = np.frombuffer(
        pinned_meta, dtype=np.int32, count=LAYERS * 40
    ).reshape(LAYERS, 40)
    ready = [cp.cuda.Event() for _ in range(LAYERS)]
    _original_gate_up, swiglu, down_kernel, reduce = serial_set
    rng = np.random.default_rng(seed)
    times = np.empty(end - begin, dtype=np.float64)
    misses = np.zeros(end - begin, dtype=np.int64)
    outputs = np.empty((end - begin, 2048), dtype=np.float32)
    copied_records = 0
    mixed_layers = 0
    for local, token in enumerate(range(begin, end)):
        state.set(rng.standard_normal(2048, dtype=np.float32), stream=compute_stream)
        wall_begin = time.perf_counter_ns()
        for layer in range(LAYERS):
            slots, hit_positions, miss_positions, miss_copies = plan_layer(
                route[token, layer], layer, layer_bases, fixed_ids, fixed_set, dynamic
            )
            hit_count = len(hit_positions)
            miss_count = len(miss_positions)
            misses[local] += miss_count
            copied_records += miss_count
            mixed_layers += int(hit_count > 0 and miss_count > 0)
            host_meta = host_metadata[layer]
            host_meta.fill(0)
            host_meta[0:8] = slots
            if hit_count:
                hp = np.asarray(hit_positions, dtype=np.int32)
                host_meta[8:8 + hit_count] = slots[hp]
                host_meta[16:16 + hit_count] = hp
            if miss_count:
                mp = np.asarray(miss_positions, dtype=np.int32)
                host_meta[24:24 + miss_count] = slots[mp]
                host_meta[32:32 + miss_count] = mp
            cp.cuda.runtime.memcpyAsync(
                metadata.data.ptr, pinned_meta.ptr + layer * 160, 160,
                cp.cuda.runtime.memcpyHostToDevice, compute_stream.ptr)
            for expert, slot in miss_copies:
                copy_expert(copy_stream, pinned, cache, layer_bases, layer, expert, slot)
            if miss_count:
                ready[layer].record(copy_stream)
            if hit_count:
                split_gate_up((hit_count * 1536,), (256,),
                    (state, cache_view, hit_slots_d, hit_positions_d, gate, up),
                    stream=compute_stream)
            if miss_count:
                compute_stream.wait_event(ready[layer])
                split_gate_up((miss_count * 1536,), (256,),
                    (state, cache_view, miss_slots_d, miss_positions_d, gate, up),
                    stream=compute_stream)
            swiglu((24,), (256,), (gate, up), stream=compute_stream)
            down_kernel((8 * 2048,), (256,),
                (gate, cache_view, full_slots_d, down), stream=compute_stream)
            reduce((8,), (256,), (down, state), stream=compute_stream)
        compute_stream.synchronize()
        copy_stream.synchronize()
        times[local] = (time.perf_counter_ns() - wall_begin) / 1e6
        outputs[local] = cp.asnumpy(state)
        if local % 64 == 0:
            print(json.dumps({"mode": "rendezvous", "domain": domain,
                              "token": token, "misses": int(misses[local]),
                              "ms": float(times[local])}), flush=True)
    return times, misses, outputs, copied_records, mixed_layers


def positive_slope(times, misses):
    x = np.asarray(misses, dtype=np.float64)
    y = np.asarray(times, dtype=np.float64)
    return float(np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)[0][1])


def run(phase: str):
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    evaluator = json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"))
    provenance = {
        PREREG: lock["preregistration_sha256"],
        BANK_RESULT_PATH: lock["p1d_bank_result_sha256"],
        P1D_VERIFY: lock["p1d_verification_sha256"],
        P2B_CAPTURE: lock["p2b_route_capture_sha256"],
        P2C_VALIDATION: lock["p2c_validation_sha256"],
        P2C_TEST: lock["p2c_test_sha256"],
    }
    if any(sha256(path) != expected for path, expected in provenance.items()):
        raise ValueError("P4B provenance mismatch")
    if sha256(LOCK_PATH) != evaluator["input_lock_sha256"] or sha256(Path(__file__)) != evaluator["evaluator_sha256"]:
        raise ValueError("P4B evaluator lock mismatch")
    output = R / f"p4b_rendezvous_async_{phase}.json"
    report = R / f"P4B_RENDEZVOUS_ASYNC_{phase.upper()}.md"
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite P4B result")
    if phase == "test":
        validation_path = R / "p4b_rendezvous_async_validation.json"
        if (not validation_path.exists() or json.loads(validation_path.read_text(encoding="utf-8"))["status"] != "p4b_validation_pass_test_authorized"):
            raise RuntimeError("P4B test not authorized")

    route_dir = ROOT / lock["route_dir"]
    routes, route_hashes = load_routes(route_dir)
    fixed = static_sets(routes)
    bank = json.loads(BANK_RESULT_PATH.read_text(encoding="utf-8"))
    pinned, pinned_hashes, pin_ms = pin_bank(bank)
    layer_bases = bases()
    serial_set = serial_kernels()
    split_gate_up = gate_up_kernel()
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
    serial_rows = {}; async_rows = {}; correctness = {}
    all_st=[]; all_sm=[]; all_at=[]; all_am=[]
    total_records=0; total_mixed=0
    for domain in selected_domains:
        seed = lock["initial_state_seed"] + DOMAINS.index(domain) * 100000 + begin
        st, sm, so = run_domain_serial(domain, routes[domain], fixed[domain], begin,
            end, pinned, cache, cache_view, layer_bases, compute_stream, serial_set, seed)
        at, am, ao, records, mixed = run_domain_rendezvous(domain, routes[domain],
            fixed[domain], begin, end, pinned, cache, cache_view, layer_bases,
            compute_stream, copy_stream, serial_set, split_gate_up, seed)
        delta = ao - so
        correctness[domain] = {
            "exact": bool(np.array_equal(ao, so)),
            "max_abs": float(np.abs(delta).max()),
            "relative_l2": float(np.linalg.norm(delta)/max(np.linalg.norm(so),1e-30)),
            "finite": bool(np.isfinite(ao).all()) and bool(np.isfinite(so).all()),
            "values": int(ao.size),
        }
        serial_rows[domain]=result_row(st,sm); async_rows[domain]=result_row(at,am)
        all_st.append(st); all_sm.append(sm); all_at.append(at); all_am.append(am)
        total_records += records; total_mixed += mixed
    st=np.concatenate(all_st); sm=np.concatenate(all_sm); at=np.concatenate(all_at); am=np.concatenate(all_am)
    aggregate = {
        "serial": {"tokens":int(st.size),"wall_ms":percentile(st),"misses":percentile(sm.astype(float)),"miss_slope_ms":positive_slope(st,sm)},
        "rendezvous_async": {"tokens":int(at.size),"wall_ms":percentile(at),"misses":percentile(am.astype(float)),"miss_slope_ms":positive_slope(at,am)},
        "speedup": float(st.mean()/at.mean()),
        "prediction_relative_error": float(abs(at.mean()-lock["prediction_ms"]["mean"])/lock["prediction_ms"]["mean"]),
    }
    gates = {
        "full_bank_pinned_and_hashed": len(pinned)==LAYERS and len(pinned)*388497408==BANK_BYTES,
        "pinned_hashes_match": pinned_hashes=={str(i):bank["manifests"][str(i)]["artifact_sha256"] for i in range(LAYERS)},
        "device_co_resident_and_scratch": free_after>=lock["gates"]["minimum_scratch_bytes"],
        "async_misses_match_serial": bool(np.array_equal(am,sm)),
        "copy_records_exact": total_records==int(am.sum()),
        "copy_bytes_exact": total_records*EXPERT_BYTES==int(am.sum())*EXPERT_BYTES,
        "outputs_exact": all(row["exact"] for row in correctness.values()),
        "outputs_within_limits": all(row["max_abs"]<=lock["gates"]["max_abs_output_error"] and row["relative_l2"]<=lock["gates"]["relative_l2_output_error"] for row in correctness.values()),
        "finite": all(row["finite"] for row in correctness.values()) and bool(np.isfinite(st).all()) and bool(np.isfinite(at).all()),
        "async_mean_le_20": at.mean()<=lock["gates"]["aggregate_mean_ms_max"],
        "async_p95_le_25": np.percentile(at,95)<=lock["gates"]["aggregate_p95_ms_max"],
        "all_domain_mean_le_22": all(row["wall_ms_stats"]["mean"]<=lock["gates"]["all_domain_mean_ms_max"] for row in async_rows.values()),
        "all_domain_p95_le_30": all(row["wall_ms_stats"]["p95"]<=lock["gates"]["all_domain_p95_ms_max"] for row in async_rows.values()),
        "serial_mean_le_35": st.mean()<=lock["gates"]["serial_mean_ms_max"],
        "serial_miss_slope_positive": aggregate["serial"]["miss_slope_ms"]>0,
        "prediction_within_15pct": aggregate["prediction_relative_error"]<=lock["prediction_ms"]["tolerance_fraction"],
    }
    gates={k:bool(v) for k,v in gates.items()}; passed=all(gates.values())
    if phase=="smoke": status="p4b_smoke_pass" if all(row["exact"] for row in correctness.values()) else "p4b_smoke_fail"
    elif phase=="validation": status="p4b_validation_pass_test_authorized" if passed else "p4b_validation_closed_test_unopened"
    else: status="p4b_rendezvous_async_pass" if passed else "p4b_rendezvous_async_closed"
    result={
        "kind":"streamq5_moe_p4b_rendezvous_async","completed_utc":datetime.now(timezone.utc).isoformat(),"phase":phase,"status":status,
        "inputs":{"preregistration_sha256":sha256(PREREG),"input_lock_sha256":sha256(LOCK_PATH),"evaluator_lock_sha256":sha256(EVALUATOR_LOCK),"evaluator_sha256":sha256(Path(__file__)),"route_capture_sha256":sha256(P2B_CAPTURE),"route_artifact_sha256":route_hashes},
        "physical":{"pinned_bank_bytes":len(pinned)*388497408,"pin_and_hash_ms":pin_ms,"cache_bytes":CACHE_BYTES,"trunk_reservation_bytes":TRUNK_BYTES,"kv_reservation_bytes":KV_BYTES,"free_before_bytes":int(free_before),"free_after_bytes":int(free_after),"total_vram_bytes":int(total_vram),"copied_records":int(total_records),"copied_bytes":int(total_records*EXPERT_BYTES),"mixed_layer_events":int(total_mixed)},
        "policy":{"static":20,"dynamic_layers_0_7":15,"dynamic_layers_8_47":14,"lookahead_layers":0,"mixed_layer_kernel_launches":5,"metadata_h2d_bytes_per_layer":160},
        "serial":serial_rows,"rendezvous_async":async_rows,"correctness":correctness,"aggregate":aggregate,"gates":gates,
        "claim_boundary":"Physical causal gate/up-rendezvous expert dataplane on independent routes only; no full-model decode."
    }
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    report.write_text(f"# STREAMQ5-MoE P4B — rendezvous async {phase}\n\nStatus: **{status}**. Serial mean/p95 {aggregate['serial']['wall_ms']['mean']:.3f}/{aggregate['serial']['wall_ms']['p95']:.3f} ms; rendezvous {aggregate['rendezvous_async']['wall_ms']['mean']:.3f}/{aggregate['rendezvous_async']['wall_ms']['p95']:.3f} ms.\n",encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=("smoke","validation","test"),required=True); args=parser.parse_args()
    result=run(args.phase)
    print(json.dumps({"status":result["status"],"aggregate":result["aggregate"],"gates":result["gates"]},indent=2))
    if result["status"].endswith("fail") or "closed" in result["status"]: raise SystemExit(1)


if __name__=="__main__": main()
