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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p3a_integrated_expert import (
    BANK_BYTES, BANK_RESULT_PATH, CACHE_BYTES, DOMAINS, EXPERT_BYTES, KV_BYTES,
    LAYERS, TRUNK_BYTES, bases, copy_expert, kernels as serial_kernels,
    pin_bank,
)
from scripts.streamq5_moe.run_p4a_causal_async import (
    async_kernels, initialize_static, percentile, result_row, sha256,
)
from scripts.streamq5_moe.run_p4b_rendezvous_async import (
    load_routes, plan_layer, positive_slope, run_domain_serial, static_sets,
)


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P4D_PACKED_METADATA_OVERLAP_PREREGISTRATION.md"
LOCK_PATH = R / "p4d_packed_overlap_input_lock.json"
EVALUATOR_LOCK = R / "p4d_packed_overlap_evaluator_lock.json"
P1D_VERIFY = R / "p1d_physical_bank_verification.json"
ROUTE_INPUT_LOCK = R / "p4d_route_input_lock.json"
ROUTE_CAPTURE = R / "p4d_route_capture_result.json"


def run_domain_packed(domain, route, fixed_ids, begin, end, pinned, cache,
                      cache_view, layer_bases, compute_stream, copy_stream,
                      kernel_set, seed):
    initialize_static(compute_stream, pinned, cache, layer_bases, fixed_ids)
    fixed_set = [frozenset(row) for row in fixed_ids]
    dynamic = [OrderedDict() for _ in range(LAYERS)]
    state = cp.empty(2048, dtype=cp.float32)
    gate = cp.empty(6144, dtype=cp.float32)
    up = cp.empty(6144, dtype=cp.float32)
    down = cp.empty(16384, dtype=cp.float32)
    metadata = cp.empty(40, dtype=cp.int32)
    hit_slots_d = metadata[8:16]
    hit_positions_d = metadata[16:24]
    miss_slots_d = metadata[24:32]
    miss_positions_d = metadata[32:40]
    pinned_meta = cp.cuda.alloc_pinned_memory(LAYERS * 160)
    host_metadata = np.frombuffer(
        pinned_meta, dtype=np.int32, count=LAYERS * 40
    ).reshape(LAYERS, 40)
    ready = [cp.cuda.Event() for _ in range(LAYERS)]
    gate_up, swiglu, down_kernel, reduce = kernel_set
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
            slots, hits, miss_positions, miss_copies = plan_layer(
                route[token, layer], layer, layer_bases, fixed_ids,
                fixed_set, dynamic
            )
            hit_count = len(hits)
            miss_count = len(miss_positions)
            misses[local] += miss_count
            copied_records += miss_count
            mixed_layers += int(hit_count > 0 and miss_count > 0)
            host = host_metadata[layer]
            host.fill(0)
            if hit_count:
                hp = np.asarray(hits, dtype=np.int32)
                host[8:8 + hit_count] = slots[hp]
                host[16:16 + hit_count] = hp
            if miss_count:
                mp = np.asarray(miss_positions, dtype=np.int32)
                host[24:24 + miss_count] = slots[mp]
                host[32:32 + miss_count] = mp
            cp.cuda.runtime.memcpyAsync(
                metadata.data.ptr, pinned_meta.ptr + layer * 160, 160,
                cp.cuda.runtime.memcpyHostToDevice, compute_stream.ptr
            )
            for expert, slot in miss_copies:
                copy_expert(
                    copy_stream, pinned, cache, layer_bases,
                    layer, expert, slot
                )
            if miss_count:
                ready[layer].record(copy_stream)
            if hit_count:
                gate_up((hit_count * 1536,), (256,),
                    (state, cache_view, hit_slots_d, hit_positions_d, gate, up),
                    stream=compute_stream)
                swiglu((hit_count * 3,), (256,),
                    (gate, up, hit_positions_d), stream=compute_stream)
                down_kernel((hit_count * 2048,), (256,),
                    (gate, cache_view, hit_slots_d, hit_positions_d, down),
                    stream=compute_stream)
            if miss_count:
                compute_stream.wait_event(ready[layer])
                gate_up((miss_count * 1536,), (256,),
                    (state, cache_view, miss_slots_d, miss_positions_d, gate, up),
                    stream=compute_stream)
                swiglu((miss_count * 3,), (256,),
                    (gate, up, miss_positions_d), stream=compute_stream)
                down_kernel((miss_count * 2048,), (256,),
                    (gate, cache_view, miss_slots_d, miss_positions_d, down),
                    stream=compute_stream)
            reduce((8,), (256,), (down, state), stream=compute_stream)
        compute_stream.synchronize()
        copy_stream.synchronize()
        times[local] = (time.perf_counter_ns() - wall_begin) / 1e6
        outputs[local] = cp.asnumpy(state)
        if local % 64 == 0:
            print(json.dumps({"mode": "packed_async", "domain": domain,
                              "token": token, "misses": int(misses[local]),
                              "ms": float(times[local])}), flush=True)
    return times, misses, outputs, copied_records, mixed_layers


def run(phase):
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    evaluator = json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"))
    provenance = {
        PREREG: lock["preregistration_sha256"],
        BANK_RESULT_PATH: lock["p1d_bank_result_sha256"],
        P1D_VERIFY: lock["p1d_verification_sha256"],
        ROUTE_INPUT_LOCK: lock["route_input_lock_sha256"],
        ROUTE_CAPTURE: lock["route_capture_sha256"],
    }
    if any(sha256(path) != expected for path, expected in provenance.items()):
        raise ValueError("P4D provenance mismatch")
    if (sha256(LOCK_PATH) != evaluator["input_lock_sha256"]
            or sha256(Path(__file__)) != evaluator["evaluator_sha256"]):
        raise ValueError("P4D evaluator mismatch")
    capture = json.loads(ROUTE_CAPTURE.read_text(encoding="utf-8"))
    route_lock = json.loads(ROUTE_INPUT_LOCK.read_text(encoding="utf-8"))
    if (capture["status"] != "route_capture_complete"
            or not all(capture["controls"].values())
            or not route_lock["exact_128_context_disjoint_from_prior_decisions"]):
        raise RuntimeError("valid fresh P4D route capture required")
    output = R / f"p4d_packed_overlap_{phase}.json"
    report = R / f"P4D_PACKED_OVERLAP_{phase.upper()}.md"
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite P4D")
    if phase == "test":
        validation_path = R / "p4d_packed_overlap_validation.json"
        if (not validation_path.exists()
                or json.loads(validation_path.read_text(encoding="utf-8"))["status"]
                != "p4d_validation_pass_test_authorized"):
            raise RuntimeError("P4D test not authorized")

    routes, route_hashes = load_routes(ROOT / lock["route_dir"])
    fixed = static_sets(routes)
    bank = json.loads(BANK_RESULT_PATH.read_text(encoding="utf-8"))
    pinned, pinned_hashes, pin_ms = pin_bank(bank)
    layer_bases = bases()
    serial_set = serial_kernels()
    packed_set = async_kernels()
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
    domains = ("general",) if phase == "smoke" else DOMAINS
    serial_rows = {}; async_rows = {}; correctness = {}
    all_st=[]; all_sm=[]; all_at=[]; all_am=[]
    records=0; mixed=0
    for domain in domains:
        seed = lock["initial_state_seed"] + DOMAINS.index(domain) * 100000 + begin
        st, sm, so = run_domain_serial(
            domain, routes[domain], fixed[domain], begin, end, pinned, cache,
            cache_view, layer_bases, compute_stream, serial_set, seed
        )
        at, am, ao, rec, mix = run_domain_packed(
            domain, routes[domain], fixed[domain], begin, end, pinned, cache,
            cache_view, layer_bases, compute_stream, copy_stream, packed_set, seed
        )
        delta = ao - so
        correctness[domain] = {
            "exact": bool(np.array_equal(ao, so)),
            "max_abs": float(np.abs(delta).max()),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(so), 1e-30)),
            "finite": bool(np.isfinite(ao).all()) and bool(np.isfinite(so).all()),
            "values": int(ao.size),
        }
        serial_rows[domain]=result_row(st,sm); async_rows[domain]=result_row(at,am)
        all_st.append(st);all_sm.append(sm);all_at.append(at);all_am.append(am)
        records += rec; mixed += mix
    st=np.concatenate(all_st);sm=np.concatenate(all_sm);at=np.concatenate(all_at);am=np.concatenate(all_am)
    aggregate = {
        "serial": {"tokens":int(st.size),"wall_ms":percentile(st),"misses":percentile(sm.astype(float)),"miss_slope_ms":positive_slope(st,sm)},
        "packed_async": {"tokens":int(at.size),"wall_ms":percentile(at),"misses":percentile(am.astype(float)),"miss_slope_ms":positive_slope(at,am)},
        "speedup":float(st.mean()/at.mean()),
        "prediction_relative_error":float(abs(at.mean()-lock["prediction_ms"]["mean"])/lock["prediction_ms"]["mean"]),
    }
    gates = {
        "fresh_route_capture": capture["status"]=="route_capture_complete" and all(capture["controls"].values()) and route_lock["exact_128_context_disjoint_from_prior_decisions"],
        "full_bank_pinned_and_hashed":len(pinned)==LAYERS and len(pinned)*388497408==BANK_BYTES,
        "pinned_hashes_match":pinned_hashes=={str(i):bank["manifests"][str(i)]["artifact_sha256"] for i in range(LAYERS)},
        "device_co_resident_and_scratch":free_after>=lock["gates"]["minimum_scratch_bytes"],
        "async_misses_match_serial":bool(np.array_equal(am,sm)),
        "copy_records_exact":records==int(am.sum()),
        "copy_bytes_exact":records*EXPERT_BYTES==int(am.sum())*EXPERT_BYTES,
        "outputs_exact":all(x["exact"] for x in correctness.values()),
        "outputs_within_limits":all(x["max_abs"]<=lock["gates"]["max_abs_output_error"] and x["relative_l2"]<=lock["gates"]["relative_l2_output_error"] for x in correctness.values()),
        "finite":all(x["finite"] for x in correctness.values()) and bool(np.isfinite(st).all()) and bool(np.isfinite(at).all()),
        "async_mean_le_20":at.mean()<=lock["gates"]["aggregate_mean_ms_max"],
        "async_p95_le_25":np.percentile(at,95)<=lock["gates"]["aggregate_p95_ms_max"],
        "all_domain_mean_le_22":all(x["wall_ms_stats"]["mean"]<=lock["gates"]["all_domain_mean_ms_max"] for x in async_rows.values()),
        "all_domain_p95_le_30":all(x["wall_ms_stats"]["p95"]<=lock["gates"]["all_domain_p95_ms_max"] for x in async_rows.values()),
        "serial_mean_le_35":st.mean()<=lock["gates"]["serial_mean_ms_max"],
        "serial_miss_slope_positive":aggregate["serial"]["miss_slope_ms"]>0,
        "prediction_within_15pct":aggregate["prediction_relative_error"]<=lock["prediction_ms"]["tolerance_fraction"],
    }
    gates={k:bool(v) for k,v in gates.items()}; passed=all(gates.values())
    if phase=="smoke": status="p4d_smoke_pass" if all(x["exact"] for x in correctness.values()) else "p4d_smoke_fail"
    elif phase=="validation": status="p4d_validation_pass_test_authorized" if passed else "p4d_validation_closed_test_unopened"
    else: status="p4d_packed_overlap_pass" if passed else "p4d_packed_overlap_closed"
    result = {
        "kind":"streamq5_moe_p4d_packed_metadata_full_overlap","completed_utc":datetime.now(timezone.utc).isoformat(),"phase":phase,"status":status,
        "inputs":{"preregistration_sha256":sha256(PREREG),"input_lock_sha256":sha256(LOCK_PATH),"evaluator_lock_sha256":sha256(EVALUATOR_LOCK),"evaluator_sha256":sha256(Path(__file__)),"route_input_lock_sha256":sha256(ROUTE_INPUT_LOCK),"route_capture_sha256":sha256(ROUTE_CAPTURE),"route_artifact_sha256":route_hashes},
        "physical":{"pinned_bank_bytes":len(pinned)*388497408,"pin_and_hash_ms":pin_ms,"cache_bytes":CACHE_BYTES,"trunk_reservation_bytes":TRUNK_BYTES,"kv_reservation_bytes":KV_BYTES,"free_before_bytes":int(free_before),"free_after_bytes":int(free_after),"total_vram_bytes":int(total_vram),"copied_records":int(records),"copied_bytes":int(records*EXPERT_BYTES),"mixed_layer_events":int(mixed)},
        "policy":{"static":20,"dynamic_layers_0_7":15,"dynamic_layers_8_47":14,"lookahead_layers":0,"mixed_layer_kernel_launches":7,"metadata_h2d_bytes_per_layer":160},
        "serial":serial_rows,"packed_async":async_rows,"correctness":correctness,"aggregate":aggregate,"gates":gates,
        "claim_boundary":"Physical causal packed-metadata full-overlap expert dataplane on fresh routes only; no full-model decode."
    }
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    report.write_text(f"# STREAMQ5-MoE P4D — packed full-overlap {phase}\n\nStatus: **{status}**. Serial mean/p95 {aggregate['serial']['wall_ms']['mean']:.3f}/{aggregate['serial']['wall_ms']['p95']:.3f} ms; packed async {aggregate['packed_async']['wall_ms']['mean']:.3f}/{aggregate['packed_async']['wall_ms']['p95']:.3f} ms.\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("smoke","validation","test"),required=True);a=p.parse_args();r=run(a.phase)
    print(json.dumps({"status":r["status"],"aggregate":r["aggregate"],"gates":r["gates"]},indent=2))
    if r["status"].endswith("fail") or "closed" in r["status"]: raise SystemExit(1)


if __name__=="__main__": main()
