from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
from safetensors import safe_open

from moe_lab.reporting import ROOT


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P2C_STATIC_DYNAMIC_REBALANCE_PREREGISTRATION.md"
LOCK_PATH = R / "p2b_route_input_lock.json"
CAPTURE_PATH = R / "p2b_route_capture_result.json"
SELECTION_PATH = R / "p2c_policy_selection.json"
EVALUATOR_LOCK = R / "p2c_h2d_evaluator_lock.json"
P1D_VERIFY = R / "p1d_physical_bank_verification.json"
BANK_RESULT = R / "p1d_physical_bank_result.json"
ROUTE_DIR = ROOT / "reports/runs/streamq5_moe/p2b_routes"
BANK_DIR = ROOT / "reports/runs/streamq5_moe/p1d_q5_bank"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, EXPERTS, TOP_K = 48, 128, 8
STATIC_SLOTS, TOTAL_SLOTS = 20, 1640
EXPERT_BYTES, LAYER_BYTES, BANK_BYTES = 3_035_136, 388_497_408, 18_647_875_584
CACHE_BYTES = 4_977_623_040
TRUNK_BYTES, KV_BYTES, MINIMUM_SCRATCH_BYTES = 1_541_093_376, 402_653_184, 402_653_184
SAMPLE_OFFSETS = (0, EXPERT_BYTES // 2, EXPERT_BYTES - 4096)
SAMPLE_BYTES = 4096


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    raw = memoryview(value).cast("B")
    for begin in range(0, len(raw), 8 * 2**20):
        digest.update(raw[begin : begin + 8 * 2**20])
    return digest.hexdigest()


def dynamic_slots(layer: int) -> int:
    return 15 if layer <= 7 else 14


def layer_bases() -> list[int]:
    bases, cursor = [], 0
    for layer in range(LAYERS):
        bases.append(cursor)
        cursor += STATIC_SLOTS + dynamic_slots(layer)
    if cursor != TOTAL_SLOTS:
        raise RuntimeError("physical slot partition mismatch")
    return bases


def load_routes() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    routes = {domain: [] for domain in DOMAINS}
    hashes = {}
    for layer in range(LAYERS):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"
        hashes[str(layer)] = sha256(path)
        with safe_open(path, framework="numpy") as handle:
            for domain in DOMAINS:
                value = handle.get_tensor(f"{domain}_router_ids").astype(np.int64)
                if value.shape != (1024, TOP_K) or value.min() < 0 or value.max() >= EXPERTS:
                    raise ValueError(f"invalid P2B routes {layer}:{domain}")
                routes[domain].append(value)
    return {domain: np.stack(values, axis=1) for domain, values in routes.items()}, hashes


def static_ids(routes: dict[str, np.ndarray]) -> tuple[dict[str, list[tuple[int, ...]]], dict[str, str]]:
    selected, count_hashes = {}, {}
    for domain in DOMAINS:
        selected[domain] = []
        packed_counts = []
        for layer in range(LAYERS):
            counts = np.bincount(routes[domain][:512, layer, :].reshape(-1), minlength=EXPERTS)
            order = np.lexsort((np.arange(EXPERTS), -counts))
            selected[domain].append(tuple(int(value) for value in order[:STATIC_SLOTS]))
            packed_counts.append(counts.astype(np.int64))
        count_hashes[domain] = hashlib.sha256(np.stack(packed_counts).tobytes()).hexdigest()
    return selected, count_hashes


def simulate(route: np.ndarray, fixed_ids: list[tuple[int, ...]], begin: int, end: int) -> np.ndarray:
    fixed = [frozenset(values) for values in fixed_ids]
    dynamic = [OrderedDict() for _ in range(LAYERS)]
    misses = np.zeros(end - begin, dtype=np.int64)
    for local, token in enumerate(range(begin, end)):
        for layer in range(LAYERS):
            lru = dynamic[layer]
            for raw in route[token, layer]:
                expert = int(raw)
                if expert in fixed[layer]:
                    continue
                if expert in lru:
                    lru.move_to_end(expert)
                else:
                    misses[local] += 1
                    lru[expert] = None
                    if len(lru) > dynamic_slots(layer):
                        lru.popitem(last=False)
    return misses


def pin_full_bank(bank_result: dict):
    pinned_layers = []
    hashes = {}
    started = time.perf_counter()
    for layer in range(LAYERS):
        memory = cp.cuda.alloc_pinned_memory(LAYER_BYTES)
        host = np.frombuffer(memory, dtype=np.uint8, count=LAYER_BYTES)
        path = BANK_DIR / f"layer_{layer:02d}.q5bin"
        with path.open("rb", buffering=8 * 2**20) as handle:
            read = handle.readinto(host)
            if read != LAYER_BYTES or handle.read(1):
                raise RuntimeError(f"pinned layer read failed {layer}")
        observed = sha256_array(host)
        expected = bank_result["manifests"][str(layer)]["artifact_sha256"]
        if observed != expected:
            raise ValueError(f"pinned bank hash mismatch {layer}")
        hashes[str(layer)] = observed
        pinned_layers.append(memory)
        print(json.dumps({"pinned_layer": layer, "pinned_bytes": (layer + 1) * LAYER_BYTES}), flush=True)
    return pinned_layers, hashes, (time.perf_counter() - started) * 1000


def copy_record(stream, pinned_layers, cache_memory, bases, layer: int, expert: int, slot: int):
    source = pinned_layers[layer].ptr + expert * EXPERT_BYTES
    destination = cache_memory.ptr + (bases[layer] + slot) * EXPERT_BYTES
    cp.cuda.runtime.memcpyAsync(destination, source, EXPERT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr)


def sample_matches(pinned_layers, cache_memory, bases, layer: int, expert: int, slot: int) -> bool:
    source = np.frombuffer(pinned_layers[layer], dtype=np.uint8, count=LAYER_BYTES)
    destination_base = (bases[layer] + slot) * EXPERT_BYTES
    source_base = expert * EXPERT_BYTES
    for offset in SAMPLE_OFFSETS:
        device = cp.ndarray((SAMPLE_BYTES,), dtype=cp.uint8, memptr=cache_memory + destination_base + offset)
        observed = cp.asnumpy(device)
        if not np.array_equal(observed, source[source_base + offset : source_base + offset + SAMPLE_BYTES]):
            return False
    return True


def percentile_stats(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def actual_domain(domain: str, route: np.ndarray, fixed_ids: list[tuple[int, ...]], begin: int, end: int, pinned_layers, cache_memory, bases, stream):
    fixed = [frozenset(values) for values in fixed_ids]
    preload_begin, preload_end = cp.cuda.Event(), cp.cuda.Event()
    preload_wall_begin = time.perf_counter_ns(); preload_begin.record(stream)
    for layer in range(LAYERS):
        for slot, expert in enumerate(fixed_ids[layer]):
            copy_record(stream, pinned_layers, cache_memory, bases, layer, expert, slot)
    preload_end.record(stream); preload_end.synchronize()
    preload_wall_ms = (time.perf_counter_ns() - preload_wall_begin) / 1e6
    preload_event_ms = float(cp.cuda.get_elapsed_time(preload_begin, preload_end))

    integrity_failures = 0
    for layer in range(LAYERS):
        integrity_failures += int(not sample_matches(pinned_layers, cache_memory, bases, layer, fixed_ids[layer][0], 0))

    dynamic = [OrderedDict() for _ in range(LAYERS)]
    misses = np.zeros(end - begin, dtype=np.int64)
    event_ms = np.zeros(end - begin, dtype=np.float64)
    wall_ms = np.zeros(end - begin, dtype=np.float64)
    for local, token in enumerate(range(begin, end)):
        begin_event, end_event = cp.cuda.Event(), cp.cuda.Event()
        wall_begin = time.perf_counter_ns(); begin_event.record(stream)
        for layer in range(LAYERS):
            lru = dynamic[layer]
            for raw in route[token, layer]:
                expert = int(raw)
                if expert in fixed[layer]:
                    continue
                if expert in lru:
                    lru.move_to_end(expert)
                else:
                    misses[local] += 1
                    if len(lru) < dynamic_slots(layer):
                        slot = STATIC_SLOTS + len(lru)
                    else:
                        _evicted, slot = lru.popitem(last=False)
                    lru[expert] = slot
                    copy_record(stream, pinned_layers, cache_memory, bases, layer, expert, slot)
        end_event.record(stream); end_event.synchronize()
        wall_ms[local] = (time.perf_counter_ns() - wall_begin) / 1e6
        event_ms[local] = cp.cuda.get_elapsed_time(begin_event, end_event)
        if local % 64 == 0:
            print(json.dumps({"domain": domain, "token": int(token), "misses": int(misses[local]), "wall_ms": float(wall_ms[local])}), flush=True)

    for layer in range(LAYERS):
        if dynamic[layer]:
            expert, slot = next(iter(dynamic[layer].items()))
            integrity_failures += int(not sample_matches(pinned_layers, cache_memory, bases, layer, expert, slot))
    return {
        "tokens": end - begin,
        "preload_wall_ms": preload_wall_ms, "preload_event_ms": preload_event_ms,
        "misses": misses.tolist(), "h2d_bytes": (misses * EXPERT_BYTES).tolist(),
        "wall_ms": wall_ms.tolist(), "event_ms": event_ms.tolist(),
        "miss_stats": percentile_stats(misses.astype(np.float64)),
        "wall_ms_stats": percentile_stats(wall_ms), "event_ms_stats": percentile_stats(event_ms),
        "sample_integrity_failures": integrity_failures,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = R / f"p2c_physical_h2d_{args.split}.json"
    report = R / f"P2C_PHYSICAL_H2D_{args.split.upper()}.md"
    if output.exists() or report.exists():
        raise FileExistsError(f"refusing to overwrite P2C {args.split}")
    validation_path = R / "p2c_physical_h2d_validation.json"
    if args.split == "test" and not validation_path.exists():
        raise RuntimeError("P2C validation required before test")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    capture = json.loads(CAPTURE_PATH.read_text(encoding="utf-8"))
    evaluator_lock = json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    p1d = json.loads(P1D_VERIFY.read_text(encoding="utf-8"))
    bank_result = json.loads(BANK_RESULT.read_text(encoding="utf-8"))
    if p1d.get("status") != "p1d_physical_bank_verification_pass" or capture.get("status") != "route_capture_complete":
        raise RuntimeError("verified P1D bank and P2B route capture required")
    if selection.get("status") != "p2c_policy_selected_test_unopened" or selection["selected"]["static_slots"] != STATIC_SLOTS:
        raise RuntimeError("locked P2C selection required")
    if sha256(Path(__file__)) != evaluator_lock["evaluator_sha256"] or sha256(LOCK_PATH) != evaluator_lock["input_lock_sha256"] or sha256(CAPTURE_PATH) != evaluator_lock["route_capture_sha256"] or sha256(SELECTION_PATH) != evaluator_lock["selection_sha256"] or sha256(PREREG) != selection["inputs"]["preregistration_sha256"]:
        raise ValueError("P2C evaluator provenance mismatch")
    if args.split == "test":
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("status") != "p2c_validation_pass_test_authorized" or (R / "p2b_physical_h2d_test.json").exists():
            raise RuntimeError("P2C test not authorized or prior test already opened")

    routes, route_hashes = load_routes()
    fixed_ids, count_hashes = static_ids(routes)
    begin, end = lock["partitions"][args.split]
    expected_misses = {domain: simulate(routes[domain], fixed_ids[domain], begin, end) for domain in DOMAINS}
    bases = layer_bases()
    started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat()
    pinned_layers, pinned_hashes, pin_load_ms = pin_full_bank(bank_result)
    pinned_bytes = len(pinned_layers) * LAYER_BYTES

    cp.get_default_memory_pool().free_all_blocks()
    free_before, total_vram = cp.cuda.runtime.memGetInfo()
    required_with_scratch = CACHE_BYTES + TRUNK_BYTES + KV_BYTES + MINIMUM_SCRATCH_BYTES
    if free_before < required_with_scratch:
        raise MemoryError(f"actual free VRAM {free_before} below preregistered co-residency requirement {required_with_scratch}")
    cache_memory = cp.cuda.alloc(CACHE_BYTES)
    trunk_memory = cp.cuda.alloc(TRUNK_BYTES)
    kv_memory = cp.cuda.alloc(KV_BYTES)
    stream = cp.cuda.Stream(non_blocking=False)
    with stream:
        cp.cuda.runtime.memsetAsync(cache_memory.ptr, 0, CACHE_BYTES, stream.ptr)
        cp.cuda.runtime.memsetAsync(trunk_memory.ptr, 0, TRUNK_BYTES, stream.ptr)
        cp.cuda.runtime.memsetAsync(kv_memory.ptr, 0, KV_BYTES, stream.ptr)
    stream.synchronize()
    free_after, _ = cp.cuda.runtime.memGetInfo()

    per_domain = {}
    for domain in DOMAINS:
        with stream:
            per_domain[domain] = actual_domain(domain, routes[domain], fixed_ids[domain], begin, end, pinned_layers, cache_memory, bases, stream)
        observed = np.asarray(per_domain[domain]["misses"], dtype=np.int64)
        per_domain[domain]["simulation_exact"] = bool(np.array_equal(observed, expected_misses[domain]))

    all_wall = np.concatenate([np.asarray(per_domain[domain]["wall_ms"], dtype=np.float64) for domain in DOMAINS])
    all_event = np.concatenate([np.asarray(per_domain[domain]["event_ms"], dtype=np.float64) for domain in DOMAINS])
    all_misses = np.concatenate([np.asarray(per_domain[domain]["misses"], dtype=np.int64) for domain in DOMAINS])
    aggregate = {
        "tokens": int(all_wall.size), "misses": percentile_stats(all_misses.astype(np.float64)),
        "wall_ms": percentile_stats(all_wall), "event_ms": percentile_stats(all_event),
        "mean_h2d_bytes": float((all_misses * EXPERT_BYTES).mean()), "p95_h2d_bytes": float(np.percentile(all_misses * EXPERT_BYTES, 95)),
    }
    gates = {
        "full_bank_pinned_and_hash_exact": pinned_bytes == BANK_BYTES and pinned_hashes == {str(layer): bank_result["manifests"][str(layer)]["artifact_sha256"] for layer in range(LAYERS)},
        "device_cache_trunk_kv_co_resident": free_before >= required_with_scratch and free_after >= MINIMUM_SCRATCH_BYTES,
        "exact_physical_cache_bytes": CACHE_BYTES == TOTAL_SLOTS * EXPERT_BYTES,
        "aggregate_mean_wall_h2d_ms_le_25": aggregate["wall_ms"]["mean"] <= 25.0,
        "aggregate_p95_wall_h2d_ms_le_35": aggregate["wall_ms"]["p95"] <= 35.0,
        "all_domain_mean_wall_h2d_ms_le_25": all(row["wall_ms_stats"]["mean"] <= 25.0 for row in per_domain.values()),
        "all_domain_p95_wall_h2d_ms_le_35": all(row["wall_ms_stats"]["p95"] <= 35.0 for row in per_domain.values()),
        "all_domain_preload_wall_ms_le_250": all(row["preload_wall_ms"] <= 250.0 for row in per_domain.values()),
        "all_miss_simulations_exact": all(row["simulation_exact"] for row in per_domain.values()),
        "all_sampled_transfers_exact": all(row["sample_integrity_failures"] == 0 for row in per_domain.values()),
    }
    all_pass = all(gates.values())
    if args.split == "validation":
        status, phase_pass = ("p2c_validation_pass_test_authorized", False) if all_pass else ("p2c_validation_closed", False)
    else:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        status = "p2c_physical_h2d_pass" if all_pass and all(validation["gates"].values()) else "p2c_physical_h2d_closed"
        phase_pass = status == "p2c_physical_h2d_pass"
    payload = {
        "kind": "streamq5_moe_p2c_rebalanced_actual_pinned_bank_fragmented_h2d", "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "split": args.split, "status": status,
        "inputs": {"preregistration_sha256": sha256(PREREG), "selection_sha256": sha256(SELECTION_PATH), "input_lock_sha256": sha256(LOCK_PATH), "route_capture_sha256": sha256(CAPTURE_PATH), "evaluator_lock_sha256": sha256(EVALUATOR_LOCK), "evaluator_sha256": sha256(Path(__file__)), "p1d_verification_sha256": sha256(P1D_VERIFY), "bank_result_sha256": sha256(BANK_RESULT), "route_artifact_sha256": route_hashes},
        "policy": {"total_slots": TOTAL_SLOTS, "static_slots_per_layer": STATIC_SLOTS, "dynamic_slots_layers_0_7": 15, "dynamic_slots_layers_8_47": 14, "calibration_count_sha256": count_hashes, "split_tokens": [begin, end]},
        "physical": {"pinned_bank_bytes": pinned_bytes, "pin_and_hash_load_ms": pin_load_ms, "pinned_layer_sha256": pinned_hashes, "cache_bytes": CACHE_BYTES, "trunk_bytes": TRUNK_BYTES, "kv_bytes": KV_BYTES, "minimum_scratch_bytes": MINIMUM_SCRATCH_BYTES, "free_before_bytes": int(free_before), "free_after_bytes": int(free_after), "total_vram_bytes": int(total_vram)},
        "per_domain": per_domain, "aggregate": aggregate, "gates": gates, "p2c_pass": phase_pass, "next_phase_authorized": phase_pass,
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": "Actual full pinned bank, co-resident GPU allocations, fragmented cache-miss H2D and integrity; no combined expert compute, trunk/attention, or end-to-end token loop.",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.write_text(f"# STREAMQ5-MoE P2C - {args.split} rebalanced fysieke H2D\n\nUitkomst: **{status}**. Wall mean/p95: {aggregate['wall_ms']['mean']:.3f}/{aggregate['wall_ms']['p95']:.3f} ms/token. Vrij na co-residentie: {free_after / 2**20:.1f} MiB.\n", encoding="utf-8")
    print(json.dumps({"status": status, "aggregate": aggregate, "physical": payload["physical"], "gates": gates}, indent=2), flush=True)
    if status.endswith("closed"):
        raise SystemExit(1)
