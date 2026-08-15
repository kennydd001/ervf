from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from moe_lab.reporting import ROOT


PREREG = ROOT / "reports/offload_roofline/P_B_LFU_PREREGISTRATION.md"
HERA_RESULT = ROOT / "reports/hera_moe/p0_multidomain_tier_result.json"
OUT_JSON = ROOT / "reports/offload_roofline/p_b_lfu_result.json"
OUT_MD = ROOT / "reports/offload_roofline/P_B_LFU_REPORT.md"
RAW_NPZ = ROOT / "reports/runs/offload_roofline/p_b_lfu_raw.npz"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
CAPACITIES = np.asarray((1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 4700, 5120, 5632, 6144), dtype=np.int32)
UNIVERSE = 6144
LAYERS = 48
EXPERTS_PER_LAYER = 128
TOP_K = 8
TOKENS = 32768
SWITCH_TOKENS = 512
GATE_CAPACITY = 4700
SCORE_FREQUENCY_MULTIPLIER = 1_000_000_000
SCORE_RECENCY_MULTIPLIER = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_routes():
    hera = json.loads(HERA_RESULT.read_text(encoding="utf-8"))
    routes = {domain: np.empty((TOKENS, LAYERS, TOP_K), dtype=np.int16) for domain in DOMAINS}
    hash_checks = []
    for layer in range(LAYERS):
        row = hera["artifacts"][str(layer)]
        path = ROOT / row["artifact"]
        actual_hash = sha256(path)
        hash_checks.append(actual_hash == row["artifact_sha256"])
        tensors = load_file(path)
        for domain in DOMAINS:
            values = tensors[f"{domain}_router_ids"]
            if values.shape != (TOKENS, TOP_K):
                raise ValueError(f"unexpected route shape {domain} layer {layer}: {values.shape}")
            routes[domain][:, layer, :] = values
    if not all(hash_checks):
        raise ValueError("HERA route hash mismatch")
    return routes, hera, len(hash_checks)


def token_keys(route_row: np.ndarray) -> np.ndarray:
    keys = (np.arange(LAYERS, dtype=np.int32)[:, None] * EXPERTS_PER_LAYER + route_row.astype(np.int32)).reshape(-1)
    if np.unique(keys).size != LAYERS * TOP_K:
        raise ValueError("router top-k contains duplicate layer/expert invocation")
    return keys


def ranking(counts: np.ndarray, last: np.ndarray, seen: np.ndarray) -> np.ndarray:
    rank = np.full(UNIVERSE, UNIVERSE, dtype=np.int16)
    indices = np.flatnonzero(seen)
    if indices.size:
        score = counts[indices] * SCORE_FREQUENCY_MULTIPLIER + (last[indices] + 1) * SCORE_RECENCY_MULTIPLIER + indices
        order = indices[np.argsort(score, kind="stable")[::-1]]
        rank[order] = np.arange(order.size, dtype=np.int16)
    return rank


def simulate_stationary(route: np.ndarray):
    counts = np.zeros(UNIVERSE, dtype=np.int64)
    last = np.full(UNIVERSE, -1, dtype=np.int64)
    seen = np.zeros(UNIVERSE, dtype=bool)
    cold = np.empty((TOKENS, CAPACITIES.size), dtype=np.int16)
    for token in range(TOKENS):
        keys = token_keys(route[token])
        rank = ranking(counts, last, seen)
        cold[token] = np.count_nonzero(rank[keys, None] >= CAPACITIES[None, :], axis=0)
        counts[keys] += 1
        last[keys] = token
        seen[keys] = True
    return cold, (counts, last, seen)


def simulate_switch(route: np.ndarray, source_state):
    counts, last, seen = (value.copy() for value in source_state)
    cold = np.empty(SWITCH_TOKENS, dtype=np.int16)
    for target_token in range(SWITCH_TOKENS):
        keys = token_keys(route[target_token])
        rank = ranking(counts, last, seen)
        cold[target_token] = np.count_nonzero(rank[keys] >= GATE_CAPACITY)
        counts[keys] += 1
        last[keys] = TOKENS + target_token
        seen[keys] = True
    return cold


def nearest_rank_p99(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.99, method="higher"))


def recovery_token(cold: np.ndarray):
    rolling = np.convolve(cold.astype(np.float64), np.ones(64) / 64, mode="valid")
    for start in range(rolling.size - 64):
        if rolling[start] <= 3.0 and rolling[start + 64] <= 3.0:
            return start, rolling
    return None, rolling


if __name__ == "__main__":
    if any(path.exists() for path in (OUT_JSON, OUT_MD, RAW_NPZ)):
        raise FileExistsError("refusing to overwrite P-B outputs")
    started = time.perf_counter()
    routes, hera, verified_files = load_routes()
    stationary_raw, states, stationary = {}, {}, {}
    for domain in DOMAINS:
        cold, state = simulate_stationary(routes[domain])
        stationary_raw[domain] = cold
        states[domain] = state
        curve = {}
        for index, capacity in enumerate(CAPACITIES.tolist()):
            values = cold[:, index]
            expert_gib = capacity * 4_718_592 * 1.930708991156684 / 8 / 2**30
            curve[str(capacity)] = {
                "mean_cold_calls_per_token": float(values.mean()),
                "p99_cold_calls_per_token": nearest_rank_p99(values),
                "maximum_cold_calls_per_token": int(values.max()),
                "total_cold_calls": int(values.sum()),
                "resident_expert_gib": expert_gib,
                "resident_plus_int4_trunk_gib": expert_gib + 0.7176275253295898,
            }
        stationary[domain] = curve

    switch_rows, switch_raw = {}, {}
    for source in DOMAINS:
        for target in DOMAINS:
            if source == target:
                continue
            name = f"{source}_to_{target}"
            cold = simulate_switch(routes[target], states[source])
            recovery, rolling = recovery_token(cold)
            switch_raw[name] = cold
            switch_rows[name] = {
                "recovery_token": recovery,
                "recovery_within_200": recovery is not None and recovery <= 200,
                "first_64_mean": float(cold[:64].mean()),
                "first_200_mean": float(cold[:200].mean()),
                "minimum_64_token_mean": float(rolling.min()),
                "cold_calls_first_512": int(cold.sum()),
            }

    RAW_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RAW_NPZ,
        capacities=CAPACITIES,
        **{f"stationary__{domain}": values for domain, values in stationary_raw.items()},
        **{f"switch__{name}": values for name, values in switch_raw.items()},
    )
    gate_rows = {domain: stationary[domain][str(GATE_CAPACITY)] for domain in DOMAINS}
    mean_gate = all(row["mean_cold_calls_per_token"] <= 3.0 for row in gate_rows.values())
    p99_gate = all(row["p99_cold_calls_per_token"] <= 12.0 for row in gate_rows.values())
    recovery_gate = all(row["recovery_within_200"] for row in switch_rows.values())
    resident_gib = gate_rows[DOMAINS[0]]["resident_plus_int4_trunk_gib"]
    memory_gate = resident_gib <= 5.75
    all_gates = mean_gate and p99_gate and recovery_gate and memory_gate
    result = {
        "kind": "offload_roofline_p_b_cumulative_lfu",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "verdict": "p_b_pass" if all_gates else "p_b_negative",
        "evidence_class": "exploratory_reanalysis_of_previously_opened_routes",
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "hera_result_sha256": sha256(HERA_RESULT),
            "verified_route_artifacts": verified_files,
            "domains": list(DOMAINS), "tokens_per_domain": TOKENS,
            "layers": LAYERS, "top_k": TOP_K, "universe_layer_expert_keys": UNIVERSE,
        },
        "policy": {
            "id": "cumulative_lfu_no_decay_token_boundary_prefetch",
            "capacities": CAPACITIES.tolist(), "gate_capacity": GATE_CAPACITY,
            "tie_break": "higher_frequency_then_more_recent_token_then_higher_global_key",
            "cold_call_semantics": "key_absent_immediately_before_token; lookahead_does_not_erase_transfer",
        },
        "stationary": stationary,
        "switches": switch_rows,
        "memory_at_4700": {"resident_plus_int4_trunk_gib": resident_gib, "gate_gib": 5.75},
        "gates": {
            "all_domain_means_le_3": mean_gate,
            "all_domain_p99_le_12": p99_gate,
            "all_20_switches_recover_le_200": recovery_gate,
            "resident_le_5_75_gib": memory_gate,
            "all_pass": all_gates,
        },
        "raw": {"path": str(RAW_NPZ.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(RAW_NPZ)},
        "claim_boundary": "Trace-level cumulative-LFU misses only; no transfer overlap, wall-clock, CE, or runtime-kernel measurement.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# P-B cumulatieve LFU — resultaat", "", f"**Uitkomst: {result['verdict']}**.", "", "Bij 4.700 slots:", ""]
    for domain in DOMAINS:
        row = gate_rows[domain]
        lines.append(f"- {domain}: mean {row['mean_cold_calls_per_token']:.4f}, p99 {row['p99_cold_calls_per_token']:.0f}, max {row['maximum_cold_calls_per_token']}.")
    passing_switches = sum(row["recovery_within_200"] for row in switch_rows.values())
    lines.extend(["", f"Domeinwissels binnen 200 tokens: {passing_switches}/20.", f"Resident inclusief INT4-trunk: {resident_gib:.6f} GiB (gate 5,75 GiB).", "", "Dit is een exploratieve routertracesimulatie; geen tok/s- of kwaliteitsmeting.", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"], "gate_rows": gate_rows, "switches_passing": f"{passing_switches}/20", "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
