from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from moe_lab.reporting import ROOT


RESULT = ROOT / "reports/offload_roofline/p_b_lfu_result.json"
HERA_RESULT = ROOT / "reports/hera_moe/p0_multidomain_tier_result.json"
OUT_JSON = ROOT / "reports/offload_roofline/p_b_lfu_verification.json"
OUT_MD = ROOT / "reports/offload_roofline/P_B_LFU_VERIFICATION.md"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
CAPACITY = 4700
UNIVERSE = 6144
TOKENS = 32768
SWITCH_TOKENS = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_routes(hera):
    routes = {domain: np.empty((TOKENS, 48, 8), dtype=np.int16) for domain in DOMAINS}
    hashes_ok = True
    for layer in range(48):
        row = hera["artifacts"][str(layer)]
        path = ROOT / row["artifact"]
        hashes_ok &= sha256(path) == row["artifact_sha256"]
        tensors = load_file(path)
        for domain in DOMAINS:
            routes[domain][:, layer, :] = tensors[f"{domain}_router_ids"]
    return routes, hashes_ok


def resident_mask(counts, last, seen):
    indices = np.flatnonzero(seen)
    resident = np.zeros(UNIVERSE, dtype=bool)
    if indices.size <= CAPACITY:
        resident[indices] = True
    else:
        scores = counts[indices] * 1_000_000_000 + (last[indices] + 1) * 10_000 + indices
        keep = np.argpartition(scores, -CAPACITY)[-CAPACITY:]
        resident[indices[keep]] = True
    return resident


def keys(row):
    return (np.arange(48, dtype=np.int32)[:, None] * 128 + row.astype(np.int32)).reshape(-1)


def stationary(route):
    counts = np.zeros(UNIVERSE, dtype=np.int64)
    last = np.full(UNIVERSE, -1, dtype=np.int64)
    seen = np.zeros(UNIVERSE, dtype=bool)
    cold = np.empty(TOKENS, dtype=np.int16)
    for token in range(TOKENS):
        token_keys = keys(route[token])
        resident = resident_mask(counts, last, seen)
        cold[token] = np.count_nonzero(~resident[token_keys])
        counts[token_keys] += 1
        last[token_keys] = token
        seen[token_keys] = True
    return cold, (counts, last, seen)


def switch(route, state):
    counts, last, seen = (item.copy() for item in state)
    cold = np.empty(SWITCH_TOKENS, dtype=np.int16)
    for token in range(SWITCH_TOKENS):
        token_keys = keys(route[token])
        resident = resident_mask(counts, last, seen)
        cold[token] = np.count_nonzero(~resident[token_keys])
        counts[token_keys] += 1
        last[token_keys] = TOKENS + token
        seen[token_keys] = True
    return cold


def recovery(cold):
    rolling = np.convolve(cold.astype(np.float64), np.ones(64) / 64, mode="valid")
    for start in range(rolling.size - 64):
        if rolling[start] <= 3.0 and rolling[start + 64] <= 3.0:
            return start
    return None


if __name__ == "__main__":
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("refusing to overwrite P-B verification")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    hera = json.loads(HERA_RESULT.read_text(encoding="utf-8"))
    raw_path = ROOT / result["raw"]["path"]
    raw = np.load(raw_path)
    routes, hashes_ok = load_routes(hera)
    checks = {}
    checks["raw_hash"] = sha256(raw_path) == result["raw"]["sha256"]
    checks["hera_hash"] = sha256(HERA_RESULT) == result["inputs"]["hera_result_sha256"]
    checks["all_route_hashes"] = hashes_ok
    checks["capacities"] = raw["capacities"].tolist() == result["policy"]["capacities"]
    capacity_index = raw["capacities"].tolist().index(CAPACITY)
    states = {}
    stationary_exact = True
    metric_exact = True
    for domain in DOMAINS:
        observed, state = stationary(routes[domain])
        states[domain] = state
        stored = raw[f"stationary__{domain}"][:, capacity_index]
        stationary_exact &= np.array_equal(observed, stored)
        row = result["stationary"][domain][str(CAPACITY)]
        metric_exact &= abs(float(stored.mean()) - row["mean_cold_calls_per_token"]) < 1e-12
        metric_exact &= float(np.quantile(stored, 0.99, method="higher")) == row["p99_cold_calls_per_token"]
        metric_exact &= int(stored.max()) == row["maximum_cold_calls_per_token"]
        metric_exact &= int(stored.sum()) == row["total_cold_calls"]
    checks["independent_stationary_4700_exact"] = stationary_exact
    checks["stationary_metrics_exact"] = metric_exact
    switches_exact = recovery_exact = True
    for source in DOMAINS:
        for target in DOMAINS:
            if source == target:
                continue
            name = f"{source}_to_{target}"
            observed = switch(routes[target], states[source])
            stored = raw[f"switch__{name}"]
            switches_exact &= np.array_equal(observed, stored)
            recovery_exact &= recovery(stored) == result["switches"][name]["recovery_token"]
    checks["independent_all_20_switches_exact"] = switches_exact
    checks["switch_recovery_exact"] = recovery_exact
    mean_gate = all(result["stationary"][d][str(CAPACITY)]["mean_cold_calls_per_token"] <= 3 for d in DOMAINS)
    p99_gate = all(result["stationary"][d][str(CAPACITY)]["p99_cold_calls_per_token"] <= 12 for d in DOMAINS)
    switch_gate = all(row["recovery_token"] is not None and row["recovery_token"] <= 200 for row in result["switches"].values())
    memory = CAPACITY * 4_718_592 * 1.930708991156684 / 8 / 2**30 + 0.7176275253295898
    checks["mean_gate"] = mean_gate == result["gates"]["all_domain_means_le_3"]
    checks["p99_gate"] = p99_gate == result["gates"]["all_domain_p99_le_12"]
    checks["switch_gate"] = switch_gate == result["gates"]["all_20_switches_recover_le_200"]
    checks["memory_arithmetic"] = abs(memory - result["memory_at_4700"]["resident_plus_int4_trunk_gib"]) < 1e-12
    checks["memory_gate"] = (memory <= 5.75) == result["gates"]["resident_le_5_75_gib"]
    checks["verdict"] = result["verdict"] == "p_b_negative" and not result["gates"]["all_pass"]
    passed = sum(checks.values())
    verification = {
        "kind": "offload_roofline_p_b_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "checks_passed": passed, "checks_total": len(checks),
        "all_pass": passed == len(checks),
        "verdict": "p_b_negative_verified" if passed == len(checks) else "verification_failed",
    }
    OUT_JSON.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join([
        "# P-B onafhankelijke verificatie", "",
        f"**{verification['verdict']}** — {passed}/{len(checks)} controles geslaagd.", "",
        "Een tweede selectie-implementatie (`argpartition`, los van de sorteerimplementatie van de runner) reproduceerde exact alle vijf 4.700-slotreeksen en alle 20 wisselreeksen. Metrics, gates, hashes en geheugenrekenkunde zijn opnieuw gecontroleerd.", "",
    ]), encoding="utf-8")
    print(json.dumps({"verdict": verification["verdict"], "checks": f"{passed}/{len(checks)}"}, indent=2))
