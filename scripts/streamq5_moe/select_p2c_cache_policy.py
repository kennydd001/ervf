from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
from safetensors import safe_open

from moe_lab.reporting import ROOT


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P2C_STATIC_DYNAMIC_REBALANCE_PREREGISTRATION.md"
P2B_VALIDATION = R / "p2b_physical_h2d_validation.json"
ROUTE_DIR = ROOT / "reports/runs/streamq5_moe/p2b_routes"
OUTPUT = R / "p2c_policy_selection.json"
REPORT = R / "P2C_POLICY_SELECTION.md"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
CANDIDATES = (8, 12, 16, 20, 24, 28, 32)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def total_slots(layer):
    return 35 if layer <= 7 else 34


def evaluate(route, static_count):
    fixed = []
    for layer in range(48):
        counts = np.bincount(route[:512, layer, :].reshape(-1), minlength=128)
        order = np.lexsort((np.arange(128), -counts))
        fixed.append(frozenset(int(value) for value in order[:static_count]))
    lru = [OrderedDict() for _ in range(48)]
    misses = np.zeros(256, dtype=np.int64)
    for local, token in enumerate(range(512, 768)):
        for layer in range(48):
            cache = lru[layer]
            dynamic_capacity = total_slots(layer) - static_count
            for raw in route[token, layer]:
                expert = int(raw)
                if expert in fixed[layer]:
                    continue
                if expert in cache:
                    cache.move_to_end(expert)
                else:
                    misses[local] += 1; cache[expert] = None
                    if len(cache) > dynamic_capacity:
                        cache.popitem(last=False)
    return misses


if __name__ == "__main__":
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite P2C policy selection")
    p2b = json.loads(P2B_VALIDATION.read_text(encoding="utf-8"))
    if p2b.get("status") != "p2b_validation_closed" or (R / "p2b_physical_h2d_test.json").exists():
        raise RuntimeError("closed P2B validation and unopened test required")
    routes = {domain: [] for domain in DOMAINS}; route_hashes = {}
    for layer in range(48):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"; route_hashes[str(layer)] = sha256(path)
        with safe_open(path, framework="numpy") as handle:
            for domain in DOMAINS:
                routes[domain].append(handle.get_tensor(f"{domain}_router_ids").astype(np.int64))
    routes = {domain: np.stack(values, axis=1) for domain, values in routes.items()}
    rows = []
    for static_count in CANDIDATES:
        per_domain, all_misses = {}, []
        for domain in DOMAINS:
            misses = evaluate(routes[domain], static_count); all_misses.append(misses)
            per_domain[domain] = {"mean_misses": float(misses.mean()), "p95_misses": float(np.percentile(misses, 95)), "max_misses": int(misses.max())}
        combined = np.concatenate(all_misses)
        rows.append({"static_slots": static_count, "dynamic_slots_layers_0_7": 35 - static_count, "dynamic_slots_layers_8_47": 34 - static_count, "worst_domain_p95_misses": max(row["p95_misses"] for row in per_domain.values()), "aggregate_mean_misses": float(combined.mean()), "aggregate_p95_misses": float(np.percentile(combined, 95)), "per_domain": per_domain})
    selected = min(rows, key=lambda row: (row["worst_domain_p95_misses"], row["aggregate_mean_misses"], -row["static_slots"]))
    payload = {"kind": "streamq5_moe_p2c_validation_only_policy_selection", "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "p2c_policy_selected_test_unopened", "inputs": {"preregistration_sha256": sha256(PREREG), "p2b_validation_sha256": sha256(P2B_VALIDATION), "route_artifact_sha256": route_hashes}, "candidates": list(CANDIDATES), "selection_rule": "min worst-domain p95 misses; then aggregate mean; then larger static count", "rows": rows, "selected": selected, "test_tokens_accessed": False}
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(f"# STREAMQ5-MoE P2C - policyselectie\n\nGeselecteerd: **{selected['static_slots']} static**, dynamic {selected['dynamic_slots_layers_0_7']}/{selected['dynamic_slots_layers_8_47']}. Worst-domain validation-p95: {selected['worst_domain_p95_misses']:.2f} misses. Test bleef ongeopend.\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected": selected, "test_tokens_accessed": False}, indent=2))
