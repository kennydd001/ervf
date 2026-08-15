from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from safetensors import safe_open

from moe_lab.reporting import ROOT


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P1A_CACHE_PREREGISTRATION.md"
LOCK_PATH = R / "p1a_route_input_lock.json"
ROUTE_EVAL_LOCK = R / "p1a_route_evaluator_lock.json"
CACHE_EVAL_LOCK = R / "p1a_cache_evaluator_lock.json"
CAPTURE_PATH = R / "p1a_route_capture_result.json"
VALIDATION_PATH = R / "p1a_cache_validation.json"
TEST_PATH = R / "p1a_cache_test.json"
ROUTE_SCRIPT = ROOT / "scripts/streamq5_moe/capture_p1a_routes.py"
CACHE_SCRIPT = ROOT / "scripts/streamq5_moe/evaluate_p1a_cache.py"
ROUTE_DIR = ROOT / "reports/runs/streamq5_moe/p1a_routes"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
EXPERT_BYTES, BANDWIDTH = 3_035_136, 26.158915272090432


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a, b, tolerance=1e-10):
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def simulate(route, fixed, begin, end, layer):
    lru = OrderedDict(); misses = np.zeros(end - begin, dtype=np.int64)
    for local, token in enumerate(range(begin, end)):
        for value in route[token]:
            expert = int(value)
            if expert in fixed:
                continue
            if expert in lru:
                lru.move_to_end(expert)
            else:
                misses[local] += 1; lru[expert] = None
                if len(lru) > (8 if layer <= 37 else 7):
                    lru.popitem(last=False)
    return misses


def main():
    checks = []
    add = lambda name, passed, detail="": checks.append({"name": name, "pass": bool(passed), "detail": detail})
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    capture = json.loads(CAPTURE_PATH.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    test = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    route_eval_lock = json.loads(ROUTE_EVAL_LOCK.read_text(encoding="utf-8"))
    cache_eval_lock = json.loads(CACHE_EVAL_LOCK.read_text(encoding="utf-8"))

    add("preregistration hash", sha256(PREREG) == lock["preregistration_sha256"])
    add("route evaluator hash", sha256(ROUTE_SCRIPT) == route_eval_lock["evaluator_sha256"])
    add("cache evaluator hash", sha256(CACHE_SCRIPT) == cache_eval_lock["evaluator_sha256"])
    add("route capture hash", sha256(CAPTURE_PATH) == cache_eval_lock["route_capture_sha256"])
    add("capture status", capture["status"] == "route_capture_complete" and all(capture["controls"].values()))
    add("48 captured layers", set(capture["manifests"]) == {str(value) for value in range(48)})

    routes = {domain: [] for domain in DOMAINS}; artifact_hashes = {}
    route_ok = True
    for layer in range(48):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"
        artifact_hashes[str(layer)] = sha256(path)
        route_ok &= artifact_hashes[str(layer)] == capture["manifests"][str(layer)]["artifact_sha256"]
        with safe_open(path, framework="numpy") as handle:
            route_ok &= set(handle.keys()) == {f"{domain}_router_ids" for domain in DOMAINS}
            for domain in DOMAINS:
                value = handle.get_tensor(f"{domain}_router_ids").astype(np.int64)
                route_ok &= value.shape == (1024, 8) and value.min() >= 0 and value.max() < 128
                route_ok &= all(len(set(row.tolist())) == 8 for row in value)
                routes[domain].append(value)
    add("route artifacts and values", route_ok)
    routes = {domain: np.stack(values, axis=1) for domain, values in routes.items()}

    recomputed = {}
    for split, (begin, end), reported in (("validation", (512, 768), validation), ("test", (768, 1024), test)):
        all_misses = []
        domain_stats = {}
        for domain in DOMAINS:
            domain_misses = np.zeros(end - begin, dtype=np.int64)
            for layer in range(48):
                counts = np.bincount(routes[domain][:512, layer, :].reshape(-1), minlength=128)
                order = np.lexsort((np.arange(128), -counts))
                domain_misses += simulate(routes[domain][:, layer, :], frozenset(int(x) for x in order[:32]), begin, end, layer)
            all_misses.append(domain_misses)
            ms = domain_misses * EXPERT_BYTES / (BANDWIDTH * 1e9) * 1000
            domain_stats[domain] = {"mean": float(ms.mean()), "p95": float(np.percentile(ms, 95))}
            add(f"{split} {domain} mean", close(ms.mean(), reported["per_domain"][domain]["mean_h2d_ms"]))
            add(f"{split} {domain} p95", close(np.percentile(ms, 95), reported["per_domain"][domain]["p95_h2d_ms"]))
        misses = np.concatenate(all_misses); ms = misses * EXPERT_BYTES / (BANDWIDTH * 1e9) * 1000
        recomputed[split] = {"mean_h2d_ms": float(ms.mean()), "p95_h2d_ms": float(np.percentile(ms, 95)), "misses": misses}
        add(f"{split} aggregate mean", close(ms.mean(), reported["aggregate"]["mean_h2d_ms"]))
        add(f"{split} aggregate p95", close(np.percentile(ms, 95), reported["aggregate"]["p95_h2d_ms"]))
        expected_gates = {
            "mean_h2d_ms_le_25": ms.mean() <= 25,
            "p95_h2d_ms_le_35": np.percentile(ms, 95) <= 35,
            "all_domain_mean_h2d_ms_le_25": all(row["mean"] <= 25 for row in domain_stats.values()),
            "all_domain_p95_h2d_ms_le_35": all(row["p95"] <= 35 for row in domain_stats.values()),
            "static_preload_ms_le_250": reported["physical_accounting"]["static_preload_ms"] <= 250,
            "resident_gib_le_gpu": reported["physical_accounting"]["resident_gib"] <= reported["physical_accounting"]["gpu_gib"],
            "host_bank_gib_le_17_45": reported["physical_accounting"]["host_bank_gib"] <= 17.45,
            "host_bank_gib_le_24": reported["physical_accounting"]["host_bank_gib"] <= 24,
        }
        add(f"{split} gates", reported["gates"] == expected_gates and all(expected_gates.values()))
        add(f"{split} route hashes", reported["inputs"]["route_artifact_sha256"] == artifact_hashes)

    slots = 38 * 40 + 10 * 39
    cache_bytes = slots * EXPERT_BYTES
    resident = 1.4352550506591797 + 0.375 + 0.75 + cache_bytes / 2**30
    preload = 32 * 48 * EXPERT_BYTES / (BANDWIDTH * 1e9) * 1000
    add("physical bank arithmetic", validation["physical_accounting"]["host_bank_bytes"] == 18_647_875_584 and close(validation["physical_accounting"]["host_bank_gib"], 17.3671875))
    add("cache resident arithmetic", validation["policy"]["total_slots"] == slots == 1910 and close(validation["physical_accounting"]["resident_gib"], resident))
    add("static preload arithmetic", close(validation["physical_accounting"]["static_preload_ms"], preload))
    add("split chronology", validation["status"] == "p1a_validation_pass_test_authorized" and test["status"] == "p1a_cache_pass")
    add("phase flags", validation["p1a_pass"] is False and test["p1a_pass"] is True and test["next_phase_authorized"] is True)

    passed = sum(row["pass"] for row in checks); total = len(checks)
    status = "p1a_cache_verification_pass" if passed == total else "p1a_cache_verification_fail"
    payload = {
        "kind": "streamq5_moe_p1a_independent_verification", "status": status,
        "checks_passed": passed, "checks_total": total, "checks": checks,
        "validation_mean_h2d_ms": recomputed["validation"]["mean_h2d_ms"], "validation_p95_h2d_ms": recomputed["validation"]["p95_h2d_ms"],
        "test_mean_h2d_ms": recomputed["test"]["mean_h2d_ms"], "test_p95_h2d_ms": recomputed["test"]["p95_h2d_ms"],
        "claim_boundary": "Independent route-cache and byte-accounting verification; physical packing, H2D timing, kernel, overlap, and wall-clock remain unproven.",
    }
    (R / "p1a_cache_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (R / "P1A_CACHE_VERIFICATION.md").write_text(f"# STREAMQ5-MoE P1A - onafhankelijke verificatie\n\nUitkomst: **{status}** ({passed}/{total}). Test mean/p95: {payload['test_mean_h2d_ms']:.3f}/{payload['test_p95_h2d_ms']:.3f} ms/token.\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": f"{passed}/{total}", "test_mean_ms": payload["test_mean_h2d_ms"], "test_p95_ms": payload["test_p95_h2d_ms"]}, indent=2))
    if status.endswith("fail"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
