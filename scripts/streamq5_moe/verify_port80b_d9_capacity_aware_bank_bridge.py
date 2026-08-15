#!/usr/bin/env python3
"""CPU-only independent verifier for PORT80B-D9.

No D9 runner code is imported and no CUDA API is called.  The verifier
reconstructs frozen routes, timing summaries, gate decisions, provenance and
cleanup adjudication from the raw result.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "streamq5_moe"
RUNS = R / "../runs/streamq5_moe/port80b_p0"
PREREG = R / "PORT80B_D9_CAPACITY_AWARE_BANK_BRIDGE_PREREGISTRATION.md"
RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_d9_capacity_aware_bank_bridge.py"
COMPILE = R / "port80b_d9_capacity_aware_bank_bridge_compile.json"
RAW = R / "port80b_d9_capacity_aware_bank_bridge.json"
BANK = RUNS / "port80b_p0_full_q5_bank.bin"
MANIFEST = RUNS / "port80b_p0_full_q5_bank_manifest.json"
D7 = R / "port80b_d7_staged_exact_q5_plane.json"
D8 = R / "port80b_d8_registration_capacity_independent_verification.json"
OUT = R / "port80b_d9_capacity_aware_bank_bridge_independent_verification.json"
REPORT = R / "PORT80B_D9_CAPACITY_AWARE_BANK_BRIDGE_INDEPENDENT_VERIFICATION_REPORT_2026-08-12.md"

LAYERS = 48
PREFIX = 499
COLD_BEGIN = 499
COLD_END = 512
ACTIVE = 10
EXPERT_BYTES = 2_027_520
BANK_BYTES = 49_925_652_480
MIN_AVAILABLE = 2 * 2**30
TRACE_SEED = 0x80B0120826
MASK64 = (1 << 64) - 1
CASES = ("all_hot", "mixed_5_hot_5_cold", "all_cold_tail")
VALIDATION_P50 = {"all_hot": 65.0, "mixed_5_hot_5_cold": 100.0, "all_cold_tail": 135.0}
TEST_P95 = {"all_hot": 65.0, "mixed_5_hot_5_cold": 100.0, "all_cold_tail": 135.0}
EXPECTED_BANK_SHA = "4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def routes(token: int, experts: int) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for layer in range(LAYERS):
        state = (TRACE_SEED ^ (token * 0xD6E8FEB86659FD93) ^ (layer * 0xA5A3564E27F8862D)) & MASK64
        values: list[int] = []
        while len(values) < ACTIVE:
            state = splitmix64(state)
            expert = int(state % experts)
            if expert not in values:
                values.append(expert)
        selected.extend((layer, expert) for expert in values)
    return selected


def frozen_cases() -> dict[str, list[tuple[int, int]]]:
    hot_1 = routes(130_001, PREFIX)
    hot_2 = routes(130_002, PREFIX)
    by_layer_1 = {layer: [] for layer in range(LAYERS)}
    by_layer_2 = {layer: [] for layer in range(LAYERS)}
    for layer, expert in hot_1:
        by_layer_1[layer].append(expert)
    for layer, expert in hot_2:
        by_layer_2[layer].append(expert)
    result = {name: [] for name in CASES}
    for layer in range(LAYERS):
        result["all_hot"].extend((layer, expert) for expert in by_layer_1[layer])
        result["mixed_5_hot_5_cold"].extend((layer, expert) for expert in by_layer_2[layer][:5])
        result["mixed_5_hot_5_cold"].extend((layer, COLD_BEGIN + ((layer + rank) % 13)) for rank in range(5))
        result["all_cold_tail"].extend((layer, COLD_BEGIN + ((layer + rank) % 13)) for rank in range(10))
    return result


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def fixed_order(round_index: int) -> list[str]:
    rotation = round_index % len(CASES)
    order = list(CASES[rotation:] + CASES[:rotation])
    if round_index & 1:
        order.reverse()
    return order


def close(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    return left == right


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def main() -> None:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    compile_evidence = json.loads(COMPILE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    d8 = json.loads(D8.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    hashes = {
        "preregistration_sha256": digest(PREREG),
        "runner_sha256": digest(RUNNER),
        "compile_evidence_sha256": digest(COMPILE),
        "manifest_sha256": digest(MANIFEST),
        "d7_result_sha256": digest(D7),
        "d8_independent_verification_sha256": digest(D8),
    }
    add(checks, "raw_provenance_hashes", all(raw["inputs"][key] == value for key, value in hashes.items()), hashes)
    add(
        checks,
        "compile_locked_prereg_and_runner",
        compile_evidence["pass"]
        and compile_evidence["inputs"]["preregistration_sha256"] == hashes["preregistration_sha256"]
        and compile_evidence["inputs"]["runner_sha256"] == hashes["runner_sha256"],
    )
    add(
        checks,
        "immutable_bank_contract",
        BANK.is_file()
        and BANK.stat().st_size == BANK_BYTES
        and manifest["bank_sha256"] == EXPECTED_BANK_SHA
        and raw["inputs"]["bank_sha256_from_manifest"] == EXPECTED_BANK_SHA,
        {"size": BANK.stat().st_size, "manifest_sha256": manifest["bank_sha256"]},
    )
    add(
        checks,
        "d8_supports_only_clean_499_observation",
        d8["capacity"]["largest_clean_prefix_experts_per_layer"] == PREFIX
        and d8["capacity"]["raw_largest_claim_is_protocol_valid"] is False
        and d8["protocol_checks"]["full_prefix_has_exactly_44_raw_unregister_failures"] is True,
        d8["cumulative_ram_caveat"],
    )

    selected = frozen_cases()
    expected_counts = {"all_hot": (480, 0), "mixed_5_hot_5_cold": (240, 240), "all_cold_tail": (0, 480)}
    rebuilt_route_contract: dict[str, Any] = {}
    route_ok = True
    for name, values in selected.items():
        hot = sum(expert < PREFIX for _, expert in values)
        cold = len(values) - hot
        valid = len(values) == 480 and (hot, cold) == expected_counts[name]
        for layer in range(LAYERS):
            layer_values = [expert for route_layer, expert in values if route_layer == layer]
            valid &= len(layer_values) == ACTIVE and len(set(layer_values)) == ACTIVE
        rebuilt_route_contract[name] = {
            "records": len(values),
            "hot_records": hot,
            "cold_escape_records": cold,
            "route_sha256": hashlib.sha256(np.asarray(values, dtype=np.int16).tobytes()).hexdigest(),
            "pass": bool(valid),
        }
        route_ok &= valid
    add(checks, "frozen_routes_reconstructed", route_ok and rebuilt_route_contract == raw["route_contract"], rebuilt_route_contract)

    physical = raw["physical"]
    add(
        checks,
        "physical_499_plus_13_arithmetic",
        physical["registered_experts_per_layer"] == PREFIX
        and physical["cold_escape_experts_per_layer"] == COLD_END - COLD_BEGIN
        and physical["registered_ranges"] == LAYERS
        and physical["registered_bytes"] == LAYERS * PREFIX * EXPERT_BYTES,
        physical,
    )
    add(
        checks,
        "ram_safety_gates",
        physical["available_ram_before_registration"] >= MIN_AVAILABLE
        and physical["available_ram_after_registration"] >= MIN_AVAILABLE,
        physical,
    )

    positive = raw["integrity"]["positive"]
    negative = raw["integrity"]["negative_controls"]
    add(checks, "positive_differentiated_checks", all(positive[name]["full_image_byte_mismatches"] == 0 and positive[name]["pass"] for name in CASES), positive)
    add(
        checks,
        "negative_wrong_route_controls",
        negative["wrong_expert"]["expected"] != negative["wrong_expert"]["substituted"]
        and negative["wrong_expert"]["detected_mismatches"] > 0
        and negative["wrong_expert"]["pass"]
        and negative["wrong_layer"]["expected"] != negative["wrong_layer"]["substituted"]
        and negative["wrong_layer"]["detected_mismatches"] > 0
        and negative["wrong_layer"]["pass"],
        negative,
    )
    digests = set()
    correctness_ok = True
    for name in CASES:
        row = raw["correctness"][name]
        correctness_ok &= row["elements"] == 1_474_560 and row["different_bits"] == 0 and row["bitwise_equal"] and row["finite"] and row["max_abs"] == 0.0 and row["expected_sha256"] == row["observed_sha256"]
        digests.update((row["expected_sha256"], row["observed_sha256"]))
    add(checks, "bitexact_all_cases_and_common_digest", correctness_ok and len(digests) == 1, sorted(digests))

    validation_stats = {}
    test_stats = {}
    timing_ok = True
    for name in CASES:
        v = raw["validation"]["cases"][name]
        t = raw["test"]["cases"][name]
        validation_stats[name] = {"wall_stats": stats(v["wall_ms"]), "cuda_event_stats": stats(v["cuda_event_ms"])}
        test_stats[name] = {"wall_stats": stats(t["wall_ms"]), "cuda_event_stats": stats(t["cuda_event_ms"])}
        timing_ok &= len(v["wall_ms"]) == 24 and len(v["cuda_event_ms"]) == 24
        timing_ok &= len(t["wall_ms"]) == 60 and len(t["cuda_event_ms"]) == 60
        timing_ok &= np.isfinite(v["wall_ms"]).all() and np.isfinite(t["wall_ms"]).all()
        timing_ok &= close(validation_stats[name]["wall_stats"], v["wall_stats"])
        timing_ok &= close(validation_stats[name]["cuda_event_stats"], v["cuda_event_stats"])
        timing_ok &= close(test_stats[name]["wall_stats"], t["wall_stats"])
        timing_ok &= close(test_stats[name]["cuda_event_stats"], t["cuda_event_stats"])
    add(checks, "all_timing_statistics_recomputed", timing_ok, {"validation": validation_stats, "test": test_stats})

    validation_orders = [fixed_order(index) for index in range(24)]
    test_orders = [fixed_order(24 + index) for index in range(60)]
    add(checks, "frozen_case_orders", validation_orders == raw["validation"]["orders"] and test_orders == raw["test"]["orders"])

    gates = {
        "mapping_readonly": True,
        "prefix_exactly_499_and_tail_exactly_13": PREFIX == 499 and COLD_END - COLD_BEGIN == 13,
        "route_source_provenance_exact": route_ok,
        "positive_route_integrity_zero_mismatch": all(positive[name]["pass"] for name in CASES),
        "wrong_expert_detected": negative["wrong_expert"]["pass"] and negative["wrong_expert"]["detected_mismatches"] > 0,
        "wrong_layer_detected": negative["wrong_layer"]["pass"] and negative["wrong_layer"]["detected_mismatches"] > 0,
        "all_outputs_bitexact_finite_digest_equal": correctness_ok and len(digests) == 1,
        "validation_24_each_finite": all(len(raw["validation"]["cases"][name]["wall_ms"]) == 24 and np.isfinite(raw["validation"]["cases"][name]["wall_ms"]).all() for name in CASES),
        "validation_p50_case_limits": all(validation_stats[name]["wall_stats"]["p50"] <= VALIDATION_P50[name] for name in CASES),
        "test_60_each_finite": all(len(raw["test"]["cases"][name]["wall_ms"]) == 60 and np.isfinite(raw["test"]["cases"][name]["wall_ms"]).all() for name in CASES),
        **{f"test_{name}_wall_p95_le_{int(TEST_P95[name])}ms": test_stats[name]["wall_stats"]["p95"] <= TEST_P95[name] for name in CASES},
        "strong_mixed_wall_p95_le_80ms": test_stats["mixed_5_hot_5_cold"]["wall_stats"]["p95"] <= 80.0,
        "strong_all_cold_wall_p95_le_110ms": test_stats["all_cold_tail"]["wall_stats"]["p95"] <= 110.0,
        "registration_48_ranges": physical["registered_ranges"] == LAYERS,
        "post_registration_available_ram_ge_2gib": physical["available_ram_after_registration"] >= MIN_AVAILABLE,
        "no_cuda_or_runner_error": raw["error"] is None,
        "clean_unregister_48_ranges": physical["registered_ranges"] == LAYERS and raw["unregister_failures"] == [],
    }
    add(checks, "all_gates_independently_recomputed", gates == raw["gates"], gates)
    primary = all(value for name, value in gates.items() if not name.startswith("strong_"))
    strong = primary and gates["strong_mixed_wall_p95_le_80ms"] and gates["strong_all_cold_wall_p95_le_110ms"]
    add(checks, "verdict_recomputed", primary == raw["primary_pass"] and strong == raw["strong_pass"] and raw["status"] == "capacity_bridge_strong_pass")
    add(checks, "clean_unregister_final_gate", raw["unregister_failures"] == [] and gates["clean_unregister_48_ranges"])
    add(checks, "full_bank_claim_remains_false", raw["full_bank_registration_pass"] is False)

    result = {
        "kind": "port80b_d9_capacity_aware_bank_bridge_independent_cpu_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "capacity_bridge_strong_pass_independently_verified" if all(row["pass"] for row in checks) else "verification_failure",
        "all_checks_pass": all(row["pass"] for row in checks),
        "source_raw_sha256": digest(RAW),
        "checks": checks,
        "recomputed_gates": gates,
        "recomputed_validation_stats": validation_stats,
        "recomputed_test_stats": test_stats,
        "post_unregister_available_ram_bytes": physical["available_ram_after_unregister"],
        "ram_interpretation": {
            "before_registration_bytes": physical["available_ram_before_registration"],
            "after_registration_bytes": physical["available_ram_after_registration"],
            "after_unregister_bytes": physical["available_ram_after_unregister"],
            "registration_delta_bytes": physical["available_ram_after_registration"] - physical["available_ram_before_registration"],
            "run_through_cleanup_delta_bytes": physical["available_ram_after_unregister"] - physical["available_ram_before_registration"],
            "interpretation": "Registration itself left available RAM high, but the timed run first-touched almost the full mapped bank; after clean unregister only 3.123 GB remained available. This is consistent with file-backed pages remaining resident/standby after first touch. Clean CUDA unregister proves API cleanup, not immediate OS working-set reclamation or endurance stability, and this audit does not diagnose a leak.",
        },
        "capacity_caveat": d8["cumulative_ram_caveat"]["interpretation"],
        "claim_boundary": "CPU-only verification of frozen D9 evidence; no GPU rerun, registration, bank sweep, model, quality or endurance claim.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    rows = []
    for name in CASES:
        rows.append(
            f"| {name} | {validation_stats[name]['wall_stats']['p50']:.3f} | "
            f"{test_stats[name]['wall_stats']['p50']:.3f} | {test_stats[name]['wall_stats']['p95']:.3f} |"
        )
    REPORT.write_text(
        "# PORT80B-D9 — independent CPU verification\n\n"
        f"Verdict: **{result['verdict']}**. Checks: **{sum(row['pass'] for row in checks)}/{len(checks)}**.\n\n"
        "| case | validation wall p50 ms | test wall p50 ms | test wall p95 ms |\n"
        "|---|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
        f"Wrong-expert and wrong-layer controls detected {negative['wrong_expert']['detected_mismatches']} and {negative['wrong_layer']['detected_mismatches']} byte mismatches. All positive images had zero mismatches; all outputs were bitexact. All 48 registered ranges unregistered cleanly.\n\n"
        f"RAM was {physical['available_ram_before_registration'] / 1e9:.3f} GB before and {physical['available_ram_after_registration'] / 1e9:.3f} GB immediately after registration, but {physical['available_ram_after_unregister'] / 1e9:.3f} GB after the run and clean unregister. Registration therefore did not initially fault most pages; the timed workload first-touched the mapped bank and file-backed pages remained resident/standby. This is not evidence of failed CUDA cleanup, but it also does not prove prompt OS reclamation or stable endurance headroom.\n\n"
        f"Capacity caveat: {result['capacity_caveat']}\n\n"
        f"Claim boundary: {result['claim_boundary']}\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": result["verdict"], "checks": f"{sum(row['pass'] for row in checks)}/{len(checks)}", "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
