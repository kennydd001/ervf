from __future__ import annotations

import ast
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
RUNS = ROOT / "reports/runs/streamq5_moe/port80b_p0"
PREREG = REPORTS / "PORT80B_D1_TOKEN_BATCH_DIRECTPATH_PREREGISTRATION.md"
RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_d1_token_batch_directpath.py"
RESULT = REPORTS / "port80b_d1_token_batch_directpath.json"
MANIFEST = RUNS / "port80b_p0_full_q5_bank_manifest.json"
BANK = RUNS / "port80b_p0_full_q5_bank.bin"
OUTPUT = REPORTS / "port80b_d1_token_batch_directpath_independent_verification.json"
REPORT = REPORTS / "PORT80B_D1_TOKEN_BATCH_DIRECTPATH_INDEPENDENT_VERIFICATION_REPORT_2026-08-12.md"

ARMS = ("record480", "layer48", "token1")
LAYERS = 48
TOP_K = 10
EXPERTS_WITH_SHARED = 513
EXPERT_BYTES = 2_027_520
BANK_BYTES = 49_925_652_480
TOKEN_BYTES = 973_209_600
TOKEN = 10_000
ROUNDS = 120
WARMUPS = 10
STAGE_TOKENS = tuple(range(10_001, 10_033))
EXPECTED_BANK_SHA256 = "4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462"
TRACE_SEED = 0x80B0120826
MASK64 = (1 << 64) - 1
TOLERANCE = 1e-9


def sha256(path: Path, chunk_bytes: int = 64 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def percentile(values: list[float], q: float) -> float:
    """Stdlib implementation of NumPy's default linear percentile."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty percentile input")
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def recompute_stats(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("non-finite or empty timing series")
    floats = [float(value) for value in values]
    return {
        "count": len(floats),
        "mean": statistics.fmean(floats),
        "p50": percentile(floats, 0.50),
        "p95": percentile(floats, 0.95),
        "p99": percentile(floats, 0.99),
        "min": min(floats),
        "max": max(floats),
    }


def close(left: float | int, right: float | int) -> bool:
    return abs(float(left) - float(right)) <= TOLERANCE


def stats_match(recomputed: dict[str, float | int], stored: dict[str, Any]) -> dict[str, bool]:
    return {name: close(value, stored[name]) for name, value in recomputed.items()}


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def route(token: int, layer: int) -> tuple[int, ...]:
    counter = (TRACE_SEED ^ (token * 0xD6E8FEB86659FD93) ^ (layer * 0xA5A3564E27F8862D)) & MASK64
    first = splitmix64(counter)
    second = splitmix64(first)
    start = first & 511
    stride = ((second & 255) << 1) | 1
    result = tuple(int((start + rank * stride) & 511) for rank in range(TOP_K))
    if len(set(result)) != TOP_K:
        raise AssertionError("independent route generator emitted duplicates")
    return result


def offsets_for(token: int) -> list[int]:
    return [
        (layer * EXPERTS_WITH_SHARED + expert) * EXPERT_BYTES
        for layer in range(LAYERS)
        for expert in route(token, layer)
    ]


def expected_orders() -> list[list[str]]:
    orders: list[list[str]] = []
    for round_index in range(ROUNDS):
        rotation = round_index % len(ARMS)
        order = list(ARMS[rotation:] + ARMS[:rotation])
        if round_index & 1:
            order.reverse()
        orders.append(order)
    return orders


def hash_source_token(bank: Path, offsets: list[int]) -> str:
    digest = hashlib.sha256()
    with bank.open("rb", buffering=0) as handle:
        for offset in offsets:
            handle.seek(offset)
            remaining = EXPERT_BYTES
            while remaining:
                block = handle.read(min(8 * 2**20, remaining))
                if not block:
                    raise EOFError(f"bank ended while reading record at {offset}")
                digest.update(block)
                remaining -= len(block)
    return digest.hexdigest()


def hash_stage_edges(bank: Path) -> str:
    digest = hashlib.sha256()
    with bank.open("rb", buffering=0) as handle:
        for token in STAGE_TOKENS:
            offsets = offsets_for(token)
            handle.seek(offsets[0])
            first = handle.read(4096)
            handle.seek(offsets[-1] + EXPERT_BYTES - 4096)
            last = handle.read(4096)
            if len(first) != 4096 or len(last) != 4096:
                raise EOFError("bank ended while reading staging edge")
            digest.update(first)
            digest.update(last)
    return digest.hexdigest()


def literal_assignments(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    assignments: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            assignments[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    return assignments


def main() -> None:
    prereg_text = PREREG.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    input_hashes = {
        "preregistration_sha256": sha256(PREREG),
        "evaluator_sha256": sha256(RUNNER),
        "bank_manifest_sha256": sha256(MANIFEST),
        "result_sha256": sha256(RESULT),
    }
    provenance_checks = {
        key: value == result["inputs"][key]
        for key, value in input_hashes.items()
        if key != "result_sha256"
    }
    provenance_checks.update(
        {
            "manifest_bank_sha_matches_result": manifest["bank_sha256"] == result["inputs"]["bank_sha256_from_verified_manifest"],
            "manifest_bank_sha_matches_frozen_value": manifest["bank_sha256"] == EXPECTED_BANK_SHA256,
            "bank_file_size_exact": BANK.stat().st_size == BANK_BYTES,
            "manifest_bank_size_exact": manifest["contract"]["bank_bytes"] == BANK_BYTES,
            "manifest_expert_size_exact": manifest["contract"]["expert_bytes"] == EXPERT_BYTES,
            "manifest_order_exact": manifest["order"] == "layer-major; expert 0..511 routed then 512 shared; gate/up/down",
        }
    )

    assignments = literal_assignments(runner_text)
    source_contract_checks = {
        "token_constant": assignments.get("TOKEN") == TOKEN,
        "arms_constant": tuple(assignments.get("ARMS", ())) == ARMS,
        "warmups_constant": assignments.get("WARMUPS") == WARMUPS,
        "rounds_constant": assignments.get("ROUNDS") == ROUNDS,
        "expected_bank_sha_constant": assignments.get("EXPECTED_BANK_SHA256") == EXPECTED_BANK_SHA256,
        "uses_cuda_events": "cp.cuda.Event()" in runner_text and "cp.cuda.get_elapsed_time" in runner_text,
        "uses_async_h2d": "memcpyAsync" in runner_text and "memcpyHostToDevice" in runner_text,
        "uses_read_only_memmap": 'mode="r"' in runner_text,
        "uses_np_copyto_gather": "np.copyto" in runner_text,
        "prereg_has_frozen_gates": all(fragment in prereg_text for fragment in ("<=45 ms", "<=0,80", "<=0,90")),
    }

    route_offsets = offsets_for(TOKEN)
    route_checks = {
        "source_token": result["inputs"]["source_token"] == TOKEN,
        "offset_count_480": len(route_offsets) == LAYERS * TOP_K == 480,
        "all_offsets_in_bank": all(0 <= offset <= BANK_BYTES - EXPERT_BYTES for offset in route_offsets),
        "ten_unique_per_layer": all(len(set(route(TOKEN, layer))) == TOP_K for layer in range(LAYERS)),
        "routed_experts_exclude_shared_512": all(0 <= expert < 512 for layer in range(LAYERS) for expert in route(TOKEN, layer)),
        "token_bytes_exact": len(route_offsets) * EXPERT_BYTES == TOKEN_BYTES == result["physical"]["token_bytes"],
        "records_per_token_exact": result["physical"]["records_per_token"] == 480,
        "pinned_bytes_exact": result["physical"]["pinned_bytes"] == TOKEN_BYTES,
        "two_device_buffers_exact": result["physical"]["device_bytes"] == 2 * TOKEN_BYTES,
    }

    actual_bank_sha = sha256(BANK)
    actual_source_sha = hash_source_token(BANK, route_offsets)
    actual_edge_sha = hash_stage_edges(BANK)
    data_hash_checks = {
        "full_bank_sha256": actual_bank_sha == EXPECTED_BANK_SHA256,
        "source_pinned_sha256": actual_source_sha == result["inputs"]["source_pinned_sha256"],
        "staging_edge_digest": actual_edge_sha == result["staging"]["edge_digest"],
    }

    stored_orders = result["protocol"]["orders"]
    wanted_orders = expected_orders()
    order_counter = Counter(tuple(order) for order in stored_orders)
    position_counts = {
        arm: [sum(order[position] == arm for order in stored_orders) for position in range(3)]
        for arm in ARMS
    }
    order_checks = {
        "warmups_per_arm_10": result["protocol"]["warmups_per_arm"] == WARMUPS,
        "rounds_120": result["protocol"]["rounds"] == ROUNDS,
        "stored_order_count_120": len(stored_orders) == ROUNDS,
        "orders_match_rotation_reverse_algorithm": stored_orders == wanted_orders,
        "each_of_six_permutations_20_times": len(order_counter) == 6 and set(order_counter.values()) == {20},
        "each_arm_each_position_40_times": all(counts == [40, 40, 40] for counts in position_counts.values()),
    }

    timing_recomputed: dict[str, dict[str, float | int]] = {}
    timing_stat_checks: dict[str, dict[str, bool]] = {}
    sample_checks: dict[str, bool] = {}
    for arm in ARMS:
        raw = result["timing"][arm]["raw_ms"]
        timing_recomputed[arm] = recompute_stats(raw)
        timing_stat_checks[arm] = stats_match(timing_recomputed[arm], result["timing"][arm]["stats"])
        sample_checks[f"{arm}_120_finite"] = len(raw) == ROUNDS and all(math.isfinite(float(value)) for value in raw)

    stage_raw = result["staging"]["raw_ms"]
    stage_recomputed = recompute_stats(stage_raw)
    staging_stat_checks = stats_match(stage_recomputed, result["staging"]["stats"])
    sample_checks["staging_32_finite"] = len(stage_raw) == len(STAGE_TOKENS) and all(
        math.isfinite(float(value)) for value in stage_raw
    )
    sample_checks["staging_tokens_exact"] = result["staging"]["tokens"] == list(STAGE_TOKENS)

    token_stats = timing_recomputed["token1"]
    record_stats = timing_recomputed["record480"]
    ratios = {
        "token1_over_record480_p50": float(token_stats["p50"]) / float(record_stats["p50"]),
        "token1_over_record480_p95": float(token_stats["p95"]) / float(record_stats["p95"]),
    }
    ratio_checks = {name: close(value, result["ratios"][name]) for name, value in ratios.items()}
    projection = {
        "ideal_overlap_p95_ms": max(float(stage_recomputed["p95"]), float(token_stats["p95"])),
        "fully_serial_p95_ms": float(stage_recomputed["p95"]) + float(token_stats["p95"]),
    }
    projection_checks = {name: close(value, result["projection"][name]) for name, value in projection.items()}

    correctness_claim_checks = {
        "three_named_arms": set(result["correctness"]) == set(ARMS),
        "all_claim_byte_equal": all(result["correctness"][arm]["byte_equal"] is True for arm in ARMS),
        "all_claim_full_token_bytes": all(result["correctness"][arm]["bytes"] == TOKEN_BYTES for arm in ARMS),
        "stored_aggregate_matches_arm_claims": result["gates"]["all_full_buffers_byte_equal"]
        == all(result["correctness"][arm]["byte_equal"] for arm in ARMS),
    }

    recomputed_gates = {
        "all_full_buffers_byte_equal": all(result["correctness"][arm]["byte_equal"] for arm in ARMS),
        "all_arms_120_finite_samples": all(sample_checks[f"{arm}_120_finite"] for arm in ARMS),
        "token1_p95_le_45ms": float(token_stats["p95"]) <= 45.0,
        "token1_p50_ratio_le_0_80": ratios["token1_over_record480_p50"] <= 0.80,
        "token1_p95_ratio_le_0_90": ratios["token1_over_record480_p95"] <= 0.90,
        "ideal_overlap_p95_le_45ms": projection["ideal_overlap_p95_ms"] <= 45.0,
    }
    gate_checks = {name: value == result["gates"][name] for name, value in recomputed_gates.items()}
    component_pass = all(
        recomputed_gates[name]
        for name in (
            "all_full_buffers_byte_equal",
            "all_arms_120_finite_samples",
            "token1_p95_le_45ms",
            "token1_p50_ratio_le_0_80",
            "token1_p95_ratio_le_0_90",
        )
    )
    overall_pass = component_pass and recomputed_gates["ideal_overlap_p95_le_45ms"]
    expected_status = "directpath_feasibility_pass" if overall_pass else (
        "h2d_component_pass_staging_closed" if component_pass else "directpath_closed"
    )
    verdict_checks = {
        "component_pass": component_pass == result["component_pass"],
        "overall_pass": overall_pass == result["overall_pass"],
        "status": expected_status == result["status"],
    }

    check_groups = {
        "provenance": provenance_checks,
        "runner_source_contract": source_contract_checks,
        "route_and_physical_order": route_checks,
        "data_hashes": data_hash_checks,
        "measurement_order": order_checks,
        "samples": sample_checks,
        "timing_stats": {arm: all(checks.values()) for arm, checks in timing_stat_checks.items()},
        "staging_stats": staging_stat_checks,
        "ratios": ratio_checks,
        "projection": projection_checks,
        "correctness_claim_internal_consistency": correctness_claim_checks,
        "gates": gate_checks,
        "verdict": verdict_checks,
    }
    flat_checks: list[bool] = []
    for group in check_groups.values():
        for value in group.values():
            flat_checks.append(bool(value))
    all_replayable_checks_pass = all(flat_checks)

    output = {
        "kind": "port80b_d1_token_batch_directpath_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "gpu_context_opened": False,
        "independent_verdict": "verified_negative" if all_replayable_checks_pass and not overall_pass else "verification_failed",
        "all_replayable_checks_pass": all_replayable_checks_pass,
        "verified_component_pass": component_pass,
        "verified_overall_pass": overall_pass,
        "verified_status": expected_status,
        "input_hashes": input_hashes,
        "verifier_sha256": sha256(Path(__file__)),
        "full_bank_sha256": actual_bank_sha,
        "source_token_sha256": actual_source_sha,
        "staging_edge_digest": actual_edge_sha,
        "checks": check_groups,
        "recomputed": {
            "timing": timing_recomputed,
            "staging": stage_recomputed,
            "ratios": ratios,
            "projection": projection,
            "gates": recomputed_gates,
        },
        "measurement_order_counts": {"/".join(order): count for order, count in sorted(order_counter.items())},
        "measurement_position_counts": position_counts,
        "limitation": (
            "The saved D1 result contains boolean full-device-buffer equality claims but no per-arm device hashes or buffers. "
            "This CPU-only audit can verify their internal aggregation and independently reconstruct all source/provenance hashes, "
            "but it cannot replay the transient GPU equality operation. The negative verdict does not depend on that limitation, "
            "because the preregistered p50/p95 ratio and staging gates fail."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    failed_gate_names = [name for name, value in recomputed_gates.items() if not value]
    report = f"""# PORT80B-D1 — onafhankelijke CPU-only verificatie

**Verdict:** `verified_negative`  
**GPU-context geopend:** nee  
**Alle replaybare checks:** {'PASS' if all_replayable_checks_pass else 'FAIL'}

## Onafhankelijk herberekend

| Arm | n | mean ms | p50 ms | p95 ms | p99 ms | min–max ms |
|---|---:|---:|---:|---:|---:|---:|
| record480 | {timing_recomputed['record480']['count']} | {timing_recomputed['record480']['mean']:.6f} | {timing_recomputed['record480']['p50']:.6f} | {timing_recomputed['record480']['p95']:.6f} | {timing_recomputed['record480']['p99']:.6f} | {timing_recomputed['record480']['min']:.6f}–{timing_recomputed['record480']['max']:.6f} |
| layer48 | {timing_recomputed['layer48']['count']} | {timing_recomputed['layer48']['mean']:.6f} | {timing_recomputed['layer48']['p50']:.6f} | {timing_recomputed['layer48']['p95']:.6f} | {timing_recomputed['layer48']['p99']:.6f} | {timing_recomputed['layer48']['min']:.6f}–{timing_recomputed['layer48']['max']:.6f} |
| token1 | {timing_recomputed['token1']['count']} | {timing_recomputed['token1']['mean']:.6f} | {timing_recomputed['token1']['p50']:.6f} | {timing_recomputed['token1']['p95']:.6f} | {timing_recomputed['token1']['p99']:.6f} | {timing_recomputed['token1']['min']:.6f}–{timing_recomputed['token1']['max']:.6f} |
| mmap→pinned staging | {stage_recomputed['count']} | {stage_recomputed['mean']:.6f} | {stage_recomputed['p50']:.6f} | {stage_recomputed['p95']:.6f} | {stage_recomputed['p99']:.6f} | {stage_recomputed['min']:.6f}–{stage_recomputed['max']:.6f} |

- `token1/record480` p50-ratio: **{ratios['token1_over_record480_p50']:.9f}**.
- `token1/record480` p95-ratio: **{ratios['token1_over_record480_p95']:.9f}**.
- Ideale overlap-p95: **{projection['ideal_overlap_p95_ms']:.6f} ms**.
- Volledig seriële p95-projectie: **{projection['fully_serial_p95_ms']:.6f} ms**.

## Poorten

| Poort | Herberekend |
|---|---|
| alle opgeslagen full-bufferclaims gelijk | {recomputed_gates['all_full_buffers_byte_equal']} |
| 120 eindige samples per H2D-arm | {recomputed_gates['all_arms_120_finite_samples']} |
| token1 p95 ≤45 ms | {recomputed_gates['token1_p95_le_45ms']} |
| token1/record480 p50 ≤0,80 | {recomputed_gates['token1_p50_ratio_le_0_80']} |
| token1/record480 p95 ≤0,90 | {recomputed_gates['token1_p95_ratio_le_0_90']} |
| ideale overlap-p95 ≤45 ms | {recomputed_gates['ideal_overlap_p95_le_45ms']} |

Gefaalde poorten: `{', '.join(failed_gate_names)}`. De onafhankelijke status is daarom `directpath_closed`, gelijk aan het bronresultaat.

## Provenance, volgorde en hashes

- Preregistratie-, runner- en manifest-SHA's matchen de in D1 opgeslagen waarden.
- Het geaudite bronresultaat heeft SHA-256 `{input_hashes['result_sha256']}`; de verifier zelf `{sha256(Path(__file__))}`.
- De fysieke bank is opnieuw volledig CPU-side gehasht: `{actual_bank_sha}`; dit matcht het bevroren manifest.
- De 480 layer-major/top-10-records van token 10.000 zijn onafhankelijk uit SplitMix64 gereconstrueerd. Hun geordende bron-SHA is `{actual_source_sha}` en matcht D1.
- De edge-digest van exact tokens 10.001–10.032 is `{actual_edge_sha}` en matcht D1.
- Alle 120 meetorders matchen het rotatie/omkeerprotocol. Elk van de zes permutaties komt 20 maal voor; elke arm staat 40 maal op iedere positie.
- De fysieke contracten zijn 48×10 records, 2.027.520 bytes per record en 973.209.600 bytes per token.

## Bewijsgrens

Het opgeslagen resultaat bevat per arm alleen `byte_equal: true` en geen devicebufferhash. Een CPU-only audit kan de tijdelijke GPU-buffers daarom niet post-hoc opnieuw vergelijken. De aggregatie van die claims is intern correct en alle bronhashes zijn onafhankelijk gereconstrueerd. Deze beperking verandert het negatieve verdict niet: de p50-/p95-ratiopoorten en de stagingpoort falen onafhankelijk van de correctness-pass.

Er is geen expertcompute, echte 80B-router, kwaliteit, dense shell, werkelijke staging/H2D-overlap of end-to-end tokens/s geverifieerd.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({
        "independent_verdict": output["independent_verdict"],
        "all_replayable_checks_pass": all_replayable_checks_pass,
        "verified_status": expected_status,
        "failed_gates": failed_gate_names,
        "output": str(OUTPUT),
        "report": str(REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
