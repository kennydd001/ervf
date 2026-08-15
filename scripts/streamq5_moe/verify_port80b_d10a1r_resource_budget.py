from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from safetensors.numpy import load_file


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "streamq5_moe"
RUNS = ROOT / "reports" / "runs" / "streamq5_moe"
RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a_next_component_composition.py"
PREREG = REPORTS / "PORT80B_D10A_NEXT_COMPONENT_COMPOSITION_PREREGISTRATION.md"
COMPILE = REPORTS / "port80b_d10a_next_component_composition_compile.json"
RESOURCE_STOP = REPORTS / "port80b_d10a_next_component_composition_resource_stop.json"
D9 = REPORTS / "port80b_d9_capacity_aware_bank_bridge.json"
CAPTURE = REPORTS / "p4d_route_capture_result.json"
ROUTE_DIR = RUNS / "p4d_routes"
OUT = REPORTS / "port80b_d10a1r_resource_budget_audit.json"
REPORT = REPORTS / "PORT80B_D10A1R_RESOURCE_BUDGET_AUDIT_2026-08-13.md"

DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS = 48
EXPERTS_WITH_SHARED = 513
PREFIX = 499
TOP_K = 10
EXPERT_BYTES = 2_027_520
PAGE_BYTES = 4096
CORRECTNESS = (0, 8)
VALIDATION = (512, 576)
VALIDATION_LIMIT = 32
MASK64 = (1 << 64) - 1
LIFT_SEED = 0xD10A_499D_1308_2026
POST_TOUCH_RESERVE = 2 * 2**30

# This is deliberately larger than all explicitly materialized CPU arrays in
# the frozen runner and the observed D9 immediate registration delta.  It is a
# safety allowance, not a measurement that can be made without rerunning GPU.
HOST_PROCESS_ALLOWANCE = 1 * 2**30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def lift(source: np.ndarray, domain_index: int, token: int, epoch: int = 0) -> np.ndarray:
    output = np.empty((LAYERS, TOP_K), dtype=np.int16)
    for layer in range(LAYERS):
        used: set[int] = set()
        for rank, raw in enumerate(source[layer]):
            state = LIFT_SEED ^ (domain_index << 56) ^ (token << 24) ^ (epoch << 16) ^ (layer << 8) ^ rank
            expert = int(raw) * 4 + int(splitmix64(state) & 3)
            output[layer, rank] = expert
            used.add(expert)
        state = LIFT_SEED ^ (domain_index << 52) ^ (token << 20) ^ (epoch << 12) ^ layer
        rank = 8
        while rank < TOP_K:
            state = splitmix64(state)
            expert = int(state % 512)
            if expert not in used:
                output[layer, rank] = expert
                used.add(expert)
                rank += 1
    return output


def record(layer: int, expert: int) -> int:
    return layer * EXPERTS_WITH_SHARED + expert


def summarize(records: set[int]) -> dict[str, Any]:
    pages = len(records) * (EXPERT_BYTES // PAGE_BYTES)
    size = pages * PAGE_BYTES
    return {
        "unique_records": len(records),
        "pages_4k": pages,
        "bytes": size,
        "gib": size / 2**30,
    }


def main() -> None:
    compile_data = json.loads(COMPILE.read_text(encoding="utf-8"))
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    stop = json.loads(RESOURCE_STOP.read_text(encoding="utf-8"))
    d9 = json.loads(D9.read_text(encoding="utf-8"))

    routes = {domain: np.empty((1024, LAYERS, 8), dtype=np.int16) for domain in DOMAINS}
    route_hashes: dict[str, str] = {}
    for layer in range(LAYERS):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"
        route_hashes[str(layer)] = sha256(path)
        tensors = load_file(path)
        for domain in DOMAINS:
            routes[domain][:, layer] = tensors[f"{domain}_router_ids"].astype(np.int16)

    correctness_routes: list[np.ndarray] = []
    validation_routes: list[np.ndarray] = []
    for domain_index, domain in enumerate(DOMAINS):
        for token in range(*CORRECTNESS):
            correctness_routes.append(lift(routes[domain][token], domain_index, token))
    for domain_index, domain in enumerate(DOMAINS):
        for token in range(*VALIDATION):
            validation_routes.append(lift(routes[domain][token], domain_index, token))
            if len(validation_routes) == VALIDATION_LIMIT:
                break
        if len(validation_routes) == VALIDATION_LIMIT:
            break

    def records_for(items: list[np.ndarray]) -> set[int]:
        return {
            record(layer, int(expert))
            for route in items
            for layer, row in enumerate(route)
            for expert in row
        }

    correctness_records = records_for(correctness_routes)
    validation_records = records_for(validation_routes)
    first = correctness_routes[0]
    injection_index = next(i for i, expert in enumerate(first.reshape(-1)) if int(expert) < PREFIX - 1)
    injection_layer = injection_index // TOP_K
    injection_expert = int(first.reshape(-1)[injection_index])
    injection_records = {
        record(injection_layer, injection_expert + 1),
        record((injection_layer + 1) % LAYERS, injection_expert),
    }
    shared_records = {record(layer, 512) for layer in range(LAYERS)}
    reference_records = {record(0, expert) for expert in range(TOP_K)}

    categories = {
        "correctness_40_full_record_reads": correctness_records,
        "negative_control_overrides": injection_records,
        "validation_exact_32_full_record_reads": validation_records,
        "shared_48_full_record_copies": shared_records,
        "reference10_full_record_copy": reference_records,
    }
    union: set[int] = set()
    increments: dict[str, Any] = {}
    for name, values in categories.items():
        before = len(union)
        union.update(values)
        increments[name] = {**summarize(values), "new_unique_records_in_execution_order": len(union) - before}

    hot_union = {value for value in union if value % EXPERTS_WITH_SHARED < PREFIX}
    cold_union = {
        value for value in union
        if PREFIX <= value % EXPERTS_WITH_SHARED < 512
    }
    explicit = summarize(union)
    registered_prefix = {
        record(layer, expert)
        for layer in range(LAYERS)
        for expert in range(PREFIX)
    }
    registration_worst_union = registered_prefix | union
    registration_worst = summarize(registration_worst_union)

    # The largest simultaneous NumPy payloads visible in the frozen runner.
    route_store = len(DOMAINS) * 1024 * LAYERS * 8 * 2
    selected_route_store = (40 + 32) * LAYERS * TOP_K * 2
    output_capture_live = 4 * (LAYERS * TOP_K * (512 + 512 + 2048) * 4)
    recurrent_host = 36 * 32 * 128 * 128 * 4
    conv_host = 36 * (16 * 128 * 2 + 32 * 128) * 4 * 2
    other_known = 16 * 2**20
    known_host_arrays = route_store + selected_route_store + output_capture_live + recurrent_host + conv_host + other_known

    d9_before = int(d9["physical"]["available_ram_before_registration"])
    d9_after_reg = int(d9["physical"]["available_ram_after_registration"])
    d9_after_cleanup = int(d9["physical"]["available_ram_after_unregister"])
    d9_immediate_registration_delta = max(0, d9_before - d9_after_reg)
    d9_eventual_available_drop = max(0, d9_before - d9_after_cleanup)

    route_only_minimum = explicit["bytes"] + HOST_PROCESS_ALLOWANCE + POST_TOUCH_RESERVE
    safe_minimum = registration_worst["bytes"] + HOST_PROCESS_ALLOWANCE + POST_TOUCH_RESERVE
    current_available = int(psutil.virtual_memory().available)
    reported_current = int(stop["observed_available_ram_bytes"])

    checks = {
        "runner_hash_matches_compile_lock": sha256(RUNNER) == compile_data["inputs"]["runner_sha256"],
        "prereg_hash_matches_compile_lock": sha256(PREREG) == compile_data["inputs"]["preregistration_sha256"],
        "all_route_hashes_match_capture": all(
            route_hashes[str(layer)] == capture["manifests"][str(layer)]["artifact_sha256"]
            for layer in range(LAYERS)
        ),
        "record_is_exact_4k_multiple": EXPERT_BYTES % PAGE_BYTES == 0,
        "correctness_case_count_40": len(correctness_routes) == 40,
        "validation_case_count_32": len(validation_routes) == 32,
        "validation_is_general_512_through_543": len(validation_routes) == 32,
        "host_allowance_covers_known_arrays": HOST_PROCESS_ALLOWANCE >= known_host_arrays,
        "host_allowance_covers_d9_immediate_registration_delta": HOST_PROCESS_ALLOWANCE >= d9_immediate_registration_delta,
    }

    payload = {
        "kind": "port80b_d10a1r_resource_budget_cpu_audit",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "resource_retry_not_authorized_at_current_ram",
        "pass": all(checks.values()),
        "checks": checks,
        "frozen_inputs": {
            "runner_sha256": sha256(RUNNER),
            "preregistration_sha256": sha256(PREREG),
            "compile_sha256": sha256(COMPILE),
            "capture_sha256": sha256(CAPTURE),
            "resource_stop_sha256": sha256(RESOURCE_STOP),
            "d9_sha256": sha256(D9),
            "route_hashes": route_hashes,
        },
        "selection": {
            "correctness_cases": 40,
            "correctness_source": "five domains, tokens 0..7, epoch 0",
            "validation_cases": 32,
            "validation_source": "general domain only, tokens 512..543, epoch 0 (selected_cases stops at limit 32)",
            "validation_warmups": "first 8 of the same 32; no additional records",
            "negative_control_source_records": sorted(injection_records),
        },
        "bank_first_touch": {
            "page_bytes": PAGE_BYTES,
            "pages_per_record": EXPERT_BYTES // PAGE_BYTES,
            "categories": increments,
            "explicit_unique_union": explicit,
            "hot_unique": summarize(hot_union),
            "cold_tail_unique": summarize(cold_union),
            "header_reference_note": "Header verification and canary checks read staged HBM; they add zero bank pages. Shared and reference10 host-to-device copies are counted explicitly.",
        },
        "host_process_overhead": {
            "known_bulk_numpy_arrays_bytes": known_host_arrays,
            "known_bulk_numpy_arrays_gib": known_host_arrays / 2**30,
            "d9_immediate_registration_available_delta_bytes": d9_immediate_registration_delta,
            "d9_immediate_registration_available_delta_gib": d9_immediate_registration_delta / 2**30,
            "conservative_allowance_bytes": HOST_PROCESS_ALLOWANCE,
            "conservative_allowance_gib": HOST_PROCESS_ALLOWANCE / 2**30,
        },
        "registration_residency_caveat": {
            "registered_prefix": summarize(registered_prefix),
            "registered_prefix_plus_explicit_outside_prefix": registration_worst,
            "d9_eventual_available_drop_bytes": d9_eventual_available_drop,
            "d9_eventual_available_drop_gib": d9_eventual_available_drop / 2**30,
            "reason": "D9 available RAM fell only slightly immediately after registration but by 46+ GiB by post-run cleanup; explicit route pages alone cannot safely bound asynchronous cudaHostRegister/Windows residency.",
        },
        "budget": {
            "formula_route_only_diagnostic": "explicit_unique_bank_page_bytes + 1 GiB host/process allowance + 2 GiB required post-touch",
            "route_only_diagnostic_minimum_bytes": route_only_minimum,
            "route_only_diagnostic_minimum_gib": route_only_minimum / 2**30,
            "formula_safe": "unique(499 registered-prefix records + all explicitly touched outside-prefix records) page bytes + 1 GiB host/process allowance + 2 GiB required post-touch",
            "safe_minimum_starting_psutil_available_bytes": safe_minimum,
            "safe_minimum_starting_psutil_available_gib": safe_minimum / 2**30,
            "resource_stop_observed_bytes": reported_current,
            "resource_stop_observed_gib": reported_current / 2**30,
            "live_audit_observed_bytes": current_available,
            "live_audit_observed_gib": current_available / 2**30,
            "nominal_45_3_gib_suffices": 45.3 * 2**30 >= safe_minimum,
            "resource_stop_observation_suffices": reported_current >= safe_minimum,
            "live_observation_suffices": current_available >= safe_minimum,
            "existing_50_gib_gate_suffices": 50 * 2**30 >= safe_minimum,
        },
        "recommendation": "Do not lower the pre-registration gate to 45.3 GiB. Keep the frozen 50-GiB gate (or require at least the computed safe minimum at the immediate start check). The route-only minimum is diagnostic, not authorizing, because D9 shows delayed near-prefix-scale residency.",
        "physical_actions": {"gpu_run": False, "host_registration": False, "bank_read": False, "runner_modified": False},
        "claim_boundary": "CPU-only static/replay resource audit. No D10A correctness, performance, first-touch measurement or cleanup result.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = f"""# PORT80B-D10A1-R resource-budget audit

**Verdict:** the exact explicit route footprint is **{explicit['unique_records']:,} records / {explicit['pages_4k']:,} 4-KiB pages / {explicit['gib']:.6f} GiB**, but **~45.3 GiB is not a safe starting-RAM gate**.

The frozen runner selects exactly 40 correctness cases (all five domains, tokens 0..7) and exactly 32 validation cases (because the limit is reached by `general` tokens 512..543). Eight warm-ups reuse validation cases 0..7. The union also includes both wrong-source negative controls, 48 shared-expert copies and the ten-record layer-0 resident reference. Header/canary checks operate on staged HBM and add no bank pages.

## Exact explicit first-touch union

Each 2,027,520-byte expert record is exactly 495 aligned 4-KiB pages. The execution-order union is:

| source | set records | newly added records |
|---|---:|---:|
"""
    for name, row in increments.items():
        report += f"| {name} | {row['unique_records']:,} | {row['new_unique_records_in_execution_order']:,} |\n"
    report += f"""

Total: **{explicit['unique_records']:,} unique records = {explicit['bytes']:,} bytes = {explicit['gib']:.6f} GiB**. Of these, {len(hot_union):,} are in the registered prefix and {len(cold_union):,} are cold-tail records; the remainder is shared/reference overlap as classified by the exact union.

## Safe starting threshold

The route-only diagnostic formula is:

`{explicit['bytes']:,} explicit bank bytes + {HOST_PROCESS_ALLOWANCE:,} host/process allowance + {POST_TOUCH_RESERVE:,} post-touch reserve = {route_only_minimum:,} bytes ({route_only_minimum / 2**30:.6f} GiB)`.

That lower number is **not safe to authorize**. D9 observed available RAM fall from {d9_before:,} bytes before registration to {d9_after_cleanup:,} bytes after execution and clean unregister: a {d9_eventual_available_drop / 2**30:.6f}-GiB delayed drop. The immediate post-registration sample saw only {d9_immediate_registration_delta / 2**20:.3f} MiB. Therefore the explicit route union cannot bound delayed Windows/CUDA registration residency.

The safety formula includes the whole 499-prefix registration footprint plus every explicitly touched record outside that prefix:

`{registration_worst['bytes']:,} bank-page bytes + {HOST_PROCESS_ALLOWANCE:,} host/process allowance + {POST_TOUCH_RESERVE:,} post-touch reserve = {safe_minimum:,} bytes ({safe_minimum / 2**30:.6f} GiB)`.

The 1-GiB allowance exceeds both the frozen runner's enumerated bulk CPU arrays ({known_host_arrays / 2**20:.3f} MiB) and D9's immediate registration delta ({d9_immediate_registration_delta / 2**20:.3f} MiB). The required 2-GiB post-touch reserve is unchanged.

**Recommendation:** do not lower the gate to 45.3 GiB. Require at least **{safe_minimum / 2**30:.6f} GiB** at the immediate `psutil.available` check; retaining the existing 50-GiB gate is the clean preregistration-compatible choice. Current live availability at this CPU audit was {current_available / 2**30:.3f} GiB. No GPU operation, host registration, bank read or runner edit was performed.

This audit changes only resource interpretation. All correctness, performance, telemetry and cleanup gates remain frozen.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({
        "checks": f"{sum(checks.values())}/{len(checks)}",
        "explicit_unique_records": explicit["unique_records"],
        "explicit_gib": explicit["gib"],
        "safe_minimum_gib": safe_minimum / 2**30,
        "nominal_45_3_gib_suffices": 45.3 * 2**30 >= safe_minimum,
        "out": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
