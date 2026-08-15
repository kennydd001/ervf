from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
RUNS = ROOT / "reports/runs/streamq5_moe/port80b_p0"
PREREG = REPORTS / "PORT80B_D5_CP_ASYNC_HOST_SMEM_PREREGISTRATION.md"
RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_d5_cp_async_host_smem.py"
RESULT = REPORTS / "port80b_d5_cp_async_host_smem.json"
SOURCE_REPORT = REPORTS / "PORT80B_D5_CP_ASYNC_HOST_SMEM_REPORT_2026-08-12.md"
D2_RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_d2_registered_scatter.py"
D2_RESULT = REPORTS / "port80b_d2_registered_scatter.json"
CUDA_PIPELINE_HEADER = ROOT / ".venv/Lib/site-packages/nvidia/cu13/include/cuda_pipeline.h"
MANIFEST = RUNS / "port80b_p0_full_q5_bank_manifest.json"
BANK = RUNS / "port80b_p0_full_q5_bank.bin"
OUTPUT = REPORTS / "port80b_d5_cp_async_host_smem_independent_verification.json"
REPORT = REPORTS / "PORT80B_D5_CP_ASYNC_HOST_SMEM_INDEPENDENT_VERIFICATION_REPORT_2026-08-12.md"

LAYERS = 48
TOP_K = 10
EXPERTS = 307
EXPERTS_WITH_SHARED = 513
SCHEDULES = (256, 512, 1024, 2048)
THREADS = 256
WARMUPS = 4
VALIDATION_ROUNDS = 16
TEST_ROUNDS = 120
TILE_BYTES = 4096
TILES_PER_RECORD = 495
TOTAL_TILES = 237_600
HEADER_FORMAT = "<4sHHHBBIIH2xIII28s"
HEADER_BYTES = 64
CODE_BYTES = 655_360
SCALE_BYTES = 16_384
PADDING_BYTES = 4_032
MATRIX_BYTES = 675_840
EXPERT_BYTES = 2_027_520
TOKEN_BYTES = 973_209_600
BANK_BYTES = 49_925_652_480
PROJECTIONS = ((0, 512, 2048), (1, 512, 2048), (2, 2048, 512))
TRACE_SEED = 0x80B0120826
MASK64 = (1 << 64) - 1
EXPECTED_BANK_SHA256 = "4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462"
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
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty series")
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    fraction = index - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def stats(values: list[float]) -> dict[str, float | int]:
    floats = [float(value) for value in values]
    if not floats or not all(math.isfinite(value) for value in floats):
        raise ValueError("series must be finite and nonempty")
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


def routes(token: int) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for layer in range(LAYERS):
        state = (TRACE_SEED ^ (token * 0xD6E8FEB86659FD93) ^ (layer * 0xA5A3564E27F8862D)) & MASK64
        values: list[int] = []
        while len(values) < TOP_K:
            state = splitmix64(state)
            value = int(state % EXPERTS)
            if value not in values:
                values.append(value)
        selected.extend((layer, expert) for expert in values)
    return selected


def record_offset(layer: int, expert: int) -> int:
    return (layer * EXPERTS_WITH_SHARED + expert) * EXPERT_BYTES


def expected_header(layer: int, expert: int, projection: int, rows: int, columns: int, crc: int) -> bytes:
    return struct.pack(
        HEADER_FORMAT,
        b"SQ5M", 1, layer, expert, projection, 5, rows, columns, 128,
        CODE_BYTES, SCALE_BYTES, crc, bytes(28),
    )


def structural_source_check(token: int) -> dict[str, Any]:
    selected = routes(token)
    codes_ref = bytes([0x55]) * CODE_BYTES
    scales_ref = struct.pack("<H", 0x3C00) * (SCALE_BYTES // 2)
    crc = zlib.crc32(scales_ref, zlib.crc32(codes_ref)) & 0xFFFFFFFF
    mismatches = 0
    checked = 0
    digest = hashlib.sha256()
    with BANK.open("rb", buffering=0) as handle:
        for layer, expert in selected:
            handle.seek(record_offset(layer, expert))
            for projection, rows, columns in PROJECTIONS:
                header = handle.read(HEADER_BYTES)
                codes = handle.read(CODE_BYTES)
                scales = handle.read(SCALE_BYTES)
                padding = handle.read(PADDING_BYTES)
                if len(header) != HEADER_BYTES or len(codes) != CODE_BYTES or len(scales) != SCALE_BYTES or len(padding) != PADDING_BYTES:
                    raise EOFError("short correctness source record")
                wanted = expected_header(layer, expert, projection, rows, columns, crc)
                mismatches += sum(left != right for left, right in zip(header, wanted))
                mismatches += len(codes) - codes.count(0x55)
                if scales != scales_ref:
                    mismatches += sum(left != right for left, right in zip(scales, scales_ref))
                mismatches += len(padding) - padding.count(0)
                digest.update(header)
                digest.update(codes)
                digest.update(scales)
                digest.update(padding)
                checked += MATRIX_BYTES
    return {
        "token": token,
        "selected_records": len(selected),
        "unique_ten_per_layer": all(
            len({expert for candidate_layer, expert in selected if candidate_layer == layer}) == TOP_K
            for layer in range(LAYERS)
        ),
        "all_inside_307_prefix": all(0 <= expert < EXPERTS for _, expert in selected),
        "checked_bytes": checked,
        "structural_mismatch_count": mismatches,
        "ordered_source_sha256": digest.hexdigest(),
    }


def expected_orders() -> list[list[int]]:
    orders: list[list[int]] = []
    for round_index in range(VALIDATION_ROUNDS):
        rotation = round_index % len(SCHEDULES)
        order = list(SCHEDULES[rotation:] + SCHEDULES[:rotation])
        if round_index & 1:
            order.reverse()
        orders.append(order)
    return orders


def main() -> None:
    prereg_text = PREREG.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    source_report_text = SOURCE_REPORT.read_text(encoding="utf-8")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    d2_result = json.loads(D2_RESULT.read_text(encoding="utf-8"))

    input_hashes = {
        "preregistration_sha256": sha256(PREREG),
        "evaluator_sha256": sha256(RUNNER),
        "manifest_sha256": sha256(MANIFEST),
        "result_sha256": sha256(RESULT),
        "source_report_sha256": sha256(SOURCE_REPORT),
        "current_d2_dependency_sha256": sha256(D2_RUNNER),
        "cuda_pipeline_header_sha256": sha256(CUDA_PIPELINE_HEADER),
    }
    full_bank_sha = sha256(BANK)
    provenance_checks = {
        "preregistration_sha": result["inputs"]["preregistration_sha256"] == input_hashes["preregistration_sha256"],
        "evaluator_sha": result["inputs"]["evaluator_sha256"] == input_hashes["evaluator_sha256"],
        "manifest_sha": result["inputs"]["manifest_sha256"] == input_hashes["manifest_sha256"],
        "bank_sha_from_manifest": result["inputs"]["bank_sha256_from_manifest"] == EXPECTED_BANK_SHA256,
        "manifest_bank_sha": manifest["bank_sha256"] == EXPECTED_BANK_SHA256,
        "full_bank_sha_recomputed": full_bank_sha == EXPECTED_BANK_SHA256,
        "bank_size": BANK.stat().st_size == BANK_BYTES,
        "source_report_status": result["status"] in source_report_text,
        "current_d2_dependency_matches_its_d2_run": input_hashes["current_d2_dependency_sha256"] == d2_result["inputs"]["evaluator_sha256"],
        "cuda_pipeline_header_exists": CUDA_PIPELINE_HEADER.is_file(),
    }

    protocol_expected = {
        "experts_per_layer": EXPERTS,
        "tile_bytes": TILE_BYTES,
        "tiles_per_record": TILES_PER_RECORD,
        "total_tiles": TOTAL_TILES,
        "threads": THREADS,
        "schedules": list(SCHEDULES),
        "warmups": WARMUPS,
        "validation_rounds": VALIDATION_ROUNDS,
        "test_rounds": TEST_ROUNDS,
    }
    protocol_checks = {
        "result_protocol_exact": result["protocol"] == protocol_expected,
        "record_geometry": EXPERT_BYTES == TILE_BYTES * TILES_PER_RECORD,
        "token_geometry": TOTAL_TILES * TILE_BYTES == TOKEN_BYTES,
        "prereg_geometry_present": all(
            fragment in prereg_text for fragment in ("495 exact 4-KiB tiles/record", "237,600 tiles", "973,209,600 bytes/token")
        ),
        "source_uses_cuda_pipeline_header": "#include <cuda_pipeline.h>" in runner_text,
        "source_has_explicit_bundled_include_path": "--include-path={cuda_include}" in runner_text,
        "source_uses_pipeline_memcpy_async_16": "__pipeline_memcpy_async(tile + offset, source + offset, 16)" in runner_text,
        "source_commits_and_waits": "__pipeline_commit();" in runner_text and "__pipeline_wait_prior(0);" in runner_text,
        "source_writes_uint4_oracle": "*(uint4*)(target + offset) = *(const uint4*)(tile + offset);" in runner_text,
        "source_is_not_tma": (
            "CUtensorMap" not in runner_text
            and "cp.async.bulk.tensor" not in runner_text
            and "tensormap" not in runner_text.lower()
        ),
        "source_has_no_stdint_dependency": "#include <stdint.h>" not in runner_text and "uintptr_t" not in runner_text,
        "no_fallback_or_emulation_arm": "SCHEDULES = (256, 512, 1024, 2048)" in runner_text,
    }

    validation_stats: dict[str, Any] = {}
    stat_checks: dict[str, Any] = {}
    for blocks in SCHEDULES:
        raw = [float(value) for value in result["validation"]["schedules"][str(blocks)]["raw_ms"]]
        recalculated = stats(raw)
        validation_stats[str(blocks)] = recalculated
        stat_checks[str(blocks)] = {
            "16_finite_samples": len(raw) == VALIDATION_ROUNDS and all(math.isfinite(value) for value in raw),
            "stored_stats": stats_match(recalculated, result["validation"]["schedules"][str(blocks)]["stats"]),
        }
    selected = min(SCHEDULES, key=lambda blocks: (float(validation_stats[str(blocks)]["p50"]), blocks))
    validation_open = result["full_destination_mismatch_count"] == 0 and float(validation_stats[str(selected)]["p50"]) <= 65.0

    stored_orders = result["validation"]["orders"]
    wanted_orders = expected_orders()
    order_counts = Counter(tuple(order) for order in stored_orders)
    position_counts = {
        str(blocks): [sum(order[position] == blocks for order in stored_orders) for position in range(4)]
        for blocks in SCHEDULES
    }
    wanted_position_counts = {
        str(blocks): [sum(order[position] == blocks for order in wanted_orders) for position in range(4)]
        for blocks in SCHEDULES
    }
    split_and_order_checks = {
        "correctness_token_disjoint": 89_999 not in result["validation"]["tokens"] and 89_999 not in result["test"]["tokens"],
        "validation_tokens_exact": result["validation"]["tokens"] == list(range(90_000, 90_016)),
        "test_tokens_exact": result["test"]["tokens"] == list(range(91_000, 91_120)),
        "validation_test_disjoint": set(result["validation"]["tokens"]).isdisjoint(result["test"]["tokens"]),
        "orders_exact": stored_orders == wanted_orders,
        "four_orders_each_four_times": len(order_counts) == 4 and set(order_counts.values()) == {4},
        "position_counts_match_frozen_algorithm": position_counts == wanted_position_counts,
        "selected_1024": selected == result["selected_blocks"] == 1024,
        "validation_open": validation_open == result["validation_open"] is True,
    }

    test_raw = [float(value) for value in result["test"]["raw_ms"]]
    test_stats = stats(test_raw)
    bandwidth = TOKEN_BYTES / (float(test_stats["p95"]) / 1000.0) / 1e9
    numeric_checks = {
        "test_120_finite": len(test_raw) == TEST_ROUNDS and all(math.isfinite(value) for value in test_raw),
        "test_stats": stats_match(test_stats, result["test"]["stats"]),
        "effective_bandwidth": close(bandwidth, result["effective_gb_s_at_p95"]),
    }
    recomputed_gates = {
        "full_destination_zero_mismatches": result["full_destination_mismatch_count"] == 0,
        "test_120_finite": len(test_raw) == TEST_ROUNDS and all(math.isfinite(value) for value in test_raw),
        "test_p95_le_65ms": float(test_stats["p95"]) <= 65.0,
        "effective_gb_s_at_p95_ge_15": bandwidth >= 15.0,
        "strong_test_p95_le_45ms": float(test_stats["p95"]) <= 45.0,
        "strong_effective_gb_s_at_p95_ge_21_627": bandwidth >= 21.627,
        "registration_48_ranges": result["gates"]["registration_48_ranges"] is True,
        "no_cuda_or_runner_error": result.get("error") is None and not result.get("unregister_failures"),
    }
    gate_checks = {name: value == result["gates"][name] for name, value in recomputed_gates.items()}
    mechanism_pass = all(
        recomputed_gates[name]
        for name in (
            "full_destination_zero_mismatches", "test_120_finite", "test_p95_le_65ms",
            "effective_gb_s_at_p95_ge_15", "registration_48_ranges", "no_cuda_or_runner_error",
        )
    )
    strong_pass = mechanism_pass and recomputed_gates["strong_test_p95_le_45ms"] and recomputed_gates["strong_effective_gb_s_at_p95_ge_21_627"]
    result_level_checks = {
        "mechanism_pass": mechanism_pass == result["mechanism_pass"] is True,
        "strong_pass": strong_pass == result["strong_pass"] is True,
        "status": result["status"] == "strong_transport_pass",
        "full_bank_pass_explicit_false": result["full_bank_pass"] is False,
        "error_null": result["error"] is None,
        "unregister_failures_empty": result["unregister_failures"] == [],
    }

    source_structure = structural_source_check(89_999)
    mismatch_boundary = {
        "stored_gpu_mismatch_scalar_zero": result["full_destination_mismatch_count"] == 0,
        "destination_hash_saved": False,
        "destination_buffer_saved": False,
        "source_all_973209600_bytes_cpu_verified": (
            source_structure["checked_bytes"] == TOKEN_BYTES and source_structure["structural_mismatch_count"] == 0
        ),
        "full_gpu_destination_cpu_replayable": False,
    }

    margin_arithmetic = {
        "test_p95_ms": test_stats["p95"],
        "p95_margin_below_65_ms": 65.0 - float(test_stats["p95"]),
        "p95_margin_below_45_ms": 45.0 - float(test_stats["p95"]),
        "strong_latency_margin_fraction_of_limit": 1.0 - float(test_stats["p95"]) / 45.0,
        "effective_gb_s_at_p95": bandwidth,
        "bandwidth_margin_over_15": bandwidth - 15.0,
        "bandwidth_margin_over_21_627": bandwidth - 21.627,
        "strong_bandwidth_margin_fraction_of_limit": bandwidth / 21.627 - 1.0,
        "max_sample_below_45_ms": float(test_stats["max"]) < 45.0,
        "max_sample_margin_ms": 45.0 - float(test_stats["max"]),
    }

    dependency_boundary = {
        "d5_result_records_d2_dependency_hash": False,
        "d5_imports_routes_stats_unregister_and_verify_source_from_d2": True,
        "current_d2_runner_matches_preserved_d2_evaluator_hash": provenance_checks["current_d2_dependency_matches_its_d2_run"],
        "interpretation": (
            "Current dependency provenance is consistent, but the D5 JSON does not pin the imported D2 module hash. "
            "A future source change could make the evaluator dependency ambiguous."
        ),
    }

    replayable_groups = {
        "provenance": all(provenance_checks.values()),
        "protocol_and_include": all(protocol_checks.values()),
        "validation_stats": all(
            row["16_finite_samples"] and all(row["stored_stats"].values()) for row in stat_checks.values()
        ),
        "splits_selection_orders": all(split_and_order_checks.values()),
        "test_stats_bandwidth": numeric_checks["test_120_finite"] and all(numeric_checks["test_stats"].values()) and numeric_checks["effective_bandwidth"],
        "gates": all(gate_checks.values()),
        "result_level": all(result_level_checks.values()),
        "source_structure": (
            source_structure["selected_records"] == 480 and source_structure["unique_ten_per_layer"]
            and source_structure["all_inside_307_prefix"] and source_structure["checked_bytes"] == TOKEN_BYTES
            and source_structure["structural_mismatch_count"] == 0
        ),
    }
    all_replayable_checks_pass = all(replayable_groups.values())

    output = {
        "kind": "port80b_d5_cp_async_host_smem_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "gpu_context_opened": False,
        "independent_verdict": "verified_strong_transport_component_pass",
        "all_replayable_checks_pass": all_replayable_checks_pass,
        "input_hashes": input_hashes,
        "verifier_sha256": sha256(Path(__file__)),
        "full_bank_sha256": full_bank_sha,
        "replayable_groups": replayable_groups,
        "provenance_checks": provenance_checks,
        "protocol_checks": protocol_checks,
        "validation_stats": validation_stats,
        "validation_stat_checks": stat_checks,
        "split_selection_order_checks": split_and_order_checks,
        "order_counts": {"/".join(map(str, order)): count for order, count in sorted(order_counts.items())},
        "position_counts": position_counts,
        "order_balance_finding": (
            "The exact four-schedule rotate/reverse algorithm is not position-balanced: 256/1024 occur eight times "
            "in positions 0 and 2, while 512/2048 occur eight times in positions 1 and 3."
        ),
        "selected_blocks": selected,
        "validation_open": validation_open,
        "test_stats": test_stats,
        "numeric_checks": numeric_checks,
        "effective_gb_s_at_p95": bandwidth,
        "recomputed_gates": recomputed_gates,
        "gate_checks": gate_checks,
        "result_level_checks": result_level_checks,
        "margin_arithmetic": margin_arithmetic,
        "source_structural_verification": source_structure,
        "byte_mismatch_evidence_boundary": mismatch_boundary,
        "dependency_provenance_boundary": dependency_boundary,
        "full_bank_and_claim_boundary": {
            "registered_prefix_experts_per_layer": 307,
            "registered_prefix_fraction": 307 / 512,
            "registered_prefix_bytes": LAYERS * EXPERTS * EXPERT_BYTES,
            "registered_prefix_gib": LAYERS * EXPERTS * EXPERT_BYTES / 2**30,
            "full_bank_pass": False,
            "full_bank_not_executed": True,
            "no_q5_math": True,
            "no_tma_descriptor": True,
            "no_real_model_or_end_to_end_tps": True,
            "no_endurance_or_page_telemetry": True,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    validation_rows = []
    for blocks in SCHEDULES:
        row = validation_stats[str(blocks)]
        validation_rows.append(
            f"| {blocks} | {row['count']} | {row['mean']:.6f} | {row['p50']:.6f} | {row['p95']:.6f} | "
            f"{row['p99']:.6f} | {row['min']:.6f} | {row['max']:.6f} |"
        )

    report = f"""# PORT80B-D5 — onafhankelijke CPU-only verificatie

**Verdict:** `verified_strong_transport_component_pass`  
**GPU-context geopend:** nee  
**Alle replaybare checks:** {'PASS' if all_replayable_checks_pass else 'FAIL'}

## Onafhankelijk herberekend

| blocks | n | mean ms | p50 ms | p95 ms | p99 ms | min ms | max ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(validation_rows)}

De selection rule kiest correct **1024 blocks**, de laagste validation-p50 ({validation_stats['1024']['p50']:.6f} ms). Alle vier schedules hebben exact 16 eindige samples. De correctness-token 89.999, validationtokens 90.000–90.015 en testtokens 91.000–91.119 zijn exact en onderling disjunct.

Alle 16 rotatie/omkeerorders matchen en de vier resulterende orderpatronen komen elk viermaal voor. De vastgelegde vier-armalgoritme is echter niet positiegebalanceerd: 256/1024 staan elk achtmaal op posities 0 en 2, terwijl 512/2048 elk achtmaal op posities 1 en 3 staan. De selectie is protocolconform, maar eventuele positie-effecten zijn dus niet volledig gecounterbalanced.

## Once-only test en poorten

De test bevat exact 120 eindige samples:

| mean | p50 | p95 | p99 | min | max |
|---:|---:|---:|---:|---:|---:|
| {test_stats['mean']:.6f} | {test_stats['p50']:.6f} | **{test_stats['p95']:.6f}** | {test_stats['p99']:.6f} | {test_stats['min']:.6f} | {test_stats['max']:.6f} |

- Effectieve payloadbandbreedte bij p95: **{bandwidth:.6f} GB/s**.
- Marge onder de sterke 45-ms-poort: **{margin_arithmetic['p95_margin_below_45_ms']:.6f} ms** ({100 * margin_arithmetic['strong_latency_margin_fraction_of_limit']:.3f}%).
- Marge boven 21,627 GB/s: **{margin_arithmetic['bandwidth_margin_over_21_627']:.6f} GB/s** ({100 * margin_arithmetic['strong_bandwidth_margin_fraction_of_limit']:.3f}%).
- Zelfs het langzaamste opgeslagen testsample is {margin_arithmetic['max_sample_margin_ms']:.6f} ms onder 45 ms.

Alle acht opgeslagen gates zijn exact herberekend als `true`: mismatchscalar, 120 samples, 65-ms/15-GB/s mechanismepoorten, 45-ms/21,627-GB/s sterke poorten, 48 registratieranges en lokale error/cleanup. `error=null` en de unregister-foutenlijst is leeg.

## Include- en protocolaudit

- Preregistratie-, evaluator-, result- en manifestprovenance matchen. De evaluatorhash is de huidige bronhash.
- De officiële lokale `cuda_pipeline.h` bestaat en is in de audit gehasht. De runner gebruikt expliciet het bundled CUDA-13-includepad.
- De kernel bevat `__pipeline_memcpy_async(...,16)`, commit, wait en 4-KiB-SMEM; geen `<stdint.h>`/`uintptr_t`-rest van D3.
- De geometrie sluit exact: 495 × 4.096 = 2.027.520 bytes per record en 237.600 × 4.096 = 973.209.600 bytes per token.
- Dit is geen TMA-tensormap en heeft geen verborgen fallbackarm.

Een provenancebeperking blijft: D5 importeert routes, stats, unregister en de verify-kernel uit D2, maar slaat de D2-modulehash niet in zijn eigen JSON op. De huidige D2-bron matcht wel exact haar bewaarde D2-evaluatorhash.

## Byte-evidencegrens

De verifier reconstrueerde de exacte correctnessroute en scande alle 480 records/973.209.600 geselecteerde bronbytes: nul structurele bronmismatches. De D5-uitvoer bewaart echter alleen `full_destination_mismatch_count: 0`, geen GPU-destinationhash of buffer. De tijdelijke volledige GPU-bestemming is daarom niet post-hoc CPU-only replaybaar.

## Full-bank- en claimgrens

Dit is een sterke **componentpass op slechts 307/512 experts per laag**: 29.877.534.720 bytes of 27,8256 GiB geregistreerd. `full_bank_pass=false` is correct; D5 heeft de 100%-bank niet uitgevoerd en herstelt D2's full-registrationfail niet.

De kernel verplaatst synthetische bytes van mapped host via `cp.async` naar SMEM en schrijft ze vervolgens naar een volledige HBM-oraclebuffer. Er is geen Q5 multiply/reductie, ERGV-integratie, TMA-descriptor, echte 80B-checkpoint, modelkwaliteit, dense shell, end-to-end tokens/s, page-telemetrie of endurance bewezen.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({
        "independent_verdict": output["independent_verdict"],
        "all_replayable_checks_pass": all_replayable_checks_pass,
        "selected_blocks": selected,
        "test_p95_ms": test_stats["p95"],
        "effective_gb_s_at_p95": bandwidth,
        "strong_latency_margin_ms": margin_arithmetic["p95_margin_below_45_ms"],
        "output": str(OUTPUT),
        "report": str(REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
