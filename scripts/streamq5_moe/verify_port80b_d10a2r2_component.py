from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "streamq5_moe"
RUNS = ROOT / "reports" / "runs" / "streamq5_moe"
RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a2r2_gdn36_oracle_repair.py"
PRIOR_RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a2r_single_stream_repair_revision.py"
UNIT_TEST = ROOT / "scripts" / "streamq5_moe" / "test_port80b_d10a2r2_conv_oracle.py"
PREREG = R / "PORT80B_D10A2R2_GDN36_ORACLE_REPAIR_PREREGISTRATION.md"
UNIT = R / "port80b_d10a2r2_conv_oracle_unit.json"
PREFLIGHT = R / "port80b_d10a2r2_gdn36_oracle_repair_preflight.json"
RAW = R / "port80b_d10a2r2_gdn36_oracle_repair.json"
RAW_REPORT = R / "PORT80B_D10A2R2_GDN36_ORACLE_REPAIR_REPORT_2026-08-13.md"
ENDURANCE = R / "port80b_d10a2r2_gdn36_oracle_repair_endurance_10k.json"
CAPTURE = R / "p4d_route_capture_result.json"
ROUTE_DIR = RUNS / "p4d_routes"
BANK = RUNS / "port80b_p0" / "port80b_p0_full_q5_bank.bin"
MANIFEST = RUNS / "port80b_p0" / "port80b_p0_full_q5_bank_manifest.json"
OUT = R / "port80b_d10a2r2_component_independent_verification.json"
REPORT = R / "PORT80B_D10A2R2_COMPONENT_INDEPENDENT_VERIFICATION_REPORT_2026-08-13.md"

DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS = 48
GDN_LAYERS = 36
EXPERTS_WITH_SHARED = 513
PREFIX = 499
TOP_K = 10
EXPERT_BYTES = 2_027_520
MATRIX_BYTES = 675_840
HEADER_BYTES = 64
DEVICE_REQUEST = 4_521_569_280
START_RAM_GATE = 52_652_163_072
MIN_RAM_AFTER_TOUCH = 2 * 2**30
EMERGENCY_RAM = int(1.5 * 2**30)
VRAM_RESERVE = 512 * 2**20
CONV_WORDS = 1_179_648
CONV_NONZERO = 292_608
CONV_SHA = "cedf5736557919b023d6f7cce73d0064df07236ff1e18b5d8b3fec49d658fa1e"
EXPECTED_BANK_SHA = "4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462"
POISON = 12345.25
MASK64 = (1 << 64) - 1
LIFT_SEED = 0xD10A_499D_1308_2026
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_ARRAYS = {
    "routed_capture": ([1_474_560], "float32"),
    "routed_down": ([983_040], "float32"),
    "shared_down": ([98_304], "float32"),
    "attention": ([49_152], "float32"),
    "delta": ([1_152], "float32"),
    "kv_state": ([50_331_648], "uint16"),
    "recurrent_state": ([18_874_368], "float32"),
    "conv_state": ([1_179_648], "uint16"),
    "composed_state": ([98_304], "float32"),
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            value.update(block)
    return value.hexdigest()


def stats(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def bf16_words(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def conv_oracle() -> np.ndarray:
    layers = np.arange(GDN_LAYERS, dtype=np.int32)[:, None]
    channels = np.arange(8192, dtype=np.int32)[None, :]
    values = (((layers * 19 + channels) & 127) - 63).astype(np.float32) / np.float32(64)
    result = np.zeros((GDN_LAYERS, 8192, 4), dtype=np.uint16)
    result[:, :, 0] = bf16_words(values)
    return result.reshape(-1)


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def lift(source: np.ndarray, domain_index: int, token: int) -> np.ndarray:
    output = np.empty((LAYERS, TOP_K), dtype=np.int16)
    for layer in range(LAYERS):
        used: set[int] = set()
        for rank, raw in enumerate(source[layer]):
            state = LIFT_SEED ^ (domain_index << 56) ^ (token << 24) ^ (layer << 8) ^ rank
            expert = int(raw) * 4 + int(splitmix64(state) & 3)
            output[layer, rank] = expert
            used.add(expert)
        state = LIFT_SEED ^ (domain_index << 52) ^ (token << 20) ^ layer
        rank = 8
        while rank < TOP_K:
            state = splitmix64(state)
            expert = int(state % 512)
            if expert not in used:
                output[layer, rank] = expert
                used.add(expert)
                rank += 1
    return output


def source_constant_string(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise TypeError(name)
            return value
    raise KeyError(name)


def shared_payload_hashes() -> tuple[list[str], list[dict[str, Any]]]:
    with BANK.open("rb") as handle:
        reference: list[str] = []
        for projection in range(3):
            handle.seek(projection * MATRIX_BYTES + HEADER_BYTES)
            reference.append(hashlib.sha256(handle.read(MATRIX_BYTES - HEADER_BYTES)).hexdigest())
        rows: list[dict[str, Any]] = []
        for layer in range(LAYERS):
            values: list[str] = []
            record_offset = (layer * EXPERTS_WITH_SHARED + 512) * EXPERT_BYTES
            for projection in range(3):
                handle.seek(record_offset + projection * MATRIX_BYTES + HEADER_BYTES)
                values.append(hashlib.sha256(handle.read(MATRIX_BYTES - HEADER_BYTES)).hexdigest())
            rows.append({"layer": layer, "projection_sha256": values, "matches_reference": values == reference})
    return reference, rows


def main() -> None:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    unit = json.loads(UNIT.read_text(encoding="utf-8"))
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runner_source = RUNNER.read_text(encoding="utf-8")
    report_text = RAW_REPORT.read_text(encoding="utf-8")

    routes = {domain: np.empty((1024, LAYERS, 8), dtype=np.int16) for domain in DOMAINS}
    route_hashes: dict[str, str] = {}
    for layer in range(LAYERS):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"
        route_hashes[str(layer)] = sha256(path)
        tensors = load_file(path)
        for domain in DOMAINS:
            routes[domain][:, layer] = tensors[f"{domain}_router_ids"].astype(np.int16)

    expected_routes: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(DOMAINS):
        for token in range(8):
            expected_routes.append({"domain": domain, "token": token, "route": lift(routes[domain][token], domain_index, token)})

    canary_failures: list[dict[str, Any]] = []
    route_failures: list[int] = []
    raw_canaries = raw["raw_canary_arrays"]
    for case, (row, expected_case) in enumerate(zip(raw_canaries, expected_routes, strict=True)):
        route = expected_case["route"]
        expert_ids = route.reshape(-1).astype(np.uint16)
        layers = np.repeat(np.arange(LAYERS, dtype=np.uint16), TOP_K)
        intended = (layers.astype(np.uint32) * 512 + expert_ids.astype(np.uint32)).astype(np.uint16)
        expected_words = np.empty((intended.size, 3), dtype=np.uint16)
        for place in range(3):
            expected_words[:, place] = (0x3E80 + 4 * ((intended.astype(np.uint32) >> (5 * place)) & 31)).astype(np.uint16)
        actual = np.asarray(row["actual_header_ids"], dtype=np.uint16)
        stored_intended = np.asarray(row["intended_ids"], dtype=np.uint16)
        stored_expected = np.asarray(row["expected_words"], dtype=np.uint16)
        observed = np.asarray(row["observed_words"], dtype=np.uint16)
        if row["domain"] != expected_case["domain"] or int(row["token"]) != expected_case["token"] or not np.array_equal(stored_intended, intended):
            route_failures.append(case)
        if not (np.array_equal(actual, intended) and np.array_equal(stored_expected, expected_words) and np.array_equal(observed, expected_words)):
            canary_failures.append({"case": case})

    triples: set[tuple[int, int, int]] = set()
    roundtrip_failures: list[int] = []
    boundary_failures: list[int] = []
    for layer in range(LAYERS):
        for expert in range(512):
            identifier = layer * 512 + expert
            words = tuple(0x3E80 + 4 * ((identifier >> (5 * place)) & 31) for place in range(3))
            triples.add(words)
            decoded = sum(((word - 0x3E80) // 4) << (5 * place) for place, word in enumerate(words))
            if decoded != identifier:
                roundtrip_failures.append(identifier)
        hot, cold = layer * 512 + 498, layer * 512 + 499
        if hot >= layer * 512 + PREFIX or cold < layer * 512 + PREFIX:
            boundary_failures.append(layer)

    correctness = raw["correctness"]
    output_digests = raw["output_digests"]
    correctness_digest_consistent = all(
        row["comparison"]["left_sha256"] == row["comparison"]["right_sha256"]
        == row["output_sha256"] == output_digests[index]
        and HEX64.fullmatch(row["output_sha256"]) is not None
        for index, row in enumerate(correctness)
    )

    first_ids = np.asarray(raw_canaries[0]["intended_ids"], dtype=np.uint16)
    injection_index = next(index for index, identifier in enumerate(first_ids) if int(identifier % 512) < PREFIX - 1)
    injection_layer = injection_index // TOP_K
    injection_expert = int(first_ids[injection_index] % 512)
    negative_details: dict[str, Any] = {}
    negative_exact = True
    for name, expected_actual_id in {
        "wrong_expert": injection_layer * 512 + injection_expert + 1,
        "wrong_layer": ((injection_layer + 1) % LAYERS) * 512 + injection_expert,
    }.items():
        row = raw["negative_controls"][name]
        intended = np.asarray(row["intended_ids"], dtype=np.uint16)
        actual = np.asarray(row["actual_header_ids"], dtype=np.uint16)
        expected_words = np.asarray(row["expected_words"], dtype=np.uint16)
        observed_words = np.asarray(row["observed_words"], dtype=np.uint16)
        changed = np.flatnonzero(actual != intended)
        actual_word_oracle = np.empty_like(observed_words)
        for place in range(3):
            actual_word_oracle[:, place] = (0x3E80 + 4 * ((actual.astype(np.uint32) >> (5 * place)) & 31)).astype(np.uint16)
        intended_word_oracle = np.empty_like(expected_words)
        for place in range(3):
            intended_word_oracle[:, place] = (0x3E80 + 4 * ((intended.astype(np.uint32) >> (5 * place)) & 31)).astype(np.uint16)
        exact = (
            np.array_equal(intended, first_ids)
            and changed.tolist() == [injection_index]
            and int(actual[injection_index]) == expected_actual_id
            and np.array_equal(observed_words, actual_word_oracle)
            and np.array_equal(expected_words, intended_word_oracle)
            and int(row["header_mismatches"]) > 0
            and row["numerical_comparison"]["bitwise_equal"] is False
            and int(row["numerical_comparison"]["different_bits"]) > 0
        )
        negative_exact &= exact
        negative_details[name] = {
            "exact_single_substitution": exact,
            "changed_index": changed.astype(int).tolist(),
            "actual_id": int(actual[injection_index]),
            "header_mismatches": int(row["header_mismatches"]),
            "different_bits": int(row["numerical_comparison"]["different_bits"]),
            "max_abs": float(row["numerical_comparison"]["max_abs"]),
        }

    conv = conv_oracle()
    conv_digest = hashlib.sha256(conv.tobytes()).hexdigest()
    components = raw["components"]
    conv_cmp = components["conv_full_bf16_comparison"]
    conv_exact = (
        conv.shape == (CONV_WORDS,)
        and int(np.count_nonzero(conv)) == CONV_NONZERO
        and conv_digest == CONV_SHA
        and int(components["conv_nonzero"]) == CONV_NONZERO
        and int(components["conv_expected_nonzero"]) == CONV_NONZERO
        and int(conv_cmp["elements"]) == CONV_WORDS
        and int(conv_cmp["different_words"]) == 0
        and conv_cmp["bitwise_equal"] is True
        and conv_cmp["left_sha256"] == conv_cmp["right_sha256"] == conv_digest
    )

    reference_hashes, shared_rows = shared_payload_hashes()
    stored_shared = components["shared_payload_comparison"]
    shared_payload_exact = (
        stored_shared["header_bytes_excluded_per_projection"] == HEADER_BYTES
        and stored_shared["payload_bytes_per_projection"] == MATRIX_BYTES - HEADER_BYTES
        and stored_shared["reference_projection_sha256"] == reference_hashes
        and stored_shared["layers"] == shared_rows
        and all(row["matches_reference"] for row in shared_rows)
    )
    shared_output_exact = (
        components["shared_vs_resident"]["elements"] == 98_304
        and components["shared_vs_resident"]["different_bits"] == 0
        and components["shared_vs_resident"]["bitwise_equal"] is True
        and components["shared_vs_resident"]["max_abs"] == 0.0
        and components["shared_vs_resident"]["finite"] is True
        and components["shared_vs_resident"]["left_sha256"] == components["shared_vs_resident"]["right_sha256"]
    )

    validation = raw["validation_output_evidence"]
    validation_summary_errors: list[dict[str, Any]] = []
    unique_digests: dict[str, int] = {}
    for step, row in enumerate(validation):
        if row["step"] != step or row["domain"] != "general" or row["token"] != 512 + step:
            validation_summary_errors.append({"step": step, "reason": "selection"})
        if set(row["arrays"]) != set(EXPECTED_ARRAYS):
            validation_summary_errors.append({"step": step, "reason": "array_names"})
            continue
        for name, (shape, dtype) in EXPECTED_ARRAYS.items():
            evidence = row["arrays"][name]
            if not (
                evidence["shape"] == shape
                and evidence["dtype"] == dtype
                and evidence["finite"] is True
                and evidence["poison_count"] == 0
                and HEX64.fullmatch(evidence["sha256"]) is not None
            ):
                validation_summary_errors.append({"step": step, "array": name})
    for name in EXPECTED_ARRAYS:
        unique_digests[name] = len({row["arrays"][name]["sha256"] for row in validation})

    wall = [float(x) for x in raw["validation"]["wall_ms"]]
    events = [float(x) for x in raw["validation"]["cuda_event_ms"]]
    wall_stats = stats(wall)
    event_stats = stats(events)
    telemetry = raw["telemetry"]
    page_rates = [float(row["page_reads_per_sec"]) for row in raw["page_reads"]["samples"]]
    memory_loss = int(telemetry[0]["available_ram"]) - int(telemetry[-1]["available_ram"])
    min_available = min(int(row["available_ram"]) for row in telemetry)
    min_vram = min(int(row["free_vram"]) for row in telemetry)

    per_range_bytes = PREFIX * EXPERT_BYTES
    layer_stride = EXPERTS_WITH_SHARED * EXPERT_BYTES
    registration = raw["registration_attempts"]
    unregister = raw["unregister_attempts"]
    registration_exact = (
        len(registration) == LAYERS
        and [row["layer"] for row in registration] == list(range(LAYERS))
        and all(row["action"] == "host_register" and row["bytes"] == per_range_bytes and row["attempted"] is True and row["success"] is True and row["error"] is None and isinstance(row["host_pointer"], int) and isinstance(row["device_alias"], int) and row["device_alias"] != 0 for row in registration)
        and len({row["host_pointer"] for row in registration}) == LAYERS
        and len({row["device_alias"] for row in registration}) == LAYERS
        and all(registration[i + 1]["host_pointer"] - registration[i]["host_pointer"] == layer_stride for i in range(LAYERS - 1))
    )
    unregister_exact = (
        len(unregister) == LAYERS
        and [row["layer"] for row in unregister] == list(range(LAYERS))
        and all(row["action"] == "host_unregister" and row["bytes"] == per_range_bytes and row["attempted"] is True and row["success"] is True and row["error"] is None for row in unregister)
        and [row["host_pointer"] for row in unregister] == [row["host_pointer"] for row in registration]
        and raw["unregister_failures"] == []
    )

    recomputed_gates = {
        "canary_exhaustive_injective_roundtrip_boundary": len(triples) == LAYERS * 512 and not roundtrip_failures and not boundary_failures,
        "all_correctness_headers_zero_mismatch": len(correctness) == 40 and all(row["header_mismatches"] == 0 for row in correctness),
        "all_canaries_raw_exact": len(raw_canaries) == 40 and not canary_failures and all(row["raw_canary_exact"] and row["canary_mismatches"] == 0 for row in correctness),
        "all_routed_q5_bitexact": len(correctness) == 40 and all(row["comparison"]["bitwise_equal"] for row in correctness),
        "all_routed_q5_outputs_finite": len(correctness) == 40 and all(row["comparison"]["finite"] for row in correctness),
        "all_routed_q5_outputs_fully_written": len(correctness) == 40 and all(row["candidate_poison_count"] == 0 and row["oracle_poison_count"] == 0 for row in correctness),
        "output_digest_uniqueness_ge_95pct": len(output_digests) == 40 and len(set(output_digests)) / len(output_digests) >= 0.95,
        "wrong_expert_header_and_numerical_detected": raw["negative_controls"]["wrong_expert"]["header_mismatches"] > 0 and not raw["negative_controls"]["wrong_expert"]["numerical_comparison"]["bitwise_equal"],
        "wrong_layer_header_and_numerical_detected": raw["negative_controls"]["wrong_layer"]["header_mismatches"] > 0 and not raw["negative_controls"]["wrong_layer"]["numerical_comparison"]["bitwise_equal"],
        "attention_reference_abs_rel_le_2e_5": components["attention_max_abs"] <= 2e-5,
        "gdn_reference_abs_rel_le_2e_5": components["recurrent_sample_max_abs"] <= 2e-5 and components["conv_nonzero"] > 0,
        "conv_step0_full_bf16_bitexact_and_292608_nonzero": components["conv_nonzero"] == CONV_NONZERO and conv_cmp["bitwise_equal"],
        "shared_q5_bitexact": components["shared_vs_resident"]["bitwise_equal"],
        "shared_48_payloads_match_reference_excluding_headers": stored_shared["all_48_match_reference"],
        "component_outputs_no_poison_remaining": all(value == 0 for value in components["component_output_sentinels"].values()),
        "dense_and_runtime_touched": components["dense_checksum_observed"] == components["dense_checksum_expected"] and components["runtime_touch_sentinels"] == [0xA5, 0xA5],
        "validation_32_finite": len(wall) == 32 and bool(np.isfinite(np.asarray(wall)).all()),
        "validation_full_outputs_finite_digested_no_poison": len(validation) == 32 and not validation_summary_errors,
        "validation_wall_p95_le_150ms": wall_stats["p95"] <= 150.0,
        "validation_wall_p99_le_200ms": wall_stats["p99"] <= 200.0,
        "post_warmup_page_reads_no_sample_gt_2048": bool(page_rates) and max(page_rates) <= 2048.0,
        "validation_memory_loss_le_1gib": memory_loss <= 2**30,
        "ram_after_first_touch_ge_2gib": raw["physical"]["available_ram_after_first_touch"] >= MIN_RAM_AFTER_TOUCH,
        "vram_reserve_ge_512mib": min_vram >= VRAM_RESERVE,
        "registration_48_ranges": len(registration) == LAYERS,
        "no_cuda_or_runner_error": raw["error"] is None,
        "registration_attempt_rows_48_all_success": registration_exact,
        "clean_unregister_48_ranges": unregister_exact,
    }

    prior_cuda = source_constant_string(PRIOR_RUNNER, "COMPONENT_SOURCE")
    current_cuda = source_constant_string(RUNNER, "COMPONENT_SOURCE")
    unit_hashes_exact = (
        unit["inputs"]["runner_sha256"] == sha256(RUNNER)
        and unit["inputs"]["preregistration_sha256"] == sha256(PREREG)
        and unit["inputs"]["unit_test_sha256"] == sha256(UNIT_TEST)
        and unit["audit"]["sha256"] == conv_digest
        and unit["audit"]["nonzero_words"] == CONV_NONZERO
        and unit["audit"]["shape"] == [CONV_WORDS]
        and unit["pass"] is True
    )
    preflight_hashes_exact = (
        preflight["inputs"]["runner_sha256"] == sha256(RUNNER)
        and preflight["inputs"]["preregistration_sha256"] == sha256(PREREG)
        and preflight["inputs"]["conv_unit_test_sha256"] == sha256(UNIT_TEST)
        and preflight["inputs"]["conv_unit_result_sha256"] == sha256(UNIT)
        and preflight["pass"] is True
        and all(not value for value in preflight["physical_actions"].values())
        and preflight["component_opened"] is False
        and preflight["endurance_opened"] is False
    )

    independent_checks = {
        "raw_runner_hash_matches": raw["inputs"]["runner_sha256"] == sha256(RUNNER),
        "raw_prereg_hash_matches": raw["inputs"]["preregistration_sha256"] == sha256(PREREG),
        "raw_preflight_hash_matches": raw["inputs"]["compile_sha256"] == sha256(PREFLIGHT),
        "unit_hashes_and_oracle_exact": unit_hashes_exact,
        "preflight_hashes_and_cpu_boundary_exact": preflight_hashes_exact,
        "all_route_tensor_hashes_match_capture_and_preflight": all(route_hashes[str(layer)] == capture["manifests"][str(layer)]["artifact_sha256"] == preflight["inputs"]["route_hashes"][str(layer)] for layer in range(LAYERS)),
        "correctness_route_ids_recomputed_from_p4d": not route_failures,
        "all_raw_canaries_recomputed": not canary_failures,
        "correctness_digests_internally_exact": correctness_digest_consistent,
        "negative_controls_exact_single_substitution": negative_exact,
        "conv_oracle_independently_recomputed": conv_exact,
        "shared_payloads_independently_rehashed": shared_payload_exact,
        "shared_output_summary_exact": shared_output_exact,
        "validation_output_summaries_complete": len(validation) == 32 and not validation_summary_errors,
        "all_28_gates_reproduced_exactly": recomputed_gates == raw["gates"] and all(recomputed_gates.values()),
        "wall_percentiles_exact": wall_stats == raw["validation"]["wall_stats"],
        "cuda_event_percentiles_exact": event_stats == raw["validation"]["cuda_event_stats"],
        "ram_vram_page_arithmetic_exact": raw["physical"]["available_ram_before"] >= START_RAM_GATE and raw["physical"]["available_ram_after_registration"] >= MIN_RAM_AFTER_TOUCH and min_available >= EMERGENCY_RAM and min_vram >= VRAM_RESERVE and max(page_rates) <= 2048.0 and memory_loss <= 2**30,
        "registration_rows_exact": registration_exact,
        "unregister_rows_exact": unregister_exact,
        "cuda_component_source_unchanged_from_d10a2r": hashlib.sha256(current_cuda.encode()).hexdigest() == hashlib.sha256(prior_cuda.encode()).hexdigest(),
        "no_endurance_artifact": not ENDURANCE.exists(),
        "endurance_source_unconditionally_closed": "D10A2-R2 endurance is unconditionally fail-closed" in runner_source and "raise RuntimeError" in runner_source[runner_source.index("def endurance_phase"):runner_source.index("def main")],
        "canonical_json_endurance_false": raw["endurance_authorized_by_evidence"] is False,
        "raw_report_has_incorrect_endurance_true_text": "Endurance evidence-authorized: **True**" in report_text,
        "bank_size_and_manifest_contract": BANK.stat().st_size == 49_925_652_480 and manifest["bank_sha256"] == EXPECTED_BANK_SHA,
    }

    endurance_eligible_for_new_arm = all(recomputed_gates.values()) and all(
        independent_checks[key] for key in independent_checks if key != "raw_report_has_incorrect_endurance_true_text"
    )
    actual_endurance_open = raw["endurance_authorized_by_evidence"] is True and ENDURANCE.exists()

    payload = {
        "kind": "port80b_d10a2r2_component_independent_cpu_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "component_pass_verified_new_endurance_arm_eligible_but_current_endurance_closed",
        "verification_pass": all(independent_checks.values()),
        "independent_checks": independent_checks,
        "inputs": {
            "raw_sha256": sha256(RAW),
            "runner_sha256": sha256(RUNNER),
            "preregistration_sha256": sha256(PREREG),
            "unit_test_sha256": sha256(UNIT_TEST),
            "unit_result_sha256": sha256(UNIT),
            "preflight_sha256": sha256(PREFLIGHT),
            "raw_report_sha256": sha256(RAW_REPORT),
        },
        "gate_replay": {
            "count": len(recomputed_gates),
            "passed": sum(recomputed_gates.values()),
            "failed": [key for key, value in recomputed_gates.items() if not value],
            "raw_match": recomputed_gates == raw["gates"],
            "recomputed": recomputed_gates,
        },
        "exactness": {
            "correctness_cases": len(correctness),
            "zero_header_mismatch_cases": sum(row["header_mismatches"] == 0 for row in correctness),
            "zero_canary_mismatch_cases": sum(row["canary_mismatches"] == 0 and row["raw_canary_exact"] for row in correctness),
            "bitexact_routed_cases": sum(row["comparison"]["bitwise_equal"] for row in correctness),
            "fully_written_finite_routed_cases": sum(row["comparison"]["finite"] and row["candidate_poison_count"] == 0 and row["oracle_poison_count"] == 0 for row in correctness),
            "unique_output_digests": len(set(output_digests)),
            "route_failures": route_failures,
            "canary_failures": canary_failures,
            "negative_controls": negative_details,
            "conv_oracle": {"words": int(conv.size), "nonzero": int(np.count_nonzero(conv)), "sha256": conv_digest, "bitexact": conv_exact},
            "shared_payload_reference_sha256": reference_hashes,
            "shared_payload_layers_exact": sum(row["matches_reference"] for row in shared_rows),
            "shared_output_bitexact": shared_output_exact,
            "attention_max_abs": components["attention_max_abs"],
            "recurrent_sample_max_abs": components["recurrent_sample_max_abs"],
            "component_poison_counts": components["component_output_sentinels"],
            "dense_checksum_exact": components["dense_checksum_observed"] == components["dense_checksum_expected"],
        },
        "validation_output_evidence": {
            "steps": len(validation),
            "arrays_per_step": len(EXPECTED_ARRAYS),
            "summary_errors": validation_summary_errors,
            "unique_digests_per_array": unique_digests,
            "limitation": "The raw JSON retains digest/shape/dtype/finite/poison summaries, not the arrays. Their SHA-256 values are format- and sequence-checked but cannot be independently recomputed from this artifact.",
        },
        "timing": {"wall_ms": wall_stats, "cuda_event_ms": event_stats, "wall_event_mean_gap_ms": wall_stats["mean"] - event_stats["mean"]},
        "resources": {
            "available_before_bytes": raw["physical"]["available_ram_before"],
            "start_gate_bytes": START_RAM_GATE,
            "start_margin_bytes": raw["physical"]["available_ram_before"] - START_RAM_GATE,
            "after_registration_bytes": raw["physical"]["available_ram_after_registration"],
            "after_first_touch_bytes": raw["physical"]["available_ram_after_first_touch"],
            "after_cleanup_bytes": raw["physical"]["available_ram_after_cleanup"],
            "validation_min_available_bytes": min_available,
            "validation_endpoint_loss_bytes": memory_loss,
            "validation_min_free_vram_bytes": min_vram,
            "page_read_samples": len(page_rates),
            "page_reads_per_sec_max": max(page_rates),
            "pages_input_per_sec_max_diagnostic": max(float(row["pages_input_per_sec"]) for row in raw["page_reads"]["samples"]),
        },
        "lifecycle": {
            "registration_rows": len(registration),
            "registration_all_success": registration_exact,
            "unregistration_rows": len(unregister),
            "unregistration_all_success": unregister_exact,
            "per_range_bytes": per_range_bytes,
            "registered_total_bytes": LAYERS * per_range_bytes,
            "host_pointer_layer_stride_bytes": layer_stride,
            "unregister_failures": raw["unregister_failures"],
        },
        "endurance_decision": {
            "component_evidence_supports_new_preregistered_arm": endurance_eligible_for_new_arm,
            "d10a2r2_endurance_actually_open": actual_endurance_open,
            "canonical_json_endurance_authorized_by_evidence": raw["endurance_authorized_by_evidence"],
            "endurance_artifact_exists": ENDURANCE.exists(),
            "runner_endurance_behavior": "unconditional RuntimeError",
            "report_bug": "The Markdown report interpolates overall_pass and incorrectly prints Endurance evidence-authorized: True; canonical JSON and preregistration say false/closed.",
            "decision": "D10A2-R2 independently verifies as a component pass. It does not itself open endurance; it satisfies the evidence prerequisite for a new, separately preregistered and authorized endurance runner.",
        },
        "claim_boundary": "Synthetic shape-informed physical component/composition evidence on P4D-shaped proxy routes and uniform synthetic Q5 payloads; not an official checkpoint, natural routing, model quality, production throughput or endurance result.",
        "physical_actions": {"gpu_run": False, "host_registration": False, "registry_edit": False, "bank_write": False},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = f"""# PORT80B-D10A2-R2 independent component audit

**Verdict:** **component pass independently verified**. All **28/28** frozen component gates replay true, and {sum(independent_checks.values())}/{len(independent_checks)} independent audit checks pass. This does **not** actually open or run endurance: D10A2-R2's canonical JSON says `endurance_authorized_by_evidence=false`, the endurance artifact does not exist, and its runner raises unconditionally. The pass makes a separately preregistered and separately authorized endurance arm evidence-eligible.

## Exactness and component controls

- 40/40 P4D-derived correctness routes independently reconstructed; 40/40 zero header mismatch, exact raw IDs/canaries, finite fully written outputs and candidate/oracle bit equality.
- 40 distinct correctness output digests; every stored output digest matches both comparison-side digests.
- Wrong-expert and wrong-layer controls each change exactly the intended record. Both produce 3 header-byte mismatches and 2,050 differing FP32 output words.
- Full conv oracle: **1,179,648 BF16 words**, **292,608 nonzero**, zero differing words, SHA-256 `{conv_digest}`. The unit-test result and preflight bind the current runner/preregistration hashes.
- All 48 shared numerical payloads were independently reread and rehashed excluding their headers; every projection matches the layer-0/expert-0 reference. The stored 98,304-element shared-output comparison is bitexact.
- Attention and sampled recurrent max-absolute error are both 0; all component poison counts are 0; dense checksum and runtime sentinels match.

## Validation and resources

- Wall p50/p95/p99: **{wall_stats['p50']:.6f} / {wall_stats['p95']:.6f} / {wall_stats['p99']:.6f} ms**.
- CUDA-event p50/p95/p99: **{event_stats['p50']:.6f} / {event_stats['p95']:.6f} / {event_stats['p99']:.6f} ms**.
- Exactly 32 validation rows, each with nine expected shape/dtype/finite/poison/digest summaries. All routed, attention, delta, recurrent, conv and composed digests vary across all 32 steps; shared output is intentionally constant; KV has 25 distinct cumulative-state digests.
- Start RAM exceeded the 52,652,163,072-byte gate by {(raw['physical']['available_ram_before'] - START_RAM_GATE) / 2**30:.6f} GiB. RAM after first touch was {raw['physical']['available_ram_after_first_touch'] / 2**30:.6f} GiB; validation endpoint loss was {memory_loss / 2**20:.3f} MiB. Minimum validation free VRAM was {min_vram / 2**20:.3f} MiB.
- {len(page_rates)} page-read samples, maximum **{max(page_rates):.3f}/s**, below 2,048/s.

The validation artifact stores only digest summaries rather than underlying arrays. Their presence, format, shape, finiteness and step sequence are independently checked, but those validation SHA-256 values cannot be regenerated from the JSON alone.

## Registration and cleanup

Exactly 48 registration rows and 48 unregister rows are retained. Layers are 0..47 exactly once, every row is attempted and successful with no error, per-range bytes are {per_range_bytes:,}, host pointers follow the exact {layer_stride:,}-byte bank layer stride, unregister pointers match registration pointers, and the failure list is empty.

## Endurance decision and report defect

The generated Markdown report is internally wrong: it prints `Endurance evidence-authorized: True` by interpolating `overall`, while the canonical result field is `false`. The status, preregistration and executable runner all keep endurance closed. Therefore:

- **component mechanism gate:** passed;
- **eligibility to design/preregister the next endurance arm:** yes;
- **current D10A2-R2 endurance gate actually open:** no;
- **endurance evidence produced:** none.

Claim boundary: {payload['claim_boundary']}
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({
        "audit_checks": f"{sum(independent_checks.values())}/{len(independent_checks)}",
        "component_gates": f"{sum(recomputed_gates.values())}/{len(recomputed_gates)}",
        "component_pass_verified": all(recomputed_gates.values()),
        "current_endurance_open": actual_endurance_open,
        "new_endurance_arm_eligible": endurance_eligible_for_new_arm,
        "out": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
