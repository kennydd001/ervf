from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "streamq5_moe"
RUNS = ROOT / "reports" / "runs" / "streamq5_moe"

RAW = R / "port80b_d10br_heldout_10000_endurance_revision.json"
PREREG = R / "PORT80B_D10BR_HELDOUT_10000_ENDURANCE_REVISION_PREREGISTRATION.md"
PREFLIGHT = R / "port80b_d10br_heldout_10000_endurance_revision_preflight.json"
RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10br_heldout_10000_endurance_revision.py"
COMPONENT = R / "port80b_d10a2r2_gdn36_oracle_repair.json"
MANIFEST = RUNS / "port80b_p0" / "port80b_p0_full_q5_bank_manifest.json"
BANK = RUNS / "port80b_p0" / "port80b_p0_full_q5_bank.bin"
CAPTURE = R / "p4d_route_capture_result.json"
ROUTE_DIR = RUNS / "p4d_routes"

D10A1R_FILES = {
    "d10a1r": R / "port80b_d10a1r_conservative_resource_retry.json",
    "d10a1r_runner": ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a1r_conservative_resource_retry.py",
    "d10a1r_prereg": R / "PORT80B_D10A1R_CONSERVATIVE_RESOURCE_RETRY_PREREGISTRATION.md",
    "counter_audit": R / "D10A1R_FAILURE_COUNTER_AUDIT_2026-08-13.md",
}
D10A2R_FILES = {
    "result": R / "port80b_d10a2r_single_stream_repair_revision.json",
    "report": R / "PORT80B_D10A2R_SINGLE_STREAM_REPAIR_REVISION_REPORT_2026-08-13.md",
    "runner": ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a2r_single_stream_repair_revision.py",
    "prereg": R / "PORT80B_D10A2R_SINGLE_STREAM_REPAIR_REVISION_PREREGISTRATION.md",
}
D10A2R2_FILES = {
    "runner": ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a2r2_gdn36_oracle_repair.py",
    "prereg": R / "PORT80B_D10A2R2_GDN36_ORACLE_REPAIR_PREREGISTRATION.md",
    "unit_test": ROOT / "scripts" / "streamq5_moe" / "test_port80b_d10a2r2_conv_oracle.py",
    "unit": R / "port80b_d10a2r2_conv_oracle_unit.json",
    "preflight": R / "port80b_d10a2r2_gdn36_oracle_repair_preflight.json",
    "preflight_report": R / "PORT80B_D10A2R2_GDN36_ORACLE_REPAIR_PREFLIGHT_REPORT_2026-08-13.md",
    "result": COMPONENT,
    "report": R / "PORT80B_D10A2R2_GDN36_ORACLE_REPAIR_REPORT_2026-08-13.md",
}
INDEPENDENT_AUDIT = R / "port80b_d10a2r2_component_independent_verification.json"
INDEPENDENT_AUDIT_REPORT = R / "PORT80B_D10A2R2_COMPONENT_INDEPENDENT_VERIFICATION_REPORT_2026-08-13.md"
D10A2R2_ERRATUM = R / "PORT80B_D10A2R2_REPORT_ERRATUM_2026-08-13.md"

FAILED_RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10b_heldout_10000_endurance.py"
FAILED_PREREG = R / "PORT80B_D10B_HELDOUT_10000_ENDURANCE_PREREGISTRATION.md"
FAILED_PREFLIGHT = R / "port80b_d10b_heldout_10000_endurance_preflight.json"
FAILED_REPORT = R / "PORT80B_D10B_HELDOUT_10000_ENDURANCE_PREFLIGHT_REPORT_2026-08-13.md"
FAILED_RAW = R / "port80b_d10b_heldout_10000_endurance.json"

OUT = R / "port80b_d10br_endurance_independent_verification.json"
REPORT = R / "PORT80B_D10BR_ENDURANCE_INDEPENDENT_VERIFICATION_REPORT_2026-08-13.md"

DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS = 48
EXPERTS_WITH_SHARED = 513
EXPERT_BYTES = 2_027_520
PREFIX = 499
TOP_K = 10
HIDDEN = 2048
INTER = 512
ENDURANCE_SOURCE = (768, 1024)
ENDURANCE_STEPS = 10_000
WARMUPS = 8
COLD_SLOTS = 32
MASK64 = (1 << 64) - 1
LIFT_SEED = 0xD10A_499D_1308_2026
EXPECTED_ROUTE_SHA = "85f12fb0020bb8568dfc3683662e8251b29bf83684beb296dbb6d8734f5ffd20"
EXPECTED_BANK_SHA = "4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462"
MIN_RAM_BEFORE = 52_652_163_072
MIN_RAM_AFTER_TOUCH = 2 * 2**30
EMERGENCY_RAM = int(1.5 * 2**30)
VRAM_RESERVE = 512 * 2**20
DENSE_BYTES = 1_933_921_280
DEVICE_REQUEST = 4_521_569_280
BANK_BYTES = 49_925_652_480
POISON_VALUE = 12345.25
DIGEST_STEPS = tuple([0] + list(range(99, ENDURANCE_STEPS, 100)))
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_ARRAYS = {
    "routed_capture": ([LAYERS * TOP_K * (INTER + INTER + HIDDEN)], "float32"),
    "routed_down": ([LAYERS * TOP_K * HIDDEN], "float32"),
    "shared_down": ([LAYERS * HIDDEN], "float32"),
    "attention": ([12 * 16 * 256], "float32"),
    "delta": ([36 * 32], "float32"),
    "kv_state": ([12 * 2 * 2 * 4096 * 256], "uint16"),
    "recurrent_state": ([36 * 32 * 128 * 128], "float32"),
    "conv_state": ([36 * (16 * 128 * 2 + 32 * 128) * 4], "uint16"),
    "composed_state": ([LAYERS * HIDDEN], "float32"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def same_stats(observed: dict[str, Any], calculated: dict[str, Any]) -> bool:
    if observed.keys() != calculated.keys():
        return False
    for key, value in calculated.items():
        if key == "count":
            if observed[key] != value:
                return False
        elif not math.isclose(float(observed[key]), float(value), rel_tol=0.0, abs_tol=1e-9):
            return False
    return True


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def lift(source: np.ndarray, domain_index: int, token: int, epoch: int) -> np.ndarray:
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


def load_routes() -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, bool]]:
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    routes = {domain: np.empty((1024, LAYERS, 8), dtype=np.int16) for domain in DOMAINS}
    hashes: dict[str, str] = {}
    locks: dict[str, bool] = {}
    for layer in range(LAYERS):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"
        layer_hash = sha256(path)
        hashes[str(layer)] = layer_hash
        locks[str(layer)] = layer_hash == capture["manifests"][str(layer)]["artifact_sha256"]
        tensors = load_file(path)
        for domain in DOMAINS:
            routes[domain][:, layer] = tensors[f"{domain}_router_ids"].astype(np.int16)
    return routes, hashes, locks


def rebuild_route_contract(routes: dict[str, np.ndarray]) -> dict[str, Any]:
    digest = hashlib.sha256()
    metadata: list[dict[str, Any]] = []
    cold_values: list[int] = []
    coverage: set[tuple[int, int]] = set()
    for epoch in range(8):
        for domain_index, domain in enumerate(DOMAINS):
            for token in range(ENDURANCE_SOURCE[0], ENDURANCE_SOURCE[1]):
                route = lift(routes[domain][token], domain_index, token, epoch)
                digest.update(route.tobytes())
                cold_values.append(int(np.count_nonzero(route >= PREFIX)))
                coverage.update((layer, int(expert)) for layer, row in enumerate(route) for expert in row)
                metadata.append({"domain": domain, "domain_index": domain_index, "token": token, "epoch": epoch})
                if len(metadata) == ENDURANCE_STEPS:
                    values = np.asarray(cold_values, dtype=np.float64)
                    return {
                        "route_sha256": digest.hexdigest(),
                        "steps": len(metadata),
                        "metadata": metadata,
                        "cold_records": int(values.sum()),
                        "cold_rate": float(values.sum() / (len(metadata) * LAYERS * TOP_K)),
                        "cold_per_step_p50": float(np.percentile(values, 50)),
                        "cold_per_step_p95": float(np.percentile(values, 95)),
                        "cold_per_step_p99": float(np.percentile(values, 99)),
                        "cold_per_step_max": int(values.max()),
                        "layer_expert_coverage": len(coverage),
                    }
    raise RuntimeError("independent held-out route reconstruction did not reach 10,000 steps")


def ast_assignment(tree: ast.Module, name: str) -> str | None:
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.dump(node.value, include_attributes=False)
    return None


class NormalizeRevisionStrings(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            value = node.value.replace(
                "port80b_d10br_heldout_10000_endurance_revision",
                "port80b_d10b_heldout_10000_endurance",
            ).replace("PORT80B-D10B-R", "PORT80B-D10B")
            return ast.copy_location(ast.Constant(value=value), node)
        return node


def normalized_function(tree: ast.Module, name: str) -> str:
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    normalized = NormalizeRevisionStrings().visit(ast.fix_missing_locations(ast.parse(ast.unparse(node)))).body[0]
    return ast.dump(normalized, include_attributes=False)


def main() -> None:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    component = json.loads(COMPONENT.read_text(encoding="utf-8"))
    failed_preflight = json.loads(FAILED_PREFLIGHT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    routes, route_hashes, route_locks = load_routes()
    route = rebuild_route_contract(routes)
    del routes

    wall = [float(value) for value in raw["latency"]["wall_ms"]]
    event = [float(value) for value in raw["latency"]["cuda_event_ms"]]
    wall_stats = stats(wall)
    event_stats = stats(event)
    first_stats = stats(wall[:1000])
    last_stats = stats(wall[-1000:])
    drift = float(last_stats["p95"]) / float(first_stats["p95"])

    state_checks = raw["state_checks"]
    telemetry = raw["telemetry"]
    checkpoints = raw["checkpoint_evidence"]
    expected_metadata = route["metadata"]
    expected_checkpoint_metadata = [expected_metadata[step] for step in DIGEST_STEPS]

    checkpoint_array_rows: list[dict[str, Any]] = []
    checkpoint_schema_ok = True
    checkpoint_metadata_ok = True
    digest_values: dict[str, list[str]] = {name: [] for name in EXPECTED_ARRAYS}
    for position, checkpoint in enumerate(checkpoints):
        step = DIGEST_STEPS[position] if position < len(DIGEST_STEPS) else -1
        expected_meta = expected_checkpoint_metadata[position] if position < len(expected_checkpoint_metadata) else {}
        checkpoint_metadata_ok = checkpoint_metadata_ok and checkpoint.get("step") == step and all(
            checkpoint.get(key) == expected_meta.get(key) for key in ("domain", "token", "epoch")
        )
        arrays = checkpoint.get("arrays", {})
        checkpoint_schema_ok = checkpoint_schema_ok and set(arrays) == set(EXPECTED_ARRAYS)
        for name, (shape, dtype) in EXPECTED_ARRAYS.items():
            value = arrays.get(name, {})
            checkpoint_array_rows.append(value)
            digest = value.get("sha256")
            if isinstance(digest, str):
                digest_values[name].append(digest)
            checkpoint_schema_ok = checkpoint_schema_ok and (
                value.get("shape") == shape
                and value.get("dtype") == dtype
                and value.get("finite") is True
                and value.get("poison_count") == 0
                and isinstance(digest, str)
                and HEX64.fullmatch(digest) is not None
            )
    digest_uniqueness = {name: len(set(values)) for name, values in digest_values.items()}

    telemetry_steps_ok = [row.get("step") for row in telemetry] == list(range(ENDURANCE_STEPS))
    telemetry_state_flags = [row.get("state_finite_and_written") for row in telemetry]
    telemetry_state_matches = telemetry_state_flags == state_checks
    telemetry_numeric = all(
        isinstance(row.get("available_ram"), int)
        and isinstance(row.get("free_vram"), int)
        and all(isinstance(row.get("process", {}).get(key), int) for key in ("rss", "peak_wset", "private", "pagefile", "num_page_faults"))
        for row in telemetry
    )
    minimum_available = min(row["available_ram"] for row in telemetry)
    minimum_vram = min(row["free_vram"] for row in telemetry)
    memory_loss = telemetry[0]["available_ram"] - telemetry[-1]["available_ram"]

    page_samples = raw["page_reads"]["samples"]
    page_reads = [float(row["page_reads_per_sec"]) for row in page_samples]
    pages_input = [float(row["pages_input_per_sec"]) for row in page_samples]
    page_samples_finite = bool(page_samples) and all(
        math.isfinite(row["monotonic_seconds"])
        and math.isfinite(row["page_reads_per_sec"])
        and math.isfinite(row["pages_input_per_sec"])
        for row in page_samples
    )

    per_call_base = (0x5A * (DENSE_BYTES * (DENSE_BYTES + 1) // 2)) & MASK64
    per_step = (0x5A * DENSE_BYTES) & MASK64
    invocation_count = WARMUPS + ENDURANCE_STEPS
    summed_steps = sum(range(WARMUPS)) + (ENDURANCE_STEPS * (ENDURANCE_STEPS - 1) // 2)
    dense_expected = (invocation_count * per_call_base + per_step * summed_steps) & MASK64

    registration = raw["registration_attempts"]
    unregistration = raw["unregister_attempts"]
    expected_bytes = PREFIX * EXPERT_BYTES
    expected_host_stride = EXPERTS_WITH_SHARED * EXPERT_BYTES
    registration_rows_ok = len(registration) == LAYERS and all(
        row.get("action") == "host_register"
        and row.get("layer") == layer
        and row.get("bytes") == expected_bytes
        and row.get("attempted") is True
        and row.get("success") is True
        and row.get("error") is None
        and isinstance(row.get("host_pointer"), int)
        and isinstance(row.get("device_alias"), int)
        for layer, row in enumerate(registration)
    )
    host_pointer_stride_ok = registration_rows_ok and all(
        registration[layer]["host_pointer"] - registration[layer - 1]["host_pointer"] == expected_host_stride
        for layer in range(1, LAYERS)
    )
    unregistration_rows_ok = len(unregistration) == LAYERS and all(
        row.get("action") == "host_unregister"
        and row.get("layer") == layer
        and row.get("bytes") == expected_bytes
        and row.get("attempted") is True
        and row.get("success") is True
        and row.get("error") is None
        and row.get("host_pointer") == registration[layer]["host_pointer"]
        and row.get("device_alias") is None
        for layer, row in enumerate(unregistration)
    )

    physical = raw["physical"]
    raw_inputs = raw["inputs"]
    input_hashes = {
        "raw": sha256(RAW),
        "preregistration": sha256(PREREG),
        "runner": sha256(RUNNER),
        "preflight": sha256(PREFLIGHT),
        "component": sha256(COMPONENT),
        "manifest": sha256(MANIFEST),
        "capture": sha256(CAPTURE),
        "failed_runner": sha256(FAILED_RUNNER),
        "failed_preregistration": sha256(FAILED_PREREG),
        "failed_preflight": sha256(FAILED_PREFLIGHT),
        "failed_preflight_report": sha256(FAILED_REPORT),
        "d10a1r": {name: sha256(path) for name, path in D10A1R_FILES.items()},
        "d10a2r": {name: sha256(path) for name, path in D10A2R_FILES.items()},
        "d10a2r2": {name: sha256(path) for name, path in D10A2R2_FILES.items()},
        "independent_audit": sha256(INDEPENDENT_AUDIT),
        "independent_audit_report": sha256(INDEPENDENT_AUDIT_REPORT),
        "d10a2r2_erratum": sha256(D10A2R2_ERRATUM),
    }

    current_tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    failed_tree = ast.parse(FAILED_RUNNER.read_text(encoding="utf-8"))
    frozen_names = (
        "DOMAINS", "PREFIX", "TOP_K", "HIDDEN", "INTER", "CORRECTNESS", "VALIDATION",
        "ENDURANCE_SOURCE", "ENDURANCE_STEPS", "ACK", "MIN_RAM_AFTER_TOUCH", "EMERGENCY_RAM",
        "VRAM_RESERVE", "DENSE_BYTES", "KV_BYTES", "RECURRENT_BYTES", "CONV_BYTES", "SHARED_BYTES",
        "RUNTIME_BYTES", "COLD_SLOTS", "COLD_BYTES", "OUTPUT_BYTES", "REFERENCE_BYTES",
        "DEVICE_REQUEST", "POISON", "DIGEST_STEPS", "EXPECTED_ENDURANCE_ROUTE_SHA256", "LIFT_SEED",
        "COMPONENT_SOURCE",
    )
    assignment_equality = {
        name: ast_assignment(current_tree, name) == ast_assignment(failed_tree, name)
        for name in frozen_names
    }
    normalized_endurance_equal = normalized_function(current_tree, "endurance_phase") == normalized_function(failed_tree, "endurance_phase")

    preflight_actions_clean = all(value is False for value in preflight["physical_actions"].values())
    failed_actions_clean = all(value is False for value in failed_preflight["physical_actions"].values())
    preflight_input_locks = (
        preflight["inputs"]["preregistration_sha256"] == input_hashes["preregistration"]
        and preflight["inputs"]["runner_sha256"] == input_hashes["runner"]
        and preflight["inputs"]["manifest_sha256"] == input_hashes["manifest"]
        and preflight["inputs"]["capture_sha256"] == input_hashes["capture"]
        and preflight["inputs"]["route_hashes"] == route_hashes
        and preflight["inputs"]["conv_unit_test_sha256"] == "02dfa87cee8ac58b54a8b71656d109a55d96298db659228c2726f447214f3650"
        and preflight["inputs"]["conv_unit_result_sha256"] == "ba7c398facaaa88b46ad95ec020bd031fed324755ed0bf7550af0c63ba9941c1"
        and preflight["inputs"]["independent_audit_sha256"] == "409c379600b733bc466b21c981f75342d6087612f3c60f6e7c4889f31828ab6d"
    )
    result_input_locks = (
        raw_inputs["preregistration_sha256"] == input_hashes["preregistration"]
        and raw_inputs["runner_sha256"] == input_hashes["runner"]
        and raw_inputs["preflight_sha256"] == input_hashes["preflight"]
        and raw_inputs["d10a2r2_component_sha256"] == input_hashes["component"]
        and raw_inputs["bank_sha256_from_manifest"] == EXPECTED_BANK_SHA
    )
    manifest_bank_hash = manifest.get("bulk_bank", {}).get("sha256") or manifest.get("bank_sha256") or manifest.get("sha256")
    full_prior_lock_chain_current = (
        input_hashes["d10a1r"] == preflight["audit"]["immutable_prior_locks"]
        and input_hashes["d10a2r"] == preflight["audit"]["immutable_d10a2r_locks"]
        and input_hashes["d10a2r2"] == preflight["audit"]["immutable_d10a2r2_locks"]
        and input_hashes["independent_audit"] == preflight["audit"]["independent_audit"]["json_sha256"]
        and input_hashes["independent_audit_report"] == preflight["audit"]["independent_audit"]["report_sha256"]
        and input_hashes["d10a2r2_erratum"] == preflight["audit"]["independent_audit"]["erratum_sha256"]
    )

    recomputed_gates = {
        "exactly_10000_heldout_cases": route["steps"] == ENDURANCE_STEPS and raw["route_contract"]["steps"] == ENDURANCE_STEPS,
        "latency_vectors_exactly_10000_finite_positive": (
            len(wall) == ENDURANCE_STEPS
            and len(event) == ENDURANCE_STEPS
            and bool(np.isfinite(wall).all())
            and bool(np.isfinite(event).all())
            and min(wall) > 0.0
            and min(event) > 0.0
        ),
        "wall_p95_le_150ms": wall_stats["p95"] <= 150.0,
        "wall_p99_le_200ms": wall_stats["p99"] <= 200.0,
        "last_first_1000_p95_ratio_le_1_20": drift <= 1.20,
        "all_10000_composed_states_finite_and_written": len(state_checks) == ENDURANCE_STEPS and all(value is True for value in state_checks),
        "exact_101_checkpoint_schedule": [row.get("step") for row in checkpoints] == list(DIGEST_STEPS),
        "checkpoint_arrays_exactly_909_finite_digested_no_poison": (
            len(checkpoints) == len(DIGEST_STEPS)
            and len(checkpoint_array_rows) == len(DIGEST_STEPS) * 9
            and checkpoint_schema_ok
        ),
        "all_composed_checkpoint_digests_unique": digest_uniqueness["composed_state"] == len(DIGEST_STEPS),
        "telemetry_exactly_10000": len(telemetry) == ENDURANCE_STEPS and telemetry_steps_ok,
        "post_warmup_page_reads_no_sample_gt_2048": page_samples_finite and max(page_reads) <= 2048.0,
        "endurance_memory_loss_le_1gib": memory_loss <= 2**30,
        "ram_after_first_touch_ge_2gib": physical["available_ram_after_first_touch"] >= MIN_RAM_AFTER_TOUCH,
        "vram_reserve_ge_512mib": minimum_vram >= VRAM_RESERVE,
        "dense_checksum_exact_and_runtime_touched": (
            raw["dense_runtime"]["dense_checksum_expected"] == dense_expected
            and raw["dense_runtime"]["dense_checksum_observed"] == dense_expected
            and raw["dense_runtime"]["runtime_sentinels"] == [0xA5, 0xA5]
        ),
        "registration_48_ranges": (
            registration_rows_ok
            and physical["registered_bytes"] == LAYERS * expected_bytes
            and host_pointer_stride_ok
        ),
        "no_cuda_or_runner_error": raw["error"] is None,
        "registration_attempt_rows_48_all_success": registration_rows_ok,
        "clean_unregister_48_ranges": unregistration_rows_ok and raw["unregister_failures"] == [],
    }

    checks = {
        "raw_status_and_overall_pass": raw["status"] == "heldout_10000_endurance_pass" and raw["overall_pass"] is True,
        "all_19_recomputed_gates_pass": len(recomputed_gates) == 19 and all(recomputed_gates.values()),
        "recomputed_gates_exactly_match_raw": recomputed_gates == raw["gates"],
        "wall_stats_exact": same_stats(raw["latency"]["wall_stats"], wall_stats),
        "event_stats_exact": same_stats(raw["latency"]["cuda_event_stats"], event_stats),
        "first_1000_stats_exact": same_stats(raw["latency"]["first_1000_wall_stats"], first_stats),
        "last_1000_stats_exact": same_stats(raw["latency"]["last_1000_wall_stats"], last_stats),
        "drift_ratio_exact": math.isclose(raw["latency"]["last_first_p95_ratio"], drift, rel_tol=0.0, abs_tol=1e-12),
        "heldout_route_sha_rebuilt": route["route_sha256"] == EXPECTED_ROUTE_SHA == raw["route_contract"]["route_sha256"],
        "route_partition_label_warmups_exact": raw["route_contract"] == {
            "label": "p4d_shaped_synthetic_proxy",
            "partition": [768, 1024],
            "route_sha256": EXPECTED_ROUTE_SHA,
            "steps": ENDURANCE_STEPS,
            "warmups": WARMUPS,
        },
        "route_tensor_hashes_locked": all(route_locks.values()) and preflight["inputs"]["route_hashes"] == route_hashes,
        "cold_slot_boundary_safe": route["cold_per_step_max"] <= COLD_SLOTS,
        "checkpoint_metadata_matches_rebuilt_route": checkpoint_metadata_ok,
        "checkpoint_schedule_exact_101": len(checkpoints) == 101 and [row["step"] for row in checkpoints] == list(DIGEST_STEPS),
        "checkpoint_909_schema_finite_digest_poison_clean": len(checkpoint_array_rows) == 909 and checkpoint_schema_ok,
        "composed_checkpoint_digests_unique": digest_uniqueness["composed_state"] == 101,
        "state_flags_exact_10000_true": len(state_checks) == 10_000 and all(value is True for value in state_checks),
        "telemetry_exact_10000_ordered_numeric": len(telemetry) == 10_000 and telemetry_steps_ok and telemetry_numeric,
        "telemetry_state_flags_match_state_vector": telemetry_state_matches and all(value is True for value in telemetry_state_flags),
        "emergency_ram_never_crossed": minimum_available >= EMERGENCY_RAM,
        "first_touch_ram_gate": physical["available_ram_after_first_touch"] >= MIN_RAM_AFTER_TOUCH,
        "vram_reserve_gate": minimum_vram >= VRAM_RESERVE and physical["free_vram_after_allocations"] >= VRAM_RESERVE,
        "device_request_exact": physical["device_request_bytes"] == DEVICE_REQUEST,
        "start_ram_gate": physical["available_ram_before"] >= MIN_RAM_BEFORE,
        "memory_loss_arithmetic_gate": memory_loss <= 2**30,
        "page_samples_valid_and_below_gate": page_samples_finite and raw["page_reads"]["error"] is None and max(page_reads) <= 2048.0,
        "dense_checksum_formula_exact": raw["dense_runtime"]["dense_checksum_expected"] == dense_expected == raw["dense_runtime"]["dense_checksum_observed"],
        "runtime_sentinels_exact": raw["dense_runtime"]["runtime_sentinels"] == [165, 165],
        "registration_rows_exact": registration_rows_ok and host_pointer_stride_ok,
        "unregistration_rows_exact": unregistration_rows_ok and raw["unregister_failures"] == [],
        "registered_byte_arithmetic_exact": physical["registered_bytes"] == LAYERS * PREFIX * EXPERT_BYTES,
        "raw_provenance_locks_current": result_input_locks,
        "preflight_provenance_locks_current": preflight_input_locks,
        "preflight_full_prior_unit_audit_lock_chain_current": full_prior_lock_chain_current,
        "preflight_internal_audit_all_true": preflight["audit"]["pass"] is True and all(preflight["audit"]["checks"].values()),
        "preflight_source_contract_all_true": all(preflight["source_mutation_contract"].values()),
        "preflight_canary_audit_clean": preflight["canary_audit"]["injective"] is True and preflight["canary_audit"]["roundtrip_pass"] is True and preflight["canary_audit"]["boundary_498_499_pass"] is True,
        "manifest_declares_expected_bank_hash": manifest_bank_hash == EXPECTED_BANK_SHA,
        "bulk_bank_exists_with_exact_frozen_size": BANK.is_file() and BANK.stat().st_size == BANK_BYTES,
        "preflight_passed_without_physical_actions": preflight["pass"] is True and preflight["error"] is None and preflight_actions_clean and preflight["component_opened"] is False and preflight["endurance_opened"] is False,
        "component_clean_pass_locked": component["overall_pass"] is True and all(component["gates"].values()) and component["error"] is None and component["unregister_failures"] == [],
        "failed_original_preflight_was_cpu_only_and_closed": failed_preflight["pass"] is False and failed_actions_clean and failed_preflight["component_opened"] is False and failed_preflight["endurance_opened"] is False,
        "failed_original_artifacts_hash_locked": (
            input_hashes["failed_runner"] == "4f8226d82d7d804195a9728bc9852cc9b75fa33ec6d8481e86d94ae90ff3cb68"
            and input_hashes["failed_preregistration"] == "8d171ac876d03681d35de9155b100ec01e3345588ca31eb926e9acddaa59b977"
            and input_hashes["failed_preflight"] == "91bb855940f0d39f241c29159ae39c011c46e4e9a3297d50bdb696e90fde985e"
            and input_hashes["failed_preflight_report"] == "c0e9fcbcc9e1010307bffe23334b9a829cf7d5f7310a55092a0bfb1eb2c4dd21"
        ),
        "revision_frozen_assignments_unchanged": all(assignment_equality.values()),
        "revision_endurance_body_only_identifier_heading_changes": normalized_endurance_equal,
        "no_original_gpu_result_exists": not FAILED_RAW.exists(),
        "current_result_is_single_immutable_output": RAW.is_file() and RAW.stat().st_size > 0,
        "raw_wall_seconds_finite_positive": math.isfinite(raw["wall_seconds"]) and raw["wall_seconds"] > 0,
        "claim_boundary_present_and_non_breakthrough": "not checkpoint" in raw["claim_boundary"] and "breakthrough" in raw["claim_boundary"],
    }

    replayability = {
        "checkpoint_summaries_persisted": True,
        "checkpoint_underlying_arrays_persisted": False,
        "checkpoint_sha_format_and_internal_summary_contract_verified": checkpoint_schema_ok,
        "checkpoint_sha_recomputed_from_underlying_arrays": False,
        "state_flags_are_booleans_not_raw_10000_state_tensors": True,
        "second_endurance_replay_digest_available": False,
        "interpretation": (
            "The 101x9 SHA-256 values are valid persisted device-array summaries and satisfy the frozen "
            "summary contract, but the source arrays were not retained. A CPU-only audit therefore cannot "
            "re-hash their bytes or prove deterministic replay; uniqueness is integrity/evolution evidence only."
        ),
    }

    overall = all(checks.values()) and all(recomputed_gates.values())
    result = {
        "kind": "port80b_d10br_endurance_independent_cpu_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "independent_endurance_pass_with_summary_replayability_boundary" if overall else "independent_endurance_audit_fail",
        "pass": overall,
        "inputs": input_hashes,
        "recomputed_gates": recomputed_gates,
        "checks": checks,
        "check_count": len(checks),
        "checks_passed": sum(checks.values()),
        "timing": {
            "wall_ms": wall_stats,
            "cuda_event_ms": event_stats,
            "first_1000_wall_ms": first_stats,
            "last_1000_wall_ms": last_stats,
            "last_first_p95_ratio": drift,
            "runner_wall_seconds": raw["wall_seconds"],
        },
        "route": {key: value for key, value in route.items() if key != "metadata"},
        "checkpoint": {
            "schedule": list(DIGEST_STEPS),
            "count": len(checkpoints),
            "array_summary_count": len(checkpoint_array_rows),
            "digest_uniqueness_by_array": digest_uniqueness,
            "expected_shapes_and_dtypes": EXPECTED_ARRAYS,
        },
        "telemetry": {
            "count": len(telemetry),
            "minimum_available_ram": minimum_available,
            "minimum_free_vram": minimum_vram,
            "available_ram_first": telemetry[0]["available_ram"],
            "available_ram_last": telemetry[-1]["available_ram"],
            "memory_loss_first_minus_last": memory_loss,
            "process_rss_first": telemetry[0]["process"]["rss"],
            "process_rss_last": telemetry[-1]["process"]["rss"],
            "process_rss_change": telemetry[-1]["process"]["rss"] - telemetry[0]["process"]["rss"],
        },
        "page_reads": {
            "sample_count": len(page_samples),
            "max_page_reads_per_sec": max(page_reads),
            "max_pages_input_per_sec_diagnostic": max(pages_input),
            "sampler_error": raw["page_reads"]["error"],
        },
        "dense": {
            "calculated_expected": dense_expected,
            "observed": raw["dense_runtime"]["dense_checksum_observed"],
            "runtime_sentinels": raw["dense_runtime"]["runtime_sentinels"],
            "invocation_count": invocation_count,
            "summed_steps": summed_steps,
        },
        "lifecycle": {
            "registration_rows": len(registration),
            "unregistration_rows": len(unregistration),
            "bytes_per_registration": expected_bytes,
            "registered_bytes": physical["registered_bytes"],
            "host_pointer_stride": expected_host_stride,
            "unregister_failures": raw["unregister_failures"],
        },
        "revision_audit": {
            "frozen_assignment_equality": assignment_equality,
            "normalized_endurance_function_equal": normalized_endurance_equal,
            "original_preflight_error": failed_preflight["error"],
            "original_gpu_result_exists": FAILED_RAW.exists(),
            "conclusion": (
                "Within the retained artifacts, the revision changed provenance plumbing/labels only; the "
                "frozen endurance constants, component CUDA source and normalized endurance body are unchanged. "
                "The original attempt stopped in CPU preflight and produced no GPU endurance result."
            ),
        },
        "replayability_boundary": replayability,
        "claim_boundary": raw["claim_boundary"],
        "notes": {
            "manifest_bank_hash_field_seen": manifest_bank_hash,
            "bulk_bank_rehashed_by_this_verifier": False,
            "bulk_bank_size_bytes": BANK.stat().st_size if BANK.is_file() else None,
            "poison_value_frozen": POISON_VALUE,
            "no_gpu_was_initialized_or_used_by_this_verifier": True,
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    verdict = "PASS" if overall else "FAIL"
    failed_checks = [name for name, passed in checks.items() if not passed]
    failed_gates = [name for name, passed in recomputed_gates.items() if not passed]
    report = f"""# PORT80B-D10B-R independent endurance verification

Verdict: **{verdict}**. The independent CPU-only verifier passes **{sum(checks.values())}/{len(checks)} checks** and independently recomputes **{sum(recomputed_gates.values())}/19 frozen gates**. Failed checks: `{failed_checks}`. Failed gates: `{failed_gates}`.

## Exact recomputation

- Held-out stream: **{route['steps']:,}** cases, independently rebuilt SHA-256 `{route['route_sha256']}`; max cold records per step **{route['cold_per_step_max']}** against **{COLD_SLOTS}** slots.
- Wall p50/p95/p99: **{wall_stats['p50']:.6f} / {wall_stats['p95']:.6f} / {wall_stats['p99']:.6f} ms**. CUDA-event p50/p95/p99: **{event_stats['p50']:.6f} / {event_stats['p95']:.6f} / {event_stats['p99']:.6f} ms**.
- First/last-1,000 wall p95: **{first_stats['p95']:.6f} / {last_stats['p95']:.6f} ms**; drift ratio **{drift:.9f}**.
- State flags: **{sum(value is True for value in state_checks):,}/10,000 true**. Telemetry rows: **{len(telemetry):,}**, in exact step order.
- Checkpoints: **{len(checkpoints)}** on the frozen schedule; **{len(checkpoint_array_rows)}** array summaries. Digest uniqueness by array: `{digest_uniqueness}`.
- Page telemetry: **{len(page_samples)}** samples; maximum Page Reads/sec **{max(page_reads):.6f}**. Maximum Pages Input/sec was **{max(pages_input):.6f}** and remains diagnostic, not the frozen gate.
- RAM: before **{physical['available_ram_before']:,} B**; after first touch **{physical['available_ram_after_first_touch']:,} B**; minimum sampled during endurance **{minimum_available:,} B**. First-minus-last sampled availability is **{memory_loss:,} B**.
- VRAM: post-allocation **{physical['free_vram_after_allocations']:,} B**; minimum sampled **{minimum_vram:,} B**.
- Dense checksum: expected/observed **{dense_expected} / {raw['dense_runtime']['dense_checksum_observed']}**; runtime sentinels `{raw['dense_runtime']['runtime_sentinels']}`.
- Host lifecycle: **48/48** registration and **48/48** unregister rows clean; **{physical['registered_bytes']:,} B** registered; no unregister failures.

## Provenance and retry/retune audit

The raw result locks the current preregistration, runner, CPU preflight, D10A2-R2 component result and the bank payload SHA declared by the manifest. The preflight locks the manifest file, all 48 route tensors and the frozen CPU unit/audit evidence. This verifier confirms the current bulk bank exists at the exact frozen **{BANK_BYTES:,} B** size, but deliberately does not rescan/hash all 49.9 GB. The original D10B attempt failed before any physical action; its four artifacts match the immutable hashes and no original D10B GPU result exists.

All checked execution/resource/route/schedule assignments are AST-identical between D10B and D10B-R. After normalizing only the revision result identifier and report heading, the full `endurance_phase` AST is identical. This is strong artifact-level evidence of **no retune** and no prior GPU retry. Absolute proof that no unrecorded external run occurred is outside filesystem provenance.

## Replayability boundary

The 101 x 9 checkpoint rows retain shape, dtype, finite flag, poison count and SHA-256, not the array bytes. This verifier confirms exact schedule/route metadata, schema, digest format and uniqueness, but **cannot recompute those 909 hashes from the underlying tensors**. Likewise, the 10,000 state records are Boolean guards rather than raw state tensors. Consequently this is a valid pass under the frozen summary contract, but it is not an independent numerical replay or a cross-run determinism proof.

## Claim boundary

{raw['claim_boundary']}
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "pass": overall,
        "checks": f"{sum(checks.values())}/{len(checks)}",
        "gates": f"{sum(recomputed_gates.values())}/19",
        "output": str(OUT),
        "report": str(REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
