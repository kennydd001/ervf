from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
S = ROOT / "scripts/streamq5_moe"
REGISTRY = R / "NEXT_WAVE_REGISTRY_2026-08-12.yaml"
OUTPUT = R / "next_wave_independent_verification_2026-08-12.json"
ROUTES = ROOT / "reports/runs/streamq5_moe/p4d_routes"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(name: str) -> dict:
    return json.loads((R / name).read_text(encoding="utf-8"))


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def percentile(values: list[float], p: float) -> float:
    rows = sorted(float(value) for value in values)
    index = (len(rows) - 1) * p / 100.0
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return rows[low]
    return rows[low] * (high - index) + rows[high] * (index - low)


def raw_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def stats_equal(values: list[float], stored: dict) -> bool:
    observed = raw_stats(values)
    return all(key not in stored or close(observed[key], stored[key]) for key in observed) and (
        "iterations" not in stored or stored["iterations"] == len(values)
    )


def recursive_event_stats(value, failures: list[str], location: str) -> int:
    checked = 0
    if isinstance(value, dict):
        if isinstance(value.get("event_ms"), list) and isinstance(value.get("stats"), dict):
            checked += 1
            if not stats_equal(value["event_ms"], value["stats"]):
                failures.append(location)
        for key, child in value.items():
            checked += recursive_event_stats(child, failures, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            checked += recursive_event_stats(child, failures, f"{location}[{index}]")
    return checked


def parse_registry(path: Path) -> list[dict]:
    items: list[dict] = []
    current = None
    evidence_list = False
    for line in path.read_text(encoding="utf-8").splitlines():
        found = re.match(r"^  - id: (N\d{3})$", line)
        if found:
            current = {"id": found.group(1), "evidence": []}
            items.append(current)
            evidence_list = False
            continue
        if current is None:
            continue
        field = re.match(r"^    ([a-z_]+):(?:\s+(.*))?$", line)
        if field:
            key, raw = field.groups()
            evidence_list = key == "evidence" and not raw
            if key == "evidence" and raw:
                current["evidence"] = [raw.strip()]
            elif key != "evidence":
                current[key] = "" if raw is None else raw.strip()
            continue
        evidence = re.match(r"^      - (.+)$", line)
        if evidence_list and evidence:
            current["evidence"].append(evidence.group(1).strip())
    return items


def aggregate_q8_audit(manifest: dict) -> dict:
    aggregate = hashlib.sha256()
    record_hashes = True
    byte_sum = 0
    for record in manifest["records"]:
        path = ROOT / record["artifact"]
        digest = hashlib.sha256()
        observed_bytes = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 2**20), b""):
                aggregate.update(chunk)
                digest.update(chunk)
                observed_bytes += len(chunk)
        byte_sum += observed_bytes
        record_hashes &= observed_bytes == record["bytes"]
        record_hashes &= digest.hexdigest() == record["artifact_sha256"]
    return {
        "aggregate_sha256": aggregate.hexdigest(),
        "record_hashes_and_sizes": record_hashes,
        "byte_sum": byte_sum,
    }


def hash_claim(checks: dict[str, bool], name: str, claimed: str, path: Path) -> None:
    checks[name] = path.exists() and claimed.lower() == sha256(path).lower()


def paired_result_audit(result: dict, warmup: int, extended: bool) -> bool:
    pairs = result["pairs"]
    baseline = [row["baseline"]["wall_ms"] for row in pairs[warmup:]]
    candidate = [row["candidate"]["wall_ms"] for row in pairs[warmup:]]
    baseline_stats = raw_stats(baseline)
    candidate_stats = raw_stats(candidate)
    stats_ok = all(close(baseline_stats[key], result["baseline"][key]) for key in baseline_stats if key in result["baseline"])
    stats_ok &= all(close(candidate_stats[key], result["candidate"][key]) for key in candidate_stats if key in result["candidate"])
    ratios = {key: candidate_stats[key] / baseline_stats[key] for key in ("mean", "p50", "p95")}
    stats_ok &= all(close(ratios[key], result["ratios"][key]) for key in ratios)
    exact_names = ["exact_prediction", "exact_misses", "exact_kv", "exact_dynamic"]
    if extended:
        exact_names += ["exact_logits", "exact_state"]
    exact_ok = all(result["exactness"][name] == all(row[name] for row in pairs) for name in exact_names)
    order_ok = all(
        row["order"] == (["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"])
        for index, row in enumerate(pairs)
    )
    expected_pass = (
        all(result["exactness"].values())
        and ratios["mean"] <= 0.98
        and ratios["p50"] <= 0.98
        and ratios["p95"] <= 1.00
        and len(pairs) == 128
        and len(baseline) == 112
    )
    return stats_ok and exact_ok and order_ok and result["overall_pass"] == expected_pass


def main() -> None:
    checks: dict[str, bool] = {}
    diagnostics: dict[str, object] = {}

    registry = parse_registry(REGISTRY)
    ids = [row["id"] for row in registry]
    maximum = max(int(value[1:]) for value in ids)
    valid_statuses = {
        "queued", "in_progress", "verified_pass", "verified_pass_component",
        "verified_baseline", "verified_negative", "superseded", "blocked_artifact",
        "blocked_hardware", "blocked_scope",
    }
    checks["registry_ids_unique_and_contiguous"] = (
        len(ids) == len(set(ids)) and ids == [f"N{index:03d}" for index in range(1, maximum + 1)]
    )
    checks["registry_status_vocabulary"] = all(row.get("status") in valid_statuses for row in registry)
    checks["registry_no_queued"] = all(row.get("status") != "queued" for row in registry)
    evidence_paths = [ROOT / name for row in registry for name in row.get("evidence", [])]
    checks["registry_evidence_exists"] = all(path.exists() for path in evidence_paths)
    terminal = {status for status in valid_statuses if status not in {"queued", "in_progress"}}
    checks["registry_terminal_has_evidence_or_blocker"] = all(
        row.get("status") not in terminal
        or bool(row.get("evidence"))
        or bool(row.get("missing"))
        for row in registry
    )
    open_items = [row["id"] for row in registry if row.get("status") in {"queued", "in_progress"}]
    diagnostics["registry"] = {
        "items": len(registry),
        "terminal": len(registry) - len(open_items),
        "open_items": open_items,
        "all_local_testable_closed": not open_items,
    }

    evidence_by_id = {row["id"]: row.get("evidence", []) for row in registry}
    primary_status_ok = True
    for row in registry:
        status = row.get("status")
        json_evidence = [ROOT / name for name in row.get("evidence", []) if name.endswith(".json")]
        if not json_evidence:
            continue
        observed = []
        for evidence in json_evidence:
            if not evidence.exists():
                continue
            primary = json.loads(evidence.read_text(encoding="utf-8"))
            verdict = primary.get("overall_pass", primary.get("summary", {}).get("pass"))
            if isinstance(verdict, bool):
                observed.append(verdict)
        if status in {"verified_pass", "verified_pass_component"}:
            primary_status_ok &= any(observed)
        elif status == "verified_negative":
            primary_status_ok &= any(not verdict for verdict in observed)
    checks["registry_primary_json_status_consistent"] = primary_status_ok

    result_files = sorted(R.glob("n[1-4]*.json"))
    registered = {Path(name).name for names in evidence_by_id.values() for name in names if name.endswith(".json")}
    phases = {row.get("phase") for row in registry if row.get("phase")}
    phase_by_file = {
        "n1a_shared_activation_ervf.json": "N1A", "n1a2_q5_staging_end_to_end.json": "N1A",
        "n1a2r_q5_staging_reverse.json": "N1A", "n1b_q5_vectorized_loads.json": "N1B",
        "n1bi_q5_vectorload_end_to_end.json": "N1BI", "n1c_generalized_exact_reduction_autotuner.json": "N1C",
        "n1c2_generalized_reduction_end_to_end.json": "N1C2", "n2a_temporal_ervf_oracle.json": "N2A",
        "n2au_route_union.json": "N2AU", "n2as_sparse_temporal_q5.json": "N2AS",
        "n2b_certified_lm_head_oracle_validation.json": "N2B", "n2c_batch_route_union_sweep.json": "N2C",
        "n2d_greedy_lm_head_compile.json": "N2D", "n3b_vram_kv_cache_pareto.json": "N3B",
        "n3a2_attention_projection_flow.json": "N3A2", "n3a3_concat_qkv_end_to_end.json": "N3A3",
        "n3a4_o_residual_fusion.json": "N3A4",
        "n3d_sequential_prefill_baseline.json": "N3D", "n4a_synthetic_80b_shape_capacity.json": "N4A",
        "n4b_synthetic_80b_gpu_shape.json": "N4B",
        "n4br_synthetic_80b_exact_replication.json": "N4BR",
    }
    companion_suffixes = ("_verification.json", "_audit.json", "_compile.json")
    untracked = [
        path.name for path in result_files
        if path.name not in registered
        and phase_by_file.get(path.name) not in phases
        and not path.name.endswith(companion_suffixes)
    ]
    checks["all_discovered_next_wave_results_registered"] = not untracked
    diagnostics["unregistered_results"] = untracked

    event_failures: list[str] = []
    event_sets = 0
    for path in result_files:
        event_sets += recursive_event_stats(json.loads(path.read_text(encoding="utf-8")), event_failures, path.name)
    checks["all_raw_event_statistics_recompute"] = not event_failures and event_sets > 0
    diagnostics["raw_event_statistics"] = {"sets_checked": event_sets, "failures": event_failures}

    manifest_path = R / "p6a_exact_runtime_bank_result.json"
    manifest = load(manifest_path.name)
    q8_audit = aggregate_q8_audit(manifest)
    checks["q8_all_242_record_hashes_sizes"] = q8_audit["record_hashes_and_sizes"]
    checks["q8_manifest_aggregate_byte_arithmetic"] = (
        q8_audit["byte_sum"] == manifest["aggregate"]["bytes"]
        and sum(row["bytes"] for row in manifest["records"]) == manifest["aggregate"]["bytes"]
        and len(manifest["records"]) == manifest["aggregate"]["records"] == 242
    )

    p7_test = R / "p7c_ervf_end_to_end_test.json"
    p13c = R / "p13c_evt_pm_32g_endurance.json"
    capture = R / "p4d_route_capture_result.json"
    data = ROOT / "reports/runs/streamq5_moe/p0c_fresh_input_ids.safetensors"

    n1a = load("n1a_shared_activation_ervf.json")
    hash_claim(checks, "n1a_prereg_hash", n1a["inputs"]["preregistration_sha256"], R / "N1A_SHARED_ACTIVATION_ERVF_PREREGISTRATION.md")
    hash_claim(checks, "n1a_script_hash", n1a["inputs"]["script_sha256"], S / "run_n1a_shared_activation_ervf.py")
    checks["n1a_q8_bank_provenance"] = (
        n1a["inputs"]["q8_manifest_sha256"] == sha256(manifest_path)
        and n1a["inputs"]["q8_aggregate_sha256"] == q8_audit["aggregate_sha256"]
    )
    checks["n1a_exact_but_joint_negative"] = (
        all(row["bitwise_equal"] and row["different"] == 0 for row in n1a["correctness"].values())
        and n1a["test"]["q8"] is None and n1a["test"]["q5"]["pass"]
        and not n1a["overall_pass"]
    )
    q5_test = n1a["test"]["q5"]
    checks["n1a_q5_ratio_arithmetic"] = (
        close(q5_test["p50_ratio"], q5_test["staged"]["p50"] / q5_test["baseline"]["p50"])
        and close(q5_test["p95_ratio"], q5_test["staged"]["p95"] / q5_test["baseline"]["p95"])
    )

    n1a2 = load("n1a2_q5_staging_end_to_end.json")
    n1a2r = load("n1a2r_q5_staging_reverse.json")
    for prefix, result, prereg_name in (
        ("n1a2", n1a2, "N1A2_Q5_STAGING_END_TO_END_PREREGISTRATION.md"),
        ("n1a2r", n1a2r, "N1A2R_Q5_STAGING_REVERSE_REPLICATION_PREREGISTRATION.md"),
    ):
        hash_claim(checks, f"{prefix}_prereg_hash", result["inputs"]["preregistration_sha256"], R / prereg_name)
        checks[f"{prefix}_ratio_arithmetic"] = all(
            close(result["ratios"][name], result["candidate"]["timing"][name] / result["baseline"]["timing"][name])
            for name in ("mean", "p95")
        )
    checks["n1a2_dependency_hashes"] = (
        n1a2["inputs"]["n1a_sha256"] == sha256(R / "n1a_shared_activation_ervf.json")
        and n1a2["inputs"]["p7_test_sha256"] == sha256(p7_test)
        and n1a2["inputs"]["p13c_result_sha256"] == sha256(p13c)
    )
    checks["n1a2r_dependency_hashes"] = (
        n1a2r["inputs"]["n1a_sha256"] == sha256(R / "n1a_shared_activation_ervf.json")
        and n1a2r["inputs"]["n1a2_sha256"] == sha256(R / "n1a2_q5_staging_end_to_end.json")
        and n1a2r["inputs"]["p7_test_sha256"] == sha256(p7_test)
    )
    checks["n1a_order_reversal_closes_claim"] = n1a2["overall_pass"] and not n1a2r["overall_pass"] and n1a2r["ratios"]["mean"] > 1.0

    n1b = load("n1b_q5_vectorized_loads.json")
    hash_claim(checks, "n1b_prereg_hash", n1b["preregistration_sha256"], R / "N1B_Q5_VECTORIZED_LOADS_PREREGISTRATION.md")
    hash_claim(checks, "n1b_script_hash", n1b["script_sha256"], S / "run_n1b_q5_vectorized_loads.py")
    selected = n1b["selected"]
    n1b_test = n1b["test"]
    checks["n1b_exact_selection_and_pass"] = (
        selected == "aligned32x2" and n1b["correctness"][selected]["bitwise_equal"]
        and n1b["correctness"][selected]["different"] == 0 and n1b["overall_pass"]
    )
    checks["n1b_test_ratio_arithmetic"] = (
        close(n1b_test["p50_ratio"], n1b_test["measurements"][selected]["stats"]["p50"] / n1b_test["measurements"]["baseline"]["stats"]["p50"])
        and close(n1b_test["p95_ratio"], n1b_test["measurements"][selected]["stats"]["p95"] / n1b_test["measurements"]["baseline"]["stats"]["p95"])
    )

    n1bi = load("n1bi_q5_vectorload_end_to_end.json")
    checks["n1bi_paired_arithmetic_exact_verdict"] = paired_result_audit(n1bi, 16, False)
    checks["n1bi_dependency_hashes"] = (
        n1bi["inputs"]["n1b_sha256"] == sha256(R / "n1b_q5_vectorized_loads.json")
        and n1bi["inputs"]["p7_test_sha256"] == sha256(p7_test)
        and n1bi["inputs"]["preregistration_sha256"] == sha256(R / "N1BI_Q5_VECTORLOAD_END_TO_END_PREREGISTRATION.md")
    )

    n1c = load("n1c_generalized_exact_reduction_autotuner.json")
    hash_claim(checks, "n1c_prereg_hash", n1c["preregistration_sha256"], R / "N1C_GENERALIZED_EXACT_REDUCTION_AUTOTUNER_PREREGISTRATION.md")
    hash_claim(checks, "n1c_script_hash", n1c["script_sha256"], S / "run_n1c_generalized_exact_reduction_autotuner.py")
    checks["n1c_q8_bank_provenance"] = (
        n1c["inputs"]["q8_manifest_sha256"] == sha256(manifest_path)
        and n1c["inputs"]["q8_pinned_aggregate_sha256"] == q8_audit["aggregate_sha256"]
    )
    expected_graph = {"q8": {"head": 16, "k": 64, "o": 16, "q": 16, "router": 64, "v": 64}, "q5": {"gate_up": 8, "down": 8}}
    checks["n1c_all_widths_and_graph_exact"] = (
        n1c["widths"] == [4, 8, 16, 32, 64]
        and all(row["bitwise_equal"] and row["different"] == 0 for bank in n1c["correctness"].values() for row in bank.values())
        and all(row["bitwise_equal"] and row["different"] == 0 for row in n1c["graph_correctness"].values())
        and n1c["selected"] == expected_graph
    )
    checks["n1c_test_arithmetic_and_pass"] = n1c["overall_pass"] and all(
        close(n1c["test"][bank]["p50_ratio"], n1c["test"][bank]["candidate"]["p50"] / n1c["test"][bank]["baseline"]["p50"])
        and close(n1c["test"][bank]["p95_ratio"], n1c["test"][bank]["candidate"]["p95"] / n1c["test"][bank]["baseline"]["p95"])
        and n1c["test"][bank]["pass"]
        for bank in ("q8", "q5")
    )

    n1c2 = load("n1c2_generalized_reduction_end_to_end.json")
    checks["n1c2_paired_arithmetic_exact_verdict"] = paired_result_audit(n1c2, 16, True)
    checks["n1c2_configuration_and_dependency"] = (
        n1c2["configuration"] == expected_graph
        and n1c2["inputs"]["n1c_sha256"] == sha256(R / "n1c_generalized_exact_reduction_autotuner.json")
        and n1c2["inputs"]["p7_test_sha256"] == sha256(p7_test)
        and n1c2["inputs"]["preregistration_sha256"] == sha256(R / "N1C2_GENERALIZED_REDUCTION_END_TO_END_PREREGISTRATION.md")
        and n1c2["inputs"]["script_sha256"] == sha256(S / "run_n1c2_generalized_reduction_end_to_end.py")
    )

    n2a = load("n2a_temporal_ervf_oracle.json")
    checks["n2a_provenance"] = (
        n2a["inputs"]["preregistration_sha256"] == sha256(R / "N2A_TEMPORAL_ERVF_ORACLE_PREREGISTRATION.md")
        and n2a["inputs"]["q8_manifest_sha256"] == sha256(manifest_path)
        and n2a["inputs"]["q8_aggregate_sha256"] == q8_audit["aggregate_sha256"]
    )
    checks["n2a_all_sizes_exact"] = all(
        row["bitwise_equal"] and row["different"] == 0
        for size in n2a["correctness"].values() for row in size.values()
    )
    temporal_p50 = n2a["test"]["q8"]["temporal"]["p50"] + n2a["test"]["q5"]["temporal"]["p50"]
    sequential_p50 = n2a["test"]["q8"]["sequential"]["p50"] + n2a["test"]["q5"]["sequential"]["p50"]
    temporal_p95 = n2a["test"]["q8"]["temporal"]["p95"] + n2a["test"]["q5"]["temporal"]["p95"]
    sequential_p95 = n2a["test"]["q8"]["sequential"]["p95"] + n2a["test"]["q5"]["sequential"]["p95"]
    checks["n2a_test_arithmetic_and_pass"] = (
        close(n2a["test"]["combined"]["p50_ratio"], temporal_p50 / sequential_p50)
        and close(n2a["test"]["combined"]["p95_ratio"], temporal_p95 / sequential_p95)
        and n2a["test"]["combined"]["pass"] and n2a["overall_pass"]
    )

    n2au = load("n2au_route_union.json")
    route_hashes_ok = len(n2au["inputs"]["route_hashes"]) == 48 and all(
        n2au["inputs"]["route_hashes"][str(layer)] == sha256(ROUTES / f"layer_{layer:02d}.safetensors")
        for layer in range(48)
    )
    checks["n2au_provenance"] = (
        n2au["inputs"]["preregistration_sha256"] == sha256(R / "N2AU_ROUTE_UNION_PREREGISTRATION.md")
        and n2au["inputs"]["capture_sha256"] == sha256(capture)
        and n2au["inputs"]["n2a_sha256"] == sha256(R / "n2a_temporal_ervf_oracle.json")
        and route_hashes_ok
    )
    projection = n2au["projection"]
    byte_linear = projection["same8_temporal_q5_p50_ms"] * projection["s4_mean_union"] / 8.0
    checks["n2au_projection_arithmetic_and_pass"] = (
        close(projection["byte_linear_q5_p50_ms"], byte_linear)
        and close(projection["byte_linear_combined_ratio"], (projection["temporal_q8_p50_ms"] + byte_linear) / projection["sequential_q8_q5_p50_ms"])
        and n2au["aggregate"]["4"]["samples"] == 5 * 48 * (1024 // 4)
        and all(n2au["gates"].values()) and n2au["overall_pass"]
    )

    n2as = load("n2as_sparse_temporal_q5.json")
    checks["n2as_provenance"] = (
        n2as["inputs"]["preregistration_sha256"] == sha256(R / "N2AS_SPARSE_TEMPORAL_Q5_PREREGISTRATION.md")
        and n2as["inputs"]["route_capture_sha256"] == sha256(capture)
    )
    checks["n2as_exact_validation_negative"] = (
        n2as["validation_correctness"]["bitwise_equal"]
        and n2as["validation_correctness"]["different"] == 0
        and close(n2as["validation"]["p50_ratio"], n2as["validation"]["candidate"]["stats"]["p50"] / n2as["validation"]["reference"]["stats"]["p50"])
        and not n2as["test_opened"] and n2as["test"] is None and not n2as["overall_pass"]
    )

    n2b = load("n2b_certified_lm_head_oracle_validation.json")
    checks["n2b_provenance"] = (
        n2b["inputs"]["preregistration_sha256"] == sha256(R / "N2B_CERTIFIED_LM_HEAD_ORACLE_PREREGISTRATION.md")
        and n2b["inputs"]["manifest_sha256"] == sha256(manifest_path)
        and n2b["inputs"]["data_sha256"] == sha256(data)
    )
    n2b_summary = n2b["summary"]
    checks["n2b_exact_certification_but_zero_skip"] = (
        n2b_summary["tokens"] == 1270 and n2b_summary["gates"]["all_certified"]
        and n2b_summary["gates"]["all_best_seen_exact"] and n2b_summary["gates"]["runtime_argmax_exact"]
        and n2b_summary["skip_fraction"]["p50"] == 0.0
        and not n2b_summary["gates"]["median_skip_ge_60pct"] and not n2b_summary["pass"]
    )

    n2c_path = R / "n2c_batch_route_union_sweep.json"
    n2c = json.loads(n2c_path.read_text(encoding="utf-8"))
    completed = n2c.get("completed_sizes", [int(size) for size in n2c.get("results", {}) if size != "16"])
    n2c_rows_ok = True
    for size in completed:
        row = n2c["results"][str(size)]
        n2c_rows_ok &= row["q8_correctness"]["bitwise_equal"] and row["q8_correctness"]["different"] == 0
        n2c_rows_ok &= row["q5_correctness"]["bitwise_equal"] and row["q5_correctness"]["different"] == 0
        for part in ("q5", "combined"):
            measured = row["validation"][part]
            n2c_rows_ok &= close(measured["p50_ratio"], measured["candidate"]["stats"]["p50"] / measured["reference"]["stats"]["p50"])
            n2c_rows_ok &= close(measured["p95_ratio"], measured["candidate"]["stats"]["p95"] / measured["reference"]["stats"]["p95"])
    checks["n2c_completed_size_arithmetic_exact"] = bool(completed) and n2c_rows_ok
    checks["n2c_s16_resource_boundary_explicit"] = n2c["results"]["16"]["status"] == "blocked_by_resource_spill_timeout" and not n2c["results"]["16"]["test_opened"]
    if n2c["kind"].endswith("_checkpoint"):
        checks["n2c_checkpoint_truthful"] = set(completed) == {
            int(size) for size, row in n2c["results"].items() if row.get("validation") is not None
        }
    else:
        inputs = n2c["inputs"]
        checks["n2c_final_provenance"] = (
            inputs["preregistration_sha256"] == sha256(R / "N2C_BATCH_ROUTE_UNION_SWEEP_PREREGISTRATION.md")
            and inputs["script_sha256"] == sha256(S / "run_n2c_batch_route_union_sweep.py")
            and inputs["route_capture_sha256"] == sha256(capture)
            and inputs["n1b_sha256"] == sha256(R / "n1b_q5_vectorized_loads.json")
            and inputs["q8_manifest_sha256"] == sha256(manifest_path)
            and inputs["q8_pinned_aggregate_sha256"] == q8_audit["aggregate_sha256"]
        )
        checks["n2c_final_scope_truthful"] = bool(not n2c["sweep_complete"] and n2c["incomplete_reason"])

    n2d = load("n2d_greedy_lm_head_compile.json")
    checks["n2d_compile_only_provenance"] = (
        n2d["inputs"]["preregistration_sha256"] == sha256(R / "N2D_GREEDY_LM_HEAD_WRITE_ELISION_PREREGISTRATION.md")
        and n2d["inputs"]["script_sha256"] == sha256(S / "run_n2d_greedy_lm_head_write_elision.py")
        and n2d["inputs"]["manifest_sha256"] == sha256(manifest_path)
        and n2d["status"] == "compile_pass_timing_sealed"
        and not n2d["gpu_kernel_launched"] and not n2d["head_loaded"] and not n2d["timing_partition_opened"]
    )
    n2dv = load("n2d_greedy_lm_head_validation.json")
    n2da = load("n2d_greedy_lm_head_audit.json")
    head_path = ROOT / n2dv["inputs"]["head_artifact"]
    checks["n2d_validation_provenance"] = (
        n2dv["inputs"]["preregistration_sha256"] == sha256(R / "N2D_GREEDY_LM_HEAD_WRITE_ELISION_PREREGISTRATION.md")
        and n2dv["inputs"]["script_sha256"] == sha256(S / "run_n2d_greedy_lm_head_write_elision.py")
        and n2dv["inputs"]["manifest_sha256"] == sha256(manifest_path)
        and n2dv["inputs"]["compile_result_sha256"] == sha256(R / "n2d_greedy_lm_head_compile.json")
        and n2dv["inputs"]["head_sha256"] == sha256(head_path)
    )
    correctness = n2dv["correctness"]
    output = n2dv["output_bytes"]
    ratios = n2dv["ratios"]
    checks["n2d_validation_exact_bytes_arithmetic_negative"] = (
        len(correctness) == 17
        and all(row["all_indices_exact"] and row["all_values_exact"] and row["finite"] for row in correctness)
        and correctness[0]["input"] == "all_zero_tie" and correctness[0]["numpy_argmax"] == 0
        and output["c_bytes_saved_vs_a"] == output["a_current"] - output["c_fused_candidates"]
        and close(output["c_fraction_eliminated_vs_a"], output["c_bytes_saved_vs_a"] / output["a_current"])
        and close(ratios["c_over_a_p50"], n2dv["timing_ms"]["c_fused_candidates"]["p50"] / n2dv["timing_ms"]["a_current"]["p50"])
        and close(ratios["c_over_a_p95"], n2dv["timing_ms"]["c_fused_candidates"]["p95"] / n2dv["timing_ms"]["a_current"]["p95"])
        and not n2dv["overall_pass"] and not n2dv["test_authorized"]
    )
    checks["n2d_independent_audit_provenance_and_pass"] = (
        n2da["inputs"]["preregistration_sha256"] == sha256(R / "N2D_GREEDY_LM_HEAD_WRITE_ELISION_PREREGISTRATION.md")
        and n2da["inputs"]["evaluator_sha256"] == sha256(S / "run_n2d_greedy_lm_head_write_elision.py")
        and n2da["inputs"]["compile_sha256"] == sha256(R / "n2d_greedy_lm_head_compile.json")
        and n2da["inputs"]["validation_sha256"] == sha256(R / "n2d_greedy_lm_head_validation.json")
        and n2da["passed_checks"] == n2da["total_checks"] == len(n2da["checks"])
        and all(n2da["checks"].values()) and n2da["overall_pass"]
    )

    n2cv = load("n2c_batch_route_union_verification.json")
    checks["n2c_companion_verifier_pass"] = (
        n2cv["status"] == "pass" and n2cv["checks_passed"] == n2cv["checks_total"] == len(n2cv["checks"])
        and all(row["pass"] for row in n2cv["checks"])
    )

    n3a = load("n3a_moe_output_flow_fusion.json")
    checks["n3a_provenance"] = n3a["inputs"]["preregistration_sha256"] == sha256(R / "N3A_MOE_OUTPUT_FLOW_FUSION_PREREGISTRATION.md")
    checks["n3a_exact_negative_arithmetic"] = (
        n3a["correctness"]["bitwise_equal"] and n3a["correctness"]["different"] == 0
        and close(n3a["validation"]["p50_ratio"], n3a["validation"]["candidate"]["stats"]["p50"] / n3a["validation"]["reference"]["stats"]["p50"])
        and close(n3a["validation"]["p95_ratio"], n3a["validation"]["candidate"]["stats"]["p95"] / n3a["validation"]["reference"]["stats"]["p95"])
        and not n3a["test_opened"] and n3a["test"] is None and not n3a["overall_pass"]
    )

    n3a2 = load("n3a2_attention_projection_flow.json")
    checks["n3a2_provenance"] = (
        n3a2["inputs"]["preregistration_sha256"] == sha256(R / "N3A2_ATTENTION_PROJECTION_FLOW_PREREGISTRATION.md")
        and n3a2["inputs"]["script_sha256"] == sha256(S / "run_n3a2_attention_projection_flow.py")
        and n3a2["inputs"]["bank_sha256"] == sha256(manifest_path)
        and n3a2["inputs"]["q8_pinned_aggregate_sha256"] == q8_audit["aggregate_sha256"]
    )
    expected_outputs = n3a2["inputs"]["layers"] * (4096 + 512 + 512)
    expected_kv = n3a2["inputs"]["layers"] * 2 * 4 * 128
    n3a2_exact = all(
        row["outputs"]["bitwise_equal"]
        and row["outputs"]["different"] == 0
        and row["outputs"]["elements"] == expected_outputs
        and row["outputs"]["finite"]
        and row["kv_bitwise_equal"]
        and row["kv_different"] == 0
        and row["kv_elements"] == expected_kv
        for row in n3a2["correctness"].values()
    )
    eligible = [name for name, row in n3a2["correctness"].items() if row["outputs"]["bitwise_equal"] and row["kv_bitwise_equal"]]
    expected_selected = min(eligible, key=lambda name: n3a2["validation"][name]["stats"]["p50"])
    checks["n3a2_exact_counts_and_selection"] = (
        n3a2_exact
        and n3a2["selected"] == expected_selected == "concat_qkv"
        and close(
            n3a2["validation_p50_ratio"],
            n3a2["validation"][expected_selected]["stats"]["p50"] / n3a2["validation"]["baseline"]["stats"]["p50"],
        )
        and n3a2["test_opened"] == (n3a2["validation_p50_ratio"] <= 0.98)
    )
    n3a2_test = n3a2["test"]
    n3a2_test_correct = n3a2["test_correctness"]
    n3a2_baseline = n3a2_test["measurements"]["baseline"]["stats"]
    n3a2_candidate = n3a2_test["measurements"][expected_selected]["stats"]
    expected_n3a2_test_pass = (
        n3a2_test["p50_ratio"] <= 0.97
        and n3a2_test["p95_ratio"] <= 1.00
        and n3a2_test_correct["outputs"]["bitwise_equal"]
        and n3a2_test_correct["kv_bitwise_equal"]
    )
    checks["n3a2_test_arithmetic_exact_gates"] = (
        n3a2_test_correct["outputs"]["elements"] == expected_outputs
        and n3a2_test_correct["outputs"]["different"] == 0
        and n3a2_test_correct["kv_elements"] == expected_kv
        and n3a2_test_correct["kv_different"] == 0
        and close(n3a2_test["p50_ratio"], n3a2_candidate["p50"] / n3a2_baseline["p50"])
        and close(n3a2_test["p95_ratio"], n3a2_candidate["p95"] / n3a2_baseline["p95"])
        and close(n3a2_test["speedup_p50"], 1.0 / n3a2_test["p50_ratio"])
        and n3a2_test["pass"] == expected_n3a2_test_pass
        and n3a2["overall_pass"] == expected_n3a2_test_pass
    )
    n3a2v = load("n3a2_attention_projection_flow_verification.json")
    checks["n3a2_companion_verifier_pass"] = (
        n3a2v["status"] == "pass"
        and n3a2v["checks_passed"] == n3a2v["checks_total"] == len(n3a2v["checks"]) == 12
        and all(row["pass"] for row in n3a2v["checks"])
        and "245,760" in n3a2v["preregistration_count_erratum"]
    )

    n3a3 = load("n3a3_concat_qkv_end_to_end.json")
    checks["n3a3_provenance_and_configuration"] = (
        n3a3["inputs"]["preregistration_sha256"] == sha256(R / "N3A3_CONCAT_QKV_END_TO_END_PREREGISTRATION.md")
        and n3a3["inputs"]["script_sha256"] == sha256(S / "run_n3a3_concat_qkv_end_to_end.py")
        and n3a3["inputs"]["n3a2_sha256"] == sha256(R / "n3a2_attention_projection_flow.json")
        and n3a3["inputs"]["p7_test_sha256"] == sha256(p7_test)
        and n3a3["configuration"] == {"candidate": "concat_qkv", "changed_launches_per_layer": "3_to_1"}
    )
    checks["n3a3_paired_arithmetic_exact_verdict"] = paired_result_audit(n3a3, 16, True)
    expected_n3a3_gates = {
        "tokens_128": len(n3a3["pairs"]) == 128,
        "warmup_16": len(n3a3["pairs"][16:]) == 112,
        **n3a3["exactness"],
        "mean_ratio_le_0_98": n3a3["ratios"]["mean"] <= 0.98,
        "p50_ratio_le_0_98": n3a3["ratios"]["p50"] <= 0.98,
        "p95_ratio_le_1_00": n3a3["ratios"]["p95"] <= 1.00,
    }
    checks["n3a3_gate_map_and_negative_verdict"] = (
        n3a3["gates"] == expected_n3a3_gates
        and n3a3["overall_pass"] == all(expected_n3a3_gates.values())
        and not n3a3["overall_pass"]
    )
    n3a3v = load("n3a3_concat_qkv_end_to_end_verification.json")
    checks["n3a3_companion_verifier_pass"] = (
        n3a3v["status"] == "pass"
        and n3a3v["checks_passed"] == n3a3v["checks_total"] == len(n3a3v["checks"]) == 17
        and all(row["pass"] for row in n3a3v["checks"])
    )

    n3a4 = load("n3a4_o_residual_fusion.json")
    checks["n3a4_provenance"] = (
        n3a4["inputs"]["preregistration_sha256"] == sha256(R / "N3A4_O_RESIDUAL_FUSION_PREREGISTRATION.md")
        and n3a4["inputs"]["script_sha256"] == sha256(S / "run_n3a4_o_residual_fusion.py")
        and n3a4["inputs"]["bank_sha256"] == sha256(manifest_path)
        and n3a4["inputs"]["q8_pinned_aggregate_sha256"] == q8_audit["aggregate_sha256"]
        and n3a4["inputs"]["layers"] == n3a4["inputs"]["o_records"] == 48
    )
    n3a4_validation = n3a4["validation"]
    expected_n3a4_p50 = n3a4_validation["candidate"]["stats"]["p50"] / n3a4_validation["baseline"]["stats"]["p50"]
    expected_n3a4_p95 = n3a4_validation["candidate"]["stats"]["p95"] / n3a4_validation["baseline"]["stats"]["p95"]
    n3a4_correct = n3a4["validation_correctness"]
    expected_n3a4_open = n3a4_correct["bitwise_equal"] and expected_n3a4_p50 <= 0.98
    checks["n3a4_exact_arithmetic_and_negative_gate"] = (
        n3a4_correct["bitwise_equal"]
        and n3a4_correct["elements"] == 48 * 2048 == 98_304
        and n3a4_correct["different"] == 0
        and n3a4_correct["max_abs"] == 0.0
        and n3a4_correct["finite"]
        and close(n3a4_validation["p50_ratio"], expected_n3a4_p50)
        and close(n3a4_validation["p95_ratio"], expected_n3a4_p95)
        and n3a4["test_opened"] == expected_n3a4_open == False
        and n3a4["test_correctness"] is None
        and n3a4["test"] is None
        and not n3a4["overall_pass"]
    )
    n3a4v = load("n3a4_o_residual_fusion_verification.json")
    checks["n3a4_companion_verifier_pass"] = (
        n3a4v["status"] == "pass"
        and n3a4v["checks_passed"] == n3a4v["checks_total"] == len(n3a4v["checks"]) == 10
        and all(row["pass"] for row in n3a4v["checks"])
    )

    n3b = load("n3b_vram_kv_cache_pareto.json")
    checks["n3b_provenance"] = (
        n3b["inputs"]["preregistration_sha256"] == sha256(R / "N3B_VRAM_KV_CACHE_PARETO_PREREGISTRATION.md")
        and n3b["inputs"]["physical_sha256"] == sha256(R / "p7c_ervf_end_to_end_smoke.json")
        and n3b["inputs"]["route_capture_sha256"] == sha256(capture)
        and all(n3b["inputs"]["route_hashes"][str(layer)] == sha256(ROUTES / f"layer_{layer:02d}.safetensors") for layer in range(48))
    )
    n3b_arithmetic = True
    for context_text, row in n3b["contexts"].items():
        context = int(context_text)
        kv = 48 * 2 * 4 * context * 128 * 2
        budget = n3b["inputs"]["free_before_bytes"] - n3b["inputs"]["trunk_device_bytes"] - n3b["inputs"]["reserve_bytes"] - kv
        slots = max(0, budget // n3b["inputs"]["expert_bytes"])
        n3b_arithmetic &= row["kv_bytes"] == kv and row["cache_budget_bytes"] == budget and row["total_slots"] == slots
        n3b_arithmetic &= row["layer_capacity_min"] == slots // 48 and row["layer_capacity_max"] == math.ceil(slots / 48)
        if "test" in row:
            n3b_arithmetic &= close(row["test"]["miss_ratio"], row["test"]["misses"] / row["test"]["accesses"])
    checks["n3b_capacity_arithmetic"] = n3b_arithmetic
    checks["n3b_declared_gates_and_scope"] = (
        n3b["gates"]["allocation_exact"] and n3b["gates"]["4k_minimum_top8"]
        and n3b["gates"]["8k_minimum_top8"] and not n3b["gates"]["32k_static20_compatible"]
        and n3b["overall_pass"]
    )

    n3d = load("n3d_sequential_prefill_baseline.json")
    n3d_arithmetic = n3d["source_sha256"] == sha256(p13c) and n3d["source_tokens"] == 10000
    for cycle in ("first_4k_cycle", "second_4k_cycle"):
        for count_text, row in n3d[cycle].items():
            count = int(count_text)
            n3d_arithmetic &= close(row["effective_input_tokens_per_second"], count * 1000.0 / row["wall_ms"])
            n3d_arithmetic &= close(row["per_token_mean_ms"], row["wall_ms"] / count)
    n3d_arithmetic &= close(n3d["service_ready_ttft_ms"], n3d["first_4k_cycle"][str(n3d["prompt_tokens"])]["wall_ms"])
    n3d_arithmetic &= close(n3d["activation_plus_ttft_ms"], n3d["domain_activation_ms"] + n3d["service_ready_ttft_ms"])
    checks["n3d_prefill_baseline_arithmetic_and_source"] = n3d_arithmetic

    n4a = load("n4a_synthetic_80b_shape_capacity.json")
    checks["n4a_provenance"] = (
        n4a["inputs"]["preregistration_sha256"] == sha256(R / "N4A_SYNTHETIC_80B_SHAPE_CAPACITY_PREREGISTRATION.md")
        and n4a["inputs"]["script_sha256"] == sha256(S / "run_n4a_synthetic_80b_shape_capacity.py")
        and n4a["inputs"]["p7c_sha256"] == sha256(p7_test)
        and n4a["inputs"]["metadata_only"] and not n4a["inputs"]["weight_payload_downloaded"]
    )
    architecture = n4a["architecture"]
    groups = n4a["shape_verification"]["groups"]
    tensor_sum = sum(row["tensors"] for row in groups.values())
    parameter_sum = sum(row["parameters"] for row in groups.values())
    q5 = n4a["q5_record_contract"]
    record = q5["gate"]
    q5_math = (
        record["weights"] == 2048 * 512
        and record["code_bytes"] == record["weights"] * 5 // 8
        and record["scale_bytes"] == record["weights"] // 128 * 2
        and record["record_bytes"] == math.ceil(record["payload_with_header_bytes"] / 4096) * 4096
        and q5["expert_record_bytes"] == 3 * record["record_bytes"]
    )
    expert = n4a["expert_accounting"]
    q5_math &= expert["full_q5_bank_bytes"] == q5["expert_record_bytes"] * 48 * (512 + 1)
    q5_math &= expert["routed_active_parameters_per_token"] == 48 * 10 * 3 * 2048 * 512
    q5_math &= expert["shared_active_parameters_per_token"] == 48 * 3 * 2048 * 512
    host = n4a["host_budget"]
    q5_math &= host["accounted_bytes"] == host["full_q5_bank_bytes"] + host["persistent_q8_embedding_bytes"] + host["eight_pinned_staging_windows_bytes"]
    q5_math &= host["accounted_plus_reserve_bytes"] == host["accounted_bytes"] + host["explicit_process_reserve_bytes"]
    checks["n4a_shape_parameter_q5_budget_arithmetic"] = (
        tensor_sum == architecture["official_tensor_count"] == 74391
        and parameter_sum == architecture["official_parameter_count"] == 79674391296
        and architecture["official_bf16_checkpoint_bytes"] == 2 * parameter_sum
        and groups["routed_experts"]["tensors"] == 48 * 512 * 3
        and groups["shared_experts"]["tensors"] == 48 * 3
        and q5_math
    )
    checks["n4a_all_cpu_shape_capacity_gates"] = all(n4a["gates"].values()) and n4a["overall_pass"] and n4a["status"] == "cpu_shape_capacity_pass_physical_performance_pending"

    n4b = load("n4b_synthetic_80b_gpu_shape.json")
    checks["n4b_dependency_provenance"] = (
        n4b["inputs"]["preregistration_sha256"] == sha256(R / "N4B_SYNTHETIC_80B_GPU_SHAPE_PREREGISTRATION.md")
        and n4b["inputs"]["n4a_sha256"] == sha256(R / "n4a_synthetic_80b_shape_capacity.json")
        and n4b["inputs"]["n1c_sha256"] == sha256(R / "n1c_generalized_exact_reduction_autotuner.json")
    )
    n4b_physical = n4b["physical"]
    expected_n4b_elements = (
        n4b_physical["layers"] * n4b_physical["active_experts"]
        * (2 * n4b_physical["intermediate"] + n4b_physical["hidden"])
    )
    checks["n4b_physical_shape_and_bank_arithmetic"] = (
        n4b_physical["layers"] == 48
        and n4b_physical["active_experts"] == 11
        and n4b_physical["hidden"] == 2048
        and n4b_physical["intermediate"] == 512
        and n4b_physical["q5_bank_bytes"] == q5["expert_record_bytes"] * 48 * 11
        and n4b_physical["q5_bank_bytes"] == 1_070_530_560
    )
    expected_widths = {"8", "16", "32"}
    n4b_correctness = n4b["correctness"]
    exact_widths = {
        width for width, row in n4b_correctness.items()
        if row["bitwise_equal"] and row["different"] == 0 and row["max_abs"] == 0.0 and row["finite"]
    }
    expected_n4b_width = min(
        (int(width) for width in exact_widths),
        key=lambda width: n4b["validation"][str(width)]["stats"]["p50"],
    )
    checks["n4b_all_widths_exact_counts_and_selection"] = (
        set(n4b_correctness) == exact_widths == expected_widths
        and all(row["elements"] == expected_n4b_elements for row in n4b_correctness.values())
        and n4b["selected_width"] == expected_n4b_width == 8
    )
    n4b_selected_key = str(n4b["selected_width"])
    n4b_expert = n4b["test"][n4b_selected_key]["stats"]
    dense = n4b["dense_projection"]
    source_bytes = n1c.get("physical", {}).get("q8_device_bytes", 1_248_931_840)
    expected_dense = n1c["test"]["q8"]["candidate"]["p95"] * n4a["device_budget"]["q8_device_shell_bytes"] / source_bytes
    checks["n4b_test_and_projection_arithmetic"] = (
        set(n4b["test"]) == {"16", n4b_selected_key}
        and all(len(row["event_ms"]) == 120 for row in n4b["test"].values())
        and n4b["expert_test_stats"] == n4b_expert
        and close(dense["n1c_q8_p95_ms"], n1c["test"]["q8"]["candidate"]["p95"])
        and dense["source_bytes"] == source_bytes
        and dense["official_80b_shell_bytes"] == n4a["device_budget"]["q8_device_shell_bytes"]
        and close(dense["byte_linear_p95_ms"], expected_dense)
        and close(dense["conservative_2x_p95_ms"], 2 * expected_dense)
        and close(dense["projected_total_p95_ms"], n4b_expert["p95"] + 2 * expected_dense)
    )
    expected_n4b_gates = {
        "all_widths_exact": exact_widths == expected_widths,
        "expert_p95_le_50ms": n4b_expert["p95"] <= 50,
        "dense_byte_linear_p95_le_40ms": expected_dense <= 40,
        "dense_2x_p95_le_40ms": 2 * expected_dense <= 40,
        "projected_total_p95_le_90ms": n4b_expert["p95"] + 2 * expected_dense <= 90,
        "n4a_host_le_58g": n4a["gates"]["host_with_1gib_reserve_le_58_gib"],
        "n4a_4k_cache": n4a["gates"]["4k_cache_at_least_32_per_layer"],
        "n4a_32k_cache": n4a["gates"]["32k_cache_at_least_32_per_layer"],
    }
    checks["n4b_gate_map_and_pass"] = (
        n4b["gates"] == expected_n4b_gates
        and n4b["overall_pass"] == all(expected_n4b_gates.values())
        and n4b["overall_pass"]
    )
    diagnostics["n4b_script_hash_recorded"] = "script_sha256" in n4b["inputs"]

    n4bv = load("n4b_synthetic_80b_gpu_shape_verification.json")
    checks["n4b_companion_provenance"] = (
        n4bv["inputs"]["preregistration_sha256"] == sha256(R / "N4B_SYNTHETIC_80B_GPU_SHAPE_PREREGISTRATION.md")
        and n4bv["inputs"]["evaluator_sha256_current"] == sha256(S / "run_n4b_synthetic_80b_gpu_shape.py")
        and n4bv["inputs"]["result_sha256"] == sha256(R / "n4b_synthetic_80b_gpu_shape.json")
        and n4bv["inputs"]["n4a_sha256"] == sha256(R / "n4a_synthetic_80b_shape_capacity.json")
        and n4bv["inputs"]["n1c_sha256"] == sha256(R / "n1c_generalized_exact_reduction_autotuner.json")
        and not n4bv["inputs"]["gpu_rerun_performed"]
    )
    n4bv_record = n4bv["recomputed_record_contract"]
    n4bv_output = n4bv["recomputed_output_contract"]
    checks["n4b_companion_contract_arithmetic"] = (
        n4bv_record["weights_per_matrix"] == 2048 * 512
        and n4bv_record["code_bytes"] == 2048 * 512 * 5 // 8
        and n4bv_record["scale_bytes"] == 2048 * 512 // 128 * 2
        and n4bv_record["matrix_record_bytes"] == 675_840
        and n4bv_record["expert_record_bytes"] == 3 * 675_840
        and n4bv_record["resident_slots"] == 48 * 11
        and n4bv_record["resident_bank_bytes"] == n4b_physical["q5_bank_bytes"]
        and n4bv_output["outputs_per_layer"] == 11 * (512 + 512 + 2048)
        and n4bv_output["outputs_all_layers"] == expected_n4b_elements
    )
    checks["n4b_companion_numeric_recalculation"] = (
        n4bv["reported_numerical_pass_confirmed"]
        and n4bv["reported_gate_recalculation"] == expected_n4b_gates
        and n4bv["recomputed_selection"]["selected_width"] == n4b["selected_width"]
        and n4bv["recomputed_selection"]["expert_test_stats"] == n4b["expert_test_stats"]
        and all(n4bv["transfer_aware_sensitivity"]["gates"].values())
    )
    n4bv_failed = {name for name, passed in n4bv["checks"].items() if not passed}
    n4bv_breach = n4bv["methodological_findings"]["silu_rounding_contract_breach"]
    checks["n4b_companion_methodological_negative_consistent"] = (
        n4bv["passed_checks"] == sum(n4bv["checks"].values()) == 19
        and n4bv["total_checks"] == len(n4bv["checks"]) == 22
        and n4bv_failed == {
            "evaluator_hash_recorded_in_result",
            "raw_width_outputs_or_digests_archived",
            "canonical_streamq5_swiglu_rounding_present",
        }
        and not n4bv["independent_verification_pass"]
        and n4bv["verdict"] == "reported_numeric_shape_timing_gates_recompute_but_independent_exact_port_gate_fails"
        and n4bv_breach["present"]
        and n4bv_breach["counterexample"]["different"]
    )
    diagnostics["n4b_independent_verification"] = {
        "reported_numeric_shape_timing_gates_recompute": n4bv["reported_numerical_pass_confirmed"],
        "independent_exact_port_gate_pass": n4bv["independent_verification_pass"],
        "failed_companion_checks": sorted(n4bv_failed),
        "silu_rounding_contract_breach": n4bv_breach["present"],
    }

    n4br = load("n4br_synthetic_80b_exact_replication.json")
    checks["n4br_provenance"] = (
        n4br["inputs"]["preregistration_sha256"] == sha256(R / "N4BR_SYNTHETIC_80B_EXACT_REPLICATION_PREREGISTRATION.md")
        and n4br["inputs"]["evaluator_sha256"] == sha256(S / "run_n4br_synthetic_80b_exact_replication.py")
        and n4br["inputs"]["n4b_sha256"] == sha256(R / "n4b_synthetic_80b_gpu_shape.json")
        and n4br["inputs"]["n4a_sha256"] == sha256(R / "n4a_synthetic_80b_shape_capacity.json")
        and n4br["inputs"]["n1c_sha256"] == sha256(R / "n1c_generalized_exact_reduction_autotuner.json")
        and bool(re.fullmatch(r"[0-9a-f]{64}", n4br["inputs"]["input_sha256"]))
    )
    n4br_physical = n4br["physical"]
    n4br_expected_elements = (
        n4br_physical["layers"] * n4br_physical["active_experts"]
        * (2 * n4br_physical["intermediate"] + n4br_physical["hidden"])
    )
    n4br_widths = {"8", "16", "32"}
    n4br_exact_widths = {
        width for width, row in n4br["correctness"].items()
        if row["bitwise_equal"] and row["different"] == 0 and row["max_abs"] == 0.0 and row["finite"]
        and row["elements"] == n4br_expected_elements
    }
    n4br_digests = n4br["output_digests"]
    checks["n4br_exact_shape_and_bound_digests"] = (
        n4br_physical["q5_bank_bytes"] == q5["expert_record_bytes"] * 48 * 11
        and n4br_expected_elements == 1_622_016
        and set(n4br["correctness"]) == n4br_exact_widths == n4br_widths
        and set(n4br_digests) == {"16_reference", "8", "16", "32"}
        and len(set(n4br_digests.values())) == 1
        and all(bool(re.fullmatch(r"[0-9a-f]{64}", digest)) for digest in n4br_digests.values())
    )
    n4br_selected = min(
        (int(width) for width in n4br_exact_widths),
        key=lambda width: n4br["validation"][str(width)]["stats"]["p50"],
    )
    n4br_expert = n4br["test"][str(n4br_selected)]["stats"]
    n4br_dense = n4br["dense_projection"]
    n4br_expected_dense = n1c["test"]["q8"]["candidate"]["p95"] * n4a["device_budget"]["q8_device_shell_bytes"] / source_bytes
    checks["n4br_selection_test_and_projection_arithmetic"] = (
        n4br["selected_width"] == n4br_selected == 8
        and set(n4br["test"]) == {"16", "8"}
        and all(len(row["event_ms"]) == 120 for row in n4br["test"].values())
        and n4br["expert_test_stats"] == n4br_expert
        and close(n4br_dense["n1c_q8_p95_ms"], n1c["test"]["q8"]["candidate"]["p95"])
        and n4br_dense["source_bytes"] == source_bytes
        and n4br_dense["official_80b_shell_bytes"] == n4a["device_budget"]["q8_device_shell_bytes"]
        and close(n4br_dense["byte_linear_p95_ms"], n4br_expected_dense)
        and close(n4br_dense["conservative_2x_p95_ms"], 2 * n4br_expected_dense)
        and close(n4br_dense["projected_total_p95_ms"], n4br_expert["p95"] + 2 * n4br_expected_dense)
    )
    expected_n4br_gates = {
        "all_widths_exact": n4br_exact_widths == n4br_widths,
        "expert_p95_le_50ms": n4br_expert["p95"] <= 50,
        "dense_byte_linear_p95_le_40ms": n4br_expected_dense <= 40,
        "dense_2x_p95_le_40ms": 2 * n4br_expected_dense <= 40,
        "projected_total_p95_le_90ms": n4br_expert["p95"] + 2 * n4br_expected_dense <= 90,
        "n4a_host_le_58g": n4a["gates"]["host_with_1gib_reserve_le_58_gib"],
        "n4a_4k_cache": n4a["gates"]["4k_cache_at_least_32_per_layer"],
        "n4a_32k_cache": n4a["gates"]["32k_cache_at_least_32_per_layer"],
        "all_output_digests_equal": len(set(n4br_digests.values())) == 1,
    }
    checks["n4br_gate_map_and_pass"] = (
        n4br["gates"] == expected_n4br_gates
        and n4br["overall_pass"] == all(expected_n4br_gates.values())
        and n4br["overall_pass"]
    )

    n4brv = load("n4br_synthetic_80b_exact_replication_verification.json")
    checks["n4br_companion_provenance"] = (
        n4brv["inputs"]["preregistration_sha256"] == sha256(R / "N4BR_SYNTHETIC_80B_EXACT_REPLICATION_PREREGISTRATION.md")
        and n4brv["inputs"]["evaluator_sha256"] == sha256(S / "run_n4br_synthetic_80b_exact_replication.py")
        and n4brv["inputs"]["result_sha256"] == sha256(R / "n4br_synthetic_80b_exact_replication.json")
        and n4brv["inputs"]["n4b_sha256"] == sha256(R / "n4b_synthetic_80b_gpu_shape.json")
        and n4brv["inputs"]["n4a_sha256"] == sha256(R / "n4a_synthetic_80b_shape_capacity.json")
        and n4brv["inputs"]["n1c_sha256"] == sha256(R / "n1c_generalized_exact_reduction_autotuner.json")
        and not n4brv["inputs"]["gpu_rerun_performed"]
    )
    checks["n4br_companion_independent_digest_reconstruction"] = (
        n4brv["output_contract"]["independent_input_sha256"] == n4br["inputs"]["input_sha256"]
        and n4brv["output_contract"]["independent_output_sha256"] == n4br_digests["16_reference"]
        and n4brv["output_contract"]["outputs_all_layers"] == n4br_expected_elements
        and n4brv["checks"]["input_digest_independently_recomputed"]
        and n4brv["checks"]["output_digest_independently_recomputed"]
        and n4brv["checks"]["canonical_two_stage_swiglu_present"]
        and n4brv["checks"]["old_noncanonical_swiglu_absent"]
    )
    checks["n4br_companion_arithmetic_and_gates"] = (
        n4brv["record_contract"]["bank_bytes"] == n4br_physical["q5_bank_bytes"]
        and n4brv["selection"]["selected_width"] == n4br["selected_width"]
        and n4brv["selection"]["expert_test_stats"] == n4br["expert_test_stats"]
        and n4brv["recomputed_gates"] == expected_n4br_gates
        and all(n4brv["transfer_aware_sensitivity"]["gates"].values())
    )
    checks["n4br_companion_all_34_pass"] = (
        n4brv["passed_checks"] == n4brv["total_checks"] == len(n4brv["checks"]) == 34
        and all(n4brv["checks"].values())
        and n4brv["overall_pass"]
        and n4brv["verdict"] == "independently_verified_exact_synthetic_shape_timing_pass"
    )

    result = {
        "kind": "streamq5_moe_next_wave_independent_cpu_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256(Path(__file__)),
        "registry_sha256": sha256(REGISTRY),
        "execution": {"gpu_used": False, "gpu_modules_imported": False, "network_used": False},
        "coverage": {
            "registry_ids": ids,
            "result_files": [path.name for path in result_files],
            "q8_records_hashed": len(manifest["records"]),
            "route_files_hashed": 48,
        },
        "checks": checks,
        "diagnostics": diagnostics,
        "scientific_findings": {
            "n4b_reported_numeric_shape_timing_pass": n4bv["reported_numerical_pass_confirmed"],
            "n4b_independent_exact_port_gate_pass": n4bv["independent_verification_pass"],
            "n4b_canonical_swiglu_contract_breach": n4bv_breach["present"],
            "n4br_repaired_exact_component_pass": n4br["overall_pass"],
            "n4br_independent_exact_component_audit_pass": n4brv["overall_pass"],
        },
        "summary": {
            "passed": sum(checks.values()),
            "total": len(checks),
            "failed": [name for name, passed in checks.items() if not passed],
            "all_pass": all(checks.values()),
            "registry_all_local_testable_closed": not open_items,
        },
        "claim_boundary": (
            "CPU-only independent registry, evidence, hash and arithmetic audit. "
            "It re-verifies stored evidence but creates no new GPU, model-quality, cross-model, cross-GPU, novelty or SOTA evidence."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
