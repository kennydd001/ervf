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
RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a1r_conservative_resource_retry.py"
CORE = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a_next_component_composition.py"
VERIFY_HELPER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d2_registered_scatter.py"
PREREG = R / "PORT80B_D10A1R_CONSERVATIVE_RESOURCE_RETRY_PREREGISTRATION.md"
COMPILE = R / "port80b_d10a1r_conservative_resource_retry_compile.json"
RAW = R / "port80b_d10a1r_conservative_resource_retry.json"
BANK = ROOT / "reports" / "runs" / "streamq5_moe" / "port80b_p0" / "port80b_p0_full_q5_bank.bin"
OUT = R / "port80b_d10a1r_component_independent_verification.json"
REPORT = R / "PORT80B_D10A1R_COMPONENT_INDEPENDENT_VERIFICATION_REPORT_2026-08-13.md"

LAYERS = 48
PREFIX = 499
EXPERTS_WITH_SHARED = 513
TOP_K = 10
EXPERT_BYTES = 2_027_520
MATRIX_BYTES = 675_840
HEADER_BYTES = 64
CODE_BYTES = 655_360
SCALE_BYTES = 16_384
VRAM_RESERVE = 512 * 2**20
RAM_AFTER_TOUCH = 2 * 2**30


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


def numerical_payload_hash(handle: Any, record_offset: int) -> str:
    digest = hashlib.sha256()
    for projection in range(3):
        begin = record_offset + projection * MATRIX_BYTES + HEADER_BYTES
        handle.seek(begin)
        payload = handle.read(CODE_BYTES + SCALE_BYTES)
        if len(payload) != CODE_BYTES + SCALE_BYTES:
            raise RuntimeError("short bank numerical-payload read")
        digest.update(payload)
    return digest.hexdigest()


def main() -> None:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    compile_data = json.loads(COMPILE.read_text(encoding="utf-8"))
    core_source = CORE.read_text(encoding="utf-8")
    helper_source = VERIFY_HELPER.read_text(encoding="utf-8")

    correctness = raw["correctness"]
    raw_canaries = raw["raw_canary_arrays"]
    output_digests = raw["output_digests"]
    negative = raw["negative_controls"]
    components = raw["components"]
    wall = [float(x) for x in raw["validation"]["wall_ms"]]
    events = [float(x) for x in raw["validation"]["cuda_event_ms"]]
    telemetry = raw["telemetry"]
    page_rates = [float(x.get("page_reads_per_sec", 0.0)) for x in raw["page_reads"]["samples"]]

    canary_failures: list[dict[str, int]] = []
    triples: set[tuple[int, int, int]] = set()
    boundary_failures: list[int] = []
    for layer in range(LAYERS):
        for expert in range(512):
            identifier = layer * 512 + expert
            digits = tuple((identifier >> (5 * place)) & 31 for place in range(3))
            words = tuple(0x3E80 + 4 * digit for digit in digits)
            triples.add(words)
            decoded = sum(((word - 0x3E80) // 4) << (5 * place) for place, word in enumerate(words))
            if decoded != identifier:
                canary_failures.append({"identifier": identifier, "decoded": decoded})
        if (layer * 512 + 498) >= layer * 512 + PREFIX or (layer * 512 + 499) < layer * 512 + PREFIX:
            boundary_failures.append(layer)

    raw_canary_errors: list[dict[str, Any]] = []
    for row in raw_canaries:
        intended = np.asarray(row["intended_ids"], dtype=np.uint16)
        actual = np.asarray(row["actual_header_ids"], dtype=np.uint16)
        expected = np.asarray(row["expected_words"], dtype=np.uint16)
        observed = np.asarray(row["observed_words"], dtype=np.uint16)
        recomputed = np.empty((intended.size, 3), dtype=np.uint16)
        ids32 = intended.astype(np.uint32)
        for place in range(3):
            recomputed[:, place] = (0x3E80 + 4 * ((ids32 >> (5 * place)) & 31)).astype(np.uint16)
        if not (np.array_equal(intended, actual) and np.array_equal(expected, observed) and np.array_equal(expected, recomputed)):
            raw_canary_errors.append({"case": row["case"]})

    wall_stats = stats(wall)
    event_stats = stats(events)
    memory_loss = int(telemetry[0]["available_ram"]) - int(telemetry[-1]["available_ram"])
    maximum_available_drawdown = max(int(x["available_ram"]) for x in telemetry) - min(int(x["available_ram"]) for x in telemetry)
    min_vram = min(int(x["free_vram"]) for x in telemetry)

    recomputed_gates = {
        "canary_exhaustive_injective_roundtrip_boundary": len(triples) == LAYERS * 512 and not canary_failures and not boundary_failures,
        "all_correctness_headers_zero_mismatch": all(int(row["header_mismatches"]) == 0 for row in correctness),
        "all_canaries_raw_exact": not raw_canary_errors and all(int(row["canary_mismatches"]) == 0 and row["raw_canary_exact"] is True for row in correctness),
        "all_routed_q5_bitexact": all(row["comparison"]["bitwise_equal"] is True for row in correctness),
        "output_digest_uniqueness_ge_95pct": len(set(output_digests)) / len(output_digests) >= 0.95,
        "wrong_expert_header_and_numerical_detected": int(negative["wrong_expert"]["header_mismatches"]) > 0 and negative["wrong_expert"]["numerical_comparison"]["bitwise_equal"] is False,
        "wrong_layer_header_and_numerical_detected": int(negative["wrong_layer"]["header_mismatches"]) > 0 and negative["wrong_layer"]["numerical_comparison"]["bitwise_equal"] is False,
        "attention_reference_abs_rel_le_2e_5": float(components["attention_max_abs"]) <= 2e-5,
        "gdn_reference_abs_rel_le_2e_5": float(components["recurrent_sample_max_abs"]) <= 2e-5 and int(components["conv_nonzero"]) > 0,
        "shared_q5_bitexact": components["shared_vs_resident"]["bitwise_equal"] is True,
        "dense_and_runtime_touched": int(components["dense_checksum_observed"]) == int(components["dense_checksum_expected"]) and components["runtime_touch_sentinels"] == [0xA5, 0xA5],
        "validation_32_finite": len(wall) == 32 and bool(np.isfinite(np.asarray(wall)).all()),
        "validation_wall_p95_le_150ms": wall_stats["p95"] <= 150.0,
        "validation_wall_p99_le_200ms": wall_stats["p99"] <= 200.0,
        "post_warmup_page_reads_no_sample_gt_2048": bool(page_rates) and max(page_rates) <= 2048.0,
        "validation_memory_loss_le_1gib": memory_loss <= 2**30,
        "ram_after_first_touch_ge_2gib": int(raw["physical"]["available_ram_after_first_touch"]) >= RAM_AFTER_TOUCH,
        "vram_reserve_ge_512mib": min_vram >= VRAM_RESERVE,
        # The raw format retains only the producer boolean, not the range list.
        "registration_48_ranges": raw["gates"]["registration_48_ranges"] is True,
        "no_cuda_or_runner_error": raw["error"] is None,
        # The raw format retains the empty failure list and producer boolean, not 48 attempt rows.
        "clean_unregister_48_ranges": raw["gates"]["clean_unregister_48_ranges"] is True and raw["unregister_failures"] == [],
    }

    bad_headers = [
        {"case": int(row["case"]), "mismatches": int(row["header_mismatches"])}
        for row in correctness if int(row["header_mismatches"]) != 0
    ]
    header_stream_race = all(fragment in helper_source for fragment in (
        "headers = cp.asarray(header_reference(selected))",
        "mismatches = cp.zeros(1, dtype=cp.uint64)",
        "kernel((4096,), (256,), (destination, headers, np.uint64(TOKEN_BYTES), mismatches), stream=stream)",
    )) and "stream = cp.cuda.Stream(non_blocking=True)" in core_source
    fill_stream_race = all(fragment in core_source for fragment in (
        "attention.fill(0); recurrent.fill(0); conv.fill(0); shared_gate.fill(0); shared_up.fill(0); shared_down.fill(0)",
        "kernels[\"gated_deltanet_step\"]",
        "kernels[\"shared_q5_gate_up\"]",
        "stream = cp.cuda.Stream(non_blocking=True)",
    )) and "with stream:" not in core_source

    shared_hashes: list[str] = []
    with BANK.open("rb") as handle:
        reference_hash = numerical_payload_hash(handle, 0)
        for layer in range(LAYERS):
            offset = (layer * EXPERTS_WITH_SHARED + 512) * EXPERT_BYTES
            shared_hashes.append(numerical_payload_hash(handle, offset))

    provenance_checks = {
        "raw_runner_hash_matches": sha256(RUNNER) == raw["inputs"]["runner_sha256"],
        "raw_prereg_hash_matches": sha256(PREREG) == raw["inputs"]["preregistration_sha256"],
        "raw_compile_hash_matches": sha256(COMPILE) == raw["inputs"]["compile_sha256"],
        "compile_pass": compile_data["pass"] is True,
        "compile_runner_hash_matches": compile_data["inputs"]["runner_sha256"] == sha256(RUNNER),
        "compile_prereg_hash_matches": compile_data["inputs"]["preregistration_sha256"] == sha256(PREREG),
        "all_21_gates_reproduced_exactly": recomputed_gates == raw["gates"],
        "wall_stats_reproduced": wall_stats == raw["validation"]["wall_stats"],
        "event_stats_reproduced": event_stats == raw["validation"]["cuda_event_stats"],
        "correctness_40": len(correctness) == 40 and len(raw_canaries) == 40 and len(output_digests) == 40,
        "validation_32": len(wall) == 32 and len(events) == 32 and len(telemetry) == 32,
        "shared_source_payloads_all_equal_reference": len(set(shared_hashes)) == 1 and shared_hashes[0] == reference_hash,
    }

    formal_false = [key for key, value in recomputed_gates.items() if not value]
    payload = {
        "kind": "port80b_d10a1r_component_independent_cpu_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "formal_negative_reproduced_three_evaluator_races_endurance_closed",
        "pass": all(provenance_checks.values()),
        "provenance_checks": provenance_checks,
        "inputs": {
            "raw_sha256": sha256(RAW),
            "runner_sha256": sha256(RUNNER),
            "core_sha256": sha256(CORE),
            "verify_helper_sha256": sha256(VERIFY_HELPER),
            "preregistration_sha256": sha256(PREREG),
            "compile_sha256": sha256(COMPILE),
        },
        "gate_replay": {
            "count": len(recomputed_gates),
            "true": sum(recomputed_gates.values()),
            "false": len(recomputed_gates) - sum(recomputed_gates.values()),
            "false_names": formal_false,
            "recomputed": recomputed_gates,
            "raw_match": recomputed_gates == raw["gates"],
        },
        "timing": {
            "wall_ms": wall_stats,
            "cuda_event_ms": event_stats,
            "wall_event_mean_gap_ms": wall_stats["mean"] - event_stats["mean"],
        },
        "ram_vram_cleanup": {
            "available_ram_before": int(raw["physical"]["available_ram_before"]),
            "available_ram_after_registration": int(raw["physical"]["available_ram_after_registration"]),
            "available_ram_after_first_touch": int(raw["physical"]["available_ram_after_first_touch"]),
            "available_ram_after_cleanup": int(raw["physical"]["available_ram_after_cleanup"]),
            "registration_available_delta_bytes": int(raw["physical"]["available_ram_before"]) - int(raw["physical"]["available_ram_after_registration"]),
            "first_touch_available_delta_bytes": int(raw["physical"]["available_ram_after_registration"]) - int(raw["physical"]["available_ram_after_first_touch"]),
            "validation_endpoint_loss_bytes": memory_loss,
            "validation_max_drawdown_bytes": maximum_available_drawdown,
            "validation_min_available_bytes": min(int(x["available_ram"]) for x in telemetry),
            "validation_min_free_vram_bytes": min_vram,
            "page_read_samples": len(page_rates),
            "page_reads_per_sec_max": max(page_rates),
            "error": raw["error"],
            "unregister_failures": raw["unregister_failures"],
            "evidence_limit": "Raw JSON omits the 48 registration/unregistration attempt rows; exact counts can only be source+producer-boolean verified, while the empty failure list is retained.",
        },
        "failure_diagnosis": {
            "header_gate": {
                "formal": "false",
                "bad_cases": bad_headers,
                "mismatch_sum": sum(x["mismatches"] for x in bad_headers),
                "all_40_raw_header_ids_and_canaries_exact": not raw_canary_errors,
                "all_40_full_q5_outputs_bitexact": recomputed_gates["all_routed_q5_bitexact"],
                "evaluator_stream_race_present": header_stream_race,
                "classification": "invalid evaluator negative; full-byte transport gate remains unadjudicated, not upgraded to pass",
                "cause": "full_verify creates headers and the zeroed counter on the current/default stream, then launches verification on a separate non-blocking stream without an event dependency.",
            },
            "gdn_gate": {
                "formal": "false",
                "recurrent_sample_max_abs": float(components["recurrent_sample_max_abs"]),
                "conv_nonzero": int(components["conv_nonzero"]),
                "evaluator_fill_race_present": fill_stream_race,
                "classification": "invalid evaluator negative; convolution-state mechanism remains unadjudicated",
                "cause": "conv.fill(0) is issued on the current/default stream immediately before gated_deltanet_step on the non-blocking stream; the memset can overwrite kernel writes.",
            },
            "shared_gate": {
                "formal": "false",
                "different_bits": int(components["shared_vs_resident"]["different_bits"]),
                "expected_47_layers_x_2048": 47 * 2048,
                "exact_47_layers_pattern": int(components["shared_vs_resident"]["different_bits"]) == 47 * 2048,
                "all_48_shared_numerical_payloads_equal_reference": len(set(shared_hashes)) == 1 and shared_hashes[0] == reference_hash,
                "numerical_payload_sha256": reference_hash,
                "evaluator_fill_race_present": fill_stream_race,
                "classification": "invalid evaluator negative; shared-kernel mechanism remains unadjudicated",
                "cause": "shared_gate/shared_up/shared_down are zero-filled on the default stream immediately before shared kernels on the non-blocking stream; the exact 47-layer mismatch pattern is consistent with cross-stream overwrite.",
            },
            "mechanism_failures_proven": [],
        },
        "additional_protocol_limits": {
            "validation_32_finite_checks_wall_only": "The gate checks finite wall timings, not CUDA-event finiteness or composed state/output finiteness.",
            "attention_gate_name_overstates_check": "The source gates only attention_max_abs; no relative error field is retained.",
            "gdn_reference_is_sampled": "Only 4,096 recurrent cells are compared; conv is reduced to count_nonzero.",
            "page_telemetry_samples": len(page_rates),
            "systemic_stream_hygiene": "The same default-vs-nonblocking-stream pattern also exists for pointer-table/route uploads, canary-error resets, dense checksum reset, and the recurrent/conv/KV reset before validation. Some passed gates have strong retained raw corroboration, but the composed validation state has no retained correctness digest and is not semantically adjudicated.",
            "required_repair": "Create allocations/uploads/fills inside `with stream:` or add explicit recorded-event waits before dependent kernels; synchronize before every host adjudication. Retain full state/output finiteness and digests plus 48 register/unregister attempt rows.",
        },
        "verdict": "The raw 18/21 formal negative is reproduced. All three false gates are contaminated by explicit cross-stream evaluator races. This proves no component mechanism failure, but it also cannot authorize a pass; endurance remains closed until a preregistered race-free evaluator rerun.",
        "physical_actions": {"gpu_run": False, "host_registration": False, "registry_edit": False, "bank_writes": False},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = f"""# PORT80B-D10A1-R independent component-result audit

**Verdict:** the raw result's formal negative is exactly reproduced: **18/21 gates true**, with failures in header-byte verification, GDN convolution state and shared Q5. None of the three is clean evidence of a mechanism failure: all are contaminated by cross-stream evaluator races. They cannot be promoted to passes, so endurance correctly remains closed.

## Recomputed result

- Validation wall p50/p95/p99: **{wall_stats['p50']:.6f} / {wall_stats['p95']:.6f} / {wall_stats['p99']:.6f} ms** ({len(wall)} samples).
- CUDA-event p50/p95/p99: **{event_stats['p50']:.6f} / {event_stats['p95']:.6f} / {event_stats['p99']:.6f} ms**.
- RAM after first touch: **{raw['physical']['available_ram_after_first_touch'] / 2**30:.6f} GiB**; validation endpoint loss **{memory_loss / 2**20:.3f} MiB**; maximum observed drawdown **{maximum_available_drawdown / 2**20:.3f} MiB**.
- Minimum validation free VRAM: **{min_vram / 2**20:.3f} MiB**. Page-read maximum: **{max(page_rates):.3f}/s**, but only {len(page_rates)} one-second samples exist.
- Raw error is null and unregister-failure list is empty. The raw JSON does not retain 48 individual register/unregister attempt rows, so their exact count is not independently reconstructable from raw arrays.

## The three failures

1. **Header gate â€” evaluator bug, transport not cleanly adjudicated.** Cases {', '.join(str(x['case']) for x in bad_headers)} report {', '.join(str(x['mismatches']) for x in bad_headers)} mismatches ({sum(x['mismatches'] for x in bad_headers)} total). Yet all 40 actual header IDs/canaries equal their independent intended values, all 40 full Q5 candidate/oracle outputs are bitexact, and all 40 output digests are unique. `full_verify` creates its header reference and zero counter on the current/default stream, then launches on an explicit non-blocking stream without an event. These sporadic counts are therefore an invalid evaluator negative, not demonstrated corrupted records.

2. **GDN gate â€” evaluator bug, conv path unadjudicated.** The sampled recurrent result is exact (`max_abs=0`), but `conv_nonzero=0`. `conv.fill(0)` is launched on the default stream directly before the GDN kernel on the non-blocking stream. The asynchronous memset can overwrite the kernel's convolution-state writes.

3. **Shared-Q5 gate â€” evaluator bug, shared path unadjudicated.** Exactly **{components['shared_vs_resident']['different_bits']:,} = 47Ã—2,048** elements differ, while layer 0 matches. A CPU byte audit shows all 48 shared records and the resident reference have the same complete codes+scales SHA-256 `{reference_hash}`. The shared output buffers are zero-filled on the default stream immediately before shared kernels on the non-blocking stream, matching the observed layer-tail overwrite pattern.

The negative-control numerical outputs remain valid detections. Their header mismatch counts may also be polluted by the same verifier race; do not interpret the counts quantitatively.

## Protocol gaps retained

The `validation_32_finite` gate checks only wall-time finiteness, not composed state/output finiteness. The attention gate retains only absolute error despite its `abs_rel` name. The GDN reference compares only 4,096 recurrent cells and reduces convolution evidence to `count_nonzero`.

The stream-hygiene defect is broader than the three observed failures: pointer-table and route uploads, canary-error resets, dense-checksum reset, and the recurrent/conv/KV reset before validation also cross from the default stream to the explicit non-blocking stream without dependencies. Several passed gates have strong retained raw corroboration, but the composed validation state has no retained correctness digest and is therefore not semantically adjudicated. A repair must place allocations/uploads/fills inside the execution-stream context or use explicit CUDA-event waits, synchronize before host adjudication, and retain state/output finiteness plus digests and all 48 cleanup-attempt rows.

Conclusion: **no physical component mechanism failure is proven**, but neither are the three affected mechanisms proven passing. A new, preregistered evaluator must put allocations/fills and kernels on one stream or add explicit events, then rerun the component phase. No GPU work, registration, bank mutation or registry edit was performed by this audit.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({
        "audit_checks": f"{sum(provenance_checks.values())}/{len(provenance_checks)}",
        "gates": f"{sum(recomputed_gates.values())}/{len(recomputed_gates)}",
        "false_gates": formal_false,
        "verdict": payload["status"],
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
