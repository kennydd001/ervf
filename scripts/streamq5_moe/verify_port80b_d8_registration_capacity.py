from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
RUNS = ROOT / "reports/runs/streamq5_moe/port80b_p0"
PREREG = REPORTS / "PORT80B_D8_REGISTRATION_CAPACITY_KNEE_PREREGISTRATION.md"
RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_d8_registration_capacity_knee.py"
RESULT = REPORTS / "port80b_d8_registration_capacity_knee.json"
SOURCE_REPORT = REPORTS / "PORT80B_D8_REGISTRATION_CAPACITY_KNEE_REPORT_2026-08-12.md"
MANIFEST = RUNS / "port80b_p0_full_q5_bank_manifest.json"
BANK = RUNS / "port80b_p0_full_q5_bank.bin"
OUTPUT = REPORTS / "port80b_d8_registration_capacity_independent_verification.json"
REPORT = REPORTS / "PORT80B_D8_REGISTRATION_CAPACITY_INDEPENDENT_VERIFICATION_REPORT_2026-08-12.md"

LAYERS = 48
EXPERT_BYTES = 2_027_520
BANK_BYTES = 49_925_652_480
PREFIXES = (435, 461, 486, 499, 512)
ENTROPY_PIN_GIB = 41.441
EXPECTED_BANK_SHA256 = "4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462"


def sha256(path: Path, chunk_bytes: int = 64 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def all_bools(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return all(all_bools(item) for item in value.values())
    if isinstance(value, list):
        return all(all_bools(item) for item in value)
    return True


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    prereg_text = PREREG.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    source_report_text = SOURCE_REPORT.read_text(encoding="utf-8")

    hashes = {
        "preregistration_sha256": sha256(PREREG),
        "current_evaluator_sha256": sha256(RUNNER),
        "raw_result_sha256": sha256(RESULT),
        "source_report_sha256": sha256(SOURCE_REPORT),
        "manifest_sha256": sha256(MANIFEST),
        "bank_sha256_recomputed": sha256(BANK),
    }
    provenance_checks = {
        "preregistration_matches_raw_result": result["inputs"]["preregistration_sha256"] == hashes["preregistration_sha256"],
        "manifest_matches_raw_result": result["inputs"]["manifest_sha256"] == hashes["manifest_sha256"],
        "manifest_bank_sha": manifest["bank_sha256"] == EXPECTED_BANK_SHA256,
        "raw_result_bank_sha_from_manifest": result["inputs"]["bank_sha256_from_manifest"] == EXPECTED_BANK_SHA256,
        "full_bank_sha_recomputed": hashes["bank_sha256_recomputed"] == EXPECTED_BANK_SHA256,
        "bank_size": BANK.stat().st_size == BANK_BYTES,
    }

    rows: list[dict[str, Any]] = []
    for expected_prefix, raw in zip(PREFIXES, result["sweep"], strict=True):
        expected_bytes = LAYERS * expected_prefix * EXPERT_BYTES
        unregister_failure_count = len(raw["unregister_failures"])
        clean_success = (
            raw["experts_per_layer"] == expected_prefix
            and raw["registered_ranges"] == LAYERS
            and raw["success"] is True
            and unregister_failure_count == 0
        )
        rows.append({
            "experts_per_layer": expected_prefix,
            "registered_bytes_recomputed": expected_bytes,
            "registered_gib_recomputed": expected_bytes / 2**30,
            "fraction_recomputed": expected_prefix / 512,
            "raw_registration_success": raw["success"],
            "registered_ranges": raw["registered_ranges"],
            "unregister_failure_count": unregister_failure_count,
            "clean_registered_and_unregistered_success": clean_success,
            "arithmetic_checks": {
                "prefix": raw["experts_per_layer"] == expected_prefix,
                "registered_bytes": raw["registered_bytes"] == expected_bytes,
                "registered_gib": raw["registered_gib"] == expected_bytes / 2**30,
                "fraction": raw["fraction"] == expected_prefix / 512,
                "registered_ranges_in_bounds": 0 <= raw["registered_ranges"] <= LAYERS,
            },
            "available_ram_bytes": {
                "before": raw["available_before"],
                "after_registration": raw["available_after_registration"],
                "after_unregister": raw["available_after_unregister"],
                "change_registration": raw["available_after_registration"] - raw["available_before"],
                "change_full_row": raw["available_after_unregister"] - raw["available_before"],
            },
        })

    clean_rows = [row for row in rows if row["clean_registered_and_unregistered_success"]]
    largest_clean = max(clean_rows, key=lambda row: row["registered_bytes_recomputed"])
    raw_largest_bytes = int(result["largest_successful_registered_bytes"])
    entropy_bytes = int(ENTROPY_PIN_GIB * 2**30)
    ram_chain_checks = [
        rows[index + 1]["available_ram_bytes"]["before"] == rows[index]["available_ram_bytes"]["after_unregister"]
        for index in range(3)
    ]
    # There is a small unrecorded interval before 512, so it is reported separately.
    final_gap = rows[4]["available_ram_bytes"]["before"] - rows[3]["available_ram_bytes"]["after_unregister"]
    capacity = {
        "raw_largest_claim_bytes": raw_largest_bytes,
        "raw_largest_claim_gib": raw_largest_bytes / 2**30,
        "raw_largest_claim_is_protocol_valid": any(
            row["registered_bytes_recomputed"] == raw_largest_bytes
            and row["clean_registered_and_unregistered_success"]
            for row in rows
        ),
        "largest_clean_prefix_experts_per_layer": largest_clean["experts_per_layer"],
        "largest_clean_fraction": largest_clean["fraction_recomputed"],
        "largest_clean_registered_bytes": largest_clean["registered_bytes_recomputed"],
        "largest_clean_registered_gib": largest_clean["registered_gib_recomputed"],
        "entropy_pin_theoretical_gib": ENTROPY_PIN_GIB,
        "entropy_pin_theoretical_bytes_floor": entropy_bytes,
        "entropy_pin_below_clean_capacity": entropy_bytes <= largest_clean["registered_bytes_recomputed"],
        "entropy_pin_clean_capacity_margin_bytes": largest_clean["registered_bytes_recomputed"] - entropy_bytes,
        "entropy_pin_clean_capacity_margin_gib": largest_clean["registered_gib_recomputed"] - ENTROPY_PIN_GIB,
    }
    protocol_checks = {
        "five_prefixes_in_frozen_order": [row["experts_per_layer"] for row in rows] == list(PREFIXES),
        "all_row_arithmetic": all(all_bools(row["arithmetic_checks"]) for row in rows),
        "first_four_clean": all(row["clean_registered_and_unregistered_success"] for row in rows[:4]),
        "full_prefix_not_clean": rows[4]["clean_registered_and_unregistered_success"] is False,
        "full_prefix_has_exactly_44_raw_unregister_failures": rows[4]["unregister_failure_count"] == 44,
        "largest_clean_is_499": largest_clean["experts_per_layer"] == 499,
        "largest_clean_bytes_exact": largest_clean["registered_bytes_recomputed"] == 48_563_159_040,
        "largest_clean_gib_exact": largest_clean["registered_gib_recomputed"] == 45.22796630859375,
        "entropy_pin_below_clean_capacity": capacity["entropy_pin_below_clean_capacity"],
        "prereg_requires_immediate_unregister": "all ranges are immediately unregistered after each success" in prereg_text,
        "current_repaired_runner_uses_clean_success": "row[\"clean_success\"]" in runner_text and "if row[\"clean_success\"]" in runner_text,
    }
    artifact_findings = {
        "stored_evaluator_sha256": result["inputs"]["evaluator_sha256"],
        "current_evaluator_sha256": hashes["current_evaluator_sha256"],
        "current_evaluator_matches_raw_result": hashes["current_evaluator_sha256"] == result["inputs"]["evaluator_sha256"],
        "interpretation": "The current evaluator is a post-run repair with clean_success semantics. The exact source whose hash is stored in the raw result is not present at this path, so original-source replay is unavailable.",
        "source_report_says_48_unregister_failures": "all 48 unregister operations" in source_report_text,
        "raw_json_unregister_failure_count": rows[4]["unregister_failure_count"],
        "source_report_unregister_count_matches_raw_json": (
            ("all 48 unregister operations" not in source_report_text)
            if rows[4]["unregister_failure_count"] != 48
            else ("all 48 unregister operations" in source_report_text)
        ),
    }
    ram_caveat = {
        "first_four_before_values_chain_from_prior_after_unregister": all(ram_chain_checks),
        "before_512_minus_prior_after_unregister_bytes": final_gap,
        "available_before_first_gib": rows[0]["available_ram_bytes"]["before"] / 2**30,
        "available_after_clean_499_gib": rows[3]["available_ram_bytes"]["after_unregister"] / 2**30,
        "available_after_512_attempt_gib": rows[4]["available_ram_bytes"]["after_unregister"] / 2**30,
        "drop_first_before_to_after_clean_499_gib": (
            rows[0]["available_ram_bytes"]["before"] - rows[3]["available_ram_bytes"]["after_unregister"]
        ) / 2**30,
        "interpretation": "The sweep is cumulative, not five independent cold-start trials. Available RAM fell sharply and was not restored after clean unregisters, consistent with persistent residency/cache/OS effects. Thus 499 is the largest clean point observed in this sequence, not a stable monotone capacity guarantee or endurance result.",
    }
    replayable_checks_pass = all_bools(provenance_checks) and all_bools(protocol_checks)
    verification = {
        "kind": "port80b_d8_registration_capacity_independent_cpu_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "largest_clean_prefix_499_entropy_pin_below_capacity_raw_512_claim_invalid",
        "all_replayable_checks_pass": replayable_checks_pass,
        "input_hashes": hashes,
        "provenance_checks": provenance_checks,
        "rows": rows,
        "capacity": capacity,
        "protocol_checks": protocol_checks,
        "artifact_findings": artifact_findings,
        "cumulative_ram_caveat": ram_caveat,
        "claim_boundary": "CPU-only audit of the stored capacity sweep. No GPU rerun. It supports one clean 499/512 registration+unregister observation and only theoretical EntropyPin size fit; it does not prove a compressed bank, decoder, working-set headroom, stable capacity, transfer performance, model, quality or endurance.",
    }
    OUTPUT.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# PORT80B-D8 independent CPU protocol audit\n\n"
        f"Verdict: **{verification['verdict']}**. All replayable checks: **{replayable_checks_pass}**. No GPU code was executed.\n\n"
        "## Corrected clean capacity\n\n"
        "The preregistration requires immediate unregister after every successful registration. Applying that full lifecycle definition, the largest clean row is **499/512 experts per layer**, "
        f"**{capacity['largest_clean_registered_bytes']:,} bytes = {capacity['largest_clean_registered_gib']:.9f} GiB ({capacity['largest_clean_fraction'] * 100:.4f}%)**. "
        f"The raw JSON's **{capacity['raw_largest_claim_gib']:.6f}-GiB** largest-success claim is invalid because the 512 row contains **{rows[4]['unregister_failure_count']}**, not zero, raw unregister failures.\n\n"
        "Important precision correction: the existing erratum report says all 48 unregister calls failed, but the immutable raw JSON contains exactly **44** failure strings. Four unregister calls therefore returned without a recorded exception. "
        "This does not rescue the row: any unregister failure violates the frozen clean-lifecycle requirement.\n\n"
        "## EntropyPin arithmetic\n\n"
        f"The theoretical **{ENTROPY_PIN_GIB:.3f} GiB** size is below the clean observed capacity by **{capacity['entropy_pin_clean_capacity_margin_gib']:.6f} GiB**. This establishes only arithmetic capacity plausibility. "
        "No compressed-bank artifact, decoder memory, decode time or working-set headroom was measured.\n\n"
        "## Cumulative RAM caveat\n\n"
        f"Available RAM fell from **{ram_caveat['available_before_first_gib']:.3f} GiB** before the first row to **{ram_caveat['available_after_clean_499_gib']:.3f} GiB** after the clean 499 unregister, a cumulative drop of "
        f"**{ram_caveat['drop_first_before_to_after_clean_499_gib']:.3f} GiB**. The first four rows chain directly from the prior row's post-unregister state. This is not a set of independent cold-start capacity trials; page residency/cache/OS state accumulated. "
        "Therefore 499 is the largest clean point observed in this run, not a stable monotone knee or endurance guarantee.\n\n"
        "## Provenance limitation\n\n"
        f"The raw result pins evaluator `{result['inputs']['evaluator_sha256']}`, while the current repaired evaluator hashes to `{hashes['current_evaluator_sha256']}`. The original hashed evaluator is no longer present at that path. "
        "The preregistration, manifest, bank size and independently recomputed full-bank SHA-256 do match.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "verdict": verification["verdict"],
        "all_replayable_checks_pass": replayable_checks_pass,
        "largest_clean_prefix": largest_clean["experts_per_layer"],
        "largest_clean_gib": largest_clean["registered_gib_recomputed"],
        "full_prefix_unregister_failures_raw": rows[4]["unregister_failure_count"],
        "entropy_pin_below_clean_capacity": capacity["entropy_pin_below_clean_capacity"],
        "current_evaluator_matches_raw_result": artifact_findings["current_evaluator_matches_raw_result"],
    }, indent=2))


if __name__ == "__main__":
    main()
