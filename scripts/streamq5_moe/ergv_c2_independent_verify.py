from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "reports" / "streamq5_moe" / "ergv_c2_performance_autotuner.json"
COMPILE = ROOT / "reports" / "streamq5_moe" / "ergv_c2_compile.json"
OUTPUT = ROOT / "reports" / "streamq5_moe" / "ergv_c2_independent_verification.json"
PREREG = ROOT / "reports" / "streamq5_moe" / "ERGV_C2_PERFORMANCE_AUTOTUNER_PREREGISTRATION.md"
COMPILER = ROOT / "src" / "moe_lab" / "ergv_compiler.py"
RUNNER = ROOT / "scripts" / "streamq5_moe" / "ergv_c2_performance_autotuner.py"
P7 = ROOT / "scripts" / "streamq5_moe" / "run_p7b_ervf_kernel.py"
N1C = ROOT / "scripts" / "streamq5_moe" / "run_n1c_generalized_exact_reduction_autotuner.py"
WIDTHS = (4, 8, 16, 32, 64)
EXPECTED_N1C_Q8 = {"head": 16, "k": 64, "o": 16, "q": 16, "router": 64, "v": 64}
EXPECTED_N1C_Q5 = {"gate_up": 8, "down": 8}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty sample")
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance * max(1.0, abs(float(right)))


def main() -> None:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    compile_data = json.loads(COMPILE.read_text(encoding="utf-8"))
    checks: list[dict] = []

    def check(name: str, passed: bool, detail=None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected_hashes = {
        "preregistration_sha256": sha256(PREREG),
        "compiler_sha256": sha256(COMPILER),
        "runner_sha256": sha256(RUNNER),
        "manual_p7_source_sha256": sha256(P7),
        "manual_n1c_source_sha256": sha256(N1C),
        "compile_lock_sha256": sha256(COMPILE),
    }
    for key, expected in expected_hashes.items():
        check(
            f"provenance_{key}",
            data["source"].get(key) == expected,
            {"reported": data["source"].get(key), "recomputed": expected},
        )
    check("compile_lock_pass", compile_data["overall_pass"] is True)
    check("compile_lock_no_banks", compile_data["physical_banks_loaded"] is False)
    check("compile_lock_no_execution", compile_data["kernels_executed"] is False)
    check("compile_lock_no_timing", compile_data["timing_executed"] is False)
    check(
        "compile_generated_source_matches",
        compile_data["source"]["generated_cuda_sha256"]
        == data["source"]["generated_cuda_sha256"],
    )
    check(
        "compile_combined_source_matches",
        compile_data["source"]["combined_cuda_sha256"]
        == data["source"]["combined_cuda_sha256"],
    )

    exact_items = []
    for bank in ("q8", "q5"):
        for width in WIDTHS:
            item = data["correctness"]["generated"][bank][str(width)]
            exact_items.append(item)
            check(
                f"generated_exact_{bank}_w{width}",
                item["bitwise_equal"]
                and item["finite"]
                and item["different"] == 0
                and item["max_abs"] == 0.0,
                item,
            )
    for reference in ("manual_p7_reproduction", "manual_n1c_reproduction"):
        for bank in ("q8", "q5"):
            item = data["correctness"][reference][bank]
            check(
                f"{reference}_{bank}",
                item["bitwise_equal"]
                and item["finite"]
                and item["different"] == 0
                and item["max_abs"] == 0.0,
                item,
            )
    for bank in ("q8", "q5"):
        item = data["frozen_correctness"][bank]
        check(
            f"frozen_graph_exact_{bank}",
            item["bitwise_equal"]
            and item["finite"]
            and item["different"] == 0
            and item["max_abs"] == 0.0,
            item,
        )

    selected_q8 = data["selected"]["q8"]
    selected_q5 = data["selected"]["q5"]
    for name, width in selected_q8.items():
        validation = data["validation"]["q8"][name]
        best = min(validation[str(candidate)]["stats"]["p50"] for candidate in WIDTHS)
        equivalent = [
            candidate
            for candidate in WIDTHS
            if validation[str(candidate)]["stats"]["p50"] <= best * 1.005
        ]
        expected = 16 if 16 in equivalent else min(equivalent)
        check(f"selection_q8_{name}", width == expected, {"reported": width, "recomputed": expected})
    for part, width in selected_q5.items():
        validation = data["validation"]["q5"][part]
        best = min(validation[str(candidate)]["stats"]["p50"] for candidate in WIDTHS)
        equivalent = [
            candidate
            for candidate in WIDTHS
            if validation[str(candidate)]["stats"]["p50"] <= best * 1.005
        ]
        expected = 16 if 16 in equivalent else min(equivalent)
        check(f"selection_q5_{part}", width == expected, {"reported": width, "recomputed": expected})

    check("manual_n1c_q8_lock", data["manual_n1c_frozen"]["q8"] == EXPECTED_N1C_Q8)
    check("manual_n1c_q5_lock", data["manual_n1c_frozen"]["q5"] == EXPECTED_N1C_Q5)

    recomputed_ratios: dict[str, dict] = {}
    raw_event_count = 0
    for comparison_name in ("versus_manual_p7", "versus_manual_n1c"):
        recomputed_ratios[comparison_name] = {}
        for bank in ("q8", "q5"):
            item = data["tests"][comparison_name][bank]
            reference = [float(value) for value in item["reference"]["event_ms"]]
            candidate = [float(value) for value in item["candidate"]["event_ms"]]
            raw_event_count += len(reference) + len(candidate)
            check(f"raw_length_{comparison_name}_{bank}", len(reference) == len(candidate) == 120)
            check(
                f"iteration_metadata_{comparison_name}_{bank}",
                item["reference"]["iterations"] == item["candidate"]["iterations"] == 120,
            )
            for label, values in (("reference", reference), ("candidate", candidate)):
                stats = item[label]["stats"]
                check(
                    f"stats_{comparison_name}_{bank}_{label}",
                    close(stats["mean"], statistics.fmean(values))
                    and close(stats["p50"], percentile(values, 50))
                    and close(stats["p95"], percentile(values, 95))
                    and close(stats["min"], min(values))
                    and close(stats["max"], max(values)),
                )
            p50_ratio = percentile(candidate, 50) / percentile(reference, 50)
            p95_ratio = percentile(candidate, 95) / percentile(reference, 95)
            recomputed_ratios[comparison_name][bank] = {
                "p50_ratio": p50_ratio,
                "p95_ratio": p95_ratio,
            }
            check(
                f"ratios_{comparison_name}_{bank}",
                close(item["p50_ratio"], p50_ratio)
                and close(item["p95_ratio"], p95_ratio)
                and close(item["p50_speedup"], 1.0 / p50_ratio)
                and close(item["p95_speedup"], 1.0 / p95_ratio),
                recomputed_ratios[comparison_name][bank],
            )
    # The runner deterministically encodes even rounds as reference->candidate
    # and odd rounds as candidate->reference. With 120 rounds this implies
    # exactly 60 AB and 60 BA rounds for each of four pairs.
    check(
        "abba_arithmetic",
        data["protocol"]["paired_test_rounds"] == 120
        and 120 // 2 == 60
        and raw_event_count == 4 * 2 * 120,
        {"pairs": 4, "ab_rounds_per_pair": 60, "ba_rounds_per_pair": 60, "raw_events": raw_event_count},
    )

    p7 = recomputed_ratios["versus_manual_p7"]
    breakthrough = [
        bank
        for bank, item in p7.items()
        if item["p50_ratio"] <= 0.98 and item["p95_ratio"] <= 1.00
    ]
    no_regression = all(
        item["p50_ratio"] <= 1.02 and item["p95_ratio"] <= 1.02
        for item in p7.values()
    )
    recomputed_gates = {
        "all_generated_widths_exact_q8_q5": all(
            item["bitwise_equal"] and item["finite"] for item in exact_items
        ),
        "manual_p7_width16_reproduced": all(
            data["correctness"]["manual_p7_reproduction"][bank]["bitwise_equal"]
            and data["correctness"]["manual_p7_reproduction"][bank]["finite"]
            for bank in ("q8", "q5")
        ),
        "at_least_one_family_p50_le_0_98_and_p95_le_1_00": bool(breakthrough),
        "no_family_regression_over_1_02": no_regression,
    }
    check("gate_dictionary", data["gates"] == recomputed_gates, recomputed_gates)
    check("breakthrough_family_list", data["breakthrough_families_vs_p7"] == breakthrough, breakthrough)
    check("overall_pass_arithmetic", data["overall_pass"] is all(recomputed_gates.values()))

    # Verify the AB/BA implementation and validation-order implementation from
    # the locked runner source rather than trusting prose in the result.
    runner_text = RUNNER.read_text(encoding="utf-8")
    order_markers = (
        'order = (("reference", reference), ("candidate", candidate))',
        "if round_index & 1:",
        "order = tuple(reversed(order))",
        "rotation = round_index % len(WIDTHS)",
        "order = list(WIDTHS[rotation:] + WIDTHS[:rotation])",
    )
    check("locked_order_implementation", all(marker in runner_text for marker in order_markers), list(order_markers))

    passed = sum(item["passed"] for item in checks)
    result = {
        "kind": "ergv_c2_independent_cpu_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": passed == len(checks),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "raw_event_values_verified": raw_event_count,
        "recomputed_ratios": recomputed_ratios,
        "recomputed_gates": recomputed_gates,
        "source": {
            "result_sha256": sha256(RESULT),
            "compile_lock_sha256": sha256(COMPILE),
            "verifier_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": "Independent CPU audit of the locked C2 artifact; no GPU rerun or new performance evidence.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "overall_pass": result["overall_pass"],
                "checks_passed": passed,
                "checks_total": len(checks),
                "raw_event_values_verified": raw_event_count,
                "failed": [item["name"] for item in checks if not item["passed"]],
            },
            indent=2,
        )
    )
    if not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
