from __future__ import annotations

import json

from moe_lab.reporting import ROOT, envelope, write_json


SOURCE_NAME = "preregistered_wikitext_offset4096_mass_budget_confirmation.json"
OLD_POLICY = "max_rank_j5_m7"
LOW_FIXED = "cache_prior_j2_lambda0p0275"
LOW_MASS = "mass_budget_j2_delta0p004"
HIGH_FIXED = "cache_prior_j2_lambda0p095"
HIGH_MASS = "mass_budget_j2_delta0p018"


def paired_front_gate(
    payload: dict[str, object], fixed_policy: str, mass_policy: str
) -> dict[str, object]:
    split_results: dict[str, object] = {}
    for split in ("validation", "test"):
        fixed = payload["final"][split][fixed_policy]
        mass = payload["final"][split][mass_policy]
        load_difference = (
            mass["expert_load_reduction_fraction"]
            - fixed["expert_load_reduction_fraction"]
        )
        relative_kl_change = (
            mass["teacher_to_candidate_kl"]
            / fixed["teacher_to_candidate_kl"]
            - 1
        )
        split_results[split] = {
            "mass_budget_minus_fixed_load_reduction_fraction": load_difference,
            "relative_kl_change": relative_kl_change,
            "load_within_two_percentage_points": load_difference >= -0.02,
            "kl_at_least_five_percent_lower": relative_kl_change <= -0.05,
        }
    passed = all(
        row["load_within_two_percentage_points"]
        and row["kl_at_least_five_percent_lower"]
        for row in split_results.values()
    )
    return {
        "fixed_policy": fixed_policy,
        "mass_budget_policy": mass_policy,
        "splits": split_results,
        "passed": passed,
    }


if __name__ == "__main__":
    source_path = ROOT / "reports" / "baseline" / SOURCE_NAME
    payload = json.loads(source_path.read_text(encoding="utf-8"))["payload"]
    original_passed = (
        payload["numerical_validation"][
            "official_original_control_max_abs_across_layers"
        ]
        <= 1e-6
        and all(
            payload["final"][split]["original"][metric] == expected
            for split in ("validation", "test")
            for metric, expected in (
                ("teacher_to_candidate_kl", 0.0),
                ("relative_cross_entropy_delta", 0.0),
                ("top1_agreement", 1.0),
            )
        )
    )
    old_test = payload["final"]["test"][OLD_POLICY]
    low_test = payload["final"]["test"][LOW_MASS]
    old_baseline_passed = (
        low_test["expert_load_reduction_fraction"]
        > old_test["expert_load_reduction_fraction"]
        and low_test["teacher_to_candidate_kl"]
        < old_test["teacher_to_candidate_kl"]
    )
    pair_results = {
        "low": paired_front_gate(payload, LOW_FIXED, LOW_MASS),
        "high": paired_front_gate(payload, HIGH_FIXED, HIGH_MASS),
    }
    paired_front_passed = any(row["passed"] for row in pair_results.values())
    ce_interval = low_test["bootstrap_95_percent_intervals"][
        "relative_cross_entropy_delta"
    ]
    ce_passed = (
        low_test["relative_cross_entropy_delta"] <= 0.005
        and ce_interval[1] <= 0.01
    )
    blocks_per_split = int(payload["blocks_per_split"])
    low_blocks = payload["total_cache_statistics"][LOW_MASS]["per_block"]
    test_blocks = low_blocks[-blocks_per_split:]
    all_test_blocks_passed = all(
        row["expert_load_reduction_fraction"] > 0 for row in test_blocks
    )
    gates = {
        "gate_1_exact_original_control": {
            "passed": original_passed,
            "maximum_layer_error": payload["numerical_validation"][
                "official_original_control_max_abs_across_layers"
            ],
        },
        "gate_2_low_mass_beats_old_baseline_on_test": {
            "passed": old_baseline_passed,
            "old_policy": OLD_POLICY,
            "mass_budget_policy": LOW_MASS,
            "old_load_reduction_fraction": old_test[
                "expert_load_reduction_fraction"
            ],
            "mass_budget_load_reduction_fraction": low_test[
                "expert_load_reduction_fraction"
            ],
            "old_kl": old_test["teacher_to_candidate_kl"],
            "mass_budget_kl": low_test["teacher_to_candidate_kl"],
        },
        "gate_3_at_least_one_paired_front_comparison": {
            "passed": paired_front_passed,
            "pairs": pair_results,
        },
        "gate_4_low_mass_cross_entropy_sanity": {
            "passed": ce_passed,
            "relative_cross_entropy_delta": low_test[
                "relative_cross_entropy_delta"
            ],
            "bootstrap_95_percent_interval": ce_interval,
        },
        "gate_5_every_low_mass_test_block_saves_loads": {
            "passed": all_test_blocks_passed,
            "passing_blocks": sum(
                row["expert_load_reduction_fraction"] > 0
                for row in test_blocks
            ),
            "test_blocks": blocks_per_split,
        },
    }
    report = {
        "status": "pass" if all(row["passed"] for row in gates.values()) else "fail",
        "source_report": f"reports/baseline/{SOURCE_NAME}",
        "gates": gates,
    }
    output = write_json(
        "preregistered_mass_budget_confirmation_gates.json",
        envelope("preregistered_mass_budget_confirmation_gates", report),
    )
    print(output)
    print(json.dumps(report, indent=2))
