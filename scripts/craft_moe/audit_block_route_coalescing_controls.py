from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from moe_lab.craft_moe.block_coalescing import highs_optimal_control
from moe_lab.reporting import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_crcq_oracle import git_state, sha256_file, write_json_once  # noqa: E402


SOURCE = ROOT / "reports/craft_moe/block_route_coalescing.json"
OUTPUT = ROOT / "reports/craft_moe/block_route_coalescing_control_audit_v2.json"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite audit: {OUTPUT}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for split, split_result in source["results"].items():
        for configuration, config_result in split_result["configurations"].items():
            for block_size, cell in config_result["block_sizes"].items():
                for method, method_result in cell["methods"].items():
                    if not method.startswith("exact_ilp"):
                        continue
                    for row in method_result["raw_blocks"]:
                        diagnostics = row["diagnostics"]
                        observed_objective = int(
                            row["cold_union_count"]
                            if int(diagnostics["cache_mask"]) != 0
                            else row["union_count"]
                        )
                        passed = highs_optimal_control(
                            diagnostics, observed_objective
                        )
                        records.append(
                            {
                                "split": split,
                                "configuration": configuration,
                                "block_size": int(block_size),
                                "method": method,
                                "block_index": int(row["block_index"]),
                                "status": int(diagnostics["status"]),
                                "success": bool(diagnostics["success"]),
                                "mip_gap": float(diagnostics["mip_gap"]),
                                "objective": float(diagnostics["objective"]),
                                "observed_objective_union_count": observed_objective,
                                "passes_preregistered_optimal_control": passed,
                            }
                        )
    if not records:
        raise RuntimeError("source contains no exact ILP records")
    all_pass = all(row["passes_preregistered_optimal_control"] for row in records)
    primary = source["gates"]["primary_by_split"]
    natural_gate = all(
        row["union_reduction_vs_natural"] >= 0.40 for row in primary.values()
    )
    mass_gate = all(
        row["additional_union_reduction_vs_mass_budget"] >= 0.25
        for row in primary.values()
    )
    kl_gate = all(row["mean_local_kl"] <= 0.001 for row in primary.values())
    original_gate = bool(
        source["gates"]["original_route_numerical_control_pass"]
    )
    controls = all_pass and original_gate
    hard = any(
        row["union_reduction_vs_natural"] < 0.25 for row in primary.values()
    )
    if natural_gate and mass_gate and kl_gate and controls:
        verdict = "layer26_positive_opens_layer23_preregistration"
    elif hard:
        verdict = "oracle_negative_hard_falsification"
    else:
        verdict = "inconclusive_negative_no_downstream"
    audit = {
        "schema_version": 1,
        "kind": "craft_moe_h2_block_route_coalescing_control_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source": str(SOURCE.resolve()),
        "source_sha256": sha256_file(SOURCE),
        "reason": (
            "The source adjudicator incorrectly required floating mip_gap == 0.0. "
            "The preregistered control requires HiGHS optimal status and objective/union "
            "closure; machine-epsilon gaps are accepted at <=1e-12. A first audit "
            "artifact compared empty-cache objectives to cold-union counts and is "
            "retained; this v2 uses union_count for empty-cache ILPs and cold_union_count "
            "only for nonempty-cache ILPs."
        ),
        "fixed_tolerances": {
            "absolute_mip_gap_le": 1e-12,
            "objective_vs_observed_cold_union_absolute_error_le": 1e-6,
        },
        "exact_records": len(records),
        "all_exact_records_pass": all_pass,
        "maximum_absolute_mip_gap": max(abs(row["mip_gap"]) for row in records),
        "maximum_objective_absolute_error": max(
            abs(row["objective"] - row["observed_objective_union_count"])
            for row in records
        ),
        "status_counts": {
            str(status): sum(row["status"] == status for row in records)
            for status in sorted({row["status"] for row in records})
        },
        "recalculated_gates": {
            "union_reduction_ge_0_40_both_splits": natural_gate,
            "additional_reduction_vs_mass_budget_ge_0_25_both_splits": mass_gate,
            "mean_local_kl_le_0_001_both_splits": kl_gate,
            "all_highs_solutions_optimal": all_pass,
            "original_route_numerical_control_pass": original_gate,
            "exact_controls_pass": controls,
        },
        "recalculated_verdict": verdict,
        "failed_records": [
            row for row in records if not row["passes_preregistered_optimal_control"]
        ],
        "machine_epsilon_nonzero_gap_records": [
            row for row in records if row["mip_gap"] != 0.0
        ],
        "reproducibility": {
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd().resolve()),
            "repository": git_state(),
        },
    }
    write_json_once(OUTPUT, audit)
    print(f"result={OUTPUT}")
    print(f"all_exact_records_pass={all_pass}")
    print(f"recalculated_verdict={verdict}")


if __name__ == "__main__":
    main()
