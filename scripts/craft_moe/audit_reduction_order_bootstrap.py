from __future__ import annotations

import json
from datetime import datetime, timezone

from evaluate_crcq_oracle import SEED, sha256_file, write_json_once
from moe_lab.craft_moe.reduction_order import paired_gap_closure_bootstrap
from moe_lab.reporting import ROOT


SOURCE = ROOT / "reports/craft_moe/reduction_order.json"
OUTPUT = ROOT / "reports/craft_moe/reduction_order_bootstrap_audit.json"
BLOCK_SIZE = 128


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite audit: {OUTPUT}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    splits = {}
    for split_index, split in enumerate(("validation", "test")):
        quality = source["exact_quality"][split]
        q3 = quality["q3_reference_vectorized_fp32"]["raw"][
            "teacher_to_candidate_kl"
        ]
        q4 = quality["q4_reference_vectorized_fp32"]["raw"][
            "teacher_to_candidate_kl"
        ]
        candidate_names = {
            "fixed": "q3_fixed_validation_order",
            "fp32_control": "q3_validation_selected_fp32_control",
            "per_token_local_mse_oracle": "q3_per_token_local_mse_oracle",
        }
        candidate_series = {
            short: quality[name]["raw"]["teacher_to_candidate_kl"]
            for short, name in candidate_names.items()
        }
        bootstrap = paired_gap_closure_bootstrap(
            q3,
            q4,
            candidate_series,
            block_size=BLOCK_SIZE,
            seed=SEED + 300 + split_index,
            resamples=10_000,
        )
        source_gap = quality["gap_analysis"]
        reconciliation = {
            "fixed_absolute_error": abs(
                bootstrap["point_closure"]["fixed"]
                - source_gap["fixed_gap_closure"]
            ),
            "fp32_absolute_error": abs(
                bootstrap["point_closure"]["fp32_control"]
                - source_gap["fp32_control_gap_closure"]
            ),
        }
        if max(reconciliation.values()) > 1e-12:
            raise RuntimeError(f"point closure reconciliation failed for {split}")
        splits[split] = {
            "bootstrap": bootstrap,
            "reconciliation": reconciliation,
        }
    report = {
        "schema_version": 1,
        "kind": "craft_moe_h10_reduction_order_paired_bootstrap_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source": {
            "path": str(SOURCE.resolve()),
            "bytes": SOURCE.stat().st_size,
            "sha256": sha256_file(SOURCE),
        },
        "splits": splits,
        "all_reconciliation_controls_pass": True,
        "interpretation": (
            "paired two-block intervals quantify window heterogeneity and do not "
            "replace the preregistered held-out point-estimate hard stop"
        ),
    }
    write_json_once(OUTPUT, report)
    print(f"result={OUTPUT}")
    for split, row in splits.items():
        point = row["bootstrap"]["point_closure"]["fixed"]
        interval = row["bootstrap"]["intervals_95"]["fixed"]
        print(
            f"{split}_fixed_closure={point:.6f} "
            f"ci=[{interval['low']:.6f},{interval['high']:.6f}]"
        )


if __name__ == "__main__":
    main()
