from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evaluate_crcq_oracle import SEED, sha256_file, write_json_once
from moe_lab.craft_moe.cache_span import paired_load_bootstrap
from moe_lab.reporting import ROOT


SOURCE = ROOT / "reports/craft_moe/cache_span.json"
OUTPUT = ROOT / "reports/craft_moe/cache_span_block_bootstrap_audit.json"


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite audit: {OUTPUT}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = {}
    for split, primary, zero in (
        (
            "validation",
            source["validation"]["selected"],
            source["validation"]["zero_fill"],
        ),
        (
            "test",
            source["heldout_test"]["selected_configuration"],
            source["heldout_test"]["zero_fill"],
        ),
    ):
        primary_blocks = primary["per_block"]
        zero_blocks = zero["per_block"]
        if len(primary_blocks) != len(zero_blocks):
            raise RuntimeError("primary and zero block counts differ")
        baseline = [int(row["baseline_misses"]) for row in primary_blocks]
        primary_avoided = [int(row["avoided_misses"]) for row in primary_blocks]
        zero_avoided = [int(row["avoided_misses"]) for row in zero_blocks]
        if baseline != [int(row["baseline_misses"]) for row in zero_blocks]:
            raise RuntimeError("paired baselines differ")
        bootstrap = paired_load_bootstrap(
            baseline,
            primary_avoided,
            zero_avoided,
            seed=SEED + (200 if split == "validation" else 201),
            resamples=10_000,
        )
        point = bootstrap["point_estimates"]
        reconciliation = {
            "baseline_misses_sum_exact": sum(baseline) == primary["baseline_misses"],
            "primary_avoided_sum_exact": (
                sum(primary_avoided) == primary["avoided_misses"]
            ),
            "zero_avoided_sum_exact": sum(zero_avoided) == zero["avoided_misses"],
            "primary_fraction_absolute_error": abs(
                point["primary_miss_reduction_fraction"]
                - primary["miss_reduction_fraction"]
            ),
            "zero_fraction_absolute_error": abs(
                point["zero_fill_miss_reduction_fraction"]
                - zero["miss_reduction_fraction"]
            ),
        }
        if not all(
            (
                reconciliation["baseline_misses_sum_exact"],
                reconciliation["primary_avoided_sum_exact"],
                reconciliation["zero_avoided_sum_exact"],
                reconciliation["primary_fraction_absolute_error"] <= 1e-15,
                reconciliation["zero_fraction_absolute_error"] <= 1e-15,
            )
        ):
            raise RuntimeError(f"block reconciliation failed for {split}")
        rows[split] = {
            "baseline_misses_by_block": baseline,
            "primary_avoided_by_block": primary_avoided,
            "zero_fill_avoided_by_block": zero_avoided,
            "reconciliation": reconciliation,
            "bootstrap": bootstrap,
        }
    report = {
        "schema_version": 1,
        "kind": "craft_moe_h8_cache_span_block_bootstrap_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source": {
            "path": str(SOURCE.resolve()),
            "bytes": SOURCE.stat().st_size,
            "sha256": sha256_file(SOURCE),
        },
        "splits": rows,
        "all_reconciliation_controls_pass": True,
        "interpretation": (
            "paired two-block intervals quantify window heterogeneity; with only two "
            "blocks they are descriptive and do not upgrade the failed point gates"
        ),
    }
    write_json_once(OUTPUT, report)
    print(f"result={OUTPUT}")
    for split, row in rows.items():
        point = row["bootstrap"]["point_estimates"]
        interval = row["bootstrap"]["intervals_95"]
        print(
            f"{split}_primary={point['primary_miss_reduction_fraction']:.6f} "
            f"ci=[{interval['primary_miss_reduction_fraction']['low']:.6f},"
            f"{interval['primary_miss_reduction_fraction']['high']:.6f}] "
            f"uplift={point['span_uplift_fraction']:.6f}"
        )


if __name__ == "__main__":
    main()
