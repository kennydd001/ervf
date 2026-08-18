from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("pro_research/results/s100_phase13b/S100_PHASE13B_ACTIVATION.json"),
    )
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    failures = []
    if result.get("status") != "measured":
        failures.append("status is not measured")
    splits = result.get("splits", {})
    if len(splits.get("calibration", [])) != 10 or len(splits.get("validation", [])) != 10:
        failures.append("split prompt counts are not 10/10")
    if result.get("tokens_per_prompt") != 64:
        failures.append("unexpected tokens_per_prompt")
    families = result.get("families", {})
    for name in ("mamba_in", "mamba_out", "attention_input", "moe_input", "final_norm"):
        if name not in families:
            failures.append(f"missing family {name}")
    measured_rows = 0
    for name, family in families.items():
        for row in family.get("ranks", []):
            if row.get("status") != "measured":
                continue
            measured_rows += 1
            for key in ("validation_residual_energy_mean", "validation_residual_energy_p95", "projected_dense_byte_reduction"):
                value = float(row.get(key, float("nan")))
                if not math.isfinite(value):
                    failures.append(f"{name} rank {row.get('rank')}: non-finite {key}")
    if measured_rows == 0:
        failures.append("no measured rank rows")
    if families.get("final_norm", {}).get("ranks"):
        failures.append("final_norm must remain activation-only")
    if result.get("screen_gate", {}).get("promotion_open") is not False:
        failures.append("activation screen must not promote without output/quality gates")
    payload = {
        "kind": "verify_s100_phase13b_activation",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "families": sorted(families),
        "measured_rank_rows": measured_rows,
        "promotion_open": False,
    }
    output = args.result.with_name("S100_PHASE13B_ACTIVATION_VERIFY.json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
