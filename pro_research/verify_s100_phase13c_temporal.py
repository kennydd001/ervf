from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, default=Path("pro_research/results/s100_phase13c/S100_PHASE13C_TEMPORAL.json"))
    args = ap.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    failures = []
    if result.get("status") != "measured":
        failures.append("status is not measured")
    if result.get("tokens_per_prompt") != 64:
        failures.append("unexpected tokens_per_prompt")
    for split in ("calibration", "validation"):
        families = result.get("splits", {}).get(split, {})
        for family in ("mamba_in", "mamba_out", "attention_input", "moe_input", "final_norm"):
            row = families.get(family, {})
            if int(row.get("samples", 0)) <= 0:
                failures.append(f"{split}/{family}: no samples")
            for key in ("cosine", "norm_ratio", "delta_norm", "int8_nrmse", "int4_nrmse"):
                value = row.get(key, {}).get("mean", float("nan"))
                if not math.isfinite(float(value)):
                    failures.append(f"{split}/{family}: non-finite {key}")
            for k in ("32", "64", "128", "256", "512"):
                if k not in row.get("topk_energy", {}):
                    failures.append(f"{split}/{family}: missing topk {k}")
    if result.get("gate", {}).get("promotion_open") is not False:
        failures.append("promotion must remain closed")
    payload = {"kind": "verify_s100_phase13c_temporal", "status": "PASS" if not failures else "FAIL", "failures": failures, "promotion_open": False}
    output = args.result.with_name("S100_PHASE13C_TEMPORAL_VERIFY.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
