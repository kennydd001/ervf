from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("pro_research/results/s100_phase13a/S100_PHASE13A_ENTROPY.json"),
    )
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    failures = []
    if result.get("status") != "measured":
        failures.append("status is not measured")
    counts = result.get("counts", {})
    if counts.get("resident_matrices") != 140:
        failures.append(f"resident_matrices={counts.get('resident_matrices')}")
    if counts.get("routed_expert_sample_matrices") != 18:
        failures.append(
            f"routed_expert_sample_matrices={counts.get('routed_expert_sample_matrices')}"
        )
    rows = result.get("matrices", [])
    if len(rows) != 158:
        failures.append(f"matrix rows={len(rows)}")
    names = [row.get("name") for row in rows]
    if len(names) != len(set(names)):
        failures.append("duplicate matrix names")
    if any(not row.get("palette_bits_per_stream") for row in rows):
        failures.append("missing palette estimates")
    if not any(row.get("analysis_sampling") != "full stream" for row in rows):
        failures.append("large tensor sampling marker missing")
    gates = result.get("gates", {})
    mamba = float(gates.get("mamba_fp8_best_palette_bits_per_weight", float("nan")))
    resident = float(gates.get("resident_best_palette_fraction", float("nan")))
    if not (mamba == mamba and resident == resident):
        failures.append("non-finite gate values")
    if gates.get("mamba_fp8_le_6_bits_per_weight") != (mamba <= 6.0):
        failures.append("mamba gate is inconsistent")
    if gates.get("resident_le_70_percent_raw_bytes") != (resident <= 0.70):
        failures.append("resident gate is inconsistent")
    payload = {
        "kind": "verify_s100_phase13a_entropy",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "resident_matrices": counts.get("resident_matrices"),
        "expert_sample_matrices": counts.get("routed_expert_sample_matrices"),
        "mamba_fp8_bits_per_weight": mamba,
        "resident_palette_fraction": resident,
        "promotion_open": bool(gates.get("promotion_open", False)),
    }
    output = args.result.with_name("S100_PHASE13A_ENTROPY_VERIFY.json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
