from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("pro_research/results/s100_phase13e/S100_PHASE13E_SHARED_BASIS.json"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    failures = []
    if result.get("status") != "measured":
        failures.append("status is not measured")
    if result.get("counts", {}).get("matrix_records") != 6:
        failures.append("expected six layer/projection records")
    if result.get("counts", {}).get("expert_matrices") != 6 * 128:
        failures.append("expected 768 routed expert matrices")
    for record in result.get("records", []):
        ranks = record.get("ranks", [])
        if len(ranks) != 5:
            failures.append(f"wrong rank count at layer {record.get('layer')} {record.get('projection')}")
        for row in ranks:
            for key in ("residual_energy_fraction", "reconstruction_nrmse", "ideal_dense_shared_byte_reduction"):
                if not math.isfinite(float(row.get(key, float("nan")))):
                    failures.append(f"non-finite {key}")
    if result.get("gates", {}).get("promotion_open") is not False:
        failures.append("expert shared-basis census must not promote without quality and validation")
    payload = {
        "kind": "verify_s100_phase13e_expert_shared_basis",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "promotion_open": False,
    }
    output = args.result.with_name("S100_PHASE13E_SHARED_BASIS_VERIFY.json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
