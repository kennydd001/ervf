from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
RESULT = R / "n2c_batch_route_union_sweep.json"
PREREG = R / "N2C_BATCH_ROUTE_UNION_SWEEP_PREREGISTRATION.md"
OUTPUT = R / "n2c_batch_route_union_verification.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = []

    def add(name: str, passed: bool) -> None:
        checks.append({"name": name, "pass": bool(passed)})

    add("kind", data["kind"] == "streamq5_moe_n2c_batch_route_union_sweep")
    add("preregistration hash", data["inputs"]["preregistration_sha256"] == sha256(PREREG))
    add("script hash", data["inputs"]["script_sha256"] == sha256(ROOT / "scripts/streamq5_moe/run_n2c_batch_route_union_sweep.py"))
    for size in (2, 4, 8):
        row = data["results"][str(size)]
        add(f"S{size} Q8 bit exact", row["q8_correctness"]["bitwise_equal"] and row["q8_correctness"]["different"] == 0)
        add(f"S{size} Q5 bit exact", row["q5_correctness"]["bitwise_equal"] and row["q5_correctness"]["different"] == 0)
        add(f"S{size} validation gate failed", row["validation"]["combined"]["p50_ratio"] > 0.98 and not row["test_opened"])
    add("S16 resource status", data["results"]["16"]["status"] == "blocked_by_resource_spill_timeout")
    add("no passing sizes", data["passing_sizes"] == [] and data["winner"] is None and not data["overall_pass"])
    passed = sum(row["pass"] for row in checks)
    payload = {
        "kind": "streamq5_moe_n2c_independent_verification",
        "status": "pass" if passed == len(checks) else "fail",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "claim_boundary": "Verifies the recorded N2C component sweep and its closed tests; does not turn the incomplete S16 arm into a measurement.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
