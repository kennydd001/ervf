from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
RESULT = R / "n3a4_o_residual_fusion.json"
PREREG = R / "N3A4_O_RESIDUAL_FUSION_PREREGISTRATION.md"
SCRIPT = ROOT / "scripts/streamq5_moe/run_n3a4_o_residual_fusion.py"
OUTPUT = R / "n3a4_o_residual_fusion_verification.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = []

    def add(name: str, passed: bool) -> None:
        checks.append({"name": name, "pass": bool(passed)})

    add("kind", data["kind"] == "streamq5_moe_n3a4_o_residual_fusion")
    add("prereg hash", data["inputs"]["preregistration_sha256"] == sha256(PREREG))
    add("script hash", data["inputs"]["script_sha256"] == sha256(SCRIPT))
    add("48 O records", data["inputs"]["layers"] == 48 and data["inputs"]["o_records"] == 48)
    correctness = data["validation_correctness"]
    add("state bit exact", correctness["bitwise_equal"] and correctness["different"] == 0)
    add("state count", correctness["elements"] == 48 * 2048)
    add("state finite", correctness["finite"])
    add("validation p50 gate failed", data["validation"]["p50_ratio"] > 0.98)
    add("test correctly closed", not data["test_opened"] and data["test"] is None and data["test_correctness"] is None)
    add("overall correctly false", not data["overall_pass"])
    passed = sum(row["pass"] for row in checks)
    payload = {"kind": "streamq5_moe_n3a4_independent_verification",
               "status": "pass" if passed == len(checks) else "fail",
               "checks_passed": passed, "checks_total": len(checks), "checks": checks,
               "verified_outcome": "Bit-exact component, but validation speed gate failed and test remained sealed.",
               "claim_boundary": data["claim_boundary"]}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
