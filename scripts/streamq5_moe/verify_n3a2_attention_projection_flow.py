from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
RESULT = R / "n3a2_attention_projection_flow.json"
PREREG = R / "N3A2_ATTENTION_PROJECTION_FLOW_PREREGISTRATION.md"
SCRIPT = ROOT / "scripts/streamq5_moe/run_n3a2_attention_projection_flow.py"
OUTPUT = R / "n3a2_attention_projection_flow_verification.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = []

    def add(name: str, passed: bool) -> None:
        checks.append({"name": name, "pass": bool(passed)})

    add("kind", data["kind"] == "streamq5_moe_n3a2_attention_projection_flow")
    add("prereg hash", data["inputs"]["preregistration_sha256"] == sha256(PREREG))
    add("script hash", data["inputs"]["script_sha256"] == sha256(SCRIPT))
    for candidate in ("concat_qkv", "head_flow"):
        row = data["correctness"][candidate]
        add(f"{candidate} outputs exact", row["outputs"]["bitwise_equal"] and row["outputs"]["different"] == 0 and row["outputs"]["elements"] == 48 * (4096 + 512 + 512))
        add(f"{candidate} KV exact", row["kv_bitwise_equal"] and row["kv_different"] == 0 and row["kv_elements"] == 48 * 2 * 4 * 128)
    add("selection", data["selected"] == "concat_qkv")
    add("validation gate", data["validation_p50_ratio"] <= 0.98 and data["test_opened"])
    test_correct = data["test_correctness"]
    add("test outputs exact", test_correct["outputs"]["bitwise_equal"] and test_correct["outputs"]["different"] == 0)
    add("test KV exact", test_correct["kv_bitwise_equal"] and test_correct["kv_different"] == 0)
    add("test timing gates", data["test"]["p50_ratio"] <= 0.97 and data["test"]["p95_ratio"] <= 1.00 and data["test"]["pass"] and data["overall_pass"])
    passed = sum(row["pass"] for row in checks)
    payload = {"kind": "streamq5_moe_n3a2_independent_verification",
               "status": "pass" if passed == len(checks) else "fail",
               "checks_passed": passed, "checks_total": len(checks), "checks": checks,
               "preregistration_count_erratum": "The prose says 294,912 Q/K/V FP32 outputs; all actual tensors are 48*(4096+512+512)=245,760. Scope and gate are all Q/K/V and were fully evaluated.",
               "claim_boundary": data["claim_boundary"]}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
