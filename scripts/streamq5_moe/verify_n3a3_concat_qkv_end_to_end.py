from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
RESULT = R / "n3a3_concat_qkv_end_to_end.json"
PREREG = R / "N3A3_CONCAT_QKV_END_TO_END_PREREGISTRATION.md"
SCRIPT = ROOT / "scripts/streamq5_moe/run_n3a3_concat_qkv_end_to_end.py"
N3A2 = R / "n3a2_attention_projection_flow.json"
OUTPUT = R / "n3a3_concat_qkv_end_to_end_verification.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = []

    def add(name: str, passed: bool) -> None:
        checks.append({"name": name, "pass": bool(passed)})

    add("kind", data["kind"] == "streamq5_moe_n3a3_concat_qkv_end_to_end")
    add("prereg hash", data["inputs"]["preregistration_sha256"] == sha256(PREREG))
    add("script hash", data["inputs"]["script_sha256"] == sha256(SCRIPT))
    add("N3A2 hash", data["inputs"]["n3a2_sha256"] == sha256(N3A2))
    add("128 pairs", len(data["pairs"]) == 128 and data["workload"]["paired_tokens"] == 128)
    add("112 timed pairs", data["workload"]["timed_pairs"] == 112 and data["workload"]["warmup_pairs"] == 16)
    for name in ("exact_prediction", "exact_misses", "exact_kv", "exact_dynamic", "exact_logits", "exact_state"):
        add(name, data["exactness"][name] and all(row[name] for row in data["pairs"]))
    add("ABBA order", all(row["order"] == (["baseline", "candidate"] if row["step"] % 2 == 0 else ["candidate", "baseline"]) for row in data["pairs"]))
    add("mean gate correctly failed", data["ratios"]["mean"] > 0.98 and not data["gates"]["mean_ratio_le_0_98"])
    add("p50 gate correctly failed", data["ratios"]["p50"] > 0.98 and not data["gates"]["p50_ratio_le_0_98"])
    add("p95 gate passed", data["ratios"]["p95"] <= 1.00 and data["gates"]["p95_ratio_le_1_00"])
    add("overall correctly false", not data["overall_pass"] and not all(data["gates"].values()))
    passed = sum(row["pass"] for row in checks)
    payload = {"kind": "streamq5_moe_n3a3_independent_verification",
               "status": "pass" if passed == len(checks) else "fail",
               "checks_passed": passed, "checks_total": len(checks), "checks": checks,
               "verified_outcome": "Exact directional improvement, but preregistered end-to-end speed gate failed.",
               "claim_boundary": data["claim_boundary"]}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
