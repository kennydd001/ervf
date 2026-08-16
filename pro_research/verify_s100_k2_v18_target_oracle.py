"""Independent verifier for PRO_S100_K2_V18_TARGET_ORACLE.json.

Recomputes every declared gate from stored raw/summary evidence. It does not
import the runner and does not execute model code.
"""
from __future__ import annotations

import json
from pathlib import Path

from common import REPO, utc_now

SRC = REPO / "pro_research" / "results" / "s100_k2_v18" / "PRO_S100_K2_V18_TARGET_ORACLE.json"
OUT = REPO / "pro_research" / "results" / "s100_k2_v18" / "PRO_S100_K2_V18_TARGET_ORACLE_VERIFICATION.json"


def main() -> int:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    errors: list[str] = []
    pp = src.get("per_prompt") or []
    arms = src.get("arms") or {}
    summary = src.get("summary") or {}
    stored = src.get("gates") or {}

    required_arms = ("SEQ_A", "K2", "SEQ_B")
    for a in required_arms:
        if a not in arms or "p50" not in arms[a]:
            errors.append(f"missing arm/p50: {a}")

    if errors:
        calc = {}
    else:
        a = float(arms["SEQ_A"]["p50"])
        k = float(arms["K2"]["p50"])
        b = float(arms["SEQ_B"]["p50"])
        mid = (a + b) / 2.0
        drift = abs(a - b)
        tps = 2000.0 / k if k else 0.0
        speedup = mid / k if k else 0.0

        calc = {
            "G1_reference_A_B_token_parity": bool(pp) and all(bool(x.get("reference_a_b_parity")) for x in pp),
            "G2_candidate_token_parity": bool(pp) and all(bool(x.get("candidate_token_parity")) for x in pp),
            "G3_deterministic": bool(pp) and all(bool(x.get("deterministic")) for x in pp),
            "G4_state_bitexact": bool(pp) and all(bool((x.get("state_compare") or {}).get("bitexact")) for x in pp),
            "G5_continuation_32": bool(pp) and all(bool(x.get("continuation_32")) for x in pp),
            "G6_control_diverges": bool(pp) and any(bool((x.get("control") or {}).get("token_diverged")) or bool((x.get("control") or {}).get("state_diverged")) for x in pp),
            "G7_no_nan_inf": bool(pp) and all(bool(x.get("finite")) for x in pp),
            "D1_seq_A_B_drift_le_1ms": drift <= 1.0,
            "P1_K2_block_lt_19_285ms": k < 19.285,
            "P2_K2_block_lt_17_500ms": k < 17.500,
            "P3_effective_verified_ge_110tps": tps >= 110.0,
            "P4_speedup_vs_seq_mid_ge_1_50x": speedup >= 1.50,
        }

        checks = {
            "seq_mid_p50_ms_per_2tok": mid,
            "k2_p50_ms_per_2tok": k,
            "k2_effective_verified_tok_s": tps,
            "k2_speedup_vs_seq_mid": speedup,
            "seq_drift_ms": drift,
        }
        for key, val in checks.items():
            if key not in summary or abs(float(summary[key]) - val) > 1e-9 * max(1.0, abs(val)):
                errors.append(f"summary mismatch {key}: stored={summary.get(key)} recomputed={val}")

        for key, val in calc.items():
            if stored.get(key) is not val:
                errors.append(f"gate mismatch {key}: stored={stored.get(key)} recomputed={val}")

    correctness_keys = (
        "G1_reference_A_B_token_parity", "G2_candidate_token_parity",
        "G3_deterministic", "G4_state_bitexact", "G5_continuation_32",
        "G6_control_diverges", "G7_no_nan_inf",
    )
    correctness = bool(calc) and all(calc.get(k) is True for k in correctness_keys)
    stable = bool(calc) and calc.get("D1_seq_A_B_drift_le_1ms") is True
    if correctness and stable:
        if calc["P1_K2_block_lt_19_285ms"]:
            expected_status = "k2_v18_feasible_candidate"
        elif float(summary.get("k2_p50_ms_per_2tok", 1e30)) >= float(summary.get("seq_mid_p50_ms_per_2tok", -1e30)):
            expected_status = "layer_major_v18_negative"
        else:
            expected_status = "k2_v18_below_s100_gate"
    elif not correctness:
        expected_status = "correctness_failed"
    else:
        expected_status = "measurement_unstable"

    if src.get("status") != expected_status:
        errors.append(f"status mismatch stored={src.get('status')} recomputed={expected_status}")

    payload = {
        "kind": "s100_k2_v18_target_oracle_independent_verification",
        "created_utc": utc_now(),
        "source": str(SRC.relative_to(REPO)),
        "source_status": src.get("status"),
        "recomputed_status": expected_status,
        "recomputed_gates": calc,
        "errors": errors,
        "passed": not errors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
