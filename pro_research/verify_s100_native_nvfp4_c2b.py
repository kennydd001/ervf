"""Independent CPU-side verifier for S100 native NVFP4 C2b results."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from common import REPO, utc_now, write_json_atomic

SRC = REPO / "pro_research" / "results" / "native_nvfp4" / "C2B_TORCH212_CONTRACT.json"
OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "C2B_TORCH212_CONTRACT_VERIFICATION.json"


def _p50(rec: dict[str, Any] | None) -> float | None:
    if not rec or rec.get("status") != "pass":
        return None
    d = rec.get("event_ms") or {}
    v = d.get("p50")
    return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None


def _expected_status(src: dict[str, Any], gates: dict[str, bool]) -> str:
    api_names = (
        "G1_torch_2_12_1_cu132", "G2_cuda_sm120_or_newer",
        "G3_public_scaled_mm_present", "G4_fp4_dtype_present",
        "G5_BlockWise1x16_present", "G6_SWIZZLE_32_4_4_present",
        "G7_scaled_mm_v2_recordable",
    )
    if not all(gates.get(k) is True for k in api_names):
        return "api_contract_failed"
    known_names = (
        "K1_M1_known_value_executes", "K2_all_known_values_exact_bf16",
        "K3_all_known_values_deterministic", "K4_all_known_values_finite",
    )
    if not all(gates.get(k) is True for k in known_names):
        return "native_execution_failed"
    return "native_execution_candidate" if gates.get("C3_real_checkpoint_open") is True else "native_executes_below_c3_gate"


def main() -> int:
    errors: list[str] = []
    if not SRC.exists():
        raise SystemExit(f"missing result: {SRC}")
    src = json.loads(SRC.read_text(encoding="utf-8"))
    if src.get("kind") != "s100_native_nvfp4_c2b_torch212_contract":
        errors.append("kind mismatch")

    api = src.get("api") or {}
    gates: dict[str, bool] = {
        "G1_torch_2_12_1_cu132": bool(
            str(api.get("torch_version", "")).startswith("2.12.1")
            and str(api.get("torch_cuda_version", "")).startswith("13.2")
        ),
        "G2_cuda_sm120_or_newer": bool(
            api.get("cuda_available") is True
            and isinstance(api.get("capability"), list)
            and len(api["capability"]) >= 2
            and tuple(api["capability"][:2]) >= (12, 0)
        ),
        "G3_public_scaled_mm_present": api.get("F_scaled_mm_present") is True,
        "G4_fp4_dtype_present": api.get("float4_e2m1fn_x2_present") is True,
        "G5_BlockWise1x16_present": api.get("ScalingType_BlockWise1x16_present") is True,
        "G6_SWIZZLE_32_4_4_present": api.get("SwizzleType_32_4_4_present") is True,
        "G7_scaled_mm_v2_recordable": bool(
            api.get("torch__scaled_mm_v2_present") is True
            or not str(api.get("aten_scaled_mm_v2_schema", "ERROR")).startswith("ERROR")
        ),
    }

    api_ok = all(gates.values())
    known = src.get("known_value") or []
    byk = {x.get("name"): x for x in known if isinstance(x, dict)}
    if api_ok:
        km1 = byk.get("KNOWN_M1") or {}
        gates.update({
            "K1_M1_known_value_executes": km1.get("status") == "pass",
            "K2_all_known_values_exact_bf16": len(known) == 4 and all(
                x.get("status") == "pass" and x.get("all_equal_expected") is True for x in known
            ),
            "K3_all_known_values_deterministic": len(known) == 4 and all(
                x.get("status") == "pass" and x.get("deterministic") is True for x in known
            ),
            "K4_all_known_values_finite": len(known) == 4 and all(
                x.get("status") == "pass" and x.get("finite") is True for x in known
            ),
        })

    known_ok = api_ok and all(gates.get(k) is True for k in (
        "K1_M1_known_value_executes", "K2_all_known_values_exact_bf16",
        "K3_all_known_values_deterministic", "K4_all_known_values_finite"
    ))
    if known_ok:
        perf = src.get("performance") or []
        by = {x.get("name"): x for x in perf if isinstance(x, dict)}
        q1, q2 = _p50(by.get("M1_QLIKE")), _p50(by.get("M2_QLIKE"))
        mi1, mi2 = _p50(by.get("M1_MAMBA_IN")), _p50(by.get("M2_MAMBA_IN"))
        gates.update({
            "P1_M1_QLIKE_lt_0_20ms": bool(q1 is not None and q1 < 0.20),
            "P2_M2_QLIKE_lt_0_25ms": bool(q2 is not None and q2 < 0.25),
            "P3_M2_vs_M1_QLIKE_ratio_le_1_40": bool(q1 and q2 is not None and q2 / q1 <= 1.40),
            "P4_M1_MAMBA_IN_lt_0_30ms": bool(mi1 is not None and mi1 < 0.30),
            "P5_M2_vs_M1_MAMBA_IN_ratio_le_1_40": bool(mi1 and mi2 is not None and mi2 / mi1 <= 1.40),
        })
        gates["C3_real_checkpoint_open"] = bool(
            (gates["P1_M1_QLIKE_lt_0_20ms"] and gates["P3_M2_vs_M1_QLIKE_ratio_le_1_40"])
            or (gates["P4_M1_MAMBA_IN_lt_0_30ms"] and gates["P5_M2_vs_M1_MAMBA_IN_ratio_le_1_40"])
        )

    source_gates = src.get("gates") or {}
    for k, v in gates.items():
        if source_gates.get(k) is not v:
            errors.append(f"gate mismatch {k}: source={source_gates.get(k)!r} recomputed={v!r}")

    if src.get("status") == "technical_failure":
        expected = "technical_failure"
    else:
        expected = _expected_status(src, gates)
    if src.get("status") != expected:
        errors.append(f"status mismatch source={src.get('status')!r} recomputed={expected!r}")

    out = {
        "kind": "s100_native_nvfp4_c2b_independent_verification",
        "created_utc": utc_now(),
        "source": str(SRC.relative_to(REPO)),
        "source_status": src.get("status"),
        "recomputed_status": expected,
        "recomputed_gates": gates,
        "errors": errors,
        "passed": not errors,
    }
    write_json_atomic(OUT, out, archive=True)
    print(json.dumps(out, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
