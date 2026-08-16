"""C2: prove/measure the local PyTorch native SM120 FP4 scaled-MM contract.

Synthetic known-value FP4 data only.  No Lightning activation quantization or
model-quality claim.  See S100_NATIVE_NVFP4_C2_TORCH_CONTRACT_PREREGISTRATION.md.
"""
from __future__ import annotations

import inspect
import json
import math
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, require_gpu_free, utc_now

OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "C2_TORCH_CONTRACT.json"
PREREG = REPO / "pro_research" / "S100_NATIVE_NVFP4_C2_TORCH_CONTRACT_PREREGISTRATION.md"


def _write(payload: dict[str, Any]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _ceil(x: int, q: int) -> int:
    return ((int(x) + q - 1) // q) * q


def _scale_shapes(m: int, n: int, k: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Frozen C1 SWIZZLE_32_4_4 physical shape convention for all-one scales."""
    sf = k // 16
    sfp = _ceil(sf, 4)
    # C1 pads scale rows in 128-row blocks. B is supplied transposed to the GEMM,
    # so its scale storage is natural [Npad,SFpad] then transposed.
    return (_ceil(m, 128), sfp), (sfp, _ceil(n, 128))


def _make_case(torch, m: int, n: int, k: int):
    if k % 16 or k % 2:
        raise ValueError("K must be divisible by 16 and 2")
    # E2M1 +1 is code 0b0010; two packed +1 values => 0x22.
    au8 = torch.full((m, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    # Natural checkpoint-like weight orientation [N,K/2], then logical [K,N].
    bu8 = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    afp4 = au8.view(torch.float4_e2m1fn_x2)
    bfp4 = bu8.view(torch.float4_e2m1fn_x2).t()
    ash, bsh = _scale_shapes(m, n, k)
    # Exact E4M3 1.0. All-one bytes make the C1 permutation value-invariant,
    # while the allocated physical count/shape still follows the frozen layout.
    sa = torch.ones(ash, dtype=torch.float8_e4m3fn, device="cuda")
    sb_nat = torch.ones((bsh[1], bsh[0]), dtype=torch.float8_e4m3fn, device="cuda")
    sb = sb_nat.t()
    return {"au8": au8, "bu8": bu8, "a": afp4, "b": bfp4,
            "scale_a": sa, "scale_b": sb,
            "scale_a_shape": list(ash), "scale_b_shape": list(bsh)}


def _call(F, ScalingType, SwizzleType, case, torch):
    return F.scaled_mm(
        case["a"], case["b"],
        scale_a=case["scale_a"],
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=case["scale_b"],
        scale_recipe_b=ScalingType.BlockWise1x16,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
        use_fast_accum=False,
    )


def _time(torch, fn, reps: int) -> float:
    fn(); torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); b.synchronize()
    return float(a.elapsed_time(b)) / reps


def _run_shape(torch, F, ScalingType, SwizzleType,
               name: str, m: int, n: int, k: int, reps: int,
               correctness: bool = True) -> dict[str, Any]:
    rec: dict[str, Any] = {"name": name, "M": m, "N": n, "K": k, "reps": reps}
    try:
        c = _make_case(torch, m, n, k)
        rec.update({
            "a_storage_shape_u8": list(c["au8"].shape),
            "a_fp4_shape_reported": list(c["a"].shape),
            "a_fp4_stride": list(c["a"].stride()),
            "b_storage_shape_u8": list(c["bu8"].shape),
            "b_fp4_shape_reported": list(c["b"].shape),
            "b_fp4_stride": list(c["b"].stride()),
            "scale_a_shape": c["scale_a_shape"],
            "scale_b_shape": c["scale_b_shape"],
        })
        out1 = _call(F, ScalingType, SwizzleType, c, torch)
        out2 = _call(F, ScalingType, SwizzleType, c, torch)
        torch.cuda.synchronize()
        rec["output_shape"] = list(out1.shape)
        rec["output_dtype"] = str(out1.dtype)
        rec["finite"] = bool(torch.isfinite(out1).all().item())
        rec["deterministic"] = bool(torch.equal(out1, out2))
        if correctness:
            expected = torch.tensor(float(k), dtype=torch.bfloat16, device="cuda")
            rec["expected_bf16"] = float(expected.float().item())
            rec["all_equal_expected"] = bool(torch.all(out1 == expected).item())
            if out1.numel():
                diff = (out1.float() - expected.float()).abs()
                rec["max_abs_error"] = float(diff.max().item())
                rec["mean_abs_error"] = float(diff.mean().item())
        else:
            rec["all_equal_expected"] = None
        rec["event_ms"] = _time(torch, lambda: _call(F, ScalingType, SwizzleType, c, torch), reps)
        # Physical B payload diagnostic: packed data + padded local scales.
        b_bytes = int(c["bu8"].numel() + c["scale_b"].numel())
        rec["b_physical_bytes"] = b_bytes
        rec["b_payload_gb_s"] = b_bytes / (rec["event_ms"] * 1e6) if rec["event_ms"] else None
        rec["status"] = "pass"
        del out1, out2, c
        torch.cuda.empty_cache()
    except Exception as exc:
        rec.update({
            "status": "execution_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        torch.cuda.empty_cache()
    return rec


def main() -> int:
    payload: dict[str, Any] = {
        "kind": "s100_native_nvfp4_c2_torch_contract",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "synthetic native FP4 execution/API/timing only; no Lightning numerical quality or end-to-end speed claim",
    }
    try:
        require_gpu_free()
        import torch
        import torch.nn.functional as F

        public = hasattr(F, "scaled_mm")
        fp4 = hasattr(torch, "float4_e2m1fn_x2")
        ScalingType = getattr(F, "ScalingType", None)
        SwizzleType = getattr(F, "SwizzleType", None)
        enums = bool(
            ScalingType is not None and SwizzleType is not None and
            hasattr(ScalingType, "BlockWise1x16") and
            hasattr(SwizzleType, "SWIZZLE_32_4_4")
        )
        api = {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "F_scaled_mm_present": public,
            "float4_e2m1fn_x2_present": fp4,
            "ScalingType_BlockWise1x16_present": bool(ScalingType is not None and hasattr(ScalingType, "BlockWise1x16")),
            "SwizzleType_32_4_4_present": bool(SwizzleType is not None and hasattr(SwizzleType, "SWIZZLE_32_4_4")),
        }
        if public:
            try:
                api["F_scaled_mm_signature"] = str(inspect.signature(F.scaled_mm))
            except Exception as exc:
                api["F_scaled_mm_signature_error"] = f"{type(exc).__name__}: {exc}"
        # Record the private op schema for diagnosis only; C2 does not fall back to it.
        try:
            api["aten_scaled_mm_v2_schema"] = str(torch.ops.aten._scaled_mm_v2.default._schema)
        except Exception as exc:
            api["aten_scaled_mm_v2_schema_error"] = f"{type(exc).__name__}: {exc}"

        payload["environment"] = environment_snapshot((Path(__file__), PREREG))
        payload["api"] = api
        gates = {
            "G1_public_scaled_mm_present": bool(public),
            "G2_fp4_dtype_present": bool(fp4),
            "G3_scaling_and_swizzle_enums_present": bool(enums),
        }
        if not all(gates.values()) or not torch.cuda.is_available():
            payload.update({"gates": gates, "status": "api_contract_failed",
                            "completed_utc": utc_now()})
            _write(payload)
            print(json.dumps(payload, indent=2))
            return 0

        known = []
        for m in (1, 2, 16, 128):
            known.append(_run_shape(torch, F, ScalingType, SwizzleType,
                                    f"KNOWN_M{m}", m, 128, 256, 100, True))
        payload["known_value"] = known
        m1 = next(x for x in known if x["name"] == "KNOWN_M1")
        gates.update({
            "G4_M1_known_value_executes": m1.get("status") == "pass",
            "G5_all_executed_known_value_outputs_equal_expected_bf16": bool(known) and all(
                x.get("status") != "pass" or bool(x.get("all_equal_expected")) for x in known
            ),
            "G6_deterministic_repeat": bool(known) and all(
                x.get("status") != "pass" or bool(x.get("deterministic")) for x in known
            ),
            "G7_no_nan_inf": bool(known) and all(
                x.get("status") != "pass" or bool(x.get("finite")) for x in known
            ),
        })

        perf_specs = [
            ("M1_QLIKE", 1, 4096, 2688, 100),
            ("M2_QLIKE", 2, 4096, 2688, 100),
            ("M1_MAMBA_IN", 1, 10304, 2688, 100),
            ("M2_MAMBA_IN", 2, 10304, 2688, 100),
            ("M1_LM_HEAD", 1, 131072, 2688, 30),
            ("M2_LM_HEAD", 2, 131072, 2688, 30),
        ]
        perf = []
        # Only time larger shapes if the decisive M1 known-value contract is valid.
        if all(gates.values()):
            for name, m, n, k, reps in perf_specs:
                try:
                    free, _total = torch.cuda.mem_get_info()
                    # Conservative data+scale estimate; skip instead of risking OOM.
                    sfp = _ceil(k // 16, 4)
                    npad = _ceil(n, 128)
                    est = n * (k // 2) + npad * sfp + m * (k // 2) + _ceil(m, 128) * sfp
                    if est > int(free * 0.70):
                        perf.append({"name": name, "M": m, "N": n, "K": k,
                                     "status": "not_run_memory_gate",
                                     "estimated_input_bytes": int(est),
                                     "free_bytes": int(free)})
                        continue
                except Exception:
                    pass
                perf.append(_run_shape(torch, F, ScalingType, SwizzleType,
                                       name, m, n, k, reps, True))
        payload["performance"] = perf
        by = {r["name"]: r for r in perf}
        def ms(name):
            r = by.get(name) or {}
            return float(r["event_ms"]) if r.get("status") == "pass" else None
        q1, q2 = ms("M1_QLIKE"), ms("M2_QLIKE")
        mi1, mi2 = ms("M1_MAMBA_IN"), ms("M2_MAMBA_IN")
        perf_gates = {
            "P1_M1_QLIKE_lt_0_20ms": bool(q1 is not None and q1 < 0.20),
            "P2_M2_QLIKE_lt_0_25ms": bool(q2 is not None and q2 < 0.25),
            "P3_M2_vs_M1_QLIKE_time_ratio_le_1_40": bool(q1 and q2 is not None and q2 / q1 <= 1.40),
            "P4_M1_MAMBA_IN_lt_0_30ms": bool(mi1 is not None and mi1 < 0.30),
            "P5_M2_vs_M1_MAMBA_IN_ratio_le_1_40": bool(mi1 and mi2 is not None and mi2 / mi1 <= 1.40),
        }
        gates.update(perf_gates)
        correction = all(gates[k] for k in (
            "G1_public_scaled_mm_present", "G2_fp4_dtype_present",
            "G3_scaling_and_swizzle_enums_present", "G4_M1_known_value_executes",
            "G5_all_executed_known_value_outputs_equal_expected_bf16",
            "G6_deterministic_repeat", "G7_no_nan_inf"))
        c3_open = bool(correction and (
            (perf_gates["P1_M1_QLIKE_lt_0_20ms"] and perf_gates["P3_M2_vs_M1_QLIKE_time_ratio_le_1_40"])
            or (perf_gates["P4_M1_MAMBA_IN_lt_0_30ms"] and perf_gates["P5_M2_vs_M1_MAMBA_IN_ratio_le_1_40"])
        ))
        gates["C3_real_checkpoint_open"] = c3_open
        if not correction:
            status = "api_contract_failed"
        elif c3_open:
            status = "native_execution_candidate"
        else:
            status = "native_executes_below_c3_gate"
        payload.update({"gates": gates, "status": status,
                        "completed_utc": utc_now()})
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })

    _write(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "api": payload.get("api"),
        "known_value": payload.get("known_value"),
        "performance": payload.get("performance"),
        "gates": payload.get("gates"),
        "error": payload.get("error"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") != "technical_failure" else 2


if __name__ == "__main__":
    raise SystemExit(main())
