"""C2b: isolated PyTorch 2.12.1+cu132 native SM120 FP4 contract/timing probe.

Synthetic +1 FP4 values and +1 BlockWise1x16 local scales only.  This file
must be run from .venv-fp4-c2b; it deliberately does not import the Nemotron
runtime.  See S100_NATIVE_NVFP4_C2B_PREREGISTRATION.md.
"""
from __future__ import annotations

import inspect
import json
import math
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, require_gpu_free, utc_now, write_json_atomic

OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "C2B_TORCH212_CONTRACT.json"
PREREG = REPO / "pro_research" / "S100_NATIVE_NVFP4_C2B_PREREGISTRATION.md"


def _ceil(x: int, q: int) -> int:
    return ((int(x) + q - 1) // q) * q


def _scale_shapes(m: int, n: int, k: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Frozen C1 SWIZZLE_32_4_4 physical local-scale convention."""
    if k % 16:
        raise ValueError("K must be divisible by 16")
    sfp = _ceil(k // 16, 4)
    return (_ceil(m, 128), sfp), (sfp, _ceil(n, 128))


def _private_schema(torch) -> tuple[bool, str | None, str | None]:
    """Record both public top-level and dispatcher schema paths without calling them."""
    top = hasattr(torch, "_scaled_mm_v2")
    top_repr = None
    if top:
        try:
            top_repr = str(inspect.signature(torch._scaled_mm_v2))
        except Exception:
            top_repr = repr(torch._scaled_mm_v2)
    disp = None
    try:
        disp = str(torch.ops.aten._scaled_mm_v2.default._schema)
    except Exception as exc:
        disp = f"ERROR {type(exc).__name__}: {exc}"
    return top, top_repr, disp


def _make_case(torch, m: int, n: int, k: int) -> dict[str, Any]:
    if k % 16 or k % 2:
        raise ValueError("K must be divisible by 16 and 2")

    # E2M1 +1 is 0b0010. Two packed +1 values => byte 0x22.
    au8 = torch.full((m, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    bu8 = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    a = au8.view(torch.float4_e2m1fn_x2)
    b = bu8.view(torch.float4_e2m1fn_x2).t()

    ash, bsh = _scale_shapes(m, n, k)
    scale_a = torch.ones(ash, dtype=torch.float8_e4m3fn, device="cuda")
    # ABI FIX 2026-08-16 (first C2b run on the target machine): the previous
    # form built scale_b as `torch.ones((bsh[1], bsh[0])).t()`, a transposed
    # VIEW with stride (1, sfp). Torch 2.12.1 rejects that outright --
    #   ValueError: For Blockwise scaling both scales should be contiguous
    # -- so all four known-value cases failed with the API contract itself
    # fully present (G1-G7 all green). Mirroring B's transpose onto B's SCALE
    # was the error: `b` is transposed because a GEMM wants K-major operands,
    # but the block-scale tensor is a separate, independently laid out buffer
    # and the blockwise path requires it contiguous in its logical (sfp,
    # ceil(n,128)) shape. Values are synthetic all-ones, so only shape and
    # layout are under test here and this changes nothing else.
    scale_b = torch.ones(bsh, dtype=torch.float8_e4m3fn, device="cuda")

    return {
        "au8": au8,
        "bu8": bu8,
        "a": a,
        "b": b,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "scale_a_shape": list(ash),
        "scale_b_shape": list(bsh),
    }


def _call(torch, F, ScalingType, SwizzleType, c):
    return F.scaled_mm(
        c["a"], c["b"],
        scale_a=c["scale_a"],
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=c["scale_b"],
        scale_recipe_b=ScalingType.BlockWise1x16,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
        use_fast_accum=False,
    )


def _event_samples_ms(torch, fn, reps: int, rounds: int = 7) -> list[float]:
    # Separate warm-up from recorded rounds.
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    vals: list[float] = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        b.synchronize()
        vals.append(float(a.elapsed_time(b)) / reps)
    return vals


def _pct(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"count": 0, "min": None, "p50": None, "max": None}
    s = sorted(float(x) for x in vals)
    n = len(s)
    if n % 2:
        med = s[n // 2]
    else:
        med = 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {"count": n, "min": s[0], "p50": med, "max": s[-1]}


def _run_shape(torch, F, ScalingType, SwizzleType, name: str,
               m: int, n: int, k: int, reps: int,
               correctness: bool = True) -> dict[str, Any]:
    rec: dict[str, Any] = {"name": name, "M": m, "N": n, "K": k, "reps": reps}
    c = None
    try:
        c = _make_case(torch, m, n, k)
        rec.update({
            "a_storage_u8_shape": list(c["au8"].shape),
            "a_fp4_shape": list(c["a"].shape),
            "a_fp4_stride": list(c["a"].stride()),
            "b_storage_u8_shape": list(c["bu8"].shape),
            "b_fp4_shape": list(c["b"].shape),
            "b_fp4_stride": list(c["b"].stride()),
            "scale_a_shape": c["scale_a_shape"],
            "scale_b_shape": c["scale_b_shape"],
            "scale_a_stride": list(c["scale_a"].stride()),
            "scale_b_stride": list(c["scale_b"].stride()),
            # Recorded so the contiguity requirement that killed the first run
            # cannot regress silently into a future "API rejected" result.
            "scale_a_contiguous": bool(c["scale_a"].is_contiguous()),
            "scale_b_contiguous": bool(c["scale_b"].is_contiguous()),
        })

        out1 = _call(torch, F, ScalingType, SwizzleType, c)
        out2 = _call(torch, F, ScalingType, SwizzleType, c)
        torch.cuda.synchronize()
        rec["output_shape"] = list(out1.shape)
        rec["output_dtype"] = str(out1.dtype)
        rec["finite"] = bool(torch.isfinite(out1).all().item())
        rec["deterministic"] = bool(torch.equal(out1, out2))
        if correctness:
            expected = torch.tensor(float(k), dtype=torch.bfloat16, device="cuda")
            rec["expected_bf16"] = float(expected.float().item())
            rec["all_equal_expected"] = bool(torch.all(out1 == expected).item())
            d = (out1.float() - expected.float()).abs()
            rec["max_abs_error"] = float(d.max().item()) if d.numel() else 0.0
            rec["mean_abs_error"] = float(d.mean().item()) if d.numel() else 0.0
        else:
            rec["all_equal_expected"] = None

        samples = _event_samples_ms(
            torch, lambda: _call(torch, F, ScalingType, SwizzleType, c), reps
        )
        rec["event_samples_ms"] = samples
        rec["event_ms"] = _pct(samples)
        p50 = rec["event_ms"]["p50"]
        b_bytes = int(c["bu8"].numel() + c["scale_b"].numel())
        rec["b_physical_bytes"] = b_bytes
        rec["b_payload_gb_s_p50"] = b_bytes / (float(p50) * 1e6) if p50 else None
        rec["status"] = "pass"
        del out1, out2
    except Exception as exc:
        rec.update({
            "status": "execution_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        if c is not None:
            del c
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return rec


def _p50(rec: dict[str, Any] | None) -> float | None:
    if not rec or rec.get("status") != "pass":
        return None
    x = rec.get("event_ms") or {}
    v = x.get("p50")
    return float(v) if v is not None else None


def main() -> int:
    payload: dict[str, Any] = {
        "kind": "s100_native_nvfp4_c2b_torch212_contract",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": (
            "synthetic single-level BlockWise1x16 native FP4 execution/timing only; "
            "no Lightning quality, real-scale, activation-quantization or token/s claim"
        ),
    }

    try:
        require_gpu_free()
        import torch
        import torch.nn.functional as F

        ScalingType = getattr(F, "ScalingType", None)
        SwizzleType = getattr(F, "SwizzleType", None)
        private_top, private_repr, dispatcher_schema = _private_schema(torch)

        cuda_available = bool(torch.cuda.is_available())
        cap = list(torch.cuda.get_device_capability(0)) if cuda_available else None
        version_ok = str(torch.__version__).startswith("2.12.1") and str(torch.version.cuda).startswith("13.2")
        public = hasattr(F, "scaled_mm")
        fp4 = hasattr(torch, "float4_e2m1fn_x2")
        block16 = bool(ScalingType is not None and hasattr(ScalingType, "BlockWise1x16"))
        swizzle = bool(SwizzleType is not None and hasattr(SwizzleType, "SWIZZLE_32_4_4"))
        private_present = bool(private_top or not str(dispatcher_schema).startswith("ERROR"))

        api = {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cuda_available": cuda_available,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "capability": cap,
            "F_scaled_mm_present": public,
            "F_scaled_mm_signature": None,
            "float4_e2m1fn_x2_present": fp4,
            "ScalingType_BlockWise1x16_present": block16,
            "SwizzleType_32_4_4_present": swizzle,
            "torch__scaled_mm_v2_present": private_top,
            "torch__scaled_mm_v2_signature_or_repr": private_repr,
            "aten_scaled_mm_v2_schema": dispatcher_schema,
        }
        if public:
            try:
                api["F_scaled_mm_signature"] = str(inspect.signature(F.scaled_mm))
            except Exception as exc:
                api["F_scaled_mm_signature"] = f"ERROR {type(exc).__name__}: {exc}"

        payload["environment"] = environment_snapshot((Path(__file__), PREREG))
        payload["api"] = api
        gates: dict[str, bool] = {
            "G1_torch_2_12_1_cu132": bool(version_ok),
            "G2_cuda_sm120_or_newer": bool(cuda_available and cap and (cap[0], cap[1]) >= (12, 0)),
            "G3_public_scaled_mm_present": bool(public),
            "G4_fp4_dtype_present": bool(fp4),
            "G5_BlockWise1x16_present": bool(block16),
            "G6_SWIZZLE_32_4_4_present": bool(swizzle),
            "G7_scaled_mm_v2_recordable": bool(private_present),
        }
        payload["gates"] = gates
        if not all(gates.values()):
            payload.update({"status": "api_contract_failed", "completed_utc": utc_now()})
            write_json_atomic(OUT, payload, archive=True)
            print(json.dumps({"status": payload["status"], "api": api, "gates": gates,
                              "output": str(OUT)}, indent=2))
            return 0

        known_specs = [
            ("KNOWN_M1", 1, 128, 256, 200),
            ("KNOWN_M2", 2, 128, 256, 200),
            ("KNOWN_M16", 16, 128, 256, 200),
            ("KNOWN_M128", 128, 128, 256, 200),
        ]
        known = [
            _run_shape(torch, F, ScalingType, SwizzleType, name, m, n, k, reps, True)
            for name, m, n, k, reps in known_specs
        ]
        payload["known_value"] = known
        km1 = next(x for x in known if x["name"] == "KNOWN_M1")
        known_pass = bool(known) and all(
            x.get("status") == "pass"
            and x.get("finite") is True
            and x.get("deterministic") is True
            and x.get("all_equal_expected") is True
            for x in known
        )
        gates.update({
            "K1_M1_known_value_executes": km1.get("status") == "pass",
            "K2_all_known_values_exact_bf16": known_pass,
            "K3_all_known_values_deterministic": bool(known) and all(
                x.get("status") == "pass" and x.get("deterministic") is True for x in known
            ),
            "K4_all_known_values_finite": bool(known) and all(
                x.get("status") == "pass" and x.get("finite") is True for x in known
            ),
        })

        if not all(gates[k] for k in (
            "K1_M1_known_value_executes", "K2_all_known_values_exact_bf16",
            "K3_all_known_values_deterministic", "K4_all_known_values_finite"
        )):
            payload.update({"gates": gates, "status": "native_execution_failed",
                            "completed_utc": utc_now()})
            write_json_atomic(OUT, payload, archive=True)
            print(json.dumps({"status": payload["status"], "known_value": known,
                              "gates": gates, "output": str(OUT)}, indent=2))
            return 0

        perf_specs = [
            ("M1_QLIKE", 1, 4096, 2688, 300),
            ("M2_QLIKE", 2, 4096, 2688, 300),
            ("M1_MAMBA_IN", 1, 10304, 2688, 200),
            ("M2_MAMBA_IN", 2, 10304, 2688, 200),
            ("M1_LM_HEAD", 1, 131072, 2688, 40),
            ("M2_LM_HEAD", 2, 131072, 2688, 40),
        ]
        perf: list[dict[str, Any]] = []
        for name, m, n, k, reps in perf_specs:
            try:
                free, _total = torch.cuda.mem_get_info()
                sfp = _ceil(k // 16, 4)
                npad = _ceil(n, 128)
                est = n * (k // 2) + npad * sfp + m * (k // 2) + _ceil(m, 128) * sfp
                if est > int(free * 0.65):
                    perf.append({"name": name, "M": m, "N": n, "K": k,
                                 "status": "not_run_memory_gate",
                                 "estimated_input_bytes": int(est), "free_bytes": int(free)})
                    continue
            except Exception:
                pass
            perf.append(_run_shape(torch, F, ScalingType, SwizzleType,
                                   name, m, n, k, reps, True))
        payload["performance"] = perf
        by = {x["name"]: x for x in perf}
        q1, q2 = _p50(by.get("M1_QLIKE")), _p50(by.get("M2_QLIKE"))
        mi1, mi2 = _p50(by.get("M1_MAMBA_IN")), _p50(by.get("M2_MAMBA_IN"))
        lh1, lh2 = _p50(by.get("M1_LM_HEAD")), _p50(by.get("M2_LM_HEAD"))

        perf_gates = {
            "P1_M1_QLIKE_lt_0_20ms": bool(q1 is not None and q1 < 0.20),
            "P2_M2_QLIKE_lt_0_25ms": bool(q2 is not None and q2 < 0.25),
            "P3_M2_vs_M1_QLIKE_ratio_le_1_40": bool(q1 and q2 is not None and q2 / q1 <= 1.40),
            "P4_M1_MAMBA_IN_lt_0_30ms": bool(mi1 is not None and mi1 < 0.30),
            "P5_M2_vs_M1_MAMBA_IN_ratio_le_1_40": bool(mi1 and mi2 is not None and mi2 / mi1 <= 1.40),
        }
        gates.update(perf_gates)
        c3_open = bool(
            (perf_gates["P1_M1_QLIKE_lt_0_20ms"] and perf_gates["P3_M2_vs_M1_QLIKE_ratio_le_1_40"])
            or (perf_gates["P4_M1_MAMBA_IN_lt_0_30ms"] and perf_gates["P5_M2_vs_M1_MAMBA_IN_ratio_le_1_40"])
        )
        gates["C3_real_checkpoint_open"] = c3_open
        ratios = {
            "qlike_M2_over_M1": (q2 / q1) if q1 and q2 is not None else None,
            "mamba_in_M2_over_M1": (mi2 / mi1) if mi1 and mi2 is not None else None,
            "lm_head_M2_over_M1": (lh2 / lh1) if lh1 and lh2 is not None else None,
        }
        payload["summary"] = {
            "M1_QLIKE_p50_ms": q1, "M2_QLIKE_p50_ms": q2,
            "M1_MAMBA_IN_p50_ms": mi1, "M2_MAMBA_IN_p50_ms": mi2,
            "M1_LM_HEAD_p50_ms": lh1, "M2_LM_HEAD_p50_ms": lh2,
            **ratios,
        }
        payload.update({
            "gates": gates,
            "status": "native_execution_candidate" if c3_open else "native_executes_below_c3_gate",
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "api": payload.get("api"),
        "summary": payload.get("summary"),
        "gates": payload.get("gates"),
        "output": str(OUT),
        "error": (payload.get("error") or {}).get("message"),
    }, indent=2))
    return 0 if payload.get("status") != "technical_failure" else 2


if __name__ == "__main__":
    raise SystemExit(main())
