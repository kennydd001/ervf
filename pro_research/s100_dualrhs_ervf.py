"""S100 DualRHS-ERVF real-shape microbenchmark.

Tests the preregistered streamed-virtual-accumulator two-RHS primitive against
V18's adopted exact single-RHS dispatch, called twice.  No runtime patching and
no token-level speed claim in this phase.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import REPO, environment_snapshot, require_gpu_free, utc_now
from dualrhs_ervf import DualRHSSVAE, cpu_reduction_selftest
from ervf_dense import DenseERVF
from graph_e1f22 import _new_runtime
from selective_ervf_v3 import BF16_ERVF_SHAPES, FP8_ERVF_SHAPES

RESULT_DIR = REPO / "pro_research" / "results" / "s100_dualrhs"
OUT = RESULT_DIR / "PRO_S100_DUALRHS_ERVF.json"
PREREG = REPO / "pro_research" / "S100_DUALRHS_ERVF_PREREGISTRATION.md"
K2_OBSERVED_P50_MS = 38.67655


@dataclass
class Case:
    name: str
    kind: str
    rows: int
    cols: int
    calls_per_token: int
    W: Any = None
    codes: Any = None
    scales: Any = None
    scale: float = 1.0
    apply_relu2: bool = False
    out_scale: float = 1.0


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _collect_cases(rt) -> list[Case]:
    cases: list[Case] = []
    if rt.attn_layers:
        i = rt.attn_layers[0]
        d = rt.layer[i]
        hq = rt.n_heads * rt.head_dim
        hkv = rt.n_kv * rt.head_dim
        n = len(rt.attn_layers)
        cases += [
            Case("attn_q_bf16", "bf16", hq, rt.hidden, n, W=d["q_proj"]),
            Case("attn_k_bf16", "bf16", hkv, rt.hidden, n, W=d["k_proj"]),
            Case("attn_v_bf16", "bf16", hkv, rt.hidden, n, W=d["v_proj"]),
            Case("attn_o_bf16", "bf16", rt.hidden, hq, n, W=d["o_proj"]),
        ]
    if rt.mamba_layers:
        i = rt.mamba_layers[0]
        d = rt.layer[i]
        n = len(rt.mamba_layers)
        if d["in_k"] == "fp8_tensor":
            cases.append(Case("mamba_in_fp8", "fp8", int(rt.proj.size), rt.hidden,
                              n, W=d["in_w8"], scale=float(d["in_s"])))
        elif d["in_k"] == "bf16":
            cases.append(Case("mamba_in_bf16", "bf16", int(rt.proj.size), rt.hidden,
                              n, W=d["in_w"]))
        if d["out_k"] == "fp8_tensor":
            cases.append(Case("mamba_out_fp8", "fp8", rt.hidden, rt.d_inner,
                              n, W=d["out_w8"], scale=float(d["out_s"])))
        elif d["out_k"] == "bf16":
            cases.append(Case("mamba_out_bf16", "bf16", rt.hidden, rt.d_inner,
                              n, W=d["out_w"]))
    if rt.moe_layers:
        i = rt.moe_layers[0]
        d = rt.layer[i]
        n = len(rt.moe_layers)
        cases += [
            Case("router_f32", "f32", rt.n_experts, rt.hidden, n, W=d["gate_w"]),
            Case("shared_up_nvfp4", "nvfp4", rt.shared_inter, rt.hidden, n,
                 codes=d["sh_up_c"], scales=d["sh_up_s"], scale=float(d["sh_up_g"]),
                 apply_relu2=True),
            Case("shared_down_nvfp4", "nvfp4", rt.hidden, rt.shared_inter, n,
                 codes=d["sh_dn_c"], scales=d["sh_dn_s"], scale=float(d["sh_dn_g"])),
        ]
    if rt.lm_head_kind == "nvfp4":
        cases.append(Case("lm_head_nvfp4", "nvfp4", rt.vocab, rt.hidden, 1,
                          codes=rt.lm_head_codes, scales=rt.lm_head_scales,
                          scale=float(rt.lm_head_g)))
    elif rt.lm_head is not None:
        cases.append(Case("lm_head_bf16", "bf16", rt.vocab, rt.hidden, 1,
                          W=rt.lm_head))
    return cases


def _family_presence(cases: list[Case]) -> dict[str, bool]:
    names = {c.name for c in cases}
    return {
        "attn_q": "attn_q_bf16" in names,
        "attn_k": "attn_k_bf16" in names,
        "attn_v": "attn_v_bf16" in names,
        "attn_o": "attn_o_bf16" in names,
        "mamba_in": any(x.startswith("mamba_in_") for x in names),
        "mamba_out": any(x.startswith("mamba_out_") for x in names),
        "router": "router_f32" in names,
        "shared_up": "shared_up_nvfp4" in names,
        "shared_down": "shared_down_nvfp4" in names,
        "lm_head": any(x.startswith("lm_head_") for x in names),
    }


def _weight_bytes(case: Case) -> int:
    if case.kind in {"bf16", "f32", "fp8"}:
        return int(case.W.nbytes)
    if case.kind == "nvfp4":
        return int(case.codes.nbytes + case.scales.nbytes)
    raise ValueError(case.kind)


def _ref_single(rt, dense: DenseERVF, case: Case, out, x) -> None:
    shape = (int(case.rows), int(case.cols))
    if case.kind == "bf16":
        if shape in BF16_ERVF_SHAPES:
            return dense.mv_bf16(out, case.W, x, case.rows, case.cols)
        return rt.k.mv_bf16(out, case.W, x, case.rows, case.cols)
    if case.kind == "f32":
        return rt.k.mv_f32(out, case.W, x, case.rows, case.cols)
    if case.kind == "fp8":
        if shape in FP8_ERVF_SHAPES:
            return dense.mv_fp8_tensor(out, case.W, x, case.scale, case.rows, case.cols)
        return rt.k.mv_fp8_tensor(out, case.W, x, case.scale, case.rows, case.cols)
    if case.kind == "nvfp4":
        return rt.fused.gemv_into(out, case.codes, case.scales, x, case.scale,
                                  case.rows, case.cols,
                                  apply_relu2=case.apply_relu2,
                                  out_scale=case.out_scale)
    raise ValueError(case.kind)


def _cand_pair(rt, dual: DualRHSSVAE, case: Case, out, X) -> None:
    if case.kind == "bf16":
        return dual.bf16(out, case.W, X, case.rows, case.cols)
    if case.kind == "f32":
        return dual.f32(out, case.W, X, case.rows, case.cols)
    if case.kind == "fp8":
        return dual.fp8(out, case.W, X, case.scale, case.rows, case.cols)
    if case.kind == "nvfp4":
        return dual.nvfp4(out, case.codes, case.scales, rt.fused.e2m1,
                          rt.fused.e4m3, X, case.scale, case.rows, case.cols,
                          apply_relu2=case.apply_relu2,
                          out_scale=case.out_scale)
    raise ValueError(case.kind)


def _event_ms(cp, fn: Callable[[], None], repeats: int) -> float:
    fn()
    cp.cuda.get_current_stream().synchronize()
    a, b = cp.cuda.Event(), cp.cuda.Event()
    a.record()
    for _ in range(repeats):
        fn()
    b.record()
    b.synchronize()
    return float(cp.cuda.get_elapsed_time(a, b)) / repeats


def _bench_case(rt, dense: DenseERVF, dual: DualRHSSVAE, case: Case,
                mode: str) -> dict[str, Any]:
    cp = rt.cp
    correctness_batches = 1 if mode == "smoke" else 3
    repeats = 6 if mode == "smoke" else 40
    rounds = 2 if mode == "smoke" else 6
    correctness = []

    # Fixed timing activation pair, separate from correctness batches.
    trng = np.random.default_rng((abs(hash(case.name)) ^ 0xD00A1) & 0xFFFFFFFF)
    X = cp.asarray(trng.standard_normal((2, case.cols)).astype(np.float32).reshape(-1))
    ref = cp.empty(2 * case.rows, dtype=cp.float32)
    cand = cp.empty_like(ref)
    cand2 = cp.empty_like(ref)

    def ref_pair():
        _ref_single(rt, dense, case, ref[:case.rows], X[:case.cols])
        _ref_single(rt, dense, case, ref[case.rows:], X[case.cols:])

    def cand_pair():
        _cand_pair(rt, dual, case, cand, X)

    # Correctness on independent activation pairs.  Each batch compares both
    # RHS streams and a deterministic candidate repeat at raw FP32 bit level.
    for b in range(correctness_batches):
        rng = np.random.default_rng((0xD0A10000 + 997 * b + (abs(hash(case.name)) & 0xFFFF)) & 0xFFFFFFFF)
        xb = cp.asarray(rng.standard_normal((2, case.cols)).astype(np.float32).reshape(-1))
        rb = cp.empty(2 * case.rows, dtype=cp.float32)
        cb = cp.empty_like(rb)
        db = cp.empty_like(rb)
        _ref_single(rt, dense, case, rb[:case.rows], xb[:case.cols])
        _ref_single(rt, dense, case, rb[case.rows:], xb[case.cols:])
        _cand_pair(rt, dual, case, cb, xb)
        _cand_pair(rt, dual, case, db, xb)
        cp.cuda.get_current_stream().synchronize()
        rh = cp.asnumpy(rb).view(np.uint32)
        ch = cp.asnumpy(cb).view(np.uint32)
        dh = cp.asnumpy(db).view(np.uint32)
        r0 = rh[:case.rows]; r1 = rh[case.rows:]
        c0 = ch[:case.rows]; c1 = ch[case.rows:]
        d0 = dh[:case.rows]; d1 = dh[case.rows:]
        correctness.append({
            "batch": b,
            "rhs0_mismatches": int(np.count_nonzero(r0 != c0)),
            "rhs1_mismatches": int(np.count_nonzero(r1 != c1)),
            "repeat_rhs0_mismatches": int(np.count_nonzero(c0 != d0)),
            "repeat_rhs1_mismatches": int(np.count_nonzero(c1 != d1)),
            "finite": bool(np.isfinite(ch.view(np.float32)).all()),
        })
        del xb, rb, cb, db

    # Compile/warm both paths before ABBA.
    ref_pair(); cand_pair(); _cand_pair(rt, dual, case, cand2, X)
    cp.cuda.get_current_stream().synchronize()

    ref_a, cand_a, cand_b, ref_b = [], [], [], []
    for _ in range(rounds):
        ref_a.append(_event_ms(cp, ref_pair, repeats))
        cand_a.append(_event_ms(cp, cand_pair, repeats))
        cand_b.append(_event_ms(cp, cand_pair, repeats))
        ref_b.append(_event_ms(cp, ref_pair, repeats))
    ra = float(np.median(ref_a)); rbm = float(np.median(ref_b))
    ca = float(np.median(cand_a)); cbm = float(np.median(cand_b))
    ref_mid = 0.5 * (ra + rbm)
    cand_mid = 0.5 * (ca + cbm)
    drift = abs(ra - rbm) / ref_mid if ref_mid > 0 else math.inf
    wb = _weight_bytes(case)
    all_exact = all(
        x["rhs0_mismatches"] == 0 and x["rhs1_mismatches"] == 0 and
        x["repeat_rhs0_mismatches"] == 0 and x["repeat_rhs1_mismatches"] == 0 and
        x["finite"] for x in correctness
    )
    return {
        "name": case.name,
        "kind": case.kind,
        "rows": int(case.rows),
        "cols": int(case.cols),
        "calls_per_token": int(case.calls_per_token),
        "matrix_weight_bytes": wb,
        "dual_dynamic_shared_bytes": dual.shared_bytes(case.cols),
        "correctness": correctness,
        "all_exact": bool(all_exact),
        "reference_a_ms_per_pair": ra,
        "reference_b_ms_per_pair": rbm,
        "reference_mid_ms_per_pair": ref_mid,
        "candidate_a_ms_per_pair": ca,
        "candidate_b_ms_per_pair": cbm,
        "candidate_mid_ms_per_pair": cand_mid,
        "speedup": ref_mid / cand_mid if cand_mid > 0 else None,
        "reference_drift_fraction": drift,
        "reference_effective_weight_gb_s": (2.0 * wb) / (ref_mid * 1e6) if ref_mid else None,
        "candidate_physical_weight_gb_s": wb / (cand_mid * 1e6) if cand_mid else None,
        "raw_reference_a_ms": ref_a,
        "raw_candidate_a_ms": cand_a,
        "raw_candidate_b_ms": cand_b,
        "raw_reference_b_ms": ref_b,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload: dict[str, Any] = {
        "kind": "s100_dualrhs_ervf",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "real-shape exact two-RHS common-weight microbenchmark only; no full K2 or S100 claim",
        "k2_observed_anchor_ms": K2_OBSERVED_P50_MS,
        "cpu_reduction_selftest": cpu_reduction_selftest(),
    }
    try:
        if not payload["cpu_reduction_selftest"]["passed"]:
            raise RuntimeError("DualRHS exact reduction CPU selftest failed")
        require_gpu_free()
        import cupy as cp

        payload["environment_start"] = environment_snapshot((
            Path(__file__), PREREG,
            REPO / "pro_research" / "dualrhs_ervf.py",
            REPO / "pro_research" / "ervf_dense.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
        ))
        rt = _new_runtime(72)
        dense = DenseERVF()
        dual = DualRHSSVAE()
        cases = _collect_cases(rt)
        presence = _family_presence(cases)
        records = []
        for case in cases:
            rec = _bench_case(rt, dense, dual, case, args.mode)
            records.append(rec)
            print(
                f"DualRHS {case.name:<22} exact={rec['all_exact']} "
                f"ref2={rec['reference_mid_ms_per_pair']:.4f}ms "
                f"cand={rec['candidate_mid_ms_per_pair']:.4f}ms "
                f"speedup={rec['speedup']:.3f}x drift={100*rec['reference_drift_fraction']:.2f}%",
                flush=True,
            )

        all_exact = bool(records) and all(r["all_exact"] for r in records)
        all_families = all(presence.values()) and len(records) == 10
        weighted_ref = sum(float(r["reference_mid_ms_per_pair"]) * int(r["calls_per_token"]) for r in records)
        weighted_cand = sum(float(r["candidate_mid_ms_per_pair"]) * int(r["calls_per_token"]) for r in records)
        weighted_speedup = weighted_ref / weighted_cand if weighted_cand > 0 else 0.0
        saving = weighted_ref - weighted_cand
        projected = K2_OBSERVED_P50_MS - saving
        max_drift = max((float(r["reference_drift_fraction"]) for r in records), default=math.inf)
        min_speed = min((float(r["speedup"]) for r in records), default=0.0)
        by_name = {r["name"]: r for r in records}
        mamba_in = next((float(r["speedup"]) for r in records if r["name"].startswith("mamba_in_")), 0.0)
        lm_head = next((float(r["speedup"]) for r in records if r["name"].startswith("lm_head_")), 0.0)

        perf = {
            "P1_weighted_common_projection_speedup_ge_1_50x": weighted_speedup >= 1.50,
            "P2_projected_common_projection_saving_ge_6_0ms_per_K2_block": saving >= 6.0,
            "P3_mamba_in_speedup_ge_1_40x": mamba_in >= 1.40,
            "P4_lm_head_speedup_ge_1_35x": lm_head >= 1.35,
            "P5_no_registered_family_below_0_90x": min_speed >= 0.90,
            "D1_max_reference_drift_le_0_07": (max_drift <= 0.07) if args.mode == "full" else None,
        }
        correctness_gates = {
            "G1_G4_all_outputs_bitexact_deterministic_finite": bool(all_exact),
            "G5_all_10_families_present": bool(all_families),
        }
        integration_open = bool(
            all_exact and all_families and perf["P1_weighted_common_projection_speedup_ge_1_50x"] and
            perf["P2_projected_common_projection_saving_ge_6_0ms_per_K2_block"] and
            perf["P5_no_registered_family_below_0_90x"] and
            (args.mode != "full" or perf["D1_max_reference_drift_le_0_07"])
        )
        if not all_exact or not all_families:
            status = "correctness_failed"
        elif args.mode == "smoke":
            status = "smoke_pass"
        elif integration_open:
            status = "dualrhs_integration_candidate"
        elif saving < 3.0:
            status = "micro_null_close_geometry"
        else:
            status = "micro_below_integration_gate"

        payload.update({
            "geometry": {"width": 16, "rows_per_block": 6, "threads_per_block": 96,
                         "virtual_reference_threads": 256},
            "kernel_attributes": dual.attributes(),
            "family_presence": presence,
            "records": records,
            "summary": {
                "weighted_reference_ms_per_K2_common_projection_subset": weighted_ref,
                "weighted_candidate_ms_per_K2_common_projection_subset": weighted_cand,
                "weighted_speedup": weighted_speedup,
                "projected_common_projection_saving_ms_per_K2_block": saving,
                "projection_only_projected_K2_block_ms_from_38_67655": projected,
                "max_reference_drift_fraction": max_drift,
                "min_family_speedup": min_speed,
                "mamba_in_speedup": mamba_in,
                "lm_head_speedup": lm_head,
            },
            "gates": {**correctness_gates, **perf, "integration_open": integration_open},
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })

        del rt, dense, dual
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
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
        "output": str(OUT),
        "summary": payload.get("summary"),
        "gates": payload.get("gates"),
        "error": payload.get("error"),
    }, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
