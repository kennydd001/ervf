"""E100-MRHS256: exact common-weight reuse with one reference tid per thread.

Primary arm N=16; N=4/8 are frozen cross-check/diagnostic arms.  Component
experiment only -- no full-model throughput claim is made here.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, require_gpu_free, utc_now
from e100_mrhs import Case, _baseline_call, _new_runtime, _time_cuda
from mrhs256_exact_kernels import ExactMRHS256

RESULT_DIR = REPO / "pro_research" / "results" / "e100_mrhs256"
OUT = RESULT_DIR / "PRO_E100_MRHS256.json"
PREREG = REPO / "pro_research" / "E100_MRHS256_PREREGISTRATION.md"


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _append_quant_case(cases, unsupported, name, kind, d, prefix, rows, cols, calls):
    if kind == "nvfp4":
        ck, sk, gk = f"{prefix}_codes", f"{prefix}_scales", f"{prefix}_g"
        if all(k in d for k in (ck, sk, gk)):
            cases.append(Case(name + "_nvfp4", "nvfp4", int(rows), int(cols), calls,
                              codes=d[ck], scales=d[sk], scale=float(d[gk])))
        else:
            unsupported.append({"name": name, "reason": f"missing {ck}/{sk}/{gk}"})
    elif kind == "fp8_tensor":
        wk, sk = f"{prefix}_w8", f"{prefix}_s"
        if wk in d and sk in d:
            cases.append(Case(name + "_fp8", "fp8", int(rows), int(cols), calls,
                              W=d[wk], scale=float(d[sk])))
        else:
            unsupported.append({"name": name, "reason": f"missing {wk}/{sk}"})
    elif kind == "bf16":
        wk = f"{prefix}_w"
        if wk in d:
            cases.append(Case(name + "_bf16", "bf16", int(rows), int(cols), calls, W=d[wk]))
        else:
            unsupported.append({"name": name, "reason": f"missing {wk}"})
    else:
        unsupported.append({"name": name, "reason": f"unsupported storage kind {kind!r}"})


def _collect_cases(rt):
    cases: list[Case] = []
    unsupported: list[dict[str, str]] = []

    if rt.attn_layers:
        d = rt.layer[rt.attn_layers[0]]
        hq = rt.n_heads * rt.head_dim
        hkv = rt.n_kv * rt.head_dim
        for name, key, rows, cols in (
            ("attn_q_bf16", "q_proj", hq, rt.hidden),
            ("attn_k_bf16", "k_proj", hkv, rt.hidden),
            ("attn_v_bf16", "v_proj", hkv, rt.hidden),
            ("attn_o_bf16", "o_proj", rt.hidden, hq),
        ):
            if key in d:
                cases.append(Case(name, "bf16", int(rows), int(cols), len(rt.attn_layers), W=d[key]))
            else:
                unsupported.append({"name": name, "reason": f"missing {key}"})

    if rt.mamba_layers:
        d = rt.layer[rt.mamba_layers[0]]
        calls = len(rt.mamba_layers)
        _append_quant_case(cases, unsupported, "mamba_in", d.get("in_k"), d, "in",
                           int(rt.proj.size), rt.hidden, calls)
        _append_quant_case(cases, unsupported, "mamba_out", d.get("out_k"), d, "out",
                           rt.hidden, rt.d_inner, calls)
    else:
        unsupported += [
            {"name": "mamba_in", "reason": "no Mamba layers"},
            {"name": "mamba_out", "reason": "no Mamba layers"},
        ]

    if rt.moe_layers:
        d = rt.layer[rt.moe_layers[0]]
        calls = len(rt.moe_layers)
        if "gate_w" in d:
            cases.append(Case("router_f32", "f32", rt.n_experts, rt.hidden, calls, W=d["gate_w"]))
        else:
            unsupported.append({"name": "router_f32", "reason": "missing gate_w"})
        if all(k in d for k in ("sh_up_c", "sh_up_s", "sh_up_g")):
            cases.append(Case("shared_up_nvfp4", "nvfp4", rt.shared_inter, rt.hidden, calls,
                              codes=d["sh_up_c"], scales=d["sh_up_s"], scale=float(d["sh_up_g"]),
                              apply_relu2=True))
        else:
            unsupported.append({"name": "shared_up_nvfp4", "reason": "missing shared-up keys"})
        if all(k in d for k in ("sh_dn_c", "sh_dn_s", "sh_dn_g")):
            cases.append(Case("shared_down_nvfp4", "nvfp4", rt.hidden, rt.shared_inter, calls,
                              codes=d["sh_dn_c"], scales=d["sh_dn_s"], scale=float(d["sh_dn_g"])))
        else:
            unsupported.append({"name": "shared_down_nvfp4", "reason": "missing shared-down keys"})

    if getattr(rt, "lm_head_kind", None) == "nvfp4":
        cases.append(Case("lm_head_nvfp4", "nvfp4", rt.vocab, rt.hidden, 1,
                          codes=rt.lm_head_codes, scales=rt.lm_head_scales,
                          scale=float(rt.lm_head_g)))
    else:
        unsupported.append({"name": "lm_head_nvfp4", "reason": f"lm_head_kind={getattr(rt, 'lm_head_kind', None)!r}"})
    return cases, unsupported


def _candidate(rt, k, n, case, out, X):
    if case.kind == "bf16":
        k.bf16(n, out, case.W, X, case.rows, case.cols)
    elif case.kind == "f32":
        k.f32(n, out, case.W, X, case.rows, case.cols)
    elif case.kind == "fp8":
        k.fp8(n, out, case.W, X, case.scale, case.rows, case.cols)
    elif case.kind == "nvfp4":
        k.nvfp4(n, out, case.codes, case.scales, rt.fused.e2m1, rt.fused.e4m3,
                 X, case.scale, case.rows, case.cols, case.apply_relu2, case.out_scale)
    else:
        raise ValueError(case.kind)


def _bench(rt, k, n, case, correctness_batches, repeats, rounds):
    import cupy as cp

    correctness = []
    first_X = first_ref = first_cand = None
    for b in range(correctness_batches):
        seed = (0xE2560000 + 8191 * n + 131 * b + sum(ord(c) for c in case.name)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        X = cp.asarray(rng.standard_normal((n, case.cols)).astype(np.float32))
        ref = cp.empty((n, case.rows), dtype=cp.float32)
        cand = cp.empty((n, case.rows), dtype=cp.float32)
        cand2 = cp.empty((n, case.rows), dtype=cp.float32)
        for r in range(n):
            _baseline_call(rt, case, ref[r], X[r])
        _candidate(rt, k, n, case, cand, X)
        _candidate(rt, k, n, case, cand2, X)
        cp.cuda.get_current_stream().synchronize()
        rr, cc, dd = cp.asnumpy(ref), cp.asnumpy(cand), cp.asnumpy(cand2)
        rb, cb, db = rr.view(np.uint32), cc.view(np.uint32), dd.view(np.uint32)
        correctness.append({
            "seed": int(seed),
            "bit_equal": bool(np.array_equal(rb, cb)),
            "mismatch_count": int(np.count_nonzero(rb != cb)),
            "deterministic": bool(np.array_equal(cb, db)),
            "determinism_mismatch_count": int(np.count_nonzero(cb != db)),
            "finite": bool(np.isfinite(rr).all() and np.isfinite(cc).all() and np.isfinite(dd).all()),
        })
        if b == 0:
            first_X, first_ref, first_cand = X, ref, cand

    def ref_call():
        for r in range(n):
            _baseline_call(rt, case, first_ref[r], first_X[r])

    def cand_call():
        _candidate(rt, k, n, case, first_cand, first_X)

    ref_call(); cand_call(); cp.cuda.get_current_stream().synchronize()
    ra_s = _time_cuda(ref_call, repeats=repeats, rounds=rounds)
    ca_s = _time_cuda(cand_call, repeats=repeats, rounds=rounds)
    cb_s = _time_cuda(cand_call, repeats=repeats, rounds=rounds)
    rb_s = _time_cuda(ref_call, repeats=repeats, rounds=rounds)
    ra = float(np.median(np.asarray(ra_s, dtype=np.float64)))
    rb = float(np.median(np.asarray(rb_s, dtype=np.float64)))
    rm = 0.5 * (ra + rb)
    cm = float(np.median(np.asarray(ca_s + cb_s, dtype=np.float64)))
    drift = abs(ra - rb) / rm if rm > 0 else math.inf
    return {
        "name": case.name,
        "kind": case.kind,
        "n_rhs": n,
        "rows": case.rows,
        "cols": case.cols,
        "calls_per_token": case.calls_per_token,
        "correctness_batches": correctness,
        "all_bit_equal": all(x["bit_equal"] for x in correctness),
        "all_deterministic": all(x["deterministic"] for x in correctness),
        "all_finite": all(x["finite"] for x in correctness),
        "reference_a_ms": ra,
        "reference_b_ms": rb,
        "reference_mid_ms": rm,
        "mrhs256_ms": cm,
        "mrhs256_ms_per_rhs": cm / n,
        "aggregate_speedup": rm / cm if cm > 0 else None,
        "reference_drift_fraction": drift,
        "raw_reference_a_ms": ra_s,
        "raw_mrhs256_a_ms": ca_s,
        "raw_mrhs256_b_ms": cb_s,
        "raw_reference_b_ms": rb_s,
    }


def _summary(records, n):
    rr = [x for x in records if x["n_rhs"] == n]
    refw = sum(x["reference_mid_ms"] * x["calls_per_token"] for x in rr)
    candw = sum(x["mrhs256_ms"] * x["calls_per_token"] for x in rr)
    by = {x["name"]: x for x in rr}
    m_in = next((x for x in rr if x["name"].startswith("mamba_in_")), None)
    m_out = next((x for x in rr if x["name"].startswith("mamba_out_")), None)
    return {
        "n_rhs": n,
        "case_count": len(rr),
        "weighted_speedup": refw / candw if candw else None,
        "min_case_speedup": min((x["aggregate_speedup"] for x in rr), default=None),
        "max_reference_drift_fraction": max((x["reference_drift_fraction"] for x in rr), default=None),
        "lm_head_speedup": by.get("lm_head_nvfp4", {}).get("aggregate_speedup"),
        "mamba_in_speedup": None if m_in is None else m_in["aggregate_speedup"],
        "mamba_out_speedup": None if m_out is None else m_out["aggregate_speedup"],
        "all_exact": all(x["all_bit_equal"] and x["all_deterministic"] and x["all_finite"] for x in rr),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload = {
        "kind": "pro_e100_mrhs256",
        "mode": args.mode,
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "full-warp exact common-weight component through N16; not a full-model E100 claim",
    }
    try:
        require_gpu_free()
        payload["environment_start"] = environment_snapshot((
            Path(__file__), REPO / "pro_research" / "mrhs256_exact_kernels.py",
            REPO / "pro_research" / "e100_mrhs.py",
        ))
        rt = _new_runtime()
        cases, unsupported = _collect_cases(rt)
        ns = (4,) if args.mode == "smoke" else (4, 8, 16)
        k = ExactMRHS256(ns)
        correctness_batches = 1 if args.mode == "smoke" else 3
        repeats = 2 if args.mode == "smoke" else 10
        rounds = 2 if args.mode == "smoke" else 4
        records = []
        for n in ns:
            for case in cases:
                r = _bench(rt, k, n, case, correctness_batches, repeats, rounds)
                records.append(r)
                print(
                    f"MRHS256 N={n:>2} {case.name:<22} exact={r['all_bit_equal']} "
                    f"ref={r['reference_mid_ms']:.4f}ms cand={r['mrhs256_ms']:.4f}ms "
                    f"speedup={r['aggregate_speedup']:.3f}x drift={100*r['reference_drift_fraction']:.2f}%",
                    flush=True,
                )
        summaries = {str(n): _summary(records, n) for n in ns}
        all_exact = all(x["all_bit_equal"] and x["all_deterministic"] and x["all_finite"] for x in records)
        names = {x.name for x in cases}
        mandatory = (
            "lm_head_nvfp4" in names
            and any(x.startswith("mamba_in_") for x in names)
            and any(x.startswith("mamba_out_") for x in names)
        )
        enough = len(cases) >= 7
        n4 = summaries.get("4", {})
        n16 = summaries.get("16", {})
        perf = {
            "n4_weighted_ge_1_50": (n4.get("weighted_speedup") or 0.0) >= 1.50,
            "n16_weighted_ge_3_0": (n16.get("weighted_speedup") or 0.0) >= 3.0 if args.mode == "full" else None,
            "n16_lm_head_ge_3_0": (n16.get("lm_head_speedup") or 0.0) >= 3.0 if args.mode == "full" else None,
            "n16_mamba_in_ge_2_5": (n16.get("mamba_in_speedup") or 0.0) >= 2.5 if args.mode == "full" else None,
            "n16_mamba_out_ge_2_5": (n16.get("mamba_out_speedup") or 0.0) >= 2.5 if args.mode == "full" else None,
            "n16_no_case_below_0_95": (n16.get("min_case_speedup") or 0.0) >= 0.95 if args.mode == "full" else None,
            "n16_all_ref_drift_le_7pct": (n16.get("max_reference_drift_fraction") or math.inf) <= 0.07 if args.mode == "full" else None,
        }
        full_perf = bool(perf["n4_weighted_ge_1_50"] and all(v is True for k2, v in perf.items() if k2.startswith("n16_"))) if args.mode == "full" else None
        if not all_exact or not mandatory or not enough:
            status = "correctness_failed"
        elif args.mode == "smoke":
            status = "smoke_pass"
        elif full_perf:
            status = "mrhs256_candidate"
        else:
            status = "micro_null"
        payload.update({
            "config": {"rhs_values": list(ns), "correctness_batches": correctness_batches,
                       "timing_repeats": repeats, "timing_rounds_per_arm": rounds},
            "supported_cases": [x.name for x in cases],
            "unsupported_cases": unsupported,
            "records": records,
            "summary_by_n": summaries,
            "gates": {"all_exact": all_exact, "mandatory_supported": mandatory,
                      "at_least_seven_cases": enough, "performance": perf,
                      "full_performance_pass": full_perf},
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })
        del rt, k
        gc.collect()
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({"status": "technical_failure",
                        "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                        "completed_utc": utc_now()})
    _write(payload)
    print(json.dumps({"status": payload.get("status"), "output": str(OUT),
                      "summary_by_n": payload.get("summary_by_n"), "gates": payload.get("gates")}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
