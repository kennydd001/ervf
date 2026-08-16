"""E100-MRHS: exact common-weight reuse across multiple active sequences.

Component experiment only.  It intentionally does not patch the model runtime;
full-model integration is gated on this runner first proving real-checkpoint
bit-exactness and useful N=4 amortisation.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import REPO, environment_snapshot, require_gpu_free, require_model_dir, utc_now
from mrhs_exact_kernels import ExactMRHS, cpu_mapping_selftest

RESULT_DIR = REPO / "pro_research" / "results" / "e100_mrhs"
OUT = RESULT_DIR / "PRO_E100_MRHS.json"
PREREG = REPO / "pro_research" / "E100_MRHS_PREREGISTRATION.md"


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

    @property
    def weight_bytes(self) -> int:
        if self.kind == "bf16":
            return int(self.rows * self.cols * 2)
        if self.kind == "f32":
            return int(self.rows * self.cols * 4)
        if self.kind == "fp8":
            return int(self.rows * self.cols)
        if self.kind == "nvfp4":
            return int(self.codes.nbytes + self.scales.nbytes)
        raise ValueError(self.kind)


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _new_runtime():
    sys.path.insert(0, str(REPO / "src"))
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    return LightningRuntime(
        require_model_dir(), contexts_max=4096, embed_on_host=True,
        fp8_kv=True, verbose=False,
    )


def _collect_cases(rt) -> tuple[list[Case], list[dict[str, str]]]:
    cases: list[Case] = []
    unsupported: list[dict[str, str]] = []

    if rt.attn_layers:
        i = rt.attn_layers[0]
        d = rt.layer[i]
        qrows = rt.n_heads * rt.head_dim
        kvrows = rt.n_kv * rt.head_dim
        for name, key, rows, cols in (
            ("attn_q_bf16", "q_proj", qrows, rt.hidden),
            ("attn_k_bf16", "k_proj", kvrows, rt.hidden),
            ("attn_v_bf16", "v_proj", kvrows, rt.hidden),
            ("attn_o_bf16", "o_proj", rt.hidden, qrows),
        ):
            if key in d:
                cases.append(Case(name, "bf16", int(rows), int(cols), len(rt.attn_layers), W=d[key]))
            else:
                unsupported.append({"name": name, "reason": f"missing layer key {key}"})
    else:
        unsupported.append({"name": "attention_family", "reason": "checkpoint has no attention layers"})

    if rt.mamba_layers:
        i = rt.mamba_layers[0]
        d = rt.layer[i]
        rows = int(rt.proj.size)
        kind = d.get("in_k")
        if kind == "fp8_tensor" and "in_w8" in d:
            cases.append(Case("mamba_in_fp8", "fp8", rows, rt.hidden,
                              len(rt.mamba_layers), W=d["in_w8"], scale=float(d["in_s"])))
        elif kind == "bf16" and "in_w" in d:
            cases.append(Case("mamba_in_bf16", "bf16", rows, rt.hidden,
                              len(rt.mamba_layers), W=d["in_w"]))
        else:
            unsupported.append({"name": "mamba_in", "reason": f"unsupported storage kind {kind!r}"})
    else:
        unsupported.append({"name": "mamba_in", "reason": "checkpoint has no Mamba layers"})

    if rt.moe_layers:
        i = rt.moe_layers[0]
        d = rt.layer[i]
        if "gate_w" in d:
            cases.append(Case("router_f32", "f32", rt.n_experts, rt.hidden,
                              len(rt.moe_layers), W=d["gate_w"]))
        else:
            unsupported.append({"name": "router_f32", "reason": "missing gate_w"})

        if all(k in d for k in ("sh_up_c", "sh_up_s", "sh_up_g")):
            cases.append(Case("shared_up_nvfp4", "nvfp4", rt.shared_inter, rt.hidden,
                              len(rt.moe_layers), codes=d["sh_up_c"], scales=d["sh_up_s"],
                              scale=float(d["sh_up_g"]), apply_relu2=True))
        else:
            unsupported.append({"name": "shared_up_nvfp4", "reason": "missing shared-up NVFP4 keys"})

        if all(k in d for k in ("sh_dn_c", "sh_dn_s", "sh_dn_g")):
            cases.append(Case("shared_down_nvfp4", "nvfp4", rt.hidden, rt.shared_inter,
                              len(rt.moe_layers), codes=d["sh_dn_c"], scales=d["sh_dn_s"],
                              scale=float(d["sh_dn_g"])))
        else:
            unsupported.append({"name": "shared_down_nvfp4", "reason": "missing shared-down NVFP4 keys"})
    else:
        unsupported.append({"name": "moe_family", "reason": "checkpoint has no MoE layers"})

    if getattr(rt, "lm_head_kind", None) == "nvfp4":
        cases.append(Case("lm_head_nvfp4", "nvfp4", rt.vocab, rt.hidden, 1,
                          codes=rt.lm_head_codes, scales=rt.lm_head_scales,
                          scale=float(rt.lm_head_g)))
    else:
        unsupported.append({"name": "lm_head_nvfp4", "reason": f"lm_head_kind={getattr(rt, 'lm_head_kind', None)!r}"})

    return cases, unsupported


def _baseline_call(rt, case: Case, out, x) -> None:
    if case.kind == "bf16":
        rt.k.mv_bf16(out, case.W, x, case.rows, case.cols)
    elif case.kind == "f32":
        rt.k.mv_f32(out, case.W, x, case.rows, case.cols)
    elif case.kind == "fp8":
        rt.k.mv_fp8_tensor(out, case.W, x, case.scale, case.rows, case.cols)
    elif case.kind == "nvfp4":
        rt.fused.gemv_into(out, case.codes, case.scales, x, case.scale,
                           case.rows, case.cols, apply_relu2=case.apply_relu2,
                           out_scale=case.out_scale)
    else:
        raise ValueError(case.kind)


def _candidate_call(rt, mrhs: ExactMRHS, n: int, case: Case, out, X) -> None:
    if case.kind == "bf16":
        mrhs.bf16(n, out, case.W, X, case.rows, case.cols)
    elif case.kind == "f32":
        mrhs.f32(n, out, case.W, X, case.rows, case.cols)
    elif case.kind == "fp8":
        mrhs.fp8(n, out, case.W, X, case.scale, case.rows, case.cols)
    elif case.kind == "nvfp4":
        mrhs.nvfp4(n, out, case.codes, case.scales, rt.fused.e2m1, rt.fused.e4m3,
                    X, case.scale, case.rows, case.cols,
                    apply_relu2=case.apply_relu2, out_scale=case.out_scale)
    else:
        raise ValueError(case.kind)


def _time_cuda(call: Callable[[], None], repeats: int, rounds: int) -> list[float]:
    import cupy as cp

    samples: list[float] = []
    for _ in range(rounds):
        st, en = cp.cuda.Event(), cp.cuda.Event()
        st.record()
        for _ in range(repeats):
            call()
        en.record()
        en.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(st, en)) / repeats)
    return samples


def _bench_case(rt, mrhs: ExactMRHS, case: Case, n: int,
                correctness_batches: int, repeats: int, rounds: int) -> dict[str, Any]:
    import cupy as cp

    exact_batches: list[dict[str, Any]] = []
    first_X = None
    first_ref = None
    first_cand = None

    for b in range(correctness_batches):
        seed = (0xE1000000 + 1009 * n + 9176 * b + sum(ord(c) for c in case.name)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        X_np = rng.standard_normal((n, case.cols)).astype(np.float32)
        X = cp.asarray(X_np)
        ref = cp.empty((n, case.rows), dtype=cp.float32)
        cand = cp.empty((n, case.rows), dtype=cp.float32)
        cand2 = cp.empty((n, case.rows), dtype=cp.float32)

        for r in range(n):
            _baseline_call(rt, case, ref[r], X[r])
        _candidate_call(rt, mrhs, n, case, cand, X)
        _candidate_call(rt, mrhs, n, case, cand2, X)
        cp.cuda.get_current_stream().synchronize()

        rr = cp.asnumpy(ref)
        cc = cp.asnumpy(cand)
        dd = cp.asnumpy(cand2)
        rb, cb, db = rr.view(np.uint32), cc.view(np.uint32), dd.view(np.uint32)
        mismatch = int(np.count_nonzero(rb != cb))
        det_mismatch = int(np.count_nonzero(cb != db))
        finite = bool(np.isfinite(rr).all() and np.isfinite(cc).all() and np.isfinite(dd).all())
        exact_batches.append({
            "seed": int(seed),
            "bit_equal": mismatch == 0,
            "mismatch_count": mismatch,
            "deterministic": det_mismatch == 0,
            "determinism_mismatch_count": det_mismatch,
            "finite": finite,
        })
        if b == 0:
            first_X, first_ref, first_cand = X, ref, cand

    assert first_X is not None and first_ref is not None and first_cand is not None

    def ref_call() -> None:
        for r in range(n):
            _baseline_call(rt, case, first_ref[r], first_X[r])

    def cand_call() -> None:
        _candidate_call(rt, mrhs, n, case, first_cand, first_X)

    # Warm both arms before the frozen REF/MRHS/MRHS/REF timing order.
    ref_call(); cand_call(); cp.cuda.get_current_stream().synchronize()
    ref_a = _time_cuda(ref_call, repeats, rounds)
    cand_a = _time_cuda(cand_call, repeats, rounds)
    cand_b = _time_cuda(cand_call, repeats, rounds)
    ref_b = _time_cuda(ref_call, repeats, rounds)

    med_ref_a = float(np.median(np.asarray(ref_a, dtype=np.float64)))
    med_ref_b = float(np.median(np.asarray(ref_b, dtype=np.float64)))
    ref_mid = 0.5 * (med_ref_a + med_ref_b)
    cand_ms = float(np.median(np.asarray(cand_a + cand_b, dtype=np.float64)))
    drift = abs(med_ref_a - med_ref_b) / ref_mid if ref_mid > 0 else math.inf
    speedup = ref_mid / cand_ms if cand_ms > 0 else math.inf

    return {
        "name": case.name,
        "kind": case.kind,
        "n_rhs": n,
        "rows": case.rows,
        "cols": case.cols,
        "calls_per_token": case.calls_per_token,
        "matrix_weight_bytes": case.weight_bytes,
        "correctness_batches": exact_batches,
        "all_bit_equal": all(x["bit_equal"] for x in exact_batches),
        "all_deterministic": all(x["deterministic"] for x in exact_batches),
        "all_finite": all(x["finite"] for x in exact_batches),
        "reference_a_ms": med_ref_a,
        "reference_b_ms": med_ref_b,
        "reference_mid_ms": ref_mid,
        "mrhs_ms": cand_ms,
        "mrhs_ms_per_rhs": cand_ms / n,
        "aggregate_speedup": float(speedup),
        "reference_drift_fraction": float(drift),
        "reference_logical_weight_gb_s": float((case.weight_bytes * n) / (ref_mid * 1e6)) if ref_mid else None,
        "mrhs_single_stream_weight_gb_s": float(case.weight_bytes / (cand_ms * 1e6)) if cand_ms else None,
        "raw_reference_a_ms": ref_a,
        "raw_mrhs_a_ms": cand_a,
        "raw_mrhs_b_ms": cand_b,
        "raw_reference_b_ms": ref_b,
    }


def _summarize_n(records: list[dict[str, Any]], n: int) -> dict[str, Any]:
    rr = [x for x in records if int(x["n_rhs"]) == n]
    ref_weighted = sum(float(x["reference_mid_ms"]) * int(x["calls_per_token"]) for x in rr)
    cand_weighted = sum(float(x["mrhs_ms"]) * int(x["calls_per_token"]) for x in rr)
    by_name = {x["name"]: x for x in rr}
    return {
        "n_rhs": n,
        "case_count": len(rr),
        "all_bit_equal": all(bool(x["all_bit_equal"]) for x in rr),
        "all_deterministic": all(bool(x["all_deterministic"]) for x in rr),
        "all_finite": all(bool(x["all_finite"]) for x in rr),
        "all_reference_drift_le_7pct": all(float(x["reference_drift_fraction"]) <= 0.07 for x in rr),
        "weighted_registered_reference_ms": ref_weighted,
        "weighted_registered_mrhs_ms": cand_weighted,
        "weighted_registered_aggregate_speedup": ref_weighted / cand_weighted if cand_weighted > 0 else None,
        "min_case_speedup": min((float(x["aggregate_speedup"]) for x in rr), default=None),
        "lm_head_speedup": by_name.get("lm_head_nvfp4", {}).get("aggregate_speedup"),
        "mamba_in_speedup": next((x["aggregate_speedup"] for x in rr if x["name"].startswith("mamba_in_")), None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    cpu = cpu_mapping_selftest()
    if args.selftest:
        print(json.dumps(cpu, indent=2))
        return 0 if cpu["passed"] else 2

    payload: dict[str, Any] = {
        "kind": "pro_e100_exact_mrhs",
        "mode": args.mode,
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "component/oracle test for exact aggregate multi-sequence common-weight reuse; not a full-model or single-stream E100 claim",
        "cpu_mapping_selftest": cpu,
    }

    try:
        if not cpu["passed"]:
            raise RuntimeError("width-32 virtual reduction CPU selftest failed")
        require_gpu_free()
        payload["environment_start"] = environment_snapshot((
            Path(__file__),
            REPO / "pro_research" / "mrhs_exact_kernels.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "gpu_kernels.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "fused_nvfp4.py",
        ))

        rt = _new_runtime()
        cases, unsupported = _collect_cases(rt)
        rhs_values = (2, 4) if args.mode == "smoke" else (2, 4, 8)
        mrhs = ExactMRHS(rhs_values)
        correctness_batches = 1 if args.mode == "smoke" else 3
        repeats = 2 if args.mode == "smoke" else 10
        rounds = 2 if args.mode == "smoke" else 4

        records: list[dict[str, Any]] = []
        for n in rhs_values:
            for case in cases:
                rec = _bench_case(rt, mrhs, case, n, correctness_batches, repeats, rounds)
                records.append(rec)
                print(
                    f"MRHS N={n:>2} {case.name:<22} exact={rec['all_bit_equal']} "
                    f"det={rec['all_deterministic']} ref={rec['reference_mid_ms']:.4f}ms "
                    f"mrhs={rec['mrhs_ms']:.4f}ms speedup={rec['aggregate_speedup']:.3f}x "
                    f"drift={100.0*rec['reference_drift_fraction']:.2f}%",
                    flush=True,
                )

        summaries = {str(n): _summarize_n(records, n) for n in rhs_values}
        all_exact = all(bool(x["all_bit_equal"]) and bool(x["all_deterministic"]) and bool(x["all_finite"]) for x in records)
        supported_names = {x.name for x in cases}
        mamba_supported = any(x.startswith("mamba_in_") for x in supported_names)
        mandatory_supported = "lm_head_nvfp4" in supported_names and mamba_supported
        at_least_six_families = len(cases) >= 6

        n4 = summaries.get("4", {})
        n4_perf = {
            "weighted_speedup_ge_1_75": bool((n4.get("weighted_registered_aggregate_speedup") or 0.0) >= 1.75),
            "lm_head_speedup_ge_1_50": bool((n4.get("lm_head_speedup") or 0.0) >= 1.50),
            "mamba_in_speedup_ge_1_50": bool((n4.get("mamba_in_speedup") or 0.0) >= 1.50),
            "no_n4_case_regression_gt_5pct": bool((n4.get("min_case_speedup") or 0.0) >= 0.95),
            "all_n4_reference_drift_le_7pct": bool(n4.get("all_reference_drift_le_7pct", False)),
        }
        perf_pass = all(n4_perf.values())

        if not all_exact or not mandatory_supported or not at_least_six_families:
            status = "correctness_failed"
        elif args.mode == "smoke":
            status = "smoke_pass"
        elif perf_pass:
            status = "mrhs_candidate"
        else:
            status = "micro_null"

        payload.update({
            "config": {
                "rhs_values": list(rhs_values),
                "correctness_batches": correctness_batches,
                "timing_repeats": repeats,
                "timing_rounds_per_ABBA_arm": rounds,
            },
            "supported_cases": [x.name for x in cases],
            "unsupported_cases": unsupported,
            "records": records,
            "summary_by_n": summaries,
            "gates": {
                "all_real_outputs_bit_exact_deterministic_finite": all_exact,
                "at_least_six_case_families_supported": at_least_six_families,
                "lm_head_and_mamba_in_supported": mandatory_supported,
                "n4_performance": n4_perf,
                "full_n4_performance_pass": perf_pass if args.mode == "full" else None,
            },
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })

        del rt, mrhs
        gc.collect()
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })

    _write(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "output": str(OUT),
        "summary_by_n": payload.get("summary_by_n"),
        "gates": payload.get("gates"),
    }, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
