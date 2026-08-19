from __future__ import annotations

from dataclasses import dataclass
import json
import statistics
import sys
import traceback

import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now
from s100_lightning15_common import RESULTS, ensure_results, identity
from s100_lightning15_native import NativeSplitEngine

OUT = RESULTS / "S100_LIGHTNING15_COMPONENT.json"
BLOCKS = (1, 2, 4, 8)
TERMS = (1, 2, 3)

@dataclass
class Case:
    name: str
    family: str
    layer: int
    weight: object
    rows: int
    cols: int
    weight_bytes: int

def collect(rt):
    cases = []
    for layer in rt.attn_layers:
        data = rt.layer[int(layer)]
        hq = int(rt.n_heads * rt.head_dim)
        for family, key, rows, cols in (
            ("q", "q_proj", hq, int(rt.hidden)),
            ("k", "k_proj", int(rt.kv_dim), int(rt.hidden)),
            ("v", "v_proj", int(rt.kv_dim), int(rt.hidden)),
            ("o", "o_proj", int(rt.hidden), hq),
        ):
            if key in data:
                weight = data[key]
                cases.append(Case(
                    f"attention_{layer}_{family}",
                    family, int(layer), weight,
                    rows, cols, int(weight.nbytes),
                ))
    return cases

def measure_stream(cp, fn, reps=10):
    for _ in range(2):
        fn()
    cp.cuda.get_current_stream().synchronize()
    values = []
    for _ in range(reps):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(cp.cuda.get_elapsed_time(start, end)))
    return {
        "median_ms": statistics.median(values),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "raw_ms": values,
    }

def main():
    ensure_results()
    payload = {
        "kind": "s100_lightning15_component",
        "status": "started",
        "started_utc": utc_now(),
    }
    try:
        import torch
        import cupy as cp
        sys.path.insert(0, str(REPO / "src"))
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

        ident = identity()
        rt = LightningRuntime(
            require_model_dir(),
            contexts_max=4096,
            embed_on_host=True,
            fp8_kv=True,
            verbose=False,
        )
        cases = collect(rt)
        if not cases:
            raise RuntimeError("no Lightning BF16 attention matrices")

        props = cp.cuda.runtime.getDeviceProperties(0)
        l2 = int(props.get("l2CacheSize", 32 * 1024**2))
        rotation = sum(case.weight_bytes for case in cases)
        if rotation <= 4 * l2:
            raise RuntimeError(
                f"rotation {rotation} is not >4x L2 {l2}"
            )

        engine = NativeSplitEngine()
        results = {}
        family_results = {}

        for block in BLOCKS:
            rng = cp.random.RandomState(20260819 + block)
            x = {
                case.name: rng.standard_normal(
                    (block, case.cols), dtype=cp.float32
                )
                for case in cases
            }
            reference = {
                case.name: cp.empty(
                    (block, case.rows), dtype=cp.float32
                )
                for case in cases
            }
            candidate = {
                (case.name, terms): cp.empty(
                    (block, case.rows), dtype=cp.float32
                )
                for case in cases for terms in TERMS
            }
            for case in cases:
                for terms in TERMS:
                    engine.prepare(
                        case.weight, case.rows, case.cols,
                        block, terms,
                    )

            def baseline_stream():
                for case in cases:
                    for row in range(block):
                        rt.k.mv_bf16(
                            reference[case.name][row],
                            case.weight,
                            x[case.name][row],
                            case.rows,
                            case.cols,
                        )

            def candidate_stream(terms):
                for case in cases:
                    engine.run(
                        case.weight,
                        x[case.name],
                        candidate[(case.name, terms)],
                        case.rows,
                        case.cols,
                        block,
                        terms,
                    )

            base_a = measure_stream(cp, baseline_stream)
            base_b = measure_stream(cp, baseline_stream)
            base_mid = (
                base_a["median_ms"] + base_b["median_ms"]
            ) / 2.0

            per_terms = {}
            for terms in TERMS:
                cand_a = measure_stream(
                    cp, lambda t=terms: candidate_stream(t)
                )
                cand_b = measure_stream(
                    cp, lambda t=terms: candidate_stream(t)
                )
                cand_mid = (
                    cand_a["median_ms"] + cand_b["median_ms"]
                ) / 2.0

                failures = []
                case_metrics = []
                candidate_stream(terms)
                baseline_stream()
                cp.cuda.get_current_stream().synchronize()
                for case in cases:
                    ref = cp.asnumpy(reference[case.name]).astype(
                        np.float64
                    )
                    cand = cp.asnumpy(
                        candidate[(case.name, terms)]
                    ).astype(np.float64)
                    diff = cand - ref
                    nrmse = float(
                        np.linalg.norm(diff)
                        / max(np.linalg.norm(ref), 1e-30)
                    )
                    case_metrics.append({
                        "case": case.name,
                        "family": case.family,
                        "layer": case.layer,
                        "shape": [case.rows, case.cols],
                        "weight_bytes": case.weight_bytes,
                        "nrmse": nrmse,
                        "max_abs": float(np.max(np.abs(diff))),
                        "row_argmax_agreement": float(np.mean(
                            np.argmax(ref, axis=1)
                            == np.argmax(cand, axis=1)
                        )),
                        "finite": bool(np.isfinite(cand).all()),
                    })
                    if not np.isfinite(cand).all():
                        failures.append(case.name)

                max_nrmse = max(row["nrmse"] for row in case_metrics)
                aggregate_gbs = (
                    rotation / (cand_mid * 1e-3) / 1e9
                )
                speedup = base_mid / cand_mid
                per_terms[str(terms)] = {
                    "base_a": base_a,
                    "base_b": base_b,
                    "candidate_a": cand_a,
                    "candidate_b": cand_b,
                    "base_mid_ms": base_mid,
                    "candidate_mid_ms": cand_mid,
                    "useful_row_speedup": speedup,
                    "candidate_effective_weight_gbs": aggregate_gbs,
                    "case_metrics": case_metrics,
                    "max_case_nrmse": max_nrmse,
                    "finite": not failures,
                    "b4_gate_pass": bool(
                        block == 4
                        and terms == 2
                        and speedup >= 2.5
                        and max_nrmse <= 0.005
                        and not failures
                    ),
                }

                for family in ("q", "k", "v", "o"):
                    family_cases = [
                        row for row in case_metrics
                        if row["family"] == family
                    ]
                    family_results.setdefault(
                        str(block), {}
                    ).setdefault(str(terms), {})[family] = {
                        "max_nrmse": max(
                            row["nrmse"] for row in family_cases
                        ),
                        "mean_nrmse": float(np.mean([
                            row["nrmse"] for row in family_cases
                        ])),
                        "weight_bytes": sum(
                            row["weight_bytes"] for row in family_cases
                        ),
                    }

            results[str(block)] = {
                "baseline": {
                    "A": base_a, "B": base_b,
                    "mid_ms": base_mid,
                },
                "terms": per_terms,
            }

        payload.update({
            "status": "measured",
            "identity": ident,
            "case_count": len(cases),
            "rotation_bytes": rotation,
            "l2_bytes": l2,
            "rotation_over_l2": rotation / l2,
            "per_block": results,
            "family_metrics": family_results,
            "BF16X2_COLD_STREAM_OPEN": bool(
                results["4"]["terms"]["2"]["b4_gate_pass"]
            ),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "rotation_over_l2": payload.get("rotation_over_l2"),
        "per_block": {
            block: {
                terms: {
                    "speedup": row.get("useful_row_speedup"),
                    "nrmse": row.get("max_case_nrmse"),
                    "candidate_ms": row.get("candidate_mid_ms"),
                    "gate": row.get("b4_gate_pass"),
                }
                for terms, row in value.get("terms", {}).items()
            }
            for block, value in payload.get("per_block", {}).items()
        },
        "BF16X2_COLD_STREAM_OPEN": payload.get(
            "BF16X2_COLD_STREAM_OPEN"
        ),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
