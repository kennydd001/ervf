from __future__ import annotations

from dataclasses import dataclass
import json
import statistics
import time
import traceback

import numpy as np

from common import REPO, write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_phase14_common import RESULTS, ensure_results

OUT = RESULTS / "S100_PHASE14D_NATIVE_EXTENDED.json"
BLOCKS = (2, 4, 8)

@dataclass
class Case:
    name: str
    family: str
    W: object
    rows: int
    cols: int
    weight_bytes: int

def collect(rt):
    cases = []
    for layer in rt.mamba_layers:
        d = rt.layer[int(layer)]
        if d.get("in_k") == "bf16":
            cases.append(Case(
                f"mamba_{layer}_in", "mamba", d["in_w"],
                int(rt.proj.size), int(rt.hidden), int(d["in_w"].nbytes)
            ))
        if d.get("out_k") == "bf16":
            cases.append(Case(
                f"mamba_{layer}_out", "mamba", d["out_w"],
                int(rt.hidden), int(rt.d_inner), int(d["out_w"].nbytes)
            ))

    for layer in rt.attn_layers:
        d = rt.layer[int(layer)]
        hq = int(rt.n_heads * rt.head_dim)
        specs = []
        if "q_proj" in d:
            specs.append(("q", d["q_proj"], hq, int(rt.hidden)))
        if "k_proj" in d:
            specs.append(("k", d["k_proj"], int(rt.kv_dim), int(rt.hidden)))
        if "v_proj" in d:
            specs.append(("v", d["v_proj"], int(rt.kv_dim), int(rt.hidden)))
        if "o_proj" in d:
            specs.append(("o", d["o_proj"], int(rt.hidden), hq))
        for side, W, rows, cols in specs:
            cases.append(Case(
                f"attention_{layer}_{side}", "attention", W,
                rows, cols, int(W.nbytes)
            ))

    if getattr(rt, "lm_head_kind", None) != "nvfp4" and hasattr(rt, "lm_head"):
        cases.append(Case(
            "lm_head", "lm_head", rt.lm_head,
            int(rt.vocab), int(rt.hidden), int(rt.lm_head.nbytes)
        ))
    return cases

def percentile(values, q):
    return float(np.percentile(np.asarray(values, np.float64), q))

def main():
    ensure_results()
    p = {
        "kind": "s100_phase14d_native_extended",
        "status": "started",
        "blocks": list(BLOCKS),
        "started_utc": utc_now(),
    }
    try:
        import torch
        import cupy as cp

        if not torch.cuda.is_available():
            raise RuntimeError("torch CUDA unavailable")

        bundle = build()
        rt = bundle.rt
        cases = collect(rt)
        if not cases:
            raise RuntimeError("no BF16 matrices found")

        props = cp.cuda.runtime.getDeviceProperties(0)
        l2 = int(props.get("l2CacheSize", 32 * 1024**2))
        rotation = sum(c.weight_bytes for c in cases)
        if rotation <= 4 * l2:
            raise RuntimeError(f"rotation {rotation} is not >4x L2 {l2}")

        per_b = {}
        for B in BLOCKS:
            rows = []
            for ci, case in enumerate(cases):
                # One copy at a time keeps peak VRAM bounded.
                wt = (
                    torch.utils.dlpack.from_dlpack(case.W)
                    .view(torch.bfloat16)
                    .reshape(case.rows, case.cols)
                    .clone()
                )
                wtt = wt.t().contiguous()
                generator = torch.Generator(device="cuda")
                generator.manual_seed(20260818 + B * 1000 + ci)
                x = torch.randn(
                    B, case.cols, device="cuda", dtype=torch.float32,
                    generator=generator,
                )
                xb = x.to(torch.bfloat16)
                xcp = cp.from_dlpack(x)
                ref = cp.empty((B, case.rows), cp.float32)

                def baseline():
                    for j in range(B):
                        rt.k.mv_bf16(
                            ref[j], case.W, xcp[j], case.rows, case.cols
                        )

                def native():
                    return torch.mm(xb, wtt).float()

                for fn in (baseline, native):
                    for _ in range(2):
                        fn()
                    torch.cuda.synchronize()

                bt, nt = [], []
                for _ in range(12):
                    t0 = time.perf_counter_ns()
                    baseline()
                    torch.cuda.synchronize()
                    bt.append((time.perf_counter_ns() - t0) / 1e6)

                    t0 = time.perf_counter_ns()
                    native()
                    torch.cuda.synchronize()
                    nt.append((time.perf_counter_ns() - t0) / 1e6)

                baseline()
                torch.cuda.synchronize()
                candidate = native()
                torch.cuda.synchronize()
                r = cp.asnumpy(ref).astype(np.float64)
                c = candidate.detach().float().cpu().numpy().astype(np.float64)
                diff = c - r
                rn = np.linalg.norm(r)
                row_rn = np.linalg.norm(r, axis=1)
                row_dn = np.linalg.norm(diff, axis=1)
                rows.append({
                    "case": case.name,
                    "family": case.family,
                    "shape": [case.rows, case.cols],
                    "weight_bytes": case.weight_bytes,
                    "baseline_ms": {
                        "median": statistics.median(bt),
                        "p10": percentile(bt, 10),
                        "p90": percentile(bt, 90),
                    },
                    "native_ms": {
                        "median": statistics.median(nt),
                        "p10": percentile(nt, 10),
                        "p90": percentile(nt, 90),
                    },
                    "speedup": statistics.median(bt) / statistics.median(nt),
                    "error_vs_current_ervf": {
                        "nrmse": float(np.linalg.norm(diff) / max(rn, 1e-30)),
                        "max_abs": float(np.max(np.abs(diff))),
                        "p95_relative_row_error": float(np.percentile(
                            row_dn / np.maximum(row_rn, 1e-30), 95
                        )),
                        "row_argmax_agreement": float(np.mean(
                            np.argmax(r, axis=1) == np.argmax(c, axis=1)
                        )),
                        "finite": bool(np.isfinite(c).all()),
                    },
                })
                del wt, wtt, x, xb, xcp, ref, candidate
                torch.cuda.empty_cache()

            base_sum = sum(x["baseline_ms"]["median"] for x in rows)
            native_sum = sum(x["native_ms"]["median"] for x in rows)
            max_nrmse = max(
                x["error_vs_current_ervf"]["nrmse"] for x in rows
            )
            mean_argmax = float(np.mean([
                x["error_vs_current_ervf"]["row_argmax_agreement"]
                for x in rows
            ]))
            speed = base_sum / native_sum
            per_b[str(B)] = {
                "case_count": len(rows),
                "cases": rows,
                "aggregate": {
                    "sum_independent_ervf_ms": base_sum,
                    "sum_native_ms": native_sum,
                    "useful_row_speedup": speed,
                    "max_case_nrmse": max_nrmse,
                    "mean_row_argmax_agreement": mean_argmax,
                },
                "component_gate_pass": bool(
                    speed >= 2.5
                    and max_nrmse <= 0.005
                    and mean_argmax >= 0.97
                ),
            }

        p.update({
            "status": "measured",
            "matrix_count": len(cases),
            "rotation_bytes": rotation,
            "l2_bytes": l2,
            "rotation_over_l2": rotation / l2,
            "per_block": per_b,
            "b4_component_gate_pass": bool(
                per_b["4"]["component_gate_pass"]
            ),
            "completed_utc": utc_now(),
        })
        bundle.restore_combined()
        bundle.restore_sel()
    except Exception as exc:
        p.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })

    write_json_atomic(OUT, p, archive=True)
    print(json.dumps({
        "status": p.get("status"),
        "rotation_over_l2": p.get("rotation_over_l2"),
        "per_block": {
            key: {
                "speedup": value.get("aggregate", {}).get(
                    "useful_row_speedup"
                ),
                "max_nrmse": value.get("aggregate", {}).get(
                    "max_case_nrmse"
                ),
                "gate": value.get("component_gate_pass"),
            }
            for key, value in p.get("per_block", {}).items()
        },
        "error": (p.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if p.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
