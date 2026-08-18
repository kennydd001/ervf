from __future__ import annotations

from dataclasses import dataclass
import json
import statistics
import traceback

import numpy as np

from common import REPO, write_json_atomic, utc_now
from ervf_dense import DenseERVF
from s100_phase10a_runtime import build
from s100_phase12c_ervfm_kernels import ERVFM, MS

OUT = (
    REPO / "pro_research" / "results" / "s100_phase12c"
    / "S100_PHASE12C_DENSE.json"
)
GATES = {2: 1.75, 4: 3.20, 8: 5.50}

@dataclass
class Case:
    name: str
    family: str
    kind: str
    W: object
    rows: int
    cols: int
    weight_bytes: int
    scale: float = 1.0
    scales: object | None = None

def _nbytes(x) -> int:
    return int(getattr(x, "nbytes", 0))

def collect(rt):
    cases = []

    for layer in rt.mamba_layers:
        d = rt.layer[int(layer)]
        if d.get("in_k") == "fp8_tensor":
            cases.append(Case(
                f"mamba_{layer}_in", "mamba", "fp8",
                d["in_w8"], int(rt.proj.size), int(rt.hidden),
                _nbytes(d["in_w8"]), float(d["in_s"]),
            ))
        if d.get("out_k") == "fp8_tensor":
            cases.append(Case(
                f"mamba_{layer}_out", "mamba", "fp8",
                d["out_w8"], int(rt.hidden), int(rt.d_inner),
                _nbytes(d["out_w8"]), float(d["out_s"]),
            ))

    for layer in rt.attn_layers:
        d = rt.layer[int(layer)]
        hq = int(rt.n_heads * rt.head_dim)
        if d.get("q_kind") == "nvfp4":
            cases.append(Case(
                f"attn_{layer}_q", "attention", "nvfp4",
                d["q_codes"], hq, int(rt.hidden),
                _nbytes(d["q_codes"]) + _nbytes(d["q_scales"]),
                float(d["q_g"]), d["q_scales"],
            ))
        elif "q_proj" in d:
            cases.append(Case(
                f"attn_{layer}_q", "attention", "bf16",
                d["q_proj"], hq, int(rt.hidden), _nbytes(d["q_proj"]),
            ))
        for name, rows, cols in (
            ("k", int(rt.kv_dim), int(rt.hidden)),
            ("v", int(rt.kv_dim), int(rt.hidden)),
            ("o", int(rt.hidden), hq),
        ):
            key = f"{name}_proj"
            if key in d:
                cases.append(Case(
                    f"attn_{layer}_{name}", "attention", "bf16",
                    d[key], rows, cols, _nbytes(d[key]),
                ))

    for layer in rt.moe_layers:
        d = rt.layer[int(layer)]
        if "gate_w" in d:
            cases.append(Case(
                f"router_{layer}", "router", "f32",
                d["gate_w"], int(rt.n_experts), int(rt.hidden),
                _nbytes(d["gate_w"]),
            ))
        for name, rows, cols in (
            ("sh_up", int(rt.shared_inter), int(rt.hidden)),
            ("sh_dn", int(rt.hidden), int(rt.shared_inter)),
        ):
            ck, sk, gk = f"{name}_c", f"{name}_s", f"{name}_g"
            if ck in d:
                cases.append(Case(
                    f"{name}_{layer}", "shared_expert", "nvfp4",
                    d[ck], rows, cols,
                    _nbytes(d[ck]) + _nbytes(d[sk]),
                    float(d[gk]), d[sk],
                ))

    if rt.lm_head_kind == "nvfp4":
        cases.append(Case(
            "lm_head", "lm_head", "nvfp4",
            rt.lm_head_codes, int(rt.vocab), int(rt.hidden),
            _nbytes(rt.lm_head_codes) + _nbytes(rt.lm_head_scales),
            float(rt.lm_head_g), rt.lm_head_scales,
        ))
    elif hasattr(rt, "lm_head"):
        cases.append(Case(
            "lm_head", "lm_head", "bf16",
            rt.lm_head, int(rt.vocab), int(rt.hidden),
            _nbytes(rt.lm_head),
        ))
    return cases

def main() -> int:
    payload = {
        "kind": "s100_phase12c_dense",
        "status": "started",
        "gates": GATES,
        "started_utc": utc_now(),
    }
    try:
        import cupy as cp

        parent = build()
        rt = parent.rt
        ref = DenseERVF()
        cand = ERVFM()
        cases = collect(rt)
        if not cases:
            raise RuntimeError("no real dense cases collected")

        props = cp.cuda.runtime.getDeviceProperties(0)
        l2 = int(props.get("l2CacheSize", 32 * 1024**2))
        total_bytes = sum(c.weight_bytes for c in cases)
        if total_bytes < 4 * l2:
            raise RuntimeError(
                f"real-weight rotation too small: {total_bytes} < {4*l2}"
            )

        by_family = {}
        for c in cases:
            by_family[c.family] = by_family.get(c.family, 0) + c.weight_bytes

        per_m = {}
        rng = cp.random.RandomState(20260818)

        for m in (2, 4, 8):
            xs = {}
            ref_out = {}
            cand_out = {}
            for c in cases:
                if c.cols not in xs:
                    xs[c.cols] = rng.standard_normal(
                        (m, c.cols), dtype=cp.float32
                    )
                ref_out[c.name] = cp.empty((m, c.rows), dtype=cp.float32)
                cand_out[c.name] = cp.empty((m, c.rows), dtype=cp.float32)

            def ref_case(c):
                X = xs[c.cols]
                O = ref_out[c.name]
                for j in range(m):
                    if c.kind == "bf16":
                        ref.mv_bf16(O[j], c.W, X[j], c.rows, c.cols)
                    elif c.kind == "f32":
                        ref.mv_f32(O[j], c.W, X[j], c.rows, c.cols)
                    elif c.kind == "fp8":
                        ref.mv_fp8_tensor(
                            O[j], c.W, X[j], c.scale, c.rows, c.cols
                        )
                    elif c.kind == "nvfp4":
                        rt.fused.gemv_into(
                            O[j], c.W, c.scales, X[j], c.scale,
                            c.rows, c.cols,
                        )
                    else:
                        raise ValueError(c.kind)

            def cand_case(c):
                cand.run(
                    c.kind, m, cand_out[c.name], c.W, xs[c.cols],
                    c.rows, c.cols, scale=c.scale, scales=c.scales,
                    e2=rt.fused.e2m1, e4=rt.fused.e4m3,
                )

            exact_failures = []
            for c in cases:
                ref_case(c)
                cand_case(c)
                cp.cuda.Stream.null.synchronize()
                if not bool(cp.array_equal(
                    ref_out[c.name], cand_out[c.name]
                )):
                    diff = cp.abs(
                        ref_out[c.name] - cand_out[c.name]
                    )
                    exact_failures.append({
                        "case": c.name,
                        "kind": c.kind,
                        "max_abs": float(cp.max(diff).item()),
                        "mismatch_count": int(cp.count_nonzero(diff).item()),
                    })

            def ref_stream():
                for c in cases:
                    ref_case(c)

            def cand_stream():
                for c in cases:
                    cand_case(c)

            def measure(fn, reps=16):
                for _ in range(3):
                    fn()
                cp.cuda.Stream.null.synchronize()
                values = []
                for _ in range(reps):
                    a = cp.cuda.Event()
                    b = cp.cuda.Event()
                    a.record()
                    fn()
                    b.record()
                    b.synchronize()
                    values.append(float(cp.cuda.get_elapsed_time(a, b)))
                return {
                    "median_ms": statistics.median(values),
                    "p10_ms": float(np.percentile(values, 10)),
                    "p90_ms": float(np.percentile(values, 90)),
                    "raw_ms": values,
                }

            baseline = measure(ref_stream)
            candidate = measure(cand_stream)
            speedup = baseline["median_ms"] / candidate["median_ms"]
            per_m[str(m)] = {
                "exact": not exact_failures,
                "exact_failures": exact_failures,
                "baseline_independent_m1": baseline,
                "candidate_ervfm": candidate,
                "useful_row_speedup": speedup,
                "gate": GATES[m],
                "gate_pass": bool(
                    not exact_failures and speedup >= GATES[m]
                ),
                "effective_candidate_weight_gbs": (
                    total_bytes
                    / (candidate["median_ms"] * 1e-3)
                    / 1e9
                ),
            }

        payload.update({
            "status": "measured",
            "matrix_count": len(cases),
            "families": by_family,
            "weight_rotation_bytes": total_bytes,
            "l2_bytes": l2,
            "rotation_over_l2": total_bytes / l2,
            "case_manifest": [
                {
                    "name": c.name,
                    "family": c.family,
                    "kind": c.kind,
                    "shape": [c.rows, c.cols],
                    "weight_bytes": c.weight_bytes,
                }
                for c in cases
            ],
            "per_m": per_m,
            "dense_b4_gate_pass": bool(per_m["4"]["gate_pass"]),
            "completed_utc": utc_now(),
        })
        parent.restore_combined()
        parent.restore_sel()
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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "matrix_count": payload.get("matrix_count"),
        "rotation_over_l2": payload.get("rotation_over_l2"),
        "per_m": {
            k: {
                "exact": v.get("exact"),
                "speedup": v.get("useful_row_speedup"),
                "gate": v.get("gate"),
                "pass": v.get("gate_pass"),
            }
            for k, v in payload.get("per_m", {}).items()
        },
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
