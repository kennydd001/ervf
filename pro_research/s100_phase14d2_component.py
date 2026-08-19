from __future__ import annotations

import json
import statistics
import traceback
import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now
from ervf_dense import DenseERVF
from s100_phase14d2_native import collect_bf16_cases

OUT = (
    REPO / "pro_research" / "results" / "s100_phase14d2"
    / "S100_PHASE14D2_COMPONENT.json"
)
BLOCKS = (1, 2, 4, 8)

def main():
    payload = {
        "kind": "s100_phase14d2_component",
        "status": "started",
        "blocks": list(BLOCKS),
        "started_utc": utc_now(),
        "claim_boundary": "cold native BF16 component ceiling; not end-to-end",
    }
    try:
        import torch
        import cupy as cp

        # This component needs only the resident BF16 matrices. The former
        # Phase-10A build also allocated the full mapped expert bank and cache,
        # which is unrelated here and can exhaust Windows mapped-pinned memory
        # before the native BF16 benchmark starts.
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
        rt = LightningRuntime(
            require_model_dir(), contexts_max=4096, embed_on_host=True,
            fp8_kv=True, verbose=False,
        )
        rt.deterministic_accum = True
        ref = DenseERVF()
        cases = collect_bf16_cases(rt)
        if not cases:
            raise RuntimeError("no live BF16 cases")

        props = cp.cuda.runtime.getDeviceProperties(0)
        l2 = int(props.get("l2CacheSize", 32 * 1024**2))
        scrub_n = max(4 * l2, 160 * 1024**2)
        scrub = torch.empty(scrub_n, dtype=torch.uint8, device="cuda")

        def cold():
            # Touch one byte per cache line. The reduction is outside the timed
            # interval; it exists only to evict the previous weight working set.
            _ = scrub[::128].sum()
            torch.cuda.synchronize()

        all_rows = []
        aggregate = {}
        rng = np.random.default_rng(20260819)

        for B in BLOCKS:
            rows = []
            for ci, case in enumerate(cases):
                wt = (
                    torch.utils.dlpack.from_dlpack(case.W)
                    .view(torch.bfloat16)
                    .reshape(case.rows, case.cols)
                    .clone()
                )
                # The native path used by Phase 13D.
                wtt = wt.t().contiguous()
                xh = rng.standard_normal(
                    (B, case.cols), dtype=np.float32
                )
                xcp = cp.asarray(xh)
                xt = torch.from_numpy(xh).to("cuda")
                xb = xt.to(torch.bfloat16)
                ro = cp.empty((B, case.rows), cp.float32)

                def baseline():
                    for j in range(B):
                        ref.mv_bf16(
                            ro[j], case.W, xcp[j], case.rows, case.cols
                        )

                def native():
                    return torch.mm(xb, wtt).float()

                # Numerical comparison.
                baseline()
                torch.cuda.synchronize()
                cand = native()
                torch.cuda.synchronize()
                rr = cp.asnumpy(ro).astype(np.float64)
                cc = cand.detach().cpu().numpy().astype(np.float64)
                diff = cc - rr
                denom = max(float(np.linalg.norm(rr)), 1e-30)
                nrmse = float(np.linalg.norm(diff) / denom)
                argmax = float(np.mean(
                    np.argmax(rr, axis=1) == np.argmax(cc, axis=1)
                ))

                def measure(fn):
                    values = []
                    for _ in range(12):
                        cold()
                        a = torch.cuda.Event(enable_timing=True)
                        b = torch.cuda.Event(enable_timing=True)
                        a.record()
                        fn()
                        b.record()
                        b.synchronize()
                        values.append(float(a.elapsed_time(b)))
                    return {
                        "median_ms": statistics.median(values),
                        "p10_ms": float(np.percentile(values, 10)),
                        "p90_ms": float(np.percentile(values, 90)),
                        "raw_ms": values,
                    }

                bm = measure(baseline)
                nm = measure(native)
                rows.append({
                    "case": case.name,
                    "family": case.family,
                    "shape": [case.rows, case.cols],
                    "weight_bytes": case.weight_bytes,
                    "baseline_independent_ervf": bm,
                    "native_bf16": nm,
                    "speedup": bm["median_ms"] / nm["median_ms"],
                    "error_vs_current_ervf": {
                        "nrmse": nrmse,
                        "max_abs": float(np.max(np.abs(diff))),
                        "row_argmax_agreement": argmax,
                        "finite": bool(np.isfinite(cc).all()),
                    },
                })
                del wt, wtt, xcp, xt, xb, ro, cand
                torch.cuda.empty_cache()

            base = sum(x["baseline_independent_ervf"]["median_ms"] for x in rows)
            cand = sum(x["native_bf16"]["median_ms"] for x in rows)
            speed = base / cand
            max_nrmse = max(x["error_vs_current_ervf"]["nrmse"] for x in rows)
            mean_argmax = float(np.mean([
                x["error_vs_current_ervf"]["row_argmax_agreement"]
                for x in rows
            ]))
            gate_speed = 1.10 if B == 1 else (2.50 if B == 4 else 0.0)
            gate = bool(
                max_nrmse <= 0.005
                and mean_argmax >= 0.97
                and all(x["error_vs_current_ervf"]["finite"] for x in rows)
                and (speed >= gate_speed if gate_speed else True)
            )
            aggregate[str(B)] = {
                "case_count": len(rows),
                "baseline_sum_ms": base,
                "native_sum_ms": cand,
                "useful_row_speedup": speed,
                "max_case_nrmse": max_nrmse,
                "mean_row_argmax_agreement": mean_argmax,
                "speed_gate": gate_speed,
                "gate_pass": gate,
                "cases": rows,
            }
            all_rows.extend(rows)

        weight_rotation = sum(c.weight_bytes for c in cases)
        payload.update({
            "status": "measured",
            "case_count": len(cases),
            "weight_rotation_bytes": weight_rotation,
            "l2_bytes": l2,
            "enumerated_weight_rotation_over_l2": weight_rotation / l2,
            "cache_scrub_bytes": scrub_n,
            "per_B": aggregate,
            "B1_DIRECT_COMPONENT_PASS": bool(aggregate["1"]["gate_pass"]),
            "B4_BLOCK_COMPONENT_PASS": bool(aggregate["4"]["gate_pass"]),
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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "B1": (payload.get("per_B") or {}).get("1", {}),
        "B4": (payload.get("per_B") or {}).get("4", {}),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2)[:12000])
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
