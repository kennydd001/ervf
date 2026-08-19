from __future__ import annotations

import json
import statistics
import traceback
import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now
from ervf_dense import DenseERVF
from s100_phase15_native import collect_bf16_cases

OUT = (
    REPO / "pro_research" / "results" / "s100_phase15"
    / "S100_PHASE15A_COMPONENT_VARIANTS.json"
)
BLOCKS = (1, 4)
VARIANTS = ("mm_bf16out", "mm_fp32out", "mm_fp32out_comp2")

def main():
    payload = {
        "kind": "s100_phase15a_component_variants",
        "status": "started",
        "started_utc": utc_now(),
        "blocks": list(BLOCKS),
        "variants": list(VARIANTS),
        "claim_boundary": "cold component contract benchmark; not end-to-end",
    }
    try:
        import cupy as cp
        import torch
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

        rt = LightningRuntime(
            require_model_dir(), contexts_max=512, embed_on_host=True,
            fp8_kv=True, verbose=False,
        )
        ref = DenseERVF()
        cases = collect_bf16_cases(rt)
        if not cases:
            raise RuntimeError("no live BF16 cases")

        props = cp.cuda.runtime.getDeviceProperties(0)
        l2 = int(props.get("l2CacheSize", 32 * 1024**2))
        scrub_n = max(4 * l2, 160 * 1024**2)
        scrub = torch.empty(scrub_n, dtype=torch.uint8, device="cuda")
        rng = np.random.default_rng(20260819)

        def cold():
            _ = scrub[::128].sum()
            torch.cuda.synchronize()

        def measure(fn):
            vals = []
            for _ in range(10):
                cold()
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                b.record()
                b.synchronize()
                vals.append(float(a.elapsed_time(b)))
            return {
                "median_ms": statistics.median(vals),
                "p10_ms": float(np.percentile(vals, 10)),
                "p90_ms": float(np.percentile(vals, 90)),
                "raw_ms": vals,
            }

        results = {}
        fp32_capability = None
        for B in BLOCKS:
            aggregate = {
                v: {"baseline_ms": 0.0, "candidate_ms": 0.0,
                    "max_nrmse": 0.0, "cases": []}
                for v in VARIANTS
            }
            for case in cases:
                raw = (
                    torch.utils.dlpack.from_dlpack(case.W)
                    .view(torch.bfloat16)
                    .reshape(case.rows, case.cols)
                )
                wtt = raw.t().contiguous()
                xh = rng.standard_normal((B, case.cols), dtype=np.float32)
                xcp = cp.asarray(xh)
                xt = torch.from_numpy(xh).to("cuda")
                ro = cp.empty((B, case.rows), cp.float32)

                def baseline():
                    for j in range(B):
                        ref.mv_bf16(
                            ro[j], case.W, xcp[j], case.rows, case.cols
                        )

                baseline()
                torch.cuda.synchronize()
                rr = cp.asnumpy(ro).astype(np.float64)
                bm = measure(baseline)

                def native(v):
                    xb = xt.to(torch.bfloat16)
                    if v == "mm_bf16out":
                        return torch.mm(xb, wtt).float()
                    if v == "mm_fp32out":
                        return torch.mm(
                            xb, wtt, out_dtype=torch.float32
                        )
                    if v == "mm_fp32out_comp2":
                        hi = xt.to(torch.bfloat16)
                        lo = (xt - hi.float()).to(torch.bfloat16)
                        return (
                            torch.mm(hi, wtt, out_dtype=torch.float32)
                            + torch.mm(lo, wtt, out_dtype=torch.float32)
                        )
                    raise ValueError(v)

                for v in VARIANTS:
                    rec = {
                        "case": case.name,
                        "family": case.family,
                        "shape": [case.rows, case.cols],
                    }
                    try:
                        cand = native(v)
                        torch.cuda.synchronize()
                        cc = cand.detach().cpu().numpy().astype(np.float64)
                        diff = cc - rr
                        nrmse = float(
                            np.linalg.norm(diff)
                            / max(float(np.linalg.norm(rr)), 1e-30)
                        )
                        cm = measure(lambda v=v: native(v))
                        rec.update({
                            "status": "measured",
                            "baseline_ms": bm["median_ms"],
                            "candidate_ms": cm["median_ms"],
                            "speedup": bm["median_ms"] / cm["median_ms"],
                            "nrmse": nrmse,
                            "max_abs": float(np.max(np.abs(diff))),
                            "row_argmax_agreement": float(np.mean(
                                np.argmax(rr, axis=1)
                                == np.argmax(cc, axis=1)
                            )),
                        })
                        a = aggregate[v]
                        a["baseline_ms"] += bm["median_ms"]
                        a["candidate_ms"] += cm["median_ms"]
                        a["max_nrmse"] = max(a["max_nrmse"], nrmse)
                        a["cases"].append(rec)
                        if v.startswith("mm_fp32out"):
                            fp32_capability = True
                    except TypeError as exc:
                        rec.update({
                            "status": "unsupported",
                            "error": str(exc),
                        })
                        aggregate[v]["cases"].append(rec)
                        if v.startswith("mm_fp32out"):
                            fp32_capability = False
                    except RuntimeError as exc:
                        if "out_dtype" in str(exc):
                            rec.update({
                                "status": "unsupported",
                                "error": str(exc),
                            })
                            aggregate[v]["cases"].append(rec)
                            if v.startswith("mm_fp32out"):
                                fp32_capability = False
                        else:
                            raise

                del raw, wtt, xcp, xt, ro
                torch.cuda.empty_cache()

            for v, a in aggregate.items():
                measured = [
                    x for x in a["cases"] if x.get("status") == "measured"
                ]
                a["case_count_measured"] = len(measured)
                a["aggregate_speedup"] = (
                    a["baseline_ms"] / a["candidate_ms"]
                    if a["candidate_ms"] > 0 else None
                )
                a["mean_nrmse"] = (
                    float(np.mean([x["nrmse"] for x in measured]))
                    if measured else None
                )
            results[str(B)] = aggregate

        payload.update({
            "status": "measured",
            "case_count": len(cases),
            "l2_bytes": l2,
            "cache_scrub_bytes": scrub_n,
            "torch_mm_bf16_fp32_out_supported": fp32_capability,
            "per_B": results,
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
        "fp32_out": payload.get("torch_mm_bf16_fp32_out_supported"),
        "B1": (payload.get("per_B") or {}).get("1"),
        "B4": (payload.get("per_B") or {}).get("4"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2)[:16000])
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
