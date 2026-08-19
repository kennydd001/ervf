from __future__ import annotations

import json
import math
import traceback
import numpy as np

from common import REPO, write_json_atomic, utc_now
from s100_phase15_native import (
    make_runtime, release_runtime, prompt_rows, collect_bf16_cases,
    logsumexp_np
)

OUT = (
    REPO / "pro_research" / "results" / "s100_phase15"
    / "S100_PHASE15C_MATRIX_SENSITIVITY.json"
)
TOKENS = 8

def exact_reference(rows):
    import cupy as cp
    rt, _ = make_runtime(None)
    result = {}
    try:
        for p in rows:
            rt.reset()
            for t in p["prompt_ids"]:
                rt.step(int(t))
            recs = []
            for step in range(TOKENS):
                logits = cp.asnumpy(rt.logits).astype(np.float64)
                lse = logsumexp_np(logits)
                ids = np.argpartition(logits, -32)[-32:]
                ids = ids[np.argsort(-logits[ids])]
                target = int(ids[0])
                lp = logits[ids] - lse
                rest = max(1.0 - float(np.exp(lp).sum()), 1e-30)
                recs.append({
                    "target": target,
                    "ids": ids.astype(np.int32),
                    "logprob": lp.astype(np.float32),
                    "target_logprob": float(logits[target] - lse),
                    "rest": rest,
                })
                if step + 1 < TOKENS:
                    rt.step(target)
            result[p["id"]] = recs
    finally:
        release_runtime(rt)
    return result

def load_component_savings():
    p = (
        REPO / "pro_research" / "results" / "s100_phase14d2"
        / "S100_PHASE14D2_COMPONENT.json"
    )
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        out = {}
        for B in ("1", "4"):
            for row in data.get("per_B", {}).get(B, {}).get("cases", []):
                out.setdefault(row["case"], {})[f"B{B}_saving_ms"] = (
                    float(row["baseline_independent_ervf"]["median_ms"])
                    - float(row["native_bf16"]["median_ms"])
                )
                out[row["case"]][f"B{B}_speedup"] = float(row["speedup"])
        return out
    except Exception:
        return {}

def run_case(rows, exact, case_name):
    import cupy as cp
    rt, dispatch = make_runtime("mm_fp32out", f"name:{case_name}")
    rec = []
    try:
        for p in rows:
            rt.reset()
            for t in p["prompt_ids"]:
                rt.step(int(t))
            for step in range(TOKENS):
                e = exact[p["id"]][step]
                logits = cp.asnumpy(rt.logits).astype(np.float64)
                lse = logsumexp_np(logits)
                target = int(e["target"])
                cand_top = np.argpartition(logits, -5)[-5:]
                cand_top = cand_top[np.argsort(-logits[cand_top])]

                exact_ids = np.asarray(e["ids"], np.int32)
                plog = np.asarray(e["logprob"], np.float64)
                qlog = logits[exact_ids] - lse
                pp, qq = np.exp(plog), np.exp(qlog)
                pr = max(float(e["rest"]), 1e-30)
                qr = max(1.0 - float(qq.sum()), 1e-30)
                kl = max(float(
                    np.sum(pp * (plog - qlog))
                    + pr * (math.log(pr) - math.log(qr))
                ), 0.0)

                rec.append({
                    "top1": int(cand_top[0]) == target,
                    "top5": bool(np.any(cand_top == target)),
                    "ce": float(
                        e["target_logprob"]
                        - (logits[target] - lse)
                    ),
                    "kl": kl,
                })
                if step + 1 < TOKENS:
                    rt.step(target)
    finally:
        release_runtime(rt)

    ce = np.asarray([x["ce"] for x in rec], np.float64)
    kl = np.asarray([x["kl"] for x in rec], np.float64)
    return {
        "tokens": len(rec),
        "top1": float(np.mean([x["top1"] for x in rec])),
        "top5": float(np.mean([x["top5"] for x in rec])),
        "mean_ce": float(ce.mean()),
        "mean_kl": float(kl.mean()),
        "p95_kl": float(np.percentile(kl, 95)),
        "finite": bool(np.isfinite(ce).all() and np.isfinite(kl).all()),
    }

def main():
    payload = {
        "kind": "s100_phase15c_matrix_sensitivity",
        "status": "started",
        "started_utc": utc_now(),
        "tokens_per_prompt": TOKENS,
        "claim_boundary": "calibration-only selection evidence",
    }
    try:
        rows = prompt_rows("calibration")
        exact = exact_reference(rows)

        rt, _ = make_runtime(None)
        cases = collect_bf16_cases(rt)
        release_runtime(rt)

        savings = load_component_savings()
        arms = []
        for i, case in enumerate(cases):
            m = run_case(rows, exact, case.name)
            locally_safe = bool(
                m["top1"] >= 0.995
                and m["top5"] == 1.0
                and m["mean_ce"] <= 0.005
                and m["mean_kl"] <= 0.003
                and m["finite"]
            )
            arms.append({
                "case": case.name,
                "family": case.family,
                "metrics": m,
                "locally_safe": locally_safe,
                "phase14_component": savings.get(case.name),
            })
            print(
                f"P15 matrix {i+1}/{len(cases)} "
                f"{case.name} safe={locally_safe}",
                flush=True,
            )

        safe = [x for x in arms if x["locally_safe"]]
        safe.sort(
            key=lambda x: (
                -float((x.get("phase14_component") or {}).get(
                    "B1_saving_ms", 0.0
                )),
                x["metrics"]["mean_ce"],
            )
        )
        payload.update({
            "status": "measured",
            "case_count": len(arms),
            "safe_count": len(safe),
            "safe_cases_ranked": [x["case"] for x in safe],
            "arms": arms,
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
        "case_count": payload.get("case_count"),
        "safe_count": payload.get("safe_count"),
        "safe_cases_ranked": payload.get("safe_cases_ranked"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
