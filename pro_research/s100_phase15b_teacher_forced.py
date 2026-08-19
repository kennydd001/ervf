from __future__ import annotations

import argparse
import json
import math
import traceback
import numpy as np

from common import REPO, write_json_atomic, utc_now
from s100_phase15_native import (
    make_runtime, release_runtime, prompt_rows, logsumexp_np
)

OUTDIR = REPO / "pro_research" / "results" / "s100_phase15"

STRICT = {
    "top1": 0.970,
    "top5": 0.999,
    "mean_ce": 0.025,
    "mean_kl": 0.015,
    "p95_kl": 0.060,
}

def exact_reference(rows, tokens):
    import cupy as cp
    rt, _ = make_runtime(None)
    result = {}
    try:
        for p in rows:
            rt.reset()
            for token in p["prompt_ids"]:
                rt.step(int(token))
            recs = []
            for step in range(tokens):
                logits = cp.asnumpy(rt.logits).astype(np.float64)
                lse = logsumexp_np(logits)
                top64 = np.argpartition(logits, -64)[-64:]
                top64 = top64[np.argsort(-logits[top64])]
                target = int(top64[0])
                lp = logits[top64] - lse
                pp = np.exp(lp)
                rest = max(1.0 - float(pp.sum()), 1e-30)
                recs.append({
                    "target": target,
                    "top64": top64.astype(np.int32),
                    "top64_logprob": lp.astype(np.float32),
                    "target_logprob": float(logits[target] - lse),
                    "rest_prob": rest,
                })
                if step + 1 < tokens:
                    rt.step(target)
            result[p["id"]] = recs
            print(f"P15 exact {p['id']}", flush=True)
    finally:
        release_runtime(rt)
    return result

def candidate(rows, exact, tokens, variant, scope):
    import cupy as cp
    rt, dispatch = make_runtime(variant, scope)
    all_rows = []
    try:
        for p in rows:
            rt.reset()
            # Same prompt token ids as the exact parent.
            for token in p["prompt_ids"]:
                rt.step(int(token))

            for step in range(tokens):
                e = exact[p["id"]][step]
                logits = cp.asnumpy(rt.logits).astype(np.float64)
                lse = logsumexp_np(logits)
                cand_top64 = np.argpartition(logits, -64)[-64:]
                cand_top64 = cand_top64[np.argsort(-logits[cand_top64])]
                target = int(e["target"])
                clp = float(logits[target] - lse)

                exact_ids = np.asarray(e["top64"], np.int32)
                plog = np.asarray(e["top64_logprob"], np.float64)
                qlog = logits[exact_ids] - lse
                pp = np.exp(plog)
                qq = np.exp(qlog)
                pr = max(float(e["rest_prob"]), 1e-30)
                qr = max(1.0 - float(qq.sum()), 1e-30)
                kl = max(float(
                    np.sum(pp * (plog - qlog))
                    + pr * (math.log(pr) - math.log(qr))
                ), 0.0)

                row = {
                    "prompt_id": p["id"],
                    "domain": p["domain"],
                    "step": step,
                    "target": target,
                    "candidate_top1": int(cand_top64[0]),
                    "top1": int(cand_top64[0]) == target,
                    "top5": bool(np.any(cand_top64[:5] == target)),
                    "ce_delta": float(e["target_logprob"] - clp),
                    "coarse_kl": kl,
                    "candidate_margin_norm": float(
                        (logits[cand_top64[0]] - logits[cand_top64[1]])
                        / max(float(np.std(logits)), 1e-12)
                    ),
                }
                for k in (8, 16, 32, 64):
                    row[f"K{k}_includes_exact"] = bool(
                        np.any(cand_top64[:k] == target)
                    )
                all_rows.append(row)

                # CRITICAL REPAIR:
                # advance with exact parent target, never candidate argmax.
                if step + 1 < tokens:
                    rt.step(target)

            print(
                f"P15 candidate {variant}/{scope} {p['id']}",
                flush=True,
            )
    finally:
        native_calls = getattr(dispatch, "calls_native", None)
        original_calls = getattr(dispatch, "calls_original", None)
        fp32_supported = getattr(dispatch, "fp32_out_supported", None)
        release_runtime(rt)

    top1 = np.asarray([r["top1"] for r in all_rows], bool)
    top5 = np.asarray([r["top5"] for r in all_rows], bool)
    ce = np.asarray([r["ce_delta"] for r in all_rows], np.float64)
    kl = np.asarray([r["coarse_kl"] for r in all_rows], np.float64)

    summary = {
        "tokens": len(all_rows),
        "top1_agreement": float(top1.mean()),
        "target_in_top5": float(top5.mean()),
        "mean_ce_delta": float(ce.mean()),
        "p95_ce_delta": float(np.percentile(ce, 95)),
        "mean_coarse_kl": float(kl.mean()),
        "p95_coarse_kl": float(np.percentile(kl, 95)),
        "all_finite": bool(np.isfinite(ce).all() and np.isfinite(kl).all()),
        "native_calls": native_calls,
        "original_bf16_calls": original_calls,
        "fp32_out_supported": fp32_supported,
    }
    for k in (8, 16, 32, 64):
        summary[f"K{k}_inclusion"] = float(np.mean([
            r[f"K{k}_includes_exact"] for r in all_rows
        ]))
    strict = {
        "top1": summary["top1_agreement"] >= STRICT["top1"],
        "top5": summary["target_in_top5"] >= STRICT["top5"],
        "mean_ce": summary["mean_ce_delta"] <= STRICT["mean_ce"],
        "mean_kl": summary["mean_coarse_kl"] <= STRICT["mean_kl"],
        "p95_kl": summary["p95_coarse_kl"] <= STRICT["p95_kl"],
        "finite": summary["all_finite"],
    }
    return summary, strict, all_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=("mm_bf16out", "mm_fp32out", "mm_fp32out_comp2"),
        required=True,
    )
    ap.add_argument(
        "--scope",
        default="all",
        help="all|attention|mamba|mamba_in|mamba_out|name:<case>",
    )
    ap.add_argument(
        "--split",
        choices=("calibration", "validation", "heldout"),
        required=True,
    )
    ap.add_argument("--tokens-per-prompt", type=int, default=24)
    args = ap.parse_args()

    safe_scope = args.scope.replace(":", "_").replace("/", "_")
    out = OUTDIR / (
        f"S100_PHASE15B_{args.variant.upper()}_"
        f"{safe_scope.upper()}_{args.split.upper()}.json"
    )
    payload = {
        "kind": "s100_phase15b_teacher_forced",
        "status": "started",
        "variant": args.variant,
        "scope": args.scope,
        "split": args.split,
        "tokens_per_prompt": args.tokens_per_prompt,
        "started_utc": utc_now(),
        "claim_boundary": (
            "same exact token prefixes; candidate recurrent state may drift"
        ),
    }
    try:
        rows = prompt_rows(args.split)
        exact = exact_reference(rows, args.tokens_per_prompt)
        summary, strict, records = candidate(
            rows, exact, args.tokens_per_prompt,
            args.variant, args.scope,
        )
        payload.update({
            "status": "measured",
            "summary": summary,
            "strict_gates": strict,
            "strict_pass": all(strict.values()),
            "records": records,
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

    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "variant": args.variant,
        "scope": args.scope,
        "split": args.split,
        "strict_pass": payload.get("strict_pass"),
        "summary": payload.get("summary"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
