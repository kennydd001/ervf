from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback

import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now
from diag_component_marginals_graph import _reset_exact_state
from s100_phase10a_runtime import build
from s100_phase14d2_native import NativeBF16Dispatch
from s100_phase5_quality import load_trace, CAL_TH
from s100_phase3_fidelity import TH, _snap, _domain_summary, _bootstrap

OUTDIR = REPO / "pro_research" / "results" / "s100_phase14d2"

def reset_eager(rt):
    _reset_exact_state(rt)
    rt._graph = None
    rt.graph_mode = False
    rt.reset()

def feed(rt, ids):
    for token in ids:
        rt.step(int(token))

def teacher_hash(rt, prompt_ids, targets, count):
    import cupy as cp
    h = hashlib.sha256()
    reset_eager(rt)
    feed(rt, prompt_ids)
    for i in range(min(count, len(targets))):
        t = int(targets[i])
        m = cp.max(rt.logits)
        lse = float((m + cp.log(cp.exp(rt.logits - m).sum())).item())
        a = int(cp.argmax(rt.logits).item())
        lp = np.float32(float(rt.logits[t].item()) - lse)
        h.update(np.asarray([a], dtype="<i4").tobytes())
        h.update(np.asarray([lp], dtype="<f4").tobytes())
        if i + 1 < min(count, len(targets)):
            rt.step(t)
    return h.hexdigest()

def evaluate(split):
    import cupy as cp
    from transformers import AutoTokenizer

    bundle = build()
    rt = bundle.rt
    dispatch = NativeBF16Dispatch(rt)
    rt.k.mv_bf16 = dispatch
    rt._graph = None
    rt.graph_mode = False

    prompts, indices, n, d, meta = load_trace(split)
    targets = d["target_ids"].astype(np.int32)
    base_tlp = d["target_logprob"].astype(np.float32)
    base_ids = d["top_ids"].astype(np.int32)
    base_lp = d["top_logprob"].astype(np.float32)
    base_rest = d["rest_prob"].astype(np.float32)

    pc = len(prompts)
    top1 = np.zeros((pc, n), bool)
    in5 = np.zeros((pc, n), bool)
    rank = np.empty((pc, n), np.int32)
    ce = np.empty((pc, n), np.float32)
    kl = np.empty((pc, n), np.float32)
    dom = np.empty((pc, n), dtype="<U32")

    tok = AutoTokenizer.from_pretrained(
        str(require_model_dir()),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    pids_all = []

    for pi, (prompt, original_index) in enumerate(zip(prompts, indices)):
        ids = tok.encode(prompt["prompt"], add_special_tokens=False)
        pids_all.append(ids)
        dom[pi, :] = prompt["domain"]
        got = hashlib.sha256(
            np.asarray(ids, dtype="<i4").tobytes()
        ).hexdigest()
        expected = meta["prompt_records"][original_index]["prompt_ids_sha256"]
        if got != expected:
            raise RuntimeError(f"tokenizer drift {prompt['id']}")

        reset_eager(rt)
        feed(rt, ids)
        for ti in range(n):
            target = int(targets[pi, ti])
            clp, rr, ctop5, qlog = _snap(
                cp, rt.logits, target, base_ids[pi, ti]
            )
            top1[pi, ti] = int(ctop5[0]) == target
            in5[pi, ti] = target in {int(x) for x in ctop5}
            rank[pi, ti] = rr
            ce[pi, ti] = float(base_tlp[pi, ti]) - clp

            plog = base_lp[pi, ti].astype(np.float64)
            pp = np.exp(plog)
            qq = np.exp(qlog)
            pr = max(float(base_rest[pi, ti]), 1e-30)
            qr = max(1.0 - float(qq.sum()), 1e-30)
            kl[pi, ti] = max(float(
                np.sum(pp * (plog - qlog))
                + pr * (math.log(pr) - math.log(qr))
            ), 0.0)

            if ti + 1 < n:
                rt.step(target)
        print(f"14D2 quality {split} {pi+1}/{pc}: {prompt['id']}", flush=True)

    ft, fi = top1.ravel(), in5.ravel()
    fr, fc, fk, fd = rank.ravel(), ce.ravel(), kl.ravel(), dom.ravel()
    domains = _domain_summary(fd, ft, fi, fc, fk, fr)
    boot = _bootstrap(fc, ft)

    deterministic = True
    anchor = None
    if split == "heldout":
        a = hashlib.sha256()
        b = hashlib.sha256()
        count = min(4, pc)
        length = min(64, n)
        for pi in range(count):
            a.update(bytes.fromhex(teacher_hash(
                rt, pids_all[pi], targets[pi], length
            )))
        for pi in range(count):
            b.update(bytes.fromhex(teacher_hash(
                rt, pids_all[pi], targets[pi], length
            )))
        deterministic = a.hexdigest() == b.hexdigest()
        anchor = a.hexdigest()

    summary = {
        "tokens": int(ft.size),
        "top1_agreement": float(ft.mean()),
        "target_in_top5": float(fi.mean()),
        "mean_target_rank": float(fr.mean()),
        "max_target_rank": int(fr.max()),
        "mean_ce_delta": float(fc.mean()),
        "p95_ce_delta": float(np.percentile(fc, 95)),
        "mean_coarse_kl": float(fk.mean()),
        "p95_coarse_kl": float(np.percentile(fk, 95)),
        "bootstrap": boot,
        "all_finite": bool(np.isfinite(fc).all() and np.isfinite(fk).all()),
        "deterministic_anchor_repeat": bool(deterministic),
        "anchor_hash": anchor,
    }

    official = {
        "F1_top1": summary["top1_agreement"] >= TH["top1_agreement_min"],
        "F2_top5": summary["target_in_top5"] >= TH["target_in_top5_min"],
        "F3_mean_ce": summary["mean_ce_delta"] <= TH["mean_ce_delta_max"],
        "F4_bootstrap_ce": boot["mean_ce_delta_p95"] <= TH[
            "bootstrap95_mean_ce_delta_max"
        ],
        "F5_p95_ce": summary["p95_ce_delta"] <= TH["p95_ce_delta_max"],
        "F6_mean_kl": summary["mean_coarse_kl"] <= TH["mean_coarse_kl_max"],
        "F7_p95_kl": summary["p95_coarse_kl"] <= TH["p95_coarse_kl_max"],
        "F8_domain_top1": all(
            x["top1_agreement"] >= TH["per_domain_top1_min"]
            for x in domains.values()
        ),
        "F9_domain_ce": all(
            x["mean_ce_delta"] <= TH["per_domain_mean_ce_delta_max"]
            for x in domains.values()
        ),
        "F10_deterministic": bool(deterministic),
        "F11_finite": summary["all_finite"],
    }
    strict = {
        "V1_top1": summary["top1_agreement"] >= CAL_TH["top1"],
        "V2_top5": summary["target_in_top5"] >= CAL_TH["top5"],
        "V3_mean_ce": summary["mean_ce_delta"] <= CAL_TH["mean_ce"],
        "V4_mean_kl": summary["mean_coarse_kl"] <= CAL_TH["mean_kl"],
        "V5_p95_kl": summary["p95_coarse_kl"] <= CAL_TH["p95_kl"],
        "V6_domain_top1": all(
            x["top1_agreement"] >= CAL_TH["domain_top1"]
            for x in domains.values()
        ),
        "V7_domain_ce": all(
            x["mean_ce_delta"] <= CAL_TH["domain_ce"]
            for x in domains.values()
        ),
        "V8_finite": summary["all_finite"],
    }
    result = {
        "kind": "s100_phase14d2_native_quality",
        "status": "measured",
        "split": split,
        "claim_boundary": (
            "eager causal numerical fidelity of native BF16 substitution; "
            "not production-graph latency"
        ),
        "native_bf16_calls": dispatch.calls,
        "native_weight_count": len(dispatch.weights),
        "summary": summary,
        "per_domain": domains,
        "official_gates": official,
        "official_pass": all(official.values()),
        "strict_gates": strict,
        "strict_pass": all(strict.values()),
    }
    bundle.restore_combined()
    bundle.restore_sel()
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("validation", "heldout"), required=True)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"S100_PHASE14D2_{args.split.upper()}.json"
    payload = {
        "kind": "s100_phase14d2_native_quality",
        "status": "started",
        "split": args.split,
        "started_utc": utc_now(),
    }
    try:
        payload = evaluate(args.split)
        payload["completed_utc"] = utc_now()
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
    write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "split": args.split,
        "strict_pass": payload.get("strict_pass"),
        "official_pass": payload.get("official_pass"),
        "summary": payload.get("summary"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
