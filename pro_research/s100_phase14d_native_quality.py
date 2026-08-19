from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback

import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_phase14_common import RESULTS, ensure_results
from s100_phase5_quality import load_trace, CAL_TH
from s100_phase3_fidelity import TH, _snap, _domain_summary

class NativeBF16Dispatch:
    """Numerical-fidelity dispatch; timing is intentionally not claimed."""
    def __init__(self, rt):
        import torch
        self.rt = rt
        self.torch = torch
        self.cp = rt.cp
        self.original = rt.k.mv_bf16
        self.weights = {}
        self.calls = 0

    def __call__(self, out, W, x, rows, cols):
        torch = self.torch
        cp = self.cp
        key = (int(W.data.ptr), int(rows), int(cols))
        weight = self.weights.get(key)
        if weight is None:
            weight = (
                torch.utils.dlpack.from_dlpack(W)
                .view(torch.bfloat16)
                .reshape(int(rows), int(cols))
            )
            self.weights[key] = weight

        stream = torch.cuda.ExternalStream(
            cp.cuda.get_current_stream().ptr
        )
        with torch.cuda.stream(stream):
            xt = torch.utils.dlpack.from_dlpack(x)
            y = torch.mv(weight, xt.to(torch.bfloat16)).float()
        yc = cp.from_dlpack(y)
        cp.copyto(out, yc)
        self.calls += 1

def reset_eager(rt):
    rt._graph = None
    rt.graph_mode = False
    rt.reset()

def feed_prompt(rt, ids):
    for token in ids:
        rt.step(int(token))

def teacher_hash(rt, prompt_ids, targets, n):
    h = hashlib.sha256()
    reset_eager(rt)
    feed_prompt(rt, prompt_ids)
    for i in range(n):
        h.update(rt.cp.asnumpy(rt.logits).astype("<f4").tobytes())
        if i + 1 < n:
            rt.step(int(targets[i]))
    return h.hexdigest()

def evaluate(split):
    import cupy as cp
    from transformers import AutoTokenizer

    bundle = build()
    rt = bundle.rt
    rt._graph = None
    rt.graph_mode = False
    dispatch = NativeBF16Dispatch(rt)
    rt.k.mv_bf16 = dispatch

    prompts, indices, n, data, meta = load_trace(split)
    targets = data["target_ids"].astype(np.int32)
    base_tlp = data["target_logprob"].astype(np.float32)
    base_ids = data["top_ids"].astype(np.int32)
    base_lp = data["top_logprob"].astype(np.float32)
    base_rest = data["rest_prob"].astype(np.float32)

    pc = len(prompts)
    top1 = np.zeros((pc, n), bool)
    in5 = np.zeros((pc, n), bool)
    rank = np.empty((pc, n), np.int32)
    ce = np.empty((pc, n), np.float32)
    kl = np.empty((pc, n), np.float32)
    domain = np.empty((pc, n), dtype="<U32")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        str(require_model_dir()),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    prompt_ids = []

    for pi, (prompt, original_index) in enumerate(zip(prompts, indices)):
        ids = tokenizer.encode(
            prompt["prompt"], add_special_tokens=False
        )
        prompt_ids.append(ids)
        domain[pi, :] = prompt["domain"]
        expected = meta["prompt_records"][original_index][
            "prompt_ids_sha256"
        ]
        actual = hashlib.sha256(
            np.asarray(ids, dtype="<i4").tobytes()
        ).hexdigest()
        if actual != expected:
            raise RuntimeError(f"tokenizer drift for {prompt['id']}")

        reset_eager(rt)
        feed_prompt(rt, ids)
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

        print(
            f"{split} {pi+1:02d}/{pc}: {prompt['id']}",
            flush=True,
        )

    flat_top1 = top1.ravel()
    flat_in5 = in5.ravel()
    flat_rank = rank.ravel()
    flat_ce = ce.ravel()
    flat_kl = kl.ravel()
    flat_domain = domain.ravel()
    domains = _domain_summary(
        flat_domain, flat_top1, flat_in5,
        flat_ce, flat_kl, flat_rank,
    )

    deterministic = True
    anchor = None
    if split == "heldout":
        count = min(2, pc)
        length = min(32, n)
        a = hashlib.sha256()
        b = hashlib.sha256()
        for pi in range(count):
            a.update(bytes.fromhex(teacher_hash(
                rt, prompt_ids[pi], targets[pi], length
            )))
        for pi in range(count):
            b.update(bytes.fromhex(teacher_hash(
                rt, prompt_ids[pi], targets[pi], length
            )))
        deterministic = a.hexdigest() == b.hexdigest()
        anchor = a.hexdigest()

    summary = {
        "tokens": int(flat_top1.size),
        "top1_agreement": float(flat_top1.mean()),
        "target_in_top5": float(flat_in5.mean()),
        "mean_target_rank": float(flat_rank.mean()),
        "max_target_rank": int(flat_rank.max()),
        "mean_ce_delta": float(flat_ce.mean()),
        "p95_ce_delta": float(np.percentile(flat_ce, 95)),
        "mean_coarse_kl": float(flat_kl.mean()),
        "p95_coarse_kl": float(np.percentile(flat_kl, 95)),
        "all_finite": bool(
            np.isfinite(flat_ce).all() and np.isfinite(flat_kl).all()
        ),
        "deterministic_anchor_repeat": deterministic,
        "anchor_hash": anchor,
    }

    official = {
        "F1_top1": summary["top1_agreement"] >= TH[
            "top1_agreement_min"
        ],
        "F2_top5": summary["target_in_top5"] >= TH[
            "target_in_top5_min"
        ],
        "F3_mean_ce": summary["mean_ce_delta"] <= TH[
            "mean_ce_delta_max"
        ],
        "F5_p95_ce": summary["p95_ce_delta"] <= TH[
            "p95_ce_delta_max"
        ],
        "F6_mean_kl": summary["mean_coarse_kl"] <= TH[
            "mean_coarse_kl_max"
        ],
        "F7_p95_kl": summary["p95_coarse_kl"] <= TH[
            "p95_coarse_kl_max"
        ],
        "F8_domain_top1": all(
            row["top1_agreement"] >= TH["per_domain_top1_min"]
            for row in domains.values()
        ),
        "F9_domain_ce": all(
            row["mean_ce_delta"] <= TH[
                "per_domain_mean_ce_delta_max"
            ]
            for row in domains.values()
        ),
        "F10_deterministic": deterministic,
        "F11_finite": summary["all_finite"],
    }
    strict = {
        "V1_top1": summary["top1_agreement"] >= CAL_TH["top1"],
        "V2_top5": summary["target_in_top5"] >= CAL_TH["top5"],
        "V3_mean_ce": summary["mean_ce_delta"] <= CAL_TH["mean_ce"],
        "V4_mean_kl": summary["mean_coarse_kl"] <= CAL_TH["mean_kl"],
        "V5_p95_kl": summary["p95_coarse_kl"] <= CAL_TH["p95_kl"],
        "V6_domain_top1": all(
            row["top1_agreement"] >= CAL_TH["domain_top1"]
            for row in domains.values()
        ),
        "V7_domain_ce": all(
            row["mean_ce_delta"] <= CAL_TH["domain_ce"]
            for row in domains.values()
        ),
        "V8_finite": summary["all_finite"],
    }

    result = {
        "kind": "s100_phase14d_native_quality",
        "status": "measured",
        "split": split,
        "claim_boundary": (
            "native BF16 numerical fidelity in eager M=1 execution; "
            "not a B=4 latency result"
        ),
        "native_matrix_count": len(dispatch.weights),
        "native_bf16_calls": dispatch.calls,
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
    ap.add_argument(
        "--split", choices=("validation", "heldout"), required=True
    )
    args = ap.parse_args()
    ensure_results()
    output = RESULTS / f"S100_PHASE14D_NATIVE_{args.split.upper()}.json"
    payload = {
        "kind": "s100_phase14d_native_quality",
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

    write_json_atomic(output, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "split": args.split,
        "strict_pass": payload.get("strict_pass"),
        "official_pass": payload.get("official_pass"),
        "summary": payload.get("summary"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(output),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
