from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback

import torch
import numpy as np

from common import require_model_dir, write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_phase14r_common import (
    RESULTS, ensure_results, normalize_eager_moe
)
from s100_phase5_quality import load_trace, CAL_TH
from s100_phase3_fidelity import (
    TH, _snap, _domain_summary, _bootstrap
)

class ZeroCopyNativeBF16:
    """B=1 native GEMM numerical path. No weight copy is created."""
    def __init__(self, rt):
        import torch
        self.torch = torch
        self.cp = rt.cp
        self.weights = {}
        self.buffers = {}
        self.calls = 0

    def __call__(self, out, weight_cp, x_cp, rows, cols):
        torch = self.torch
        cp = self.cp
        key = (int(weight_cp.data.ptr), int(rows), int(cols))
        weight = self.weights.get(key)
        if weight is None:
            weight = (
                torch.utils.dlpack.from_dlpack(weight_cp)
                .view(torch.bfloat16)
                .reshape(int(rows), int(cols))
            )
            self.weights[key] = weight
            self.buffers[key] = {
                "x_bf16": torch.empty(
                    (1, int(cols)), device="cuda", dtype=torch.bfloat16
                ),
                "y_bf16": torch.empty(
                    (1, int(rows)), device="cuda", dtype=torch.bfloat16
                ),
                "y_f32": torch.empty(
                    (int(rows),), device="cuda", dtype=torch.float32
                ),
            }

        buffers = self.buffers[key]
        stream = torch.cuda.ExternalStream(
            int(cp.cuda.get_current_stream().ptr)
        )
        with torch.cuda.stream(stream):
            x_t = torch.utils.dlpack.from_dlpack(x_cp)
            buffers["x_bf16"][0].copy_(x_t)
            torch.mm(
                buffers["x_bf16"], weight.t(),
                out=buffers["y_bf16"],
            )
            buffers["y_f32"].copy_(buffers["y_bf16"][0])
        cp.copyto(out, cp.from_dlpack(buffers["y_f32"]))
        self.calls += 1

def reset_eager(rt):
    rt._graph = None
    rt.graph_mode = False
    rt.reset()

def feed_prompt(rt, ids):
    for token in ids:
        rt.step(int(token))

def teacher_hash(rt, prompt_ids, targets, count):
    digest = hashlib.sha256()
    reset_eager(rt)
    feed_prompt(rt, prompt_ids)
    for index in range(count):
        digest.update(
            rt.cp.asnumpy(rt.logits).astype("<f4").tobytes()
        )
        if index + 1 < count:
            rt.step(int(targets[index]))
    return digest.hexdigest()

def evaluate(split):
    import torch
    import cupy as cp
    from transformers import AutoTokenizer

    bundle = build()
    rt = bundle.rt
    rt._graph = None
    rt.graph_mode = False
    normalize_eager_moe(rt)

    dispatch = ZeroCopyNativeBF16(rt)
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
    tokenizer = AutoTokenizer.from_pretrained(
        str(require_model_dir()),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    prompt_ids = []

    for pi, (prompt, original_index) in enumerate(zip(prompts, indices)):
        ids = tokenizer.encode(prompt["prompt"], add_special_tokens=False)
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
            candidate_logprob, target_rank, candidate_top5, qlog = _snap(
                cp, rt.logits, target, base_ids[pi, ti]
            )
            top1[pi, ti] = int(candidate_top5[0]) == target
            in5[pi, ti] = target in {
                int(value) for value in candidate_top5
            }
            rank[pi, ti] = target_rank
            ce[pi, ti] = (
                float(base_tlp[pi, ti]) - candidate_logprob
            )

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
    bootstrap = _bootstrap(flat_ce, flat_top1)

    deterministic = True
    anchor = None
    if split == "heldout":
        prompt_count = min(4, pc)
        token_count = min(64, n)
        first = hashlib.sha256()
        second = hashlib.sha256()
        for pi in range(prompt_count):
            first.update(bytes.fromhex(teacher_hash(
                rt, prompt_ids[pi], targets[pi], token_count
            )))
        for pi in range(prompt_count):
            second.update(bytes.fromhex(teacher_hash(
                rt, prompt_ids[pi], targets[pi], token_count
            )))
        deterministic = first.hexdigest() == second.hexdigest()
        anchor = first.hexdigest()

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
        "bootstrap": bootstrap,
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
        "F4_bootstrap_ce": bootstrap["mean_ce_delta_p95"] <= TH[
            "bootstrap95_mean_ce_delta_max"
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
            value["top1_agreement"] >= TH["per_domain_top1_min"]
            for value in domains.values()
        ),
        "F9_domain_ce": all(
            value["mean_ce_delta"] <= TH[
                "per_domain_mean_ce_delta_max"
            ]
            for value in domains.values()
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
            value["top1_agreement"] >= CAL_TH["domain_top1"]
            for value in domains.values()
        ),
        "V7_domain_ce": all(
            value["mean_ce_delta"] <= CAL_TH["domain_ce"]
            for value in domains.values()
        ),
        "V8_finite": summary["all_finite"],
    }

    result = {
        "kind": "s100_phase14r_native_quality",
        "status": "measured",
        "split": split,
        "claim_boundary": (
            "zero-copy native BF16 B=1 numerical fidelity; "
            "B=4 speed is measured separately"
        ),
        "zero_copy_weight_aliases": len(dispatch.weights),
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("validation", "heldout"), required=True
    )
    args = parser.parse_args()
    ensure_results()
    output = RESULTS / (
        f"S100_PHASE14R_NATIVE_{args.split.upper()}.json"
    )
    payload = {
        "kind": "s100_phase14r_native_quality",
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
