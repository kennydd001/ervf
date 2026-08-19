from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback

import numpy as np

import torch  # Import before runtime/CuPy initialization.

from common import REPO, require_model_dir, sha256_file, write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_lightning15_common import (
    OFFICIAL, RESULTS, STRICT, TOP_K,
    candidate_name, ensure_results, feed_prompt, identity,
    logsumexp, model_signature, normalize_eager_moe,
    parse_families, reset_eager, trace_paths,
)
from s100_lightning15_native import SelectiveNativeDispatch

def load_trace(split):
    npz_path, meta_path = trace_paths(split)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    ident = identity()
    if metadata.get("status") != "trace_ready":
        raise RuntimeError(f"{split} trace not ready")
    if metadata.get("model_signature") != model_signature(ident):
        raise RuntimeError(f"{split} trace/model signature mismatch")
    if metadata.get("trace_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"{split} trace hash mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return metadata, arrays

def domain_summary(domains, top1, top5, ce, kl, rank):
    result = {}
    for domain in sorted(set(domains.tolist())):
        mask = domains == domain
        result[domain] = {
            "tokens": int(mask.sum()),
            "top1_agreement": float(top1[mask].mean()),
            "target_in_top5": float(top5[mask].mean()),
            "mean_ce_delta": float(ce[mask].mean()),
            "p95_ce_delta": float(np.percentile(ce[mask], 95)),
            "mean_coarse_kl": float(kl[mask].mean()),
            "p95_coarse_kl": float(np.percentile(kl[mask], 95)),
            "mean_target_rank": float(rank[mask].mean()),
            "max_target_rank": int(rank[mask].max()),
        }
    return result

def teacher_hash(rt, prompt_ids, targets, count):
    digest = hashlib.sha256()
    reset_eager(rt)
    feed_prompt(rt, prompt_ids)
    for index in range(min(count, len(targets))):
        target = int(targets[index])
        lse = logsumexp(rt.cp, rt.logits)
        argmax = int(rt.cp.argmax(rt.logits).item())
        logp = np.float32(float(rt.logits[target].item()) - lse)
        digest.update(np.asarray([argmax], "<i4").tobytes())
        digest.update(np.asarray([logp], "<f4").tobytes())
        if index + 1 < count:
            rt.step(target)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("baseline", "round_ervf", "tc1", "tc2", "tc3"),
        required=True,
    )
    parser.add_argument("--families", default="kvo")
    parser.add_argument(
        "--split",
        choices=("calibration", "validation", "heldout"),
        required=True,
    )
    args = parser.parse_args()
    families = parse_families(args.families)
    name = candidate_name(args.mode, args.families)
    ensure_results()
    output = RESULTS / (
        f"S100_LIGHTNING15_QUALITY_{name.upper()}_"
        f"{args.split.upper()}.json"
    )
    payload = {
        "kind": "s100_lightning15_quality",
        "status": "started",
        "candidate": name,
        "mode": args.mode,
        "families": sorted(families),
        "split": args.split,
        "started_utc": utc_now(),
    }

    try:
        import cupy as cp
        from transformers import AutoTokenizer

        metadata, trace = load_trace(args.split)
        bundle = build()
        rt = bundle.rt
        rt._graph = None
        rt.graph_mode = False
        normalize_eager_moe(rt)

        dispatch = None
        if args.mode != "baseline":
            dispatch = SelectiveNativeDispatch(
                rt, args.mode, families
            ).install()

        tokenizer = AutoTokenizer.from_pretrained(
            str(require_model_dir()),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
        prompt_records = metadata["prompt_records"]
        target_ids = trace["target_ids"].astype(np.int32)
        base_target_logprob = trace["target_logprob"].astype(np.float32)
        base_ids = trace["top_ids"].astype(np.int32)
        base_logprob = trace["top_logprob"].astype(np.float32)
        base_rest = trace["rest_prob"].astype(np.float32)
        prompt_count, length = target_ids.shape

        top1 = np.zeros((prompt_count, length), bool)
        top5 = np.zeros((prompt_count, length), bool)
        rank = np.empty((prompt_count, length), np.int32)
        ce = np.empty((prompt_count, length), np.float32)
        kl = np.empty((prompt_count, length), np.float32)
        domains = np.empty((prompt_count, length), dtype="<U32")
        prompt_ids = []

        prompt_manifest = json.loads(
            (
                REPO / "pro_research" / "S100_PHASE3_PROMPTS.json"
            ).read_text(encoding="utf-8")
        )["prompts"]
        by_id = {row["id"]: row for row in prompt_manifest}

        for pi, record in enumerate(prompt_records):
            prompt = by_id[record["id"]]
            ids = tokenizer.encode(
                prompt["prompt"], add_special_tokens=False
            )
            prompt_ids.append(ids)
            if hashlib.sha256(
                np.asarray(ids, "<i4").tobytes()
            ).hexdigest() != record["prompt_ids_sha256"]:
                raise RuntimeError(f"tokenizer drift {record['id']}")
            domains[pi, :] = record["domain"]

            reset_eager(rt)
            feed_prompt(rt, ids)
            for ti in range(length):
                logits = rt.logits
                if not bool(cp.isfinite(logits).all().item()):
                    raise RuntimeError(
                        f"non-finite {record['id']}:{ti}"
                    )
                target = int(target_ids[pi, ti])
                lse = logsumexp(cp, logits)
                target_value = float(logits[target].item())
                target_rank = int(
                    cp.count_nonzero(logits > target_value).item()
                )
                candidate_top = cp.argpartition(logits, -5)[-5:]
                candidate_top = candidate_top[
                    cp.argsort(-logits[candidate_top])
                ]
                candidate_top = cp.asnumpy(
                    candidate_top
                ).astype(np.int32)
                values = cp.asnumpy(
                    logits[cp.asarray(base_ids[pi, ti].astype(np.int64))]
                ).astype(np.float64)
                qlog = values - lse

                top1[pi, ti] = int(candidate_top[0]) == target
                top5[pi, ti] = target in {
                    int(value) for value in candidate_top
                }
                rank[pi, ti] = target_rank
                candidate_logprob = target_value - lse
                ce[pi, ti] = (
                    float(base_target_logprob[pi, ti])
                    - candidate_logprob
                )

                plog = base_logprob[pi, ti].astype(np.float64)
                pp = np.exp(plog)
                qq = np.exp(qlog)
                pr = max(float(base_rest[pi, ti]), 1e-30)
                qr = max(1.0 - float(qq.sum()), 1e-30)
                kl[pi, ti] = max(float(
                    np.sum(pp * (plog - qlog))
                    + pr * (math.log(pr) - math.log(qr))
                ), 0.0)

                if ti + 1 < length:
                    rt.step(target)

            print(
                f"{name} {args.split} {pi+1:02d}/{prompt_count}: "
                f"{record['id']}",
                flush=True,
            )

        flat_top1 = top1.ravel()
        flat_top5 = top5.ravel()
        flat_rank = rank.ravel()
        flat_ce = ce.ravel()
        flat_kl = kl.ravel()
        flat_domains = domains.ravel()
        per_domain = domain_summary(
            flat_domains, flat_top1, flat_top5,
            flat_ce, flat_kl, flat_rank,
        )

        deterministic = True
        anchor = None
        if args.split == "heldout":
            first = hashlib.sha256()
            second = hashlib.sha256()
            for pi in range(min(4, prompt_count)):
                first.update(bytes.fromhex(teacher_hash(
                    rt, prompt_ids[pi], target_ids[pi], min(64, length)
                )))
            for pi in range(min(4, prompt_count)):
                second.update(bytes.fromhex(teacher_hash(
                    rt, prompt_ids[pi], target_ids[pi], min(64, length)
                )))
            deterministic = first.hexdigest() == second.hexdigest()
            anchor = first.hexdigest()

        summary = {
            "tokens": int(flat_top1.size),
            "top1_agreement": float(flat_top1.mean()),
            "target_in_top5": float(flat_top5.mean()),
            "mean_target_rank": float(flat_rank.mean()),
            "max_target_rank": int(flat_rank.max()),
            "mean_ce_delta": float(flat_ce.mean()),
            "p95_ce_delta": float(np.percentile(flat_ce, 95)),
            "mean_coarse_kl": float(flat_kl.mean()),
            "p95_coarse_kl": float(np.percentile(flat_kl, 95)),
            "all_finite": bool(
                np.isfinite(flat_ce).all()
                and np.isfinite(flat_kl).all()
            ),
            "deterministic_anchor_repeat": deterministic,
            "anchor_hash": anchor,
        }

        strict_gates = {
            "top1": summary["top1_agreement"] >= STRICT["top1"],
            "top5": summary["target_in_top5"] >= STRICT["top5"],
            "mean_ce": summary["mean_ce_delta"] <= STRICT["mean_ce"],
            "mean_kl": summary["mean_coarse_kl"] <= STRICT["mean_kl"],
            "p95_kl": summary["p95_coarse_kl"] <= STRICT["p95_kl"],
            "domain_top1": all(
                row["top1_agreement"] >= STRICT["domain_top1"]
                for row in per_domain.values()
            ),
            "domain_ce": all(
                row["mean_ce_delta"] <= STRICT["domain_ce"]
                for row in per_domain.values()
            ),
            "finite": summary["all_finite"],
        }
        official_gates = {
            "top1": summary["top1_agreement"] >= OFFICIAL["top1"],
            "top5": summary["target_in_top5"] >= OFFICIAL["top5"],
            "mean_ce": summary["mean_ce_delta"] <= OFFICIAL["mean_ce"],
            "p95_ce": summary["p95_ce_delta"] <= OFFICIAL["p95_ce"],
            "mean_kl": summary["mean_coarse_kl"] <= OFFICIAL["mean_kl"],
            "p95_kl": summary["p95_coarse_kl"] <= OFFICIAL["p95_kl"],
            "domain_top1": all(
                row["top1_agreement"] >= OFFICIAL["domain_top1"]
                for row in per_domain.values()
            ),
            "domain_ce": all(
                row["mean_ce_delta"] <= OFFICIAL["domain_ce"]
                for row in per_domain.values()
            ),
            "deterministic": deterministic,
            "finite": summary["all_finite"],
        }

        baseline_exact = bool(
            args.mode != "baseline"
            or (
                summary["top1_agreement"] == 1.0
                and abs(summary["mean_ce_delta"]) <= 1e-6
                and summary["mean_coarse_kl"] <= 1e-8
            )
        )
        payload.update({
            "status": "measured",
            "identity": identity(),
            "trace_model_signature": metadata["model_signature"],
            "summary": summary,
            "per_domain": per_domain,
            "strict_gates": strict_gates,
            "strict_pass": all(strict_gates.values()),
            "official_gates": official_gates,
            "official_pass": all(official_gates.values()),
            "baseline_self_exact": baseline_exact,
            "dispatch": None if dispatch is None else {
                "native_calls": dispatch.native_calls,
                "original_calls": dispatch.original_calls,
                "torch_mm_call_style": (
                    dispatch.engine.mm.call_style
                    if dispatch.engine is not None else None
                ),
            },
            "completed_utc": utc_now(),
        })
        bundle.restore_combined()
        bundle.restore_sel()
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
        "candidate": name,
        "split": args.split,
        "summary": payload.get("summary"),
        "strict_pass": payload.get("strict_pass"),
        "official_pass": payload.get("official_pass"),
        "baseline_self_exact": payload.get("baseline_self_exact"),
        "dispatch": payload.get("dispatch"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(output),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
