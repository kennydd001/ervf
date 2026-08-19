from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import types

import numpy as np

from common import REPO, require_model_dir, sha256_file
from s100_lightning15_common import (
    RESULTS as PHASE15_RESULTS,
    identity,
    model_signature,
)

RESULTS = REPO / "pro_research" / "results" / "s100_lightning16"
PROMPTS = REPO / "pro_research" / "S100_PHASE3_PROMPTS.json"
TOP_K_TRACE = 64

STRICT = {
    "top1": 0.970,
    "top5": 0.999,
    "mean_ce": 0.025,
    "mean_kl": 0.015,
    "p95_kl": 0.060,
    "domain_top1": 0.90,
    "domain_ce": 0.080,
}
OFFICIAL = {
    "top1": 0.95,
    "top5": 0.995,
    "mean_ce": 0.05,
    "p95_ce": 0.25,
    "mean_kl": 0.02,
    "p95_kl": 0.08,
    "domain_top1": 0.90,
    "domain_ce": 0.10,
}

def ensure_results() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS

def assert_lightning() -> dict:
    record = identity()
    if record["max_position_embeddings"] != 1_048_576:
        raise RuntimeError("Phase 16 refuses non-Lightning checkpoint")
    return record

def prompt_manifest() -> dict[str, dict]:
    rows = json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"]
    return {row["id"]: row for row in rows}

def trace_paths(split: str):
    base = PHASE15_RESULTS / f"S100_LIGHTNING15_TRACE_{split.upper()}"
    return base.with_suffix(".npz"), base.with_suffix(".json")

def load_trace(split: str):
    npz_path, meta_path = trace_paths(split)
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Phase-15 Lightning trace missing: {split}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    current = assert_lightning()
    if metadata.get("status") != "trace_ready":
        raise RuntimeError(f"Lightning trace not ready: {split}")
    if metadata.get("model_signature") != model_signature(current):
        raise RuntimeError(f"Lightning trace signature mismatch: {split}")
    if metadata.get("trace_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"Lightning trace hash mismatch: {split}")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return metadata, arrays

def normalize_eager_moe(rt):
    original = rt._moe

    def safe(self, layer, out):
        result = original(layer, out)
        return (None, None) if result is None else result

    rt._moe = types.MethodType(safe, rt)
    return original

def reset_eager(rt):
    rt._graph = None
    rt.graph_mode = False
    rt.reset()

def feed_prompt(rt, ids):
    for token in ids:
        rt.step(int(token))

def logsumexp(cp, logits) -> float:
    maximum = cp.max(logits)
    return float(
        (maximum + cp.log(cp.exp(logits - maximum).sum())).item()
    )

def candidate_snapshot(cp, logits, target: int, base_ids: np.ndarray):
    if not bool(cp.isfinite(logits).all().item()):
        raise RuntimeError("non-finite candidate logits")
    lse = logsumexp(cp, logits)
    target_value = float(logits[target].item())
    rank = int(cp.count_nonzero(logits > target_value).item())
    top = cp.argpartition(logits, -5)[-5:]
    top = top[cp.argsort(-logits[top])]
    top = cp.asnumpy(top).astype(np.int32)
    values = cp.asnumpy(
        logits[cp.asarray(base_ids.astype(np.int64))]
    ).astype(np.float64)
    return target_value - lse, rank, top, values - lse

def domain_summary(domains, top1, top5, ce, kl, rank):
    result = {}
    for name in sorted(set(domains.tolist())):
        mask = domains == name
        result[name] = {
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

def evaluate_runtime(
    rt,
    *,
    split: str,
    prompt_limit: int | None = None,
    token_limit: int | None = None,
    deterministic: bool = False,
):
    import cupy as cp
    from transformers import AutoTokenizer

    metadata, trace = load_trace(split)
    manifest = prompt_manifest()
    records = metadata["prompt_records"]
    if prompt_limit is not None:
        records = records[:prompt_limit]

    target_ids = trace["target_ids"][:len(records)]
    base_target_logprob = trace["target_logprob"][:len(records)]
    base_ids = trace["top_ids"][:len(records)]
    base_logprob = trace["top_logprob"][:len(records)]
    base_rest = trace["rest_prob"][:len(records)]
    if token_limit is not None:
        target_ids = target_ids[:, :token_limit]
        base_target_logprob = base_target_logprob[:, :token_limit]
        base_ids = base_ids[:, :token_limit]
        base_logprob = base_logprob[:, :token_limit]
        base_rest = base_rest[:, :token_limit]

    prompt_count, length = target_ids.shape
    top1 = np.zeros((prompt_count, length), bool)
    top5 = np.zeros((prompt_count, length), bool)
    rank = np.empty((prompt_count, length), np.int32)
    ce = np.empty((prompt_count, length), np.float32)
    kl = np.empty((prompt_count, length), np.float32)
    domains = np.empty((prompt_count, length), dtype="<U32")
    prompt_ids = []

    tokenizer = AutoTokenizer.from_pretrained(
        str(require_model_dir()),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )

    for pi, record in enumerate(records):
        row = manifest[record["id"]]
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        prompt_ids.append(ids)
        digest = hashlib.sha256(
            np.asarray(ids, dtype="<i4").tobytes()
        ).hexdigest()
        if digest != record["prompt_ids_sha256"]:
            raise RuntimeError(f"tokenizer drift: {record['id']}")
        domains[pi, :] = record["domain"]
        reset_eager(rt)
        feed_prompt(rt, ids)

        for ti in range(length):
            target = int(target_ids[pi, ti])
            candidate_logp, target_rank, candidate_top5, qlog = (
                candidate_snapshot(
                    cp, rt.logits, target, base_ids[pi, ti]
                )
            )
            top1[pi, ti] = int(candidate_top5[0]) == target
            top5[pi, ti] = target in {
                int(value) for value in candidate_top5
            }
            rank[pi, ti] = target_rank
            ce[pi, ti] = (
                float(base_target_logprob[pi, ti]) - candidate_logp
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
    deterministic_repeat = True
    anchor_hash = None
    if deterministic:
        def run_anchor():
            digest = hashlib.sha256()
            for pi, record in enumerate(records[:min(4, prompt_count)]):
                row = manifest[record["id"]]
                ids = tokenizer.encode(
                    row["prompt"], add_special_tokens=False
                )
                reset_eager(rt)
                feed_prompt(rt, ids)
                count = min(64, length)
                for ti in range(count):
                    target = int(target_ids[pi, ti])
                    argmax = int(cp.argmax(rt.logits).item())
                    logp = np.float32(
                        float(rt.logits[target].item())
                        - logsumexp(cp, rt.logits)
                    )
                    digest.update(
                        np.asarray([argmax], dtype="<i4").tobytes()
                    )
                    digest.update(
                        np.asarray([logp], dtype="<f4").tobytes()
                    )
                    if ti + 1 < count:
                        rt.step(target)
            return digest.hexdigest()

        first = run_anchor()
        second = run_anchor()
        deterministic_repeat = first == second
        anchor_hash = first

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
        "deterministic_anchor_repeat": deterministic_repeat,
        "anchor_hash": anchor_hash,
        "first_flat_top1_divergence": (
            int(np.nonzero(~flat_top1)[0][0])
            if np.any(~flat_top1) else None
        ),
    }
    strict = {
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
    official = {
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
        "deterministic": deterministic_repeat,
        "finite": summary["all_finite"],
    }
    return {
        "summary": summary,
        "per_domain": per_domain,
        "strict_gates": strict,
        "strict_pass": all(strict.values()),
        "official_gates": official,
        "official_pass": all(official.values()),
        "trace_model_signature": metadata["model_signature"],
    }

def case_manifest(rt):
    rows = []
    for layer in rt.attn_layers:
        data = rt.layer[int(layer)]
        for family, key in (
            ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj"),
        ):
            if key in data:
                weight = data[key]
                rows.append({
                    "case": f"attention_{int(layer)}_{family}",
                    "layer": int(layer),
                    "family": family,
                    "key": key,
                    "pointer": int(weight.data.ptr),
                    "weight_bytes": int(weight.nbytes),
                    "rows": int(weight.shape[0]),
                    "cols": int(weight.shape[1]),
                })
    return rows
