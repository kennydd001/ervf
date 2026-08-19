from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import traceback

import numpy as np

from common import REPO, require_model_dir, sha256_file, write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_lightning15_common import (
    PROMPTS, TOP_K, ensure_results, feed_prompt, identity,
    logsumexp, model_signature, normalize_eager_moe,
    prompt_rows, reset_eager, trace_paths,
)

def write_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = handle.name
    os.replace(temporary, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("calibration", "validation", "heldout"),
        required=True,
    )
    args = parser.parse_args()
    ensure_results()
    npz_path, metadata_path = trace_paths(args.split)
    payload = {
        "kind": "s100_lightning15_parent_trace",
        "status": "started",
        "split": args.split,
        "started_utc": utc_now(),
    }

    try:
        import cupy as cp
        from transformers import AutoTokenizer

        ident = identity()
        prompts, length = prompt_rows(args.split)
        tokenizer = AutoTokenizer.from_pretrained(
            str(require_model_dir()),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
        bundle = build()
        rt = bundle.rt
        rt._graph = None
        rt.graph_mode = False
        normalize_eager_moe(rt)

        count = len(prompts)
        target_ids = np.empty((count, length), np.int32)
        target_logprob = np.empty((count, length), np.float32)
        top_ids = np.empty((count, length, TOP_K), np.int32)
        top_logprob = np.empty((count, length, TOP_K), np.float32)
        rest_prob = np.empty((count, length), np.float32)
        prompt_records = []

        for pi, prompt in enumerate(prompts):
            ids = tokenizer.encode(
                prompt["prompt"], add_special_tokens=False
            )
            prompt_records.append({
                "id": prompt["id"],
                "domain": prompt["domain"],
                "prompt_ids_sha256": hashlib.sha256(
                    np.asarray(ids, dtype="<i4").tobytes()
                ).hexdigest(),
                "prompt_token_count": len(ids),
            })
            reset_eager(rt)
            feed_prompt(rt, ids)

            for ti in range(length):
                logits = rt.logits
                if not bool(cp.isfinite(logits).all().item()):
                    raise RuntimeError(
                        f"non-finite parent logits {prompt['id']}:{ti}"
                    )
                lse = logsumexp(cp, logits)
                order = cp.argpartition(logits, -TOP_K)[-TOP_K:]
                order = order[cp.argsort(-logits[order])]
                ids_top = cp.asnumpy(order).astype(np.int32)
                values = cp.asnumpy(logits[order]).astype(np.float64)
                logp = values - lse
                target = int(ids_top[0])
                target_ids[pi, ti] = target
                target_logprob[pi, ti] = np.float32(logp[0])
                top_ids[pi, ti] = ids_top
                top_logprob[pi, ti] = logp.astype(np.float32)
                rest_prob[pi, ti] = np.float32(
                    max(1.0 - float(np.exp(logp).sum()), 1e-30)
                )
                if ti + 1 < length:
                    rt.step(target)

            print(
                f"Lightning trace {args.split} "
                f"{pi+1:02d}/{count}: {prompt['id']}",
                flush=True,
            )

        arrays = {
            "target_ids": target_ids,
            "target_logprob": target_logprob,
            "top_ids": top_ids,
            "top_logprob": top_logprob,
            "rest_prob": rest_prob,
        }
        write_npz(npz_path, arrays)
        payload.update({
            "status": "trace_ready",
            "identity": ident,
            "model_signature": model_signature(ident),
            "prompt_manifest_sha256": sha256_file(PROMPTS),
            "trace_path": str(npz_path.relative_to(REPO)),
            "trace_sha256": sha256_file(npz_path),
            "prompt_records": prompt_records,
            "array_shapes": {
                key: list(value.shape) for key, value in arrays.items()
            },
            "parent": "Lightning QFAST + alpha=0.0003 eager",
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

    write_json_atomic(metadata_path, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "split": args.split,
        "model_signature": payload.get("model_signature"),
        "trace_sha256": payload.get("trace_sha256"),
        "error": (payload.get("error") or {}).get("message"),
    }, indent=2))
    return 0 if payload.get("status") == "trace_ready" else 2

if __name__ == "__main__":
    raise SystemExit(main())
