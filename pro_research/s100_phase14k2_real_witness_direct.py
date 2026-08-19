from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
KS = (8, 16, 32, 64)
RATES = (0.50, 0.75, 0.90)


def prompt_sets():
    from transformers import AutoTokenizer

    rows = json.loads((REPO / "pro_research" / "S100_PHASE3_PROMPTS.json").read_text(encoding="utf-8"))["prompts"]
    model = Path(os.environ["LS_MODEL_DIR"])
    tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=True, use_fast=True)
    out = {"calibration": [], "validation": []}
    for row in rows:
        ids = tok.encode(row["prompt"], add_special_tokens=False)
        if row["id"].endswith("_01"):
            out["calibration"].append({"id": row["id"], "domain": row["domain"], "prompt_ids": [int(x) for x in ids]})
        elif row["id"].endswith("_02"):
            out["validation"].append({"id": row["id"], "domain": row["domain"], "prompt_ids": [int(x) for x in ids]})
    return out


def make_runtime(native: bool):
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(Path(os.environ["LS_MODEL_DIR"]), contexts_max=4096, embed_on_host=True, fp8_kv=True, verbose=False)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    if native:
        import cupy as cp

        original = rt.k.mv_bf16
        weights = {}

        def dispatch(out, W, x, rows, cols):
            key = (int(W.data.ptr), int(rows), int(cols))
            wt = weights.get(key)
            if wt is None:
                wt = torch.utils.dlpack.from_dlpack(W).view(torch.bfloat16).reshape(int(rows), int(cols)).clone()
                weights[key] = wt
            stream = torch.cuda.ExternalStream(cp.cuda.get_current_stream().ptr)
            with torch.cuda.stream(stream):
                xt = torch.utils.dlpack.from_dlpack(x).to(torch.bfloat16)
                yt = torch.mv(wt, xt).float()
                torch.utils.dlpack.from_dlpack(out).copy_(yt)

        rt.k.mv_bf16 = dispatch
        rt._phase14_original_bf16 = original
    return rt


def release_runtime(rt):
    import cupy as cp

    try:
        rt.bank = {}
        rt.cache = {}
        rt._dev_cache = {}
    except Exception:
        pass
    del rt
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()


def exact_reference(rows, tokens):
    import cupy as cp

    rt = make_runtime(native=False)
    result = {}
    try:
        for prompt in rows:
            rt.reset()
            nxt = None
            for token in prompt["prompt_ids"]:
                nxt = int(rt.step(int(token)))
            logits = []
            targets = []
            for step in range(tokens):
                logits.append(cp.asnumpy(rt.logits).astype(np.float32, copy=True))
                targets.append(int(nxt))
                if step + 1 < tokens:
                    nxt = int(rt.step(int(nxt)))
            result[prompt["id"]] = {"domain": prompt["domain"], "logits": np.stack(logits), "targets": targets}
            print(f"exact parent {prompt['id']}", flush=True)
    finally:
        release_runtime(rt)
    return result


def candidate_against_exact(rows, exact, tokens):
    import cupy as cp

    rt = make_runtime(native=True)
    out = {}
    try:
        for prompt in rows:
            ref = exact[prompt["id"]]
            rt.reset()
            nxt = None
            for token in prompt["prompt_ids"]:
                nxt = int(rt.step(int(token)))
            records = []
            for step in range(tokens):
                candidate_logits = cp.asnumpy(rt.logits).astype(np.float32, copy=True)
                exact_logits = ref["logits"][step]
                exact_top1 = int(np.argmax(exact_logits))
                top64 = np.argpartition(candidate_logits, -64)[-64:]
                top64 = top64[np.argsort(-candidate_logits[top64])]
                exact_short_scores = exact_logits[top64]
                row = {
                    "prompt_id": prompt["id"],
                    "domain": prompt["domain"],
                    "step": step,
                    "candidate_top1": int(top64[0]),
                    "candidate_margin_norm": float((candidate_logits[top64[0]] - candidate_logits[top64[1]]) / max(float(np.std(candidate_logits)), 1e-12)),
                    "exact_top1": exact_top1,
                    "candidate_top1_matches_exact": int(top64[0]) == exact_top1,
                }
                for k in KS:
                    shortlist = top64[:k]
                    row[f"K{k}_includes_exact"] = bool(np.any(shortlist == exact_top1))
                    row[f"K{k}_exact_witness_matches"] = int(shortlist[int(np.argmax(exact_short_scores[:k]))]) == exact_top1
                records.append(row)
                if step + 1 < tokens:
                    nxt = int(rt.step(int(nxt)))
            out[prompt["id"]] = records
            print(f"native candidate {prompt['id']}", flush=True)
    finally:
        release_runtime(rt)
    return out


def aggregate(records, threshold=None):
    rows = [r for group in records.values() for r in group]
    result = {
        "tokens": len(rows),
        "candidate_top1_agreement": float(np.mean([r["candidate_top1_matches_exact"] for r in rows])),
    }
    for k in KS:
        result[f"K{k}_inclusion"] = float(np.mean([r[f"K{k}_includes_exact"] for r in rows]))
        result[f"K{k}_witness"] = float(np.mean([r[f"K{k}_exact_witness_matches"] for r in rows]))
    if threshold is not None:
        mask = np.asarray([r["candidate_margin_norm"] >= threshold for r in rows], dtype=bool)
        result.update({"margin_threshold": float(threshold), "fast_fraction": float(mask.mean())})
        selected = [r for r, keep in zip(rows, mask) if keep]
        if selected:
            result["fast_candidate_top1_agreement"] = float(np.mean([r["candidate_top1_matches_exact"] for r in selected]))
            result["fast_K16_inclusion"] = float(np.mean([r["K16_includes_exact"] for r in selected]))
        else:
            result["fast_candidate_top1_agreement"] = None
            result["fast_K16_inclusion"] = None
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-per-prompt", type=int, default=16)
    ap.add_argument("--output", type=Path, default=Path("pro_research/results/s100_phase14k2/S100_PHASE14K2_REAL_WITNESS.json"))
    args = ap.parse_args()
    prompts = prompt_sets()
    payload = {"kind": "s100_phase14k2_real_witness", "status": "started", "created_utc": datetime.now(timezone.utc).isoformat(), "model_dir": os.environ["LS_MODEL_DIR"], "method": "direct teacher-forced exact parent versus native BF16 candidate on same prompt prefixes", "tokens_per_prompt": args.tokens_per_prompt}
    try:
        calibration_exact = exact_reference(prompts["calibration"], args.tokens_per_prompt)
        calibration_candidate = candidate_against_exact(prompts["calibration"], calibration_exact, args.tokens_per_prompt)
        cal_rows = [r for group in calibration_candidate.values() for r in group]
        margins = np.asarray([r["candidate_margin_norm"] for r in cal_rows], dtype=np.float64)
        choices = []
        for q in RATES:
            threshold = float(np.quantile(margins, q))
            ag = aggregate(calibration_candidate, threshold)
            ag["quantile"] = q
            ag["calibration_gate"] = bool(ag["fast_fraction"] >= 0.10 and ag["fast_candidate_top1_agreement"] is not None and ag["fast_candidate_top1_agreement"] >= 0.999 and ag["fast_K16_inclusion"] == 1.0)
            choices.append(ag)
        green = [x for x in choices if x["calibration_gate"]]
        selected = max(green, key=lambda x: x["fast_fraction"]) if green else None
        validation_exact = exact_reference(prompts["validation"], args.tokens_per_prompt)
        validation_candidate = candidate_against_exact(prompts["validation"], validation_exact, args.tokens_per_prompt)
        validation_all = aggregate(validation_candidate)
        validation_fast = aggregate(validation_candidate, selected["margin_threshold"]) if selected else None
        k16_green = validation_all["K16_inclusion"] == 1.0
        margin_green = bool(validation_fast and validation_fast["fast_fraction"] >= 0.10 and validation_fast["fast_candidate_top1_agreement"] is not None and validation_fast["fast_candidate_top1_agreement"] >= 0.999 and validation_fast["fast_K16_inclusion"] == 1.0)
        payload.update({"status": "measured", "calibration_thresholds": choices, "selected_margin_gate": selected, "validation_all": validation_all, "validation_fast": validation_fast, "REAL_K16_SHORTLIST_GREEN": bool(k16_green), "REAL_MARGIN_GATE_GREEN": margin_green, "WITNESS_GUARD_RESEARCH_OPEN": bool(k16_green), "CHEAP_UPSTREAM_CERTIFICATION_AVAILABLE": False, "s100_single_achieved": False, "claim_boundary": "real native-BF16 candidate logits versus exact parent on same teacher-forced prefixes; no heldout quality gate"})
    except Exception as exc:
        import traceback
        payload.update({"status": "technical_failure", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = "S100 PHASE 14K2 — REAL WITNESS\n" + f"Status: {payload.get('status')}\nREAL_K16_SHORTLIST_GREEN: {payload.get('REAL_K16_SHORTLIST_GREEN')}\nREAL_MARGIN_GATE_GREEN: {payload.get('REAL_MARGIN_GATE_GREEN')}\nWITNESS_GUARD_RESEARCH_OPEN: {payload.get('WITNESS_GUARD_RESEARCH_OPEN')}\n"
    (args.output.parent / "S100_PHASE14K2_SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text)
    if payload.get("error"): print(json.dumps(payload["error"], indent=2))
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
