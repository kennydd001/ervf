from __future__ import annotations

import hashlib
import json
import traceback
import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now
from diag_component_marginals_graph import _reset_exact_state
from s100_phase10a_runtime import build
from s100_phase14d2_native import NativeBF16Dispatch
from s100_phase5_quality import load_trace

OUT = (
    REPO / "pro_research" / "results" / "s100_phase14k2"
    / "S100_PHASE14K2_REAL_WITNESS.json"
)
KS = (8, 16, 32, 64)

def reset_eager(rt):
    _reset_exact_state(rt)
    rt._graph = None
    rt.graph_mode = False
    rt.reset()

def feed(rt, ids):
    for token in ids:
        rt.step(int(token))

def collect_candidate(split):
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
    tok = AutoTokenizer.from_pretrained(
        str(require_model_dir()), local_files_only=True,
        trust_remote_code=True, use_fast=True
    )
    rows = []
    for pi, (p, original_index) in enumerate(zip(prompts, indices)):
        ids = tok.encode(p["prompt"], add_special_tokens=False)
        got = hashlib.sha256(
            np.asarray(ids, dtype="<i4").tobytes()
        ).hexdigest()
        if got != meta["prompt_records"][original_index]["prompt_ids_sha256"]:
            raise RuntimeError(f"tokenizer drift {p['id']}")
        reset_eager(rt)
        feed(rt, ids)
        for ti in range(n):
            logits = rt.logits
            top = cp.argpartition(logits, -64)[-64:]
            top = top[cp.argsort(-logits[top])]
            ids64 = cp.asnumpy(top).astype(np.int32)
            vals2 = cp.asnumpy(logits[top[:2]]).astype(np.float64)
            std = float(cp.std(logits).item())
            rows.append({
                "prompt_index": pi,
                "target_index": ti,
                "prompt_id": p["id"],
                "domain": p["domain"],
                "candidate_top1": int(ids64[0]),
                "candidate_margin_norm": float(
                    (vals2[0] - vals2[1]) / max(std, 1e-12)
                ),
                "candidate_top64": ids64.tolist(),
            })
            if ti + 1 < n:
                rt.step(int(targets[pi, ti]))
        print(f"K2 candidate {split} {pi+1}/{len(prompts)} {p['id']}", flush=True)

    bundle.restore_combined()
    bundle.restore_sel()
    del bundle, rt
    cp.get_default_memory_pool().free_all_blocks()
    return rows, targets

def score_exact(split, candidate_rows, targets):
    import cupy as cp
    from transformers import AutoTokenizer
    bundle = build()
    rt = bundle.rt
    rt._graph = None
    rt.graph_mode = False

    prompts, indices, n, d, meta = load_trace(split)
    tok = AutoTokenizer.from_pretrained(
        str(require_model_dir()), local_files_only=True,
        trust_remote_code=True, use_fast=True
    )
    lookup = {
        (r["prompt_index"], r["target_index"]): r
        for r in candidate_rows
    }

    for pi, (p, original_index) in enumerate(zip(prompts, indices)):
        ids = tok.encode(p["prompt"], add_special_tokens=False)
        reset_eager(rt)
        feed(rt, ids)
        for ti in range(n):
            row = lookup[(pi, ti)]
            exact_top1 = int(cp.argmax(rt.logits).item())
            shortlist = np.asarray(row["candidate_top64"], np.int32)
            exact_scores = cp.asnumpy(
                rt.logits[cp.asarray(shortlist)]
            ).astype(np.float32)
            row["exact_top1"] = exact_top1
            row["candidate_top1_matches_exact"] = (
                row["candidate_top1"] == exact_top1
            )
            for k in KS:
                sub = shortlist[:k]
                row[f"K{k}_includes_exact"] = bool(
                    np.any(sub == exact_top1)
                )
                witness = int(sub[int(np.argmax(exact_scores[:k]))])
                row[f"K{k}_exact_witness_matches"] = (
                    witness == exact_top1
                )
            if ti + 1 < n:
                rt.step(int(targets[pi, ti]))
        print(f"K2 exact {split} {pi+1}/{len(prompts)} {p['id']}", flush=True)

    bundle.restore_combined()
    bundle.restore_sel()
    del bundle, rt
    cp.get_default_memory_pool().free_all_blocks()
    return candidate_rows

def aggregate(rows, threshold=None):
    out = {
        "tokens": len(rows),
        "candidate_top1_agreement": float(np.mean([
            r["candidate_top1_matches_exact"] for r in rows
        ])),
    }
    for k in KS:
        out[f"K{k}_inclusion"] = float(np.mean([
            r[f"K{k}_includes_exact"] for r in rows
        ]))
        out[f"K{k}_witness"] = float(np.mean([
            r[f"K{k}_exact_witness_matches"] for r in rows
        ]))
    if threshold is not None:
        mask = np.asarray([
            r["candidate_margin_norm"] >= threshold for r in rows
        ], bool)
        out["margin_threshold"] = float(threshold)
        out["fast_fraction"] = float(mask.mean())
        if mask.any():
            selected = [r for r, m in zip(rows, mask) if m]
            out["fast_candidate_top1_agreement"] = float(np.mean([
                r["candidate_top1_matches_exact"] for r in selected
            ]))
            out["fast_K16_inclusion"] = float(np.mean([
                r["K16_includes_exact"] for r in selected
            ]))
        else:
            out["fast_candidate_top1_agreement"] = None
            out["fast_K16_inclusion"] = None
    return out

def main():
    payload = {
        "kind": "s100_phase14k2_real_witness",
        "status": "started",
        "started_utc": utc_now(),
    }
    try:
        cal_c, cal_targets = collect_candidate("calibration")
        cal = score_exact("calibration", cal_c, cal_targets)

        margins = np.asarray([
            r["candidate_margin_norm"] for r in cal
        ], np.float64)
        choices = []
        for q in (0.50, 0.75, 0.90):
            threshold = float(np.quantile(margins, q))
            ag = aggregate(cal, threshold)
            ag["quantile"] = q
            ag["calibration_gate"] = bool(
                ag["fast_fraction"] >= 0.10
                and ag["fast_candidate_top1_agreement"] is not None
                and ag["fast_candidate_top1_agreement"] >= 0.999
                and ag["fast_K16_inclusion"] == 1.0
            )
            choices.append(ag)
        green = [x for x in choices if x["calibration_gate"]]
        selected = max(green, key=lambda x: x["fast_fraction"]) if green else None

        val_c, val_targets = collect_candidate("validation")
        val = score_exact("validation", val_c, val_targets)
        validation_all = aggregate(val)
        validation_fast = (
            aggregate(val, selected["margin_threshold"])
            if selected else None
        )

        k16_green = validation_all["K16_inclusion"] == 1.0
        margin_green = bool(
            validation_fast
            and validation_fast["fast_fraction"] >= 0.10
            and validation_fast["fast_candidate_top1_agreement"] is not None
            and validation_fast["fast_candidate_top1_agreement"] >= 0.999
            and validation_fast["fast_K16_inclusion"] == 1.0
        )

        payload.update({
            "status": "measured",
            "candidate": (
                "same checkpoint + native BF16 substitutions from 14D2"
            ),
            "calibration_thresholds": choices,
            "selected_margin_gate": selected,
            "validation_all": validation_all,
            "validation_fast": validation_fast,
            "REAL_K16_SHORTLIST_GREEN": bool(k16_green),
            "REAL_MARGIN_GATE_GREEN": margin_green,
            "WITNESS_GUARD_RESEARCH_OPEN": bool(k16_green),
            "CHEAP_UPSTREAM_CERTIFICATION_AVAILABLE": False,
            "cheap_certification_reason": (
                "native BF16 changes upstream hidden/state; exact shortlist "
                "rerank cannot certify omitted upstream computation"
            ),
            "heldout_read": False,
            "s100_single_achieved": False,
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
    text = (
        "S100 PHASE 14K2 — REAL WITNESS\n"
        f"Status: {payload.get('status')}\n"
        f"REAL_K16_SHORTLIST_GREEN: {payload.get('REAL_K16_SHORTLIST_GREEN')}\n"
        f"REAL_MARGIN_GATE_GREEN: {payload.get('REAL_MARGIN_GATE_GREEN')}\n"
        f"WITNESS_GUARD_RESEARCH_OPEN: {payload.get('WITNESS_GUARD_RESEARCH_OPEN')}\n"
        "CHEAP_UPSTREAM_CERTIFICATION_AVAILABLE: False\n"
        "S100 SINGLE ACHIEVED: False\n"
    )
    (OUT.parent / "S100_PHASE14K2_SUMMARY.txt").write_text(
        text, encoding="utf-8"
    )
    print(text)
    if payload.get("error"):
        print(json.dumps(payload["error"], indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
