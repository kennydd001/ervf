from __future__ import annotations

import argparse
import json
import traceback
import numpy as np

from common import REPO, write_json_atomic, utc_now
from s100_phase15_native import (
    make_runtime, release_runtime, prompt_rows,
    snapshot_recurrent_state, restore_recurrent_state
)

OUTDIR = REPO / "pro_research" / "results" / "s100_phase15"
HORIZONS = (1, 2, 4, 8)

def run_variant(variant, blocks_per_prompt):
    import cupy as cp

    rows = prompt_rows("validation")
    rt, dispatch = make_runtime(None)
    exact_mv = rt.k.mv_bf16
    native = None
    records = []

    try:
        # Create native dispatcher against the SAME runtime so the recurrent
        # state can be restored exactly without cross-runtime serialization.
        from s100_phase15_native import NativeBF16VariantDispatch
        native = NativeBF16VariantDispatch(rt, variant, "all")

        for H in HORIZONS:
            for p in rows:
                rt.k.mv_bf16 = exact_mv
                rt.reset()
                for token in p["prompt_ids"]:
                    rt.step(int(token))
                input_token = int(cp.argmax(rt.logits).item())

                for bi in range(blocks_per_prompt):
                    pre = snapshot_recurrent_state(rt)

                    # Exact block.
                    rt.k.mv_bf16 = exact_mv
                    exact_tokens = []
                    tok = input_token
                    for _ in range(H):
                        nxt = int(rt.step(int(tok)))
                        exact_tokens.append(nxt)
                        tok = nxt
                    exact_next = tok
                    post = snapshot_recurrent_state(rt)

                    # Native block from exact pre-state and same first token.
                    restore_recurrent_state(rt, pre)
                    del pre
                    rt.k.mv_bf16 = native
                    cand_tokens = []
                    tok = input_token
                    for _ in range(H):
                        nxt = int(rt.step(int(tok)))
                        cand_tokens.append(nxt)
                        tok = nxt

                    first = H
                    for j, (a, b) in enumerate(zip(exact_tokens, cand_tokens)):
                        if a != b:
                            first = j
                            break

                    records.append({
                        "variant": variant,
                        "horizon": H,
                        "prompt_id": p["id"],
                        "domain": p["domain"],
                        "block": bi,
                        "accepted_exact_prefix": first,
                        "full_block_match": first == H,
                        "first_prediction_match": (
                            exact_tokens[0] == cand_tokens[0]
                        ),
                        "position_agreement": float(np.mean(
                            np.asarray(exact_tokens, np.int32)
                            == np.asarray(cand_tokens, np.int32)
                        )),
                        "exact_tokens": exact_tokens,
                        "candidate_tokens": cand_tokens,
                    })

                    # Candidate state is discarded; exact state is canonical.
                    rt.k.mv_bf16 = exact_mv
                    restore_recurrent_state(rt, post)
                    del post
                    input_token = exact_next

                print(
                    f"P15 horizon {variant} H={H} {p['id']}",
                    flush=True,
                )

        cp.cuda.get_current_stream().synchronize()
    finally:
        rt.k.mv_bf16 = exact_mv
        release_runtime(rt)

    summary = {}
    for H in HORIZONS:
        x = [r for r in records if r["horizon"] == H]
        accepted = np.asarray(
            [r["accepted_exact_prefix"] for r in x], np.float64
        )
        summary[str(H)] = {
            "blocks": len(x),
            "first_prediction_agreement": float(np.mean([
                r["first_prediction_match"] for r in x
            ])),
            "mean_accepted_exact_prefix": float(accepted.mean()),
            "p50_accepted_exact_prefix": float(np.percentile(accepted, 50)),
            "p10_accepted_exact_prefix": float(np.percentile(accepted, 10)),
            "full_block_match_rate": float(np.mean([
                r["full_block_match"] for r in x
            ])),
            "mean_position_agreement": float(np.mean([
                r["position_agreement"] for r in x
            ])),
        }
    h4 = summary["4"]
    gate = bool(
        h4["first_prediction_agreement"] >= 0.95
        and h4["mean_accepted_exact_prefix"] >= 1.5
        and h4["full_block_match_rate"] >= 0.25
    )
    return summary, gate, records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=("mm_fp32out", "mm_fp32out_comp2"),
        required=True,
    )
    ap.add_argument("--blocks-per-prompt", type=int, default=4)
    args = ap.parse_args()

    out = OUTDIR / (
        f"S100_PHASE15D_HORIZON_{args.variant.upper()}.json"
    )
    payload = {
        "kind": "s100_phase15d_exact_state_horizon",
        "status": "started",
        "variant": args.variant,
        "started_utc": utc_now(),
        "claim_boundary": (
            "native greedy draft from exact recurrent state at every block "
            "boundary; no throughput claim"
        ),
    }
    try:
        summary, gate, records = run_variant(
            args.variant, args.blocks_per_prompt
        )
        payload.update({
            "status": "measured",
            "summary": summary,
            "H4_BLOCK_RESEARCH_GO": gate,
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
        "H4_BLOCK_RESEARCH_GO": payload.get("H4_BLOCK_RESEARCH_GO"),
        "summary": payload.get("summary"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
