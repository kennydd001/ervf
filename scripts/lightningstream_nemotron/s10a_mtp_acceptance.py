"""S10-A: measure the MTP acceptance rate `A`. No speculative loop is built.

Preregistered in S10A_MTP_ACCEPTANCE_PREREGISTRATION_2026-08-15.md.

A1 resolves the two wiring ambiguities (concat order into eh_proj, and which
backbone hidden feeds hnorm) by teacher-forced NLL on frozen held-out text.
A2 then counts, per decode step, how many of D=4 chained MTP drafts match the
tokens the backbone itself produces greedily, up to the first difference.

The backbone runs unmodified: this file only reads `rt.h` / `rt.normed` after
`step()`.  Nothing here is a throughput measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402
from moe_lab.lightningstream_nemotron.mtp import MTPBlock  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GIB = 1024 ** 3

D_DRAFT = 4
GATE_MEAN_A = 1.5           # G-S10-1
A1_NLL_CEILING = 7.0        # above this no variant is a plausible wiring
VARIANTS = [(o, s) for o in ("eh", "he") for s in ("pre", "post")]


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def gpu_state():
    try:
        o = subprocess.run(["nvidia-smi",
                            "--query-gpu=memory.used,memory.free,temperature.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        u, f, t = [x.strip() for x in o.stdout.strip().split(",")]
        return {"used_mib": int(u), "free_mib": int(f), "temp_c": int(t)}
    except Exception as e:                                   # pragma: no cover
        return {"error": str(e)}


def leading_match(drafts, truth) -> int:
    n = 0
    for a, b in zip(drafts, truth):
        if int(a) != int(b):
            break
        n += 1
    return n


# --------------------------------------------------------------------- A1
def run_a1(rt, mtp, cp, corpus, tokenizer, max_tokens: int):
    """Capture backbone hiddens once, then score all four wirings on them."""
    passages = []
    for p in corpus["a1_passages"]:
        ids = tokenizer.encode(p["text"], add_special_tokens=False)[:max_tokens]
        if len(ids) < 8:
            continue
        passages.append({"row": p["row"], "ids": [int(v) for v in ids]})

    captured, backbone_nll = [], []
    for p in passages:
        ids = p["ids"]
        rt.reset()
        pre, post = [], []
        for i, t in enumerate(ids):
            rt.step(t)
            pre.append(cp.asnumpy(rt.h).copy())
            post.append(cp.asnumpy(rt.normed).copy())
            if i + 1 < len(ids):
                lg = rt.logits
                m = cp.max(lg)
                lse = m + cp.log(cp.sum(cp.exp(lg - m)))
                backbone_nll.append(float(cp.asnumpy(lse - lg[ids[i + 1]])))
        captured.append({"ids": ids, "pre": pre, "post": post, "row": p["row"]})
        print(f"    a1 captured row {p['row']}: {len(ids)} positions", flush=True)

    results = {}
    hbuf = cp.zeros(rt.hidden, dtype=cp.float32)
    for order, hsrc in VARIANTS:
        mtp.concat_order = order
        nlls, hits, n = [], 0, 0
        for cap in captured:
            ids = cap["ids"]
            mtp.reset()
            for i in range(len(ids) - 2):
                hbuf.set(cap[hsrc][i])
                draft, _, _ = mtp.forward(ids[i + 1], hbuf, i)
                nlls.append(mtp.nll_of(ids[i + 2]))
                hits += int(draft == ids[i + 2])
                n += 1
        key = f"{order}_{hsrc}"
        results[key] = {
            "concat_order": order, "h_source": hsrc,
            "positions": n,
            "mean_nll": float(np.mean(nlls)),
            "median_nll": float(np.median(nlls)),
            "top1_agreement": hits / n,
        }
        print(f"    a1 {key:>9}: n={n} mean_nll={np.mean(nlls):.4f} "
              f"top1={hits / n:.4f}", flush=True)

    winner = min(results, key=lambda k: results[k]["mean_nll"])
    return {
        "positions_per_variant": results[winner]["positions"],
        "a1_rows": [c["row"] for c in captured],
        "max_tokens_per_passage": max_tokens,
        "backbone_next_token_nll_anchor": {
            "n": len(backbone_nll), "mean": float(np.mean(backbone_nll))},
        "variants": results,
        "winner": winner,
        "winner_mean_nll": results[winner]["mean_nll"],
        "nll_ceiling": A1_NLL_CEILING,
        "resolved": results[winner]["mean_nll"] <= A1_NLL_CEILING,
        "decision_rule": (
            "argmin mean NLL over the four wirings on frozen held-out text; "
            "declared unresolved if the winner exceeds the preregistered "
            "ceiling of 7.0 nats. Choosing the argmin is deliberately the "
            "most MTP-favourable option, which makes a later gate FAILURE "
            "conservative and a gate PASS non-final."),
    }


# --------------------------------------------------------------------- A2
def run_one_prompt(rt, mtp, cp, prompt_ids, steps, order, hsrc, chain_from,
                   label, timing: bool):
    mtp.concat_order = order
    rt.reset()
    mtp.reset()
    S = [int(v) for v in prompt_ids]
    m = len(S)
    first_measured = m - 1
    last_i = first_measured + steps - 1
    need = last_i + 1 + D_DRAFT            # S index of the last truth token
    drafts_by_step, mtp_ms = {}, []

    i = 0
    while len(S) <= need:
        nxt = rt.step(S[i])
        if i + 1 == len(S):
            S.append(int(nxt))
        h_dev = rt.h if hsrc == "pre" else rt.normed
        measured = i >= first_measured and i <= last_i
        n_chain = D_DRAFT if measured else 1
        t0 = time.perf_counter_ns()
        tok_in, h_in = S[i + 1], h_dev
        chain = []
        for j in range(n_chain):
            draft, x, y = mtp.forward(tok_in, h_in, i + j)
            chain.append(int(draft))
            tok_in = draft
            h_in = x if chain_from == "x" else y
        if measured:
            cp.cuda.Device(0).synchronize()
            mtp_ms.append((time.perf_counter_ns() - t0) / 1e6)
            drafts_by_step[i] = chain
        i += 1
        if i % 500 == 0:
            print(f"      {label}: pos {i}/{need}", flush=True)

    per_step = []
    for i in sorted(drafts_by_step):
        truth = [S[i + 2 + j] for j in range(D_DRAFT)]
        per_step.append({"i": i, "drafts": drafts_by_step[i], "truth": truth,
                         "A": leading_match(drafts_by_step[i], truth)})
    A = [r["A"] for r in per_step]
    return {
        "label": label,
        "prompt_tokens": m,
        "first_measured_index": first_measured,
        "measured_steps": len(per_step),
        "sequence": S,
        "per_step": per_step,
        "mean_A": float(np.mean(A)),
        "histogram_A": {str(v): int(sum(1 for a in A if a == v)) for v in range(D_DRAFT + 1)},
        "mtp_chain_ms": ({"n": len(mtp_ms), "p50": float(np.percentile(mtp_ms, 50)),
                          "mean": float(np.mean(mtp_ms))} if timing and mtp_ms else None),
    }


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=32)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--a1-tokens", type=int, default=160)
    ap.add_argument("--long-ctx-tokens", type=int, default=4096)
    ap.add_argument("--long-ctx-steps", type=int, default=60)
    args = ap.parse_args()

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    foreign = [l for l in o.stdout.strip().splitlines()
               if l.strip() and int(l.split(",")[0]) != os.getpid()]
    if foreign:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    corpus_path = OUT_DIR / "s10a_corpus.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

    free0, total = cp.cuda.runtime.memGetInfo()
    rt = LightningRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                          embed_on_host=True, fp8_kv=True)
    free_shell, _ = cp.cuda.runtime.memGetInfo()
    cache_bytes = rt.enable_cache(args.capacity)
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    # Backbone sanity: the same check N6-A/N7-B froze. If this drifts, nothing
    # downstream means anything.
    ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    rt.reset()
    nxt = None
    for t in ids:
        nxt = rt.step(t)
    cp.cuda.Device(0).synchronize()
    first = tokenizer.decode([nxt])
    coherent = "paris" in first.strip().lower()
    print(f"backbone sanity: -> {first!r} coherent={coherent}", flush=True)
    if not coherent:
        print("STOP: backbone incoherent, refusing to measure MTP against it.")
        return 3

    mtp = MTPBlock(rt, max_ctx=args.max_ctx, concat_order="eh")
    free_all, _ = cp.cuda.runtime.memGetInfo()
    print(f"shell {(free0 - free_shell) / GIB:.3f} GiB | cache {cache_bytes / GIB:.3f} GiB | "
          f"mtp {(mtp.exp_up.nbytes + mtp.exp_dn.nbytes) / GIB:.3f} GiB experts | "
          f"free {free_all / GIB:.3f} GiB", flush=True)

    # ------------------------------------------------------------------ A1
    print("\nA1 wiring resolution", flush=True)
    a1 = run_a1(rt, mtp, cp, corpus, tokenizer, args.a1_tokens)
    order, hsrc = a1["variants"][a1["winner"]]["concat_order"], \
        a1["variants"][a1["winner"]]["h_source"]
    chain_from = "x" if hsrc == "pre" else "y"
    print(f"  winner: {a1['winner']} mean_nll={a1['winner_mean_nll']:.4f} "
          f"resolved={a1['resolved']}", flush=True)

    (OUT_DIR / "s10a_wiring_resolution.json").write_text(
        json.dumps({"kind": "lightningstream_nemotron_s10a_wiring_resolution",
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "model_dir": MODEL_DIR.name,
                    "corpus_sha256": sha256_path(corpus_path),
                    **a1}, indent=2) + "\n", encoding="utf-8")

    if not a1["resolved"]:
        print("STOP: no plausible wiring; G-S10-1 is NOT evaluable and S10 is "
              "NOT closed on this evidence.")
        return 5

    # ------------------------------------------------------------------ A2
    print("\nA2 acceptance", flush=True)
    gate_rows = []
    for p in corpus["gate_prompts"]:
        pid = tokenizer.encode(p["text"], add_special_tokens=False)
        row = run_one_prompt(rt, mtp, cp, pid, args.steps, order, hsrc,
                             chain_from, p["id"], timing=True)
        gate_rows.append(row)
        print(f"  {p['id']:<11} steps={row['measured_steps']} "
              f"mean_A={row['mean_A']:.4f} hist={row['histogram_A']}", flush=True)

    pooled = [r["A"] for row in gate_rows for r in row["per_step"]]
    mean_A = float(np.mean(pooled))
    hist = {str(v): int(sum(1 for a in pooled if a == v)) for v in range(D_DRAFT + 1)}
    cond = []
    for j in range(D_DRAFT):
        denom = sum(1 for a in pooled if a >= j)
        cond.append((sum(1 for a in pooled if a >= j + 1) / denom) if denom else None)

    print(f"\n  pooled steps={len(pooled)} mean_A={mean_A:.4f} "
          f"gate {GATE_MEAN_A} -> {'PASS' if mean_A >= GATE_MEAN_A else 'FAIL'}", flush=True)

    # secondary, non-gating: does A move with context depth?
    long_ids = tokenizer.encode(corpus["long_ctx_text"],
                                add_special_tokens=False)[:args.long_ctx_tokens]
    print(f"\n  long-context arm: {len(long_ids)} prompt tokens", flush=True)
    long_row = run_one_prompt(rt, mtp, cp, long_ids, args.long_ctx_steps, order,
                              hsrc, chain_from, "long_ctx", timing=True)
    print(f"  long_ctx    steps={long_row['measured_steps']} "
          f"mean_A={long_row['mean_A']:.4f} hist={long_row['histogram_A']}", flush=True)

    free_end, _ = cp.cuda.runtime.memGetInfo()
    result = {
        "kind": "lightningstream_nemotron_s10a_mtp_acceptance",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "S10_A_MTP_ACCEPTANCE",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "mtp_module_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/mtp.py"),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "corpus_sha256": sha256_path(corpus_path),
        "config": {
            "D": D_DRAFT, "capacity_per_layer": args.capacity,
            "max_ctx": args.max_ctx, "steps_per_prompt": args.steps,
            "embed_on_host": True, "backbone_fp8_kv": True, "mtp_kv_dtype": "float32",
            "mtp_experts_resident": True, "decode": "greedy_argmax",
            "concat_order": order, "h_source": hsrc, "chain_from": chain_from,
        },
        "memory": {
            "shell_bytes": int(free0 - free_shell), "cache_bytes": int(cache_bytes),
            "mtp_expert_bytes": int(mtp.exp_up.nbytes + mtp.exp_dn.nbytes),
            "device_free_end_bytes": int(free_end), "device_total_bytes": int(total),
        },
        "backbone_sanity": {"top1": first, "coherent": coherent},
        "a1_wiring": a1,
        "gate_prompts": gate_rows,
        "pooled": {
            "measured_steps": len(pooled), "mean_A": mean_A, "histogram_A": hist,
            "conditional_accept": cond,
            "tokens_per_sweep_if_built": mean_A + 1.0,
        },
        "gate_G_S10_1": {
            "required_mean_A": GATE_MEAN_A,
            "required_min_steps": 200,
            "required_prompts": 3,
            "measured_mean_A": mean_A,
            "measured_steps": len(pooled),
            "measured_prompts": len(gate_rows),
            "passed": bool(mean_A >= GATE_MEAN_A and len(pooled) >= 200
                           and len(gate_rows) >= 3),
        },
        "secondary_long_context": long_row,
        "gpu": gpu_state(),
        "claim_boundary": (
            "Measured acceptance count A of D=4 chained MTP drafts against the "
            "greedy tokens this very runtime produces, on this GPU, at short to "
            "medium context. NOT a throughput result: no speculative loop was "
            "built and the MTP chain timing recorded here is a component "
            "measurement with all 128 MTP experts device-resident, which a built "
            "system cannot afford (0.000 GiB free at the N7-B configuration). "
            "The wiring was resolved empirically in A1, not from a reference "
            "implementation; the most MTP-favourable of four candidates was "
            "chosen, so a gate failure is conservative and a gate pass is not "
            "final. A at 262K context is NOT measured; the long-context arm is "
            "a single 4K-token observation and gates nothing."),
    }
    (OUT_DIR / "s10a_mtp_acceptance.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nwritten s10a_mtp_acceptance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
