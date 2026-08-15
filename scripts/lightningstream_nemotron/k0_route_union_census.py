"""K0/K1/K2: route-union census, draft-chain decomposition, active vocabulary.

Preregistered in K0_ROUTE_UNION_AND_VOCAB_PREREGISTRATION_2026-08-15.md, which
adopts Kimi's correction that raw-load parity is U* = top_k x (A+1) = 18.684 and
that a five-token union of 12 is a 36% improvement, not a failure.

Three phases against one model load:

K0  official routes captured with step(capture_routes=...) during real greedy
    generation, then per-layer unions over windows of B consecutive tokens and
    an LRU replay at six capacities in both AR and round order.
K2  a MicroSpec-style context-local draft vocabulary taken from the top-N rows
    of the backbone's own logits; acceptance, recall and chain time.
K1  where the draft chain spends its time, measured in the loop with bracketed
    baselines, never in isolation (S8).

No new kernel, no speculative loop, no change to runtime.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
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
CONCAT_ORDER, H_SOURCE, CHAIN_FROM = "eh", "post", "y"      # frozen by S10-A1
B_VALUES = [2, 3, 5, 7, 9, 13]
CAPACITIES = [32, 48, 56, 60, 64, 72]
VOCAB_N = [None, 4096, 2048, 1024]
S10A_MEAN_A = 2.1139
S10A_HIST = {0: 77, 1: 70, 2: 58, 3: 45, 4: 110}
GATE_RECALL = 0.995
GATE_A_DROP = 0.05
GATE_CHAIN_CUT = 0.30
CHAIN_BASE_MS = 19.10


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def pctl(v):
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def emitted_curve(hist: dict[int, int]) -> list[float]:
    """emitted(D) = 1 + sum_{k=1..D} P(A >= k), from the measured S10-A histogram."""
    total = sum(hist.values())
    out, acc = [], 1.0
    for k in range(1, D_DRAFT + 1):
        acc += sum(c for a, c in hist.items() if a >= k) / total
        out.append(acc)
    return out


def leading_match(drafts, truth) -> int:
    n = 0
    for a, b in zip(drafts, truth):
        if int(a) != int(b):
            break
        n += 1
    return n


# --------------------------------------------------------------------- K0
def union_census(routes: dict[str, list[list[int]]], top_k: int):
    """Unique experts per layer over every window of B consecutive tokens."""
    out = {}
    for B in B_VALUES:
        per_window = []
        for layer, steps in routes.items():
            for s in range(len(steps) - B + 1):
                u = set()
                for t in range(B):
                    u.update(int(e) for e in steps[s + t])
                per_window.append(len(u))
        if not per_window:
            continue
        out[str(B)] = {"B": B, "windows": len(per_window),
                       "unique_experts": pctl(per_window),
                       "multiplicity": (B * top_k) / float(np.mean(per_window))}
    return out


def lru_replay(routes: dict[str, list[list[int]]], capacity: int, B: int | None):
    """Replay the per-layer LRU. B=None is AR order; otherwise round order.

    Round order asks for the union of B consecutive tokens in one go, which is
    what a verifier sweep would do. Only the accepted path is committed, but the
    LRU sees the whole union either way, so this counts what the round costs.
    """
    hits = misses = lookups = 0
    for layer, steps in routes.items():
        cache: OrderedDict[int, int] = OrderedDict()

        def touch(e: int):
            nonlocal hits, misses, lookups
            lookups += 1
            if e in cache:
                cache.move_to_end(e)
                hits += 1
                return
            misses += 1
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[e] = 1

        if B is None:
            for step in steps:
                for e in step:
                    touch(int(e))
        else:
            for s in range(0, len(steps) - B + 1, B):
                u = sorted({int(e) for t in range(B) for e in steps[s + t]})
                for e in u:
                    touch(e)
    return {"hits": hits, "misses": misses, "lookups": lookups,
            "hit_rate": hits / max(1, lookups)}


# --------------------------------------------------------------------- K2
def run_prompt(rt, mtp, cp, prompt_ids, steps, vocab_n, capture=None):
    mtp.concat_order = CONCAT_ORDER
    rt.reset()
    mtp.reset()
    S = [int(v) for v in prompt_ids]
    m = len(S)
    first, last = m - 1, m - 1 + steps - 1
    need = last + 1 + D_DRAFT
    drafts, chain_ms, in_vocab_hit, in_vocab_tot = {}, [], 0, 0

    i = 0
    while len(S) <= need:
        nxt = rt.step(S[i], capture_routes=capture)
        if i + 1 == len(S):
            S.append(int(nxt))
        measured = first <= i <= last
        mtp.set_active_vocab(rt.logits, vocab_n)
        vocab_host = (set(int(v) for v in cp.asnumpy(mtp.active_idx))
                      if (measured and vocab_n is not None) else None)
        h_dev = rt.normed if H_SOURCE == "post" else rt.h
        n_chain = D_DRAFT if measured else 1
        t0 = time.perf_counter_ns()
        tok_in, h_in, chain = S[i + 1], h_dev, []
        for j in range(n_chain):
            draft, x, y = mtp.forward(tok_in, h_in, i + j)
            chain.append(int(draft))
            tok_in = draft
            h_in = x if CHAIN_FROM == "x" else y
        if measured:
            cp.cuda.Device(0).synchronize()
            chain_ms.append((time.perf_counter_ns() - t0) / 1e6)
            drafts[i] = (chain, vocab_host)
        i += 1
        if i % 1000 == 0:
            print(f"      pos {i}/{need}", flush=True)

    per_step, A = [], []
    for i in sorted(drafts):
        chain, vocab_host = drafts[i]
        truth = [S[i + 2 + j] for j in range(D_DRAFT)]
        a = leading_match(chain, truth)
        A.append(a)
        per_step.append({"i": i, "drafts": chain, "truth": truth, "A": a})
        if vocab_host is not None:
            for t in truth:
                in_vocab_tot += 1
                in_vocab_hit += int(int(t) in vocab_host)
    return {"prompt_tokens": m, "first_measured_index": first,
            "measured_steps": len(per_step), "sequence": S, "per_step": per_step,
            "mean_A": float(np.mean(A)),
            "histogram_A": {str(v): int(sum(1 for a in A if a == v))
                            for v in range(D_DRAFT + 1)},
            "chain_ms": pctl(chain_ms),
            "vocab_recall": (in_vocab_hit / in_vocab_tot) if in_vocab_tot else None,
            "vocab_recall_counts": {"hit": in_vocab_hit, "total": in_vocab_tot}}


# --------------------------------------------------------------------- K1
def chain_decomposition(rt, mtp, cp, prompt_ids, reps, probes):
    """Bracketed replication, S12-R1 protocol, applied to the draft chain."""
    hidden, moe_inter = mtp.hidden, mtp.moe_inter
    p_act = cp.zeros(max(mtp.moe_inter, mtp.shared_inter), dtype=cp.float32)
    p_hid = cp.zeros(hidden, dtype=cp.float32)

    def replay(probe):
        idx = mtp.last_idx
        if probe == "head":
            mtp.fused.gemv_into(mtp.logits, rt.lm_head_codes, rt.lm_head_scales,
                                mtp.normed, rt.lm_head_g, mtp.vocab, hidden)
        elif probe == "experts":
            for e in idx:
                e = int(e)
                mtp.k.mv_bf16(p_act[:moe_inter],
                              mtp.exp_up[e * mtp.up_stride:(e + 1) * mtp.up_stride],
                              mtp.normed, moe_inter, hidden)
                mtp._relu2(p_act[:moe_inter], p_act[:moe_inter])
                mtp.k.mv_bf16(p_hid,
                              mtp.exp_dn[e * mtp.dn_stride:(e + 1) * mtp.dn_stride],
                              p_act[:moe_inter], hidden, moe_inter)
        elif probe == "attn":
            mtp.k.mv_bf16(mtp.qv, mtp.q_proj, mtp.normed,
                          mtp.n_heads * mtp.head_dim, hidden)
            mtp.k.attention(mtp.ctx, mtp.qv, mtp.kc, mtp.vc, 1, mtp.n_heads,
                            mtp.head_dim, mtp.groups, mtp.max_ctx,
                            1.0 / float(np.sqrt(mtp.head_dim)),
                            mtp.part_acc, mtp.part_ml)
            mtp.k.mv_bf16(p_hid, mtp.o_proj, mtp.ctx, hidden,
                          mtp.n_heads * mtp.head_dim)
        elif probe == "ehproj":
            mtp.k.mv_bf16(p_hid, mtp.eh_proj, mtp.cat, hidden, 2 * hidden)
        elif probe == "shared":
            mtp.k.mv_bf16(p_act[:mtp.shared_inter], mtp.sh_up, mtp.normed,
                          mtp.shared_inter, hidden)
            mtp._relu2(p_act[:mtp.shared_inter], p_act[:mtp.shared_inter])
            mtp.k.mv_bf16(p_hid, mtp.sh_dn, p_act[:mtp.shared_inter],
                          hidden, mtp.shared_inter)

    def arm(probe):
        rt.reset()
        mtp.reset()
        mtp.set_active_vocab(None, None)
        S = [int(v) for v in prompt_ids]
        for i in range(len(S) - 1):
            rt.step(S[i])
            mtp.forward(S[i + 1], rt.normed, i)
        base = len(S) - 1
        nxt = rt.step(S[-1])
        S.append(int(nxt))
        samples = []
        for r in range(reps):
            t0 = time.perf_counter_ns()
            tok_in, h_in = S[base + 1], rt.normed
            for j in range(D_DRAFT):
                draft, x, y = mtp.forward(tok_in, h_in, base + j)
                if probe:
                    replay(probe)
                tok_in, h_in = draft, y
            cp.cuda.Device(0).synchronize()
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        return pctl(samples)

    schedule, rows = [], {}
    for k, p in enumerate(probes):
        schedule += [(f"base{k}", None), (p, p)]
    schedule.append((f"base{len(probes)}", None))
    for name, probe in schedule:
        rows[name] = arm(probe)
        print(f"    {name:<9} chain p50 {rows[name]['p50']:7.3f} ms", flush=True)

    marginal, local_drift, reported = {}, {}, {}
    for k, p in enumerate(probes):
        b0, b1 = rows[f"base{k}"]["p50"], rows[f"base{k + 1}"]["p50"]
        marginal[p] = rows[p]["p50"] - 0.5 * (b0 + b1)
        local_drift[p] = abs(b1 - b0)
        reported[p] = abs(marginal[p]) > local_drift[p]
    return {"schedule": [n for n, _ in schedule], "arms": rows,
            "marginal_ms_per_chain": marginal, "local_drift_ms": local_drift,
            "reported_above_noise": reported,
            "global_drift_ms": abs(rows[f"base{len(probes)}"]["p50"] - rows["base0"]["p50"])}


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=32)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--long-ctx-tokens", type=int, default=4096)
    ap.add_argument("--long-ctx-steps", type=int, default=60)
    ap.add_argument("--chain-reps", type=int, default=40)
    args = ap.parse_args()

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    corpus_path = OUT_DIR / "s10a_corpus.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

    free0, total = cp.cuda.runtime.memGetInfo()
    rt = LightningRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                          embed_on_host=True, fp8_kv=True)
    rt.enable_cache(args.capacity)
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    mtp = MTPBlock(rt, max_ctx=args.max_ctx, concat_order=CONCAT_ORDER)
    free_all, _ = cp.cuda.runtime.memGetInfo()
    print(f"free {free_all / GIB:.3f} GiB", flush=True)

    # ------------------------------------------------------------ K2 + K0
    vocab_arms, routes_full = {}, {}
    for n in VOCAB_N:
        label = "full" if n is None else str(n)
        print(f"\nvocab arm {label}", flush=True)
        rows, pooled, recall_hit, recall_tot, chain_p50 = [], [], 0, 0, []
        for p in corpus["gate_prompts"]:
            cap = {} if (n is None) else None
            pid = tokenizer.encode(p["text"], add_special_tokens=False)
            r = run_prompt(rt, mtp, cp, pid, args.steps, n, capture=cap)
            r["label"] = p["id"]
            rows.append(r)
            pooled += [s["A"] for s in r["per_step"]]
            chain_p50.append(r["chain_ms"]["p50"])
            if r["vocab_recall_counts"]["total"]:
                recall_hit += r["vocab_recall_counts"]["hit"]
                recall_tot += r["vocab_recall_counts"]["total"]
            if cap is not None:
                for layer, steps in cap.items():
                    routes_full.setdefault(f"{p['id']}|{layer}", []).extend(steps)
            print(f"  {p['id']:<11} A={r['mean_A']:.4f} chain p50={r['chain_ms']['p50']:.3f} ms"
                  + (f" recall={r['vocab_recall']:.4f}" if r["vocab_recall"] else ""),
                  flush=True)
        vocab_arms[label] = {
            "vocab_n": n, "prompts": rows,
            "pooled_steps": len(pooled), "pooled_mean_A": float(np.mean(pooled)),
            "pooled_histogram_A": {str(v): int(sum(1 for a in pooled if a == v))
                                   for v in range(D_DRAFT + 1)},
            "chain_p50_mean_ms": float(np.mean(chain_p50)),
            "recall": (recall_hit / recall_tot) if recall_tot else None,
            "recall_counts": {"hit": recall_hit, "total": recall_tot}}
        print(f"  pooled A={vocab_arms[label]['pooled_mean_A']:.4f} "
              f"chain {vocab_arms[label]['chain_p50_mean_ms']:.3f} ms", flush=True)

    # long-context routes, K0 only
    print("\nK0 long-context route capture", flush=True)
    long_ids = tokenizer.encode(corpus["long_ctx_text"],
                                add_special_tokens=False)[:args.long_ctx_tokens]
    cap_long: dict = {}
    r_long = run_prompt(rt, mtp, cp, long_ids, args.long_ctx_steps, None, capture=cap_long)
    r_long["label"] = "long_ctx"
    long_routes = {f"long_ctx|{k}": v[-(args.long_ctx_steps + D_DRAFT + 2):]
                   for k, v in cap_long.items()}
    print(f"  long_ctx A={r_long['mean_A']:.4f}", flush=True)

    # ------------------------------------------------------------------ K0
    print("\nK0 census", flush=True)
    census_short = union_census(routes_full, rt.top_k)
    census_long = union_census(long_routes, rt.top_k)
    emitted = emitted_curve(S10A_HIST)
    per_emitted = {}
    for B, row in census_short.items():
        D = int(B) - 1
        u = row["unique_experts"]["mean"]
        e = emitted[D - 1] if 1 <= D <= D_DRAFT else None
        per_emitted[B] = {
            "B": int(B), "D": D, "mean_union": u, "emitted_tokens": e,
            "records_per_emitted": (u / e) if e else None,
            "ar_baseline_records": rt.top_k,
            "acceptance_needed_for_parity": (u / rt.top_k) if e is None else None}
        print(f"  B={B:>2} union {u:6.2f}  emitted {str(e):>6}  "
              f"records/emitted {str(round(u / e, 3)) if e else 'n/a':>6}  (AR {rt.top_k})",
              flush=True)

    replay = {}
    for capn in CAPACITIES:
        ar = lru_replay(routes_full, capn, None)
        rnd = lru_replay(routes_full, capn, 5)
        e5 = emitted[D_DRAFT - 1]
        n_tokens = sum(len(v) for v in routes_full.values()) / max(1, len(routes_full))
        replay[str(capn)] = {
            "capacity": capn, "ar": ar, "round_B5": rnd,
            "ar_misses_per_emitted": ar["misses"] / n_tokens,
            "round_misses_per_emitted": rnd["misses"] / (n_tokens / 5 * e5),
        }
        print(f"  cap {capn:>2}: AR hit {ar['hit_rate']:.4f} miss/tok "
              f"{replay[str(capn)]['ar_misses_per_emitted']:.3f} | "
              f"B5 hit {rnd['hit_rate']:.4f} miss/emitted "
              f"{replay[str(capn)]['round_misses_per_emitted']:.3f}", flush=True)

    # ------------------------------------------------------------------ K1
    print("\nK1 chain decomposition (bracketed)", flush=True)
    pid0 = tokenizer.encode(corpus["gate_prompts"][0]["text"], add_special_tokens=False)
    decomp = chain_decomposition(rt, mtp, cp, pid0, args.chain_reps,
                                 ["head", "experts", "attn", "shared", "ehproj"])

    # ---------------------------------------------------------------- gates
    b5 = per_emitted["5"]
    g_k0_1 = b5["records_per_emitted"] < rt.top_k
    r72 = replay["72"]
    g_k0_2 = r72["round_misses_per_emitted"] < r72["ar_misses_per_emitted"]
    full = vocab_arms["full"]
    best_vocab = None
    for n in VOCAB_N:
        if n is None:
            continue
        a = vocab_arms[str(n)]
        ok = (a["recall"] >= GATE_RECALL
              and a["pooled_mean_A"] >= full["pooled_mean_A"] - GATE_A_DROP
              and a["chain_p50_mean_ms"] <= CHAIN_BASE_MS * (1 - GATE_CHAIN_CUT))
        if ok and best_vocab is None:
            best_vocab = n
    gates = {
        "G_K0_1_records_per_emitted_below_AR": {
            "B": 5, "records_per_emitted": b5["records_per_emitted"],
            "ar_baseline": rt.top_k, "kimi_parity_union": 6 * emitted[D_DRAFT - 1],
            "passed": bool(g_k0_1)},
        "G_K0_2_round_misses_below_AR": {
            "capacity": 72, "round": r72["round_misses_per_emitted"],
            "ar": r72["ar_misses_per_emitted"], "passed": bool(g_k0_2)},
        "G_K0_3_window_coverage": {
            "min_windows": min(v["windows"] for v in census_short.values()),
            "passed": bool(min(v["windows"] for v in census_short.values()) >= 300)},
        "G_K2_R1_recall": {n: vocab_arms[str(n)]["recall"] for n in VOCAB_N if n},
        "G_K2_A1_acceptance": {n: vocab_arms[str(n)]["pooled_mean_A"] for n in VOCAB_N if n},
        "G_K2_T1_chain_cut": {n: 1 - vocab_arms[str(n)]["chain_p50_mean_ms"] / CHAIN_BASE_MS
                              for n in VOCAB_N if n},
        "G_K2_combined_pass_smallest_n": best_vocab,
        "G_K1_D1_reported": decomp["reported_above_noise"],
    }

    result = {
        "kind": "lightningstream_nemotron_k0_route_union_and_vocab",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "K0_K1_K2_LIGHTNINGSPEC_CENSUS",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "mtp_module_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/mtp.py"),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "corpus_sha256": sha256_path(corpus_path),
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "steps_per_prompt": args.steps, "D": D_DRAFT,
                   "concat_order": CONCAT_ORDER, "h_source": H_SOURCE,
                   "chain_from": CHAIN_FROM, "top_k": rt.top_k,
                   "moe_layers": len(rt.moe_layers), "vocab_n": VOCAB_N,
                   "B_values": B_VALUES, "capacities": CAPACITIES},
        "s10a_reference": {"mean_A": S10A_MEAN_A, "histogram": S10A_HIST,
                           "emitted_curve_D1_D4": emitted,
                           "chain_ms_p50": CHAIN_BASE_MS},
        "k0_census_short_context": census_short,
        "k0_census_long_context": census_long,
        "k0_records_per_emitted": per_emitted,
        "k0_lru_replay": replay,
        "k0_raw_routes": routes_full,
        "k2_vocab_arms": vocab_arms,
        "k2_long_context_arm": r_long,
        "k1_chain_decomposition": decomp,
        "gates": gates,
        "claim_boundary": (
            "Route unions and LRU replays are counted from the OFFICIAL routes "
            "this runtime emits during real greedy generation, not from a "
            "recomputed top-k. The LRU replay is a simulation over those routes, "
            "not a timed measurement. Acceptance and recall are measured; chain "
            "times are component measurements with all 128 MTP experts "
            "device-resident and are NOT converted to tokens per second. No "
            "speculative loop was built, so nothing here is a throughput result. "
            "Acceptance for D>4 is NOT measured, so records-per-emitted is left "
            "undefined for B>5 and only the union is reported there."),
    }
    (OUT_DIR / "k0_route_union_census.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\ngates")
    print(f"  G-K0-1 records/emitted {b5['records_per_emitted']:.3f} < {rt.top_k}: {g_k0_1}")
    print(f"  G-K0-2 round misses/emitted {r72['round_misses_per_emitted']:.3f} "
          f"< AR {r72['ar_misses_per_emitted']:.3f}: {g_k0_2}")
    for n in VOCAB_N:
        if n is None:
            continue
        a = vocab_arms[str(n)]
        print(f"  vocab {n:>5}: recall {a['recall']:.5f}  A {a['pooled_mean_A']:.4f}  "
              f"chain {a['chain_p50_mean_ms']:.3f} ms "
              f"({(1 - a['chain_p50_mean_ms'] / CHAIN_BASE_MS) * 100:+.1f}%)")
    print(f"  smallest vocab passing all three K2 gates: {best_vocab}")
    print("\nwritten k0_route_union_census.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
