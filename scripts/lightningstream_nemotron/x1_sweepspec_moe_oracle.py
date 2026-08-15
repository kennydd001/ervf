"""X1: does expert-major block execution make the MoE term cheap enough?

Preregistered in X1_SWEEPSPEC_MOE_ORACLE_PREREGISTRATION_2026-08-15.md.

The MoE term is 39.523 of the 54.277 ms per token, so it alone decides whether a
block verifier can fit in the round budgets both packs derived (62.280 ms for 50
tok/s short, 77.850 for 40 at 128K, 103.800 for 30 at 262K). This runner builds
only that term, on real hidden states and real official routes, and times

  SEQ   : B sequential token-major passes -- the path the runtime runs today
  SWEEP : expert-major, each unique expert's record read once per block

Exactness first: at nchunks=1 the two must be bit-identical, because the union
panel mask only adds columns whose activation is exactly zero and the panel walk
stays in ascending order. Only then are timings reported.
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

from moe_lab.lightningstream_nemotron.runtime import (  # noqa: E402
    LightningRuntime, UP_CODE, UP_SCALE, DOWN_PANEL_BYTES)
from moe_lab.lightningstream_nemotron.sweepspec import SweepMoE  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GIB = 1024 ** 3

S8_MOE_TERM_MS = 39.523
EMITTED_PER_ROUND = 3.1139          # measured S10-A: 1 + A


class CapturingRuntime(LightningRuntime):
    """The real runtime, plus a tap on each MoE layer's input and route."""

    capture = None

    def _moe_cached(self, i, out):
        idx, w = super()._moe_cached(i, out)
        if self.capture is not None:
            self.capture.setdefault(i, []).append(
                (self.cp.asnumpy(self.normed).copy(),
                 np.asarray(idx, dtype=np.int64).copy(),
                 np.asarray(w, dtype=np.float64).copy()))
        return idx, w


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
            "min": float(a.min()), "max": float(a.max())}


def ensure_cached(rt, layer, experts):
    """Make every expert of the block resident, outside any timed region."""
    c, bank = rt.cache[layer], rt.bank[layer]
    for e in experts:
        e = int(e)
        if e in c["map"]:
            c["map"].move_to_end(e)
            continue
        if len(c["map"]) < c["cap"]:
            slot = len(c["map"])
        else:
            _, slot = c["map"].popitem(last=False)
        c["map"][e] = slot
        c["codes"][slot * UP_CODE:(slot + 1) * UP_CODE].set(
            bank["up_codes"][e * UP_CODE:(e + 1) * UP_CODE])
        c["scales"][slot * UP_SCALE:(slot + 1) * UP_SCALE].set(
            bank["up_scales"][e * UP_SCALE:(e + 1) * UP_SCALE])


def seq_layer(rt, layer, xs, routes, weights, out2d, act, tmp, mstate, B, nchunks):
    """Token-major: exactly the loop _moe_cached runs, once per node."""
    bank, cache = rt.bank[layer], rt.cache[layer]
    hidden, inter = rt.hidden, rt.moe_inter
    for b in range(B):
        out = out2d[b * hidden:(b + 1) * hidden]
        out.fill(0)
        for s in range(rt.top_k):
            e = int(routes[b][s])
            slot = cache["map"][e]
            rt.fused.gemv_into(act[:inter],
                               cache["codes"][slot * UP_CODE:(slot + 1) * UP_CODE],
                               cache["scales"][slot * UP_SCALE:(slot + 1) * UP_SCALE],
                               xs[b * hidden:(b + 1) * hidden],
                               float(bank["globals"][e, 1]), inter, hidden,
                               apply_relu2=True)
            rt.fused.down_masked_into(
                tmp, bank["down_base_ptr"] + e * DOWN_PANEL_BYTES,
                act[:inter], mstate, float(bank["globals"][e, 0]),
                hidden, inter, nchunks=nchunks)
            rt.fused.accumulate_into(out, tmp, float(weights[b][s]), hidden)


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=72)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--b-values", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    corpus = json.loads((OUT_DIR / "s10a_corpus.json").read_text(encoding="utf-8"))

    free0, total = cp.cuda.runtime.memGetInfo()
    rt = CapturingRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                          embed_on_host=True, fp8_kv=True)
    rt.enable_cache(args.capacity)
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    sweep = SweepMoE(rt.fused, rt.k)
    print(f"opt-in shared memory: {sweep.max_shared} B "
          f"(B=5 needs {5 * rt.hidden * 4} B)", flush=True)

    # ------------------------------------------------- capture real material
    ids = tokenizer.encode(corpus["gate_prompts"][0]["text"], add_special_tokens=False)
    rt.reset()
    rt.capture = {}
    nxt = None
    for t in ids:
        nxt = rt.step(t)
    cur = int(nxt)
    for _ in range(args.gen_tokens):
        cur = int(rt.step(cur))
    cp.cuda.Device(0).synchronize()
    rt.capture, cap = None, rt.capture
    layers = sorted(cap)
    n_steps = min(len(v) for v in cap.values())
    print(f"captured {len(layers)} MoE layers x {n_steps} steps", flush=True)

    hidden, inter, top_k = rt.hidden, rt.moe_inter, rt.top_k
    act = cp.zeros(inter, dtype=cp.float32)
    tmp = cp.zeros(hidden, dtype=cp.float32)
    mstate = rt.fused.alloc_masked_state(hidden, inter)

    results, exact = {}, {}
    for B in args.b_values:
        if B * hidden * 4 > sweep.max_shared:
            print(f"B={B}: needs {B * hidden * 4} B shared, skipped")
            continue
        starts = np.linspace(8, n_steps - B - 1, args.blocks).astype(int).tolist()
        state = sweep.alloc_state(hidden, inter, top_k, B)
        xs = cp.zeros(B * hidden, dtype=cp.float32)
        out_seq = cp.zeros(B * hidden, dtype=cp.float32)
        out_swp = cp.zeros(B * hidden, dtype=cp.float32)

        # ---------------------------------------------------- exactness pass
        mism, checked, worst = 0, 0, 0.0
        state1 = sweep.alloc_state(hidden, inter, top_k, B)
        state1["nchunks"] = 1
        for st in starts[:4]:
            for layer in layers:
                rows = cap[layer][st:st + B]
                routes = [r[1] for r in rows]
                weights = [r[2] for r in rows]
                for b, r in enumerate(rows):
                    xs[b * hidden:(b + 1) * hidden] = cp.asarray(r[0])
                ensure_cached(rt, layer, {int(e) for r in routes for e in r})
                seq_layer(rt, layer, xs, routes, weights, out_seq, act, tmp,
                          mstate, B, nchunks=1)
                plan = sweep.compile_plan(sweep.plan(routes, top_k), top_k)
                wh = np.array([[float(w) for w in weights[b]] for b in range(B)],
                              dtype=np.float32).reshape(-1)
                state1["weights"].set(wh)
                state1["contrib"].fill(0)
                out_swp.fill(0)
                sweep.moe_block(out_swp, rt, layer, xs, plan, state1, B)
                a, b_ = cp.asnumpy(out_seq), cp.asnumpy(out_swp)
                checked += 1
                if not np.array_equal(a, b_):
                    mism += 1
                    den = float(np.linalg.norm(a)) or 1.0
                    worst = max(worst, float(np.linalg.norm(a - b_)) / den)
        exact[str(B)] = {"layer_blocks_checked": checked, "mismatches": mism,
                         "worst_rel_l2": worst, "bit_identical": mism == 0}
        print(f"  B={B} exactness: {checked} layer-blocks, {mism} mismatches, "
              f"worst rel_l2 {worst:.3e}", flush=True)

        # ------------------------------------------------------ timing, both
        def run_seq():
            for layer in layers:
                rows = cap[layer][st0:st0 + B]
                seq_layer(rt, layer, xs, [r[1] for r in rows], [r[2] for r in rows],
                          out_seq, act, tmp, mstate, B, nchunks=rt.fused.nchunks)

        def run_sweep():
            for layer, plan in plans.items():
                out_swp.fill(0)
                state["contrib"].fill(0)
                state["weights"].set(wall[layer])
                sweep.moe_block(out_swp, rt, layer, xs, plan, state, B)

        seq_ms, swp_ms, seq2_ms = [], [], []
        for st0 in starts:
            plans, wall = {}, {}
            for layer in layers:
                rows = cap[layer][st0:st0 + B]
                routes = [r[1] for r in rows]
                for b, r in enumerate(rows):
                    xs[b * hidden:(b + 1) * hidden] = cp.asarray(r[0])
                ensure_cached(rt, layer, {int(e) for r in routes for e in r})
                plans[layer] = sweep.compile_plan(sweep.plan(routes, top_k), top_k)
                wall[layer] = np.array([[float(w) for w in r[2]] for r in rows],
                                       dtype=np.float32).reshape(-1)
            for arm, fn, sink in (("seq", run_seq, seq_ms), ("sweep", run_sweep, swp_ms),
                                  ("seq2", run_seq, seq2_ms)):
                fn()                       # warm
                cp.cuda.Device(0).synchronize()
                for _ in range(args.reps):
                    t0 = time.perf_counter_ns()
                    fn()
                    cp.cuda.Device(0).synchronize()
                    sink.append((time.perf_counter_ns() - t0) / 1e6)

        s_seq, s_swp, s_seq2 = pctl(seq_ms), pctl(swp_ms), pctl(seq2_ms)
        base = 0.5 * (s_seq["p50"] + s_seq2["p50"])
        drift = abs(s_seq2["p50"] - s_seq["p50"])
        unions = []
        for st0 in starts:
            for layer in layers:
                rows = cap[layer][st0:st0 + B]
                unions.append(len({int(e) for r in rows for e in r[1]}))
        results[str(B)] = {
            "B": B, "blocks": len(starts), "reps": args.reps,
            "seq_ms": s_seq, "sweep_ms": s_swp, "seq_repeat_ms": s_seq2,
            "seq_bracketed_p50_ms": base, "local_drift_ms": drift,
            "ratio_sweep_over_seq": s_swp["p50"] / base,
            "mean_union_per_layer": float(np.mean(unions)),
            "conclusive": bool(abs(s_swp["p50"] - base) > drift),
        }
        print(f"  B={B}: seq {base:8.3f} ms | sweep {s_swp['p50']:8.3f} ms | "
              f"ratio {s_swp['p50'] / base:.4f} | drift {drift:.3f} | "
              f"union {np.mean(unions):.2f}", flush=True)

    # ------------------------------------------------------------------ gates
    r5 = results.get("5")
    threshold = EMITTED_PER_ROUND / 5.0
    gates = {
        "G_X1_C1_batched_equals_gemv_at_B1": {
            "required": "gemm_nvfp4_rows_b with B=1 bit-identical to the runtime path",
            "passed": bool(exact.get("1", {}).get("bit_identical", False))},
        "G_X1_C2_sweep_equals_sequential": {
            "required": "expert-major output bit-identical to token-major, nchunks=1",
            "per_B": exact,
            "passed": bool(all(v["bit_identical"] for v in exact.values()))},
        "G_X1_P1_ratio": {
            "required_ratio_below": threshold,
            "reason": "a round emits 3.1139 tokens, so the sweep must cost less "
                      "than 3.1139/5 of five sequential passes to beat AR",
            "measured_ratio": r5["ratio_sweep_over_seq"] if r5 else None,
            "passed": bool(r5 and r5["ratio_sweep_over_seq"] < threshold)},
        "G_X1_D1_drift": {
            "conclusive_per_B": {k: v["conclusive"] for k, v in results.items()}},
    }

    free_end, _ = cp.cuda.runtime.memGetInfo()
    payload = {
        "kind": "lightningstream_nemotron_x1_sweepspec_moe_oracle",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "X1_SWEEPSPEC_MOE_ORACLE",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "sweepspec_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/sweepspec.py"),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "gen_tokens": args.gen_tokens, "blocks": args.blocks,
                   "reps": args.reps, "b_values": args.b_values,
                   "moe_layers": len(layers), "top_k": top_k,
                   "opt_in_shared_bytes": sweep.max_shared,
                   "emitted_per_round": EMITTED_PER_ROUND,
                   "s8_moe_term_ms": S8_MOE_TERM_MS},
        "exactness": exact,
        "timing": results,
        "gates": gates,
        "device_free_bytes": int(free_end),
        "claim_boundary": (
            "Measured cost of the ROUTED-EXPERT part of all 23 MoE layers for B "
            "nodes, on real hidden states and real official routes captured from "
            "a real greedy generation, with every expert of the block already "
            "resident so that the one variable is token-major versus expert-major "
            "execution. It is NOT a token time and NOT a throughput result: the "
            "router, the shared expert, Mamba, attention and the LM head are not "
            "in either arm, and no speculative loop exists. Exactness is proven "
            "at nchunks=1, where the union panel mask can only add exactly-zero "
            "columns; the timed configuration uses the production nchunks and "
            "differs from the sequential path by float reassociation only."),
    }
    (OUT_DIR / "x1_sweepspec_moe_oracle.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\ngates")
    print(f"  G-X1-C1 batched==gemv at B=1 : {gates['G_X1_C1_batched_equals_gemv_at_B1']['passed']}")
    print(f"  G-X1-C2 sweep==sequential    : {gates['G_X1_C2_sweep_equals_sequential']['passed']}")
    if r5:
        print(f"  G-X1-P1 ratio {r5['ratio_sweep_over_seq']:.4f} < {threshold:.4f} : "
              f"{gates['G_X1_P1_ratio']['passed']}")
    print("\nwritten x1_sweepspec_moe_oracle.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
