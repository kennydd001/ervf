"""Z1: TreeSweep Oracle A -- the ceiling of any tree verifier on this target.

Preregistered in Z1_TREE_VERIFIER_CEILING_PREREGISTRATION_2026-08-15.md.

A tree of N verified positions can commit at most N tokens, so for ANY drafter
and ANY topology, throughput <= N / T_v(N). X1 measured the dominant term as
linear in positions; if T_moe(N) = c*N then N/T_moe(N) = 1/c, a constant no tree
can beat. This measures c over N up to 32 instead of extrapolating from five
points, on the sequential path -- which X1 showed expert-major grouping does not
undercut.
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

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"

X1_T1_MS = 22.454
GATE_P2C_TOK_S = 250.0
GATE_R2 = 0.99


class CapturingRuntime(LightningRuntime):
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


def ensure_cached(rt, layer, experts):
    c, bank = rt.cache[layer], rt.bank[layer]
    for e in experts:
        e = int(e)
        if e in c["map"]:
            c["map"].move_to_end(e)
            continue
        slot = len(c["map"]) if len(c["map"]) < c["cap"] else c["map"].popitem(last=False)[1]
        c["map"][e] = slot
        c["codes"][slot * UP_CODE:(slot + 1) * UP_CODE].set(
            bank["up_codes"][e * UP_CODE:(e + 1) * UP_CODE])
        c["scales"][slot * UP_SCALE:(slot + 1) * UP_SCALE].set(
            bank["up_scales"][e * UP_SCALE:(e + 1) * UP_SCALE])


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=72)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--gen-tokens", type=int, default=80)
    ap.add_argument("--n-values", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--reps", type=int, default=3)
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
    rt = CapturingRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                          embed_on_host=True, fp8_kv=True)
    rt.enable_cache(args.capacity)
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

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
    cap, rt.capture = rt.capture, None
    layers = sorted(cap)
    n_steps = min(len(v) for v in cap.values())
    print(f"captured {len(layers)} MoE layers x {n_steps} steps", flush=True)

    hidden, inter, top_k = rt.hidden, rt.moe_inter, rt.top_k
    act = cp.zeros(inter, dtype=cp.float32)
    tmp = cp.zeros(hidden, dtype=cp.float32)
    outv = cp.zeros(hidden, dtype=cp.float32)
    mstate = rt.fused.alloc_masked_state(hidden, inter)

    def seq_pass(start, N):
        for layer in layers:
            rows = cap[layer][start:start + N]
            bank, cache = rt.bank[layer], rt.cache[layer]
            for r in rows:
                x = cp.asarray(r[0])
                outv.fill(0)
                for s in range(top_k):
                    e = int(r[1][s])
                    slot = cache["map"][e]
                    rt.fused.gemv_into(
                        act[:inter],
                        cache["codes"][slot * UP_CODE:(slot + 1) * UP_CODE],
                        cache["scales"][slot * UP_SCALE:(slot + 1) * UP_SCALE],
                        x, float(bank["globals"][e, 1]), inter, hidden,
                        apply_relu2=True)
                    rt.fused.down_masked_into(
                        tmp, bank["down_base_ptr"] + e * DOWN_PANEL_BYTES,
                        act[:inter], mstate, float(bank["globals"][e, 0]),
                        hidden, inter)
                    rt.fused.accumulate_into(outv, tmp, float(r[2][s]), hidden)

    rows_out, unions = {}, {}
    for N in args.n_values:
        if N + 8 >= n_steps:
            print(f"  N={N}: not enough captured steps, skipped")
            continue
        start = 8
        for layer in layers:
            r = cap[layer][start:start + N]
            ensure_cached(rt, layer, {int(e) for x in r for e in x[1]})
        u = [len({int(e) for x in cap[layer][start:start + N] for e in x[1]})
             for layer in layers]
        unions[str(N)] = float(np.mean(u))

        samples = {"a1": [], "run": [], "a2": []}
        seq_pass(start, N)
        cp.cuda.Device(0).synchronize()
        for _ in range(args.reps):
            for key in ("a1", "run", "a2"):
                t0 = time.perf_counter_ns()
                seq_pass(start, N)
                cp.cuda.Device(0).synchronize()
                samples[key].append((time.perf_counter_ns() - t0) / 1e6)
        p = {k: float(np.percentile(v, 50)) for k, v in samples.items()}
        base = 0.5 * (p["a1"] + p["a2"])
        drift = abs(p["a2"] - p["a1"])
        rows_out[str(N)] = {"N": N, "ms_p50": p["run"], "bracket_p50": base,
                            "local_drift_ms": drift, "raw": samples,
                            "mean_union_per_layer": unions[str(N)],
                            "ms_per_position": p["run"] / N}
        print(f"  N={N:>3}: {p['run']:9.3f} ms  ({p['run'] / N:6.3f} ms/position)  "
              f"drift {drift:5.3f}  union {unions[str(N)]:6.2f}", flush=True)

    ns = np.array([r["N"] for r in rows_out.values()], dtype=np.float64)
    ts = np.array([r["ms_p50"] for r in rows_out.values()], dtype=np.float64)
    c, d = np.polyfit(ns, ts, 1)
    pred = c * ns + d
    ss_res = float(((ts - pred) ** 2).sum())
    ss_tot = float(((ts - ts.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    nmax = max(rows_out, key=lambda k: int(k))
    scaling = rows_out[nmax]["ms_p50"] / rows_out["1"]["ms_p50"]
    ceiling = 1000.0 / c

    linear = r2 >= GATE_R2 and abs(scaling - int(nmax)) / int(nmax) <= 0.10
    sanity = abs(rows_out["1"]["ms_p50"] - X1_T1_MS) / X1_T1_MS <= 0.10

    gates = {
        "G_Z1_L1_linearity": {"r2": r2, "scaling_at_Nmax": scaling,
                              "N_max": int(nmax), "passed": bool(linear)},
        "G_Z1_P2C": {"required_tok_s": GATE_P2C_TOK_S,
                     "ceiling_tok_s": ceiling if linear else None,
                     "passed": bool(linear and ceiling >= GATE_P2C_TOK_S)},
        "G_Z1_S1_sanity": {"x1_t1_ms": X1_T1_MS, "measured_t1_ms": rows_out["1"]["ms_p50"],
                           "passed": bool(sanity)},
    }

    payload = {
        "kind": "lightningstream_nemotron_z1_tree_verifier_ceiling",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "Z1_TREE_VERIFIER_CEILING",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "n_values": args.n_values, "reps": args.reps,
                   "moe_layers": len(layers), "top_k": top_k,
                   "path": "sequential (X1 measured expert-major at ratio 1.0017)"},
        "rows": rows_out,
        "linear_fit": {"ms_per_position": float(c), "intercept_ms": float(d), "r2": r2},
        "ceiling_tok_s": ceiling,
        "gates": gates,
        "claim_boundary": (
            "Measured cost of the ROUTED-EXPERT part of all 23 MoE layers for N "
            "verified positions, on real hidden states and real official routes, "
            "every expert resident, short context. The ceiling 1/c is an UPPER "
            "BOUND on any tree verifier's throughput for this target on this GPU: "
            "it already assumes perfect coverage (every one of the N positions "
            "committed), zero draft cost, and free Mamba, attention, LM head, "
            "router, shared expert and state commit. A real implementation pays "
            "all of those on top. It is not a measured throughput and no runtime "
            "achieving it exists."),
    }
    (OUT_DIR / "z1_tree_verifier_ceiling.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n  fit: {c:.4f} ms per position + {d:.3f} ms   R2={r2:.5f}")
    print(f"  scaling T({nmax})/T(1) = {scaling:.3f} against {nmax}")
    print(f"  ceiling 1/c = {ceiling:.2f} tok/s   (pack gate {GATE_P2C_TOK_S})")
    print(f"  G-Z1-L1 linearity : {gates['G_Z1_L1_linearity']['passed']}")
    print(f"  G-Z1-S1 sanity    : {gates['G_Z1_S1_sanity']['passed']}")
    print(f"  G-Z1-P2C          : {gates['G_Z1_P2C']['passed']}")
    print("\nwritten z1_tree_verifier_ceiling.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
