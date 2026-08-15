"""S12: attribute the MoE term by replication INSIDE the real decode loop.

Preregistered in S12_IN_LOOP_ATTRIBUTION_PREREGISTRATION_2026-08-15.md.

S8 showed isolated component timing overcounts here: it forces a sync that hides
the overlap the real loop has, so the sum of parts exceeded the measured token by
16.9 ms at 262K. This runner therefore never times a component on its own. It
runs the real loop and adds exactly one extra invocation of one component per
occurrence, writing to scratch, and reads the end-to-end delta.

`runtime.py` is not touched: the probe lives in a subclass that calls
`super()._moe_cached()` and then does the extra work, so the measured loop is
provably the shipped loop.
"""

from __future__ import annotations

import argparse
import gc
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
    LightningRuntime, DOWN_PANEL_BYTES, UP_CODE, UP_SCALE, cp_asnumpy)

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GIB = 1024 ** 3

PROMPTS = ["The capital of France is", "The history of computing began when"]
GEN_TOKENS = 32
ARMS = ["base1", "up", "down", "router", "shared", "accum", "base2"]
S8_MOE_TERM_MS = 39.523      # measured, 262K, for the G-S12-S1 sanity gate


class ProbedRuntime(LightningRuntime):
    """The real runtime plus one optional replicated component per MoE layer."""

    probe: str | None = None

    def alloc_probe(self):
        cp = self.cp
        self.p_act = cp.zeros(max(self.moe_inter, self.shared_inter), dtype=cp.float32)
        self.p_out = cp.zeros(self.hidden, dtype=cp.float32)
        self.p_state = self.fused.alloc_masked_state(self.hidden, self.moe_inter)

    def _moe_cached(self, i, out):
        idx, w = super()._moe_cached(i, out)
        p = self.probe
        if p is None:
            return idx, w

        d, c, bank = self.layer[i], self.cache[i], self.bank[i]
        if p == "router":
            packed = self._route_device(i)
            cp_asnumpy(self.cp, packed)          # the readback is part of the cost
            return idx, w
        if p == "shared":
            self.fused.gemv_into(self.p_act[:self.shared_inter], d["sh_up_c"],
                                 d["sh_up_s"], self.normed, d["sh_up_g"],
                                 self.shared_inter, self.hidden, apply_relu2=True)
            self.fused.gemv_into(self.p_out, d["sh_dn_c"], d["sh_dn_s"],
                                 self.p_act[:self.shared_inter], d["sh_dn_g"],
                                 self.hidden, self.shared_inter)
            return idx, w

        for s, e in enumerate(idx):
            e = int(e)
            sl = c["map"][e]
            if p == "up":
                self.fused.gemv_into(
                    self.p_act[:self.moe_inter],
                    c["codes"][sl * UP_CODE:(sl + 1) * UP_CODE],
                    c["scales"][sl * UP_SCALE:(sl + 1) * UP_SCALE],
                    self.normed, float(bank["globals"][e, 1]),
                    self.moe_inter, self.hidden, apply_relu2=True)
            elif p == "down":
                # self.act still holds a real ReLU^2 activation from this layer,
                # so the panel mask has realistic sparsity; the record offset is
                # this expert's own, so the gather touches its own host pages.
                self.fused.down_masked_into(
                    self.p_out, bank["down_base_ptr"] + e * DOWN_PANEL_BYTES,
                    self.act[:self.moe_inter], self.p_state,
                    float(bank["globals"][e, 0]), self.hidden, self.moe_inter)
            elif p == "accum":
                self.fused.accumulate_into(self.p_out, self.tmp, float(w[s]), self.hidden)
            else:
                raise ValueError(f"unknown probe {p!r}")
        return idx, w


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def pct(v):
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()), "min": float(a.min())}


def generate(rt, cp, tokenizer):
    out = []
    for text in PROMPTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = rt.step(t)
        gen = [int(nxt)]
        for _ in range(GEN_TOKENS - 1):
            gen.append(int(rt.step(gen[-1])))
        cp.cuda.Device(0).synchronize()
        out.append(gen)
    return out


def context_sweep(rt, cp, contexts, max_ctx):
    rng = np.random.default_rng(11)
    varied = [int(v) for v in rng.integers(1000, 60000, size=4096)]
    rows = {}
    for target in contexts:
        if target >= max_ctx - 8:
            continue
        rt.reset()
        for j in range(min(target, 64)):
            rt.step(varied[j % len(varied)])
        rt.pos = target
        for j in range(32):
            rt.step(varied[(j + 64) % len(varied)])
        cp.cuda.Device(0).synchronize()
        samples = []
        for j in range(16):
            t0 = time.perf_counter_ns()
            rt.step(varied[(j + 96) % len(varied)])
            cp.cuda.Device(0).synchronize()
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        s = pct(samples)
        rows[str(target)] = {"context": target, "ms": s, "raw_ms": samples}
        print(f"    ctx {target:>6}: p50 {s['p50']:7.3f} ms", flush=True)
    return rows


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=70)
    ap.add_argument("--max-ctx", type=int, default=262144)
    ap.add_argument("--contexts", type=int, nargs="*", default=[0, 262100])
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
    free0, total = cp.cuda.runtime.memGetInfo()
    rt = ProbedRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                       embed_on_host=True, fp8_kv=True)
    free_shell, _ = cp.cuda.runtime.memGetInfo()
    cache_bytes = rt.enable_cache(args.capacity)
    rt.alloc_probe()
    rt.load_routed_bank()
    gc.collect()
    free_ready, _ = cp.cuda.runtime.memGetInfo()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    print(f"shell {(free0 - free_shell) / GIB:.3f} GiB | cache {cache_bytes / GIB:.3f} GiB | "
          f"free {free_ready / GIB:.3f} GiB", flush=True)

    arms, reference_gen = {}, None
    for name in ARMS:
        rt.probe = None if name.startswith("base") else name
        print(f"\narm {name}: probe={rt.probe}", flush=True)
        gen = generate(rt, cp, tokenizer)
        if reference_gen is None:
            reference_gen = gen
        identical = gen == reference_gen
        print(f"  identity: {identical}", flush=True)
        rows = context_sweep(rt, cp, args.contexts, args.max_ctx)
        arms[name] = {"arm": name, "probe": rt.probe,
                      "identical_to_base1": bool(identical),
                      "generation_token_ids": gen,
                      "context_sweep": rows}
    rt.probe = None

    contexts = [str(c) for c in args.contexts]
    marginal, drift = {}, {}
    for ctx in contexts:
        b1 = arms["base1"]["context_sweep"][ctx]["ms"]["p50"]
        b2 = arms["base2"]["context_sweep"][ctx]["ms"]["p50"]
        drift[ctx] = abs(b2 - b1)
        marginal[ctx] = {
            a: arms[a]["context_sweep"][ctx]["ms"]["p50"] - b1
            for a in ARMS if not a.startswith("base")
        }

    deep = contexts[-1]
    identity_ok = all(a["identical_to_base1"] for a in arms.values())
    smallest = min(abs(v) for v in marginal[deep].values())
    marg_sum = {c: sum(marginal[c].values()) for c in contexts}

    gates = {
        "G_S12_C1_identity": {
            "required": "generation bit-identical in all seven arms",
            "passed": bool(identity_ok)},
        "G_S12_D1_drift": {
            "required": "|base2-base1| < smallest reported marginal, per context",
            "drift_ms": drift,
            "smallest_marginal_ms_at_deep": smallest,
            "passed_at_deep": bool(drift[deep] < smallest),
            "below_noise_floor": {c: [a for a, v in marginal[c].items()
                                      if abs(v) <= drift[c]] for c in contexts}},
        "G_S12_S1_sanity": {
            "required": f"sum of marginals <= S8 MoE term {S8_MOE_TERM_MS} ms at 262K",
            "sum_ms": marg_sum,
            "s8_moe_term_ms": S8_MOE_TERM_MS,
            "passed": bool(marg_sum[deep] <= S8_MOE_TERM_MS)},
    }

    result = {
        "kind": "lightningstream_nemotron_s12_in_loop_attribution",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "S12_IN_LOOP_ATTRIBUTION",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "contexts": args.contexts, "embed_on_host": True, "fp8_kv": True,
                   "gen_tokens": GEN_TOKENS, "prompts": PROMPTS,
                   "moe_layers": len(rt.moe_layers), "top_k": rt.top_k,
                   "cache_bytes": int(cache_bytes),
                   "replications_per_moe_layer": {
                       "up": rt.top_k, "down": rt.top_k, "accum": rt.top_k,
                       "router": 1, "shared": 1}},
        "arms": arms,
        "marginal_ms_per_token": marginal,
        "baseline_drift_ms": drift,
        "gates": gates,
        "method": (
            "Real decode loop, unmodified, plus exactly one extra invocation of "
            "one component per occurrence, output to scratch. The end-to-end p50 "
            "delta is the MARGINAL in-loop cost of that component."),
        "claim_boundary": (
            "Marginal in-loop costs, measured end-to-end on this GPU at capacity "
            "70 (not 72: the probe needs scratch and 72 leaves 0.000 GiB free), "
            "so absolute figures here are NOT comparable to n7b and only the "
            "arm-to-arm differences are. Every marginal is a LOWER BOUND: the "
            "replicated call finds part of its data warm in L2, and extra work "
            "at the end of a layer gives the copy stream more room to hide. "
            "Marginals are not shares of the MoE term and do not sum to it; "
            "overlapping components have marginals that under-sum by "
            "construction. Nothing here is converted to tokens per second, and "
            "the part the marginals do not cover is reported as a number and "
            "not given a name."),
    }
    (OUT_DIR / "s12_in_loop_attribution.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nmarginal in-loop cost, ms per token")
    for ctx in contexts:
        print(f"  ctx {ctx:>7}  base p50 {arms['base1']['context_sweep'][ctx]['ms']['p50']:7.3f} ms"
              f"  drift {drift[ctx]:.3f} ms")
        for a, v in sorted(marginal[ctx].items(), key=lambda kv: -kv[1]):
            flag = "  (below noise floor)" if abs(v) <= drift[ctx] else ""
            print(f"     {a:<8} {v:+7.3f}{flag}")
        print(f"     {'sum':<8} {marg_sum[ctx]:+7.3f}")
    print(f"\n  G-S12-C1 identity: {gates['G_S12_C1_identity']['passed']}")
    print(f"  G-S12-D1 drift   : {gates['G_S12_D1_drift']['passed_at_deep']}")
    print(f"  G-S12-S1 sanity  : {gates['G_S12_S1_sanity']['passed']}")
    print("\nwritten s12_in_loop_attribution.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
