"""S12-R1: the S12 attribution again, with every probe bracketed by baselines.

Preregistered in S12R1_BRACKETED_ATTRIBUTION_PREREGISTRATION_2026-08-15.md.

S12 failed its own drift gate at 262100: base2 was 5.057 ms slower than base1,
one-sidedly, which is what five heavier probe arms in a row do to a laptop GPU's
clocks. The fix is a schedule, not a looser gate. Each probe now sits between two
baseline arms and is measured against their mean, so a linear trend in time
cancels, and each probe gets its own local noise floor instead of one global one.

Everything else is identical to S12, including the probe subclass, so
`runtime.py` still describes the loop being measured.
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
PROBES = ["up", "down", "router", "shared", "accum"]
S8_MOE_TERM_MS = 39.523


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
            cp_asnumpy(self.cp, packed)
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


def gpu_state():
    try:
        o = subprocess.run(["nvidia-smi",
                            "--query-gpu=temperature.gpu,clocks.sm,power.draw",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        t, c, p = [x.strip() for x in o.stdout.strip().split(",")]
        return {"temp_c": int(t), "sm_mhz": int(c), "power_w": float(p)}
    except Exception as e:                                   # pragma: no cover
        return {"error": str(e)}


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
        rows[str(target)] = {"context": target, "ms": s, "raw_ms": samples,
                             "gpu": gpu_state()}
        print(f"    ctx {target:>6}: p50 {s['p50']:7.3f} ms  "
              f"{rows[str(target)]['gpu'].get('temp_c', '?')}C", flush=True)
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

    # base0 up base1 down base2 router base3 shared base4 accum base5
    schedule = []
    for k, probe in enumerate(PROBES):
        schedule.append((f"base{k}", None))
        schedule.append((probe, probe))
    schedule.append((f"base{len(PROBES)}", None))

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
    for name, probe in schedule:
        rt.probe = probe
        print(f"\narm {name}: probe={probe}", flush=True)
        gen = generate(rt, cp, tokenizer)
        if reference_gen is None:
            reference_gen = gen
        identical = gen == reference_gen
        rows = context_sweep(rt, cp, args.contexts, args.max_ctx)
        arms[name] = {"arm": name, "probe": probe,
                      "identical_to_base0": bool(identical),
                      "generation_token_ids": gen, "context_sweep": rows}
    rt.probe = None

    contexts = [str(c) for c in args.contexts]

    def p50_of(arm, ctx):
        return arms[arm]["context_sweep"][ctx]["ms"]["p50"]

    marginal, local_drift, reported = {}, {}, {}
    for ctx in contexts:
        marginal[ctx], local_drift[ctx], reported[ctx] = {}, {}, {}
        for k, probe in enumerate(PROBES):
            before, after = f"base{k}", f"base{k + 1}"
            b0, b1 = p50_of(before, ctx), p50_of(after, ctx)
            m = p50_of(probe, ctx) - 0.5 * (b0 + b1)
            marginal[ctx][probe] = m
            local_drift[ctx][probe] = abs(b1 - b0)
            reported[ctx][probe] = bool(abs(m) > abs(b1 - b0))

    deep = contexts[-1]
    global_drift = {c: abs(p50_of(f"base{len(PROBES)}", c) - p50_of("base0", c))
                    for c in contexts}
    reported_sum = {c: sum(v for p, v in marginal[c].items() if reported[c][p])
                    for c in contexts}
    identity_ok = all(a["identical_to_base0"] for a in arms.values())
    largest = {c: max(marginal[c].values()) for c in contexts}

    gates = {
        "G_S12R_C1_identity": {"required": "generation bit-identical in all arms",
                               "passed": bool(identity_ok)},
        "G_S12R_D1_local_drift": {
            "required": "a marginal is only reported as a value if it exceeds its own local drift",
            "local_drift_ms": local_drift, "reported": reported,
            "below_noise_floor": {c: [p for p in PROBES if not reported[c][p]]
                                  for c in contexts}},
        "G_S12R_S1_sanity": {
            "required": f"sum of reported marginals <= S8 MoE term {S8_MOE_TERM_MS} ms",
            "sum_ms": reported_sum, "s8_moe_term_ms": S8_MOE_TERM_MS,
            "passed": bool(reported_sum[deep] <= S8_MOE_TERM_MS)},
        "G_S12R_T1_thermal": {
            "required": "global drift |base5-base0| < largest marginal, else non-conclusive",
            "global_drift_ms": global_drift, "largest_marginal_ms": largest,
            "conclusive": {c: bool(global_drift[c] < largest[c]) for c in contexts},
            "temp_first_c": {c: arms["base0"]["context_sweep"][c]["gpu"].get("temp_c")
                             for c in contexts},
            "temp_last_c": {c: arms[f"base{len(PROBES)}"]["context_sweep"][c]["gpu"].get("temp_c")
                            for c in contexts}},
    }

    result = {
        "kind": "lightningstream_nemotron_s12r1_bracketed_attribution",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "S12_R1_BRACKETED_ATTRIBUTION",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "supersedes_schedule_of": "s12_in_loop_attribution.json",
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "contexts": args.contexts, "embed_on_host": True, "fp8_kv": True,
                   "gen_tokens": GEN_TOKENS, "prompts": PROMPTS,
                   "schedule": [n for n, _ in schedule],
                   "moe_layers": len(rt.moe_layers), "top_k": rt.top_k,
                   "cache_bytes": int(cache_bytes),
                   "replications_per_moe_layer": {
                       "up": rt.top_k, "down": rt.top_k, "accum": rt.top_k,
                       "router": 1, "shared": 1}},
        "arms": arms,
        "marginal_ms_per_token": marginal,
        "local_drift_ms": local_drift,
        "reported_above_noise": reported,
        "global_drift_ms": global_drift,
        "gates": gates,
        "method": (
            "Real decode loop plus exactly one extra invocation of one component "
            "per occurrence, output to scratch. Each probe arm is bracketed by "
            "baseline arms and measured against their mean, so a linear trend in "
            "time cancels; each probe is compared against its own local drift."),
        "claim_boundary": (
            "Marginal in-loop costs, end-to-end on this GPU at capacity 70, so "
            "absolute figures are NOT comparable to n7b and only arm-to-arm "
            "differences are. Every marginal is a LOWER BOUND: the replicated "
            "call finds part of its data warm in L2, and extra work at the end "
            "of a layer gives the copy stream more room to hide. Marginals are "
            "not shares of the MoE term and do not sum to it. Nothing here is "
            "converted to tokens per second, and the part the marginals do not "
            "cover is reported as a number and not given a name."),
    }
    (OUT_DIR / "s12r1_bracketed_attribution.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\nbracketed marginal in-loop cost, ms per token")
    for ctx in contexts:
        print(f"  ctx {ctx:>7}  base0 {p50_of('base0', ctx):7.3f} ms  "
              f"global drift {global_drift[ctx]:.3f} ms")
        for p, v in sorted(marginal[ctx].items(), key=lambda kv: -kv[1]):
            flag = "" if reported[ctx][p] else "  (below its local drift)"
            print(f"     {p:<8} {v:+7.3f}  local drift {local_drift[ctx][p]:5.3f}{flag}")
        print(f"     {'sum':<8} {reported_sum[ctx]:+7.3f}  (reported only)")
    print(f"\n  G-S12R-C1 identity : {gates['G_S12R_C1_identity']['passed']}")
    print(f"  G-S12R-S1 sanity   : {gates['G_S12R_S1_sanity']['passed']}")
    print(f"  G-S12R-T1 conclusive: {gates['G_S12R_T1_thermal']['conclusive']}")
    print("\nwritten s12r1_bracketed_attribution.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
