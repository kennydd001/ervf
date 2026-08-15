"""S14: GPU-event timeline of the MoE layer inside the real decode loop.

Preregistration: S14_MOE_LAYER_TIMELINE_PREREGISTRATION_2026-08-15.md (frozen
before this run).  S12's marginal method cannot see compute-stream waiting on
the copy stream, host time between launches, or in-loop kernel slowdown -- the
~24 ms of the 39.5 ms MoE term that has no name.  This phase timestamps every
stage with CUDA events (no host sync is added; the one sync per measured token
happens after the token) so every microsecond of each MoE layer lands in
exactly one labelled stream segment.

runtime.py is not touched: the instrumentation is a subclass in this file.

Gates:
  G-S14-C1  instrumented generation bit-identical to uninstrumented
  G-S14-P1  probe overhead reported; > 20% of baseline p50 -> inconclusive
  G-S14-S1  per-context segment sum <= probed token p50, >= half the S8 MoE
            term at 262100, no negative segments
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
    LightningRuntime, cp_asnumpy, UP_CODE, UP_SCALE, DOWN_PANEL_BYTES)

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning_v35")
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GIB = 1024 ** 3
PROMPTS = ["The capital of France is", "The history of computing began when"]
GEN_TOKENS = 32
S8_MOE_TERM_262K = 39.523

SEGMENTS = ["route", "shared_up", "shared_dn", "host_gap",
            "up", "down_masked", "accum", "layer_total"]


class TimelineRuntime(LightningRuntime):
    """The real runtime; _moe_cached optionally wrapped in timing events."""

    instrument = False

    def tl_reset(self):
        self.tl = []  # per MoE layer: (events, copy events, readback ms, waits)

    def _moe_cached(self, i, out):
        if not self.instrument:
            return super()._moe_cached(i, out)
        cp, d = self.cp, self.layer[i]
        ev = [cp.cuda.Event() for _ in range(5 + 3 * self.top_k)]

        ev[0].record()
        packed = self._route_device(i)
        ev[1].record()
        out.fill(0)
        self.fused.gemv_into(self.act[:self.shared_inter], d["sh_up_c"], d["sh_up_s"],
                             self.normed, d["sh_up_g"], self.shared_inter, self.hidden,
                             apply_relu2=True)
        ev[2].record()
        self.fused.gemv_into(out, d["sh_dn_c"], d["sh_dn_s"],
                             self.act[:self.shared_inter], d["sh_dn_g"],
                             self.hidden, self.shared_inter)
        ev[3].record()

        t0 = time.perf_counter_ns()
        host = cp_asnumpy(cp, packed)
        readback_ms = (time.perf_counter_ns() - t0) / 1e6
        idx = host[: self.top_k].astype(int)
        w = host[self.top_k:].astype(np.float64)

        bank, c = self.bank[i], self.cache[i]
        cmap, cap = c["map"], c["cap"]
        assert getattr(self, "cache_mode", "up_only") == "up_only"

        evc0, evc1 = cp.cuda.Event(), cp.cuda.Event()
        slots, needs_wait = [], []
        with self.copy_stream:
            evc0.record(self.copy_stream)
            for s, e in enumerate(idx):
                e = int(e)
                if e in cmap:
                    cmap.move_to_end(e)
                    slots.append(cmap[e])
                    needs_wait.append(False)
                    self.cache_stats["hits"] += 1
                    continue
                self.cache_stats["misses"] += 1
                if len(cmap) < cap:
                    slot = len(cmap)
                else:
                    _, slot = cmap.popitem(last=False)
                cmap[e] = slot
                slots.append(slot)
                needs_wait.append(True)
                c["codes"][slot * UP_CODE:(slot + 1) * UP_CODE].set(
                    bank["up_codes"][e * UP_CODE:(e + 1) * UP_CODE],
                    stream=self.copy_stream)
                c["scales"][slot * UP_SCALE:(slot + 1) * UP_SCALE].set(
                    bank["up_scales"][e * UP_SCALE:(e + 1) * UP_SCALE],
                    stream=self.copy_stream)
                self.evt[s].record(self.copy_stream)
            evc1.record(self.copy_stream)

        order = [s for s in range(len(idx)) if not needs_wait[s]]
        order += [s for s in range(len(idx)) if needs_wait[s]]
        ev[4].record()

        k = 5
        waits_exec = []
        for s in order:
            e = idx[s]
            if needs_wait[s]:
                cp.cuda.get_current_stream().wait_event(self.evt[s])
            waits_exec.append(bool(needs_wait[s]))
            sl = slots[s]
            self.fused.gemv_into(self.act[:self.moe_inter],
                                 c["codes"][sl * UP_CODE:(sl + 1) * UP_CODE],
                                 c["scales"][sl * UP_SCALE:(sl + 1) * UP_SCALE],
                                 self.normed, float(bank["globals"][e, 1]),
                                 self.moe_inter, self.hidden, apply_relu2=True)
            ev[k].record(); k += 1
            self.fused.down_masked_into(
                self.tmp, bank["down_base_ptr"] + int(e) * DOWN_PANEL_BYTES,
                self.act[:self.moe_inter], self.mstate,
                float(bank["globals"][e, 0]), self.hidden, self.moe_inter)
            ev[k].record(); k += 1
            self.fused.accumulate_into(out, self.tmp, float(w[s]), self.hidden)
            ev[k].record(); k += 1

        self.tl.append({"layer": i, "ev": ev, "evc": (evc0, evc1),
                        "readback_ms": readback_ms, "waits": waits_exec})
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


def harvest_token(rt, cp):
    """Read all event pairs after one sync; returns per-layer segment dict."""
    cp.cuda.Device(0).synchronize()
    layers = []
    for rec in rt.tl:
        ev = rec["ev"]
        el = lambda a, b: float(cp.cuda.get_elapsed_time(ev[a], ev[b]))
        segs = {"route": el(0, 1), "shared_up": el(1, 2), "shared_dn": el(2, 3),
                "host_gap": el(3, 4)}
        up = dn = ac = 0.0
        up_wait = dn_wait = 0.0
        n_wait = 0
        for j, waited in enumerate(rec["waits"]):
            b = 5 + 3 * j
            u, dd, aa = el(b - 1, b), el(b, b + 1), el(b + 1, b + 2)
            up += u; dn += dd; ac += aa
            if waited:
                up_wait += u; dn_wait += dd; n_wait += 1
        segs.update({"up": up, "down_masked": dn, "accum": ac,
                     "up_waited": up_wait, "down_waited": dn_wait,
                     "experts_waited": n_wait,
                     "miss_copy_batch": float(cp.cuda.get_elapsed_time(*rec["evc"])),
                     "readback_host": rec["readback_ms"]})
        segs["layer_total"] = el(0, 5 + 3 * len(rec["waits"]) - 1)
        layers.append(segs)
    return layers


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
        samples, tokens = [], []
        for j in range(16):
            rt.tl_reset()
            t0 = time.perf_counter_ns()
            rt.step(varied[(j + 96) % len(varied)])
            cp.cuda.Device(0).synchronize()
            ms = (time.perf_counter_ns() - t0) / 1e6
            layers = harvest_token(rt, cp) if rt.instrument else []
            samples.append(ms)
            if layers:
                tokens.append(layers)
        row = {"context": target, "ms": pct(samples), "raw_ms": samples,
               "gpu": gpu_state()}
        if tokens:
            # per-stage sums over the 23 MoE layers, per token
            stage_sums = []
            for layers in tokens:
                stage_sums.append({s: sum(l[s] for l in layers) for s in
                                   ["route", "shared_up", "shared_dn", "host_gap",
                                    "up", "down_masked", "accum", "layer_total",
                                    "up_waited", "down_waited", "miss_copy_batch",
                                    "readback_host"]}
                                  | {"experts_waited": sum(l["experts_waited"]
                                                           for l in layers)})
            row["token_stage_sums"] = stage_sums
        rows[str(target)] = row
        print(f"    ctx {target:>6}: p50 {pct(samples)['p50']:7.3f} ms  "
              f"{row['gpu'].get('temp_c', '?')}C", flush=True)
    return rows


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=72)
    ap.add_argument("--max-ctx", type=int, default=262144)
    ap.add_argument("--contexts", type=int, nargs="*", default=[0, 262100])
    args = ap.parse_args()

    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        foreign = [l for l in o.stdout.strip().splitlines()
                   if l.strip() and int(l.split(",")[0]) != os.getpid()]
    except Exception:
        foreign = ["query failed"]
    if foreign:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    rt = TimelineRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                         embed_on_host=True, fp8_kv=True)
    cache_bytes = rt.enable_cache(args.capacity)
    rt.load_routed_bank()
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    arms = {}
    gens = {}
    for name in ["base0", "probed", "base1"]:
        rt.instrument = name == "probed"
        rt.cache_stats = {"hits": 0, "misses": 0}
        gens[name] = generate(rt, cp, tok)
        rows = context_sweep(rt, cp, args.contexts, args.max_ctx)
        arms[name] = {"context_sweep": rows,
                      "cache": dict(rt.cache_stats)}
        print(f"arm {name}: done", flush=True)
    rt.instrument = False

    c1_pass = gens["probed"] == gens["base0"] == gens["base1"]

    # ---- gates per context
    gates = {"G_S14_C1": {"requirement": "instrumented == uninstrumented, "
                                         "2 prompts x 32 tokens",
                          "pass": c1_pass}}
    contexts = [str(c) for c in args.contexts if c < args.max_ctx - 8]
    for ctx in contexts:
        b0 = arms["base0"]["context_sweep"][ctx]["ms"]["p50"]
        b1 = arms["base1"]["context_sweep"][ctx]["ms"]["p50"]
        pr = arms["probed"]["context_sweep"][ctx]["ms"]["p50"]
        overhead = pr - 0.5 * (b0 + b1)
        gates[f"G_S14_P1_ctx{ctx}"] = {
            "requirement": "probe overhead <= 20% of baseline p50",
            "base0_p50": b0, "base1_p50": b1, "probed_p50": pr,
            "overhead_ms": overhead,
            "overhead_frac": overhead / (0.5 * (b0 + b1)),
            "conclusive": overhead <= 0.20 * 0.5 * (b0 + b1)}
        sums = [t["layer_total"] for t in
                arms["probed"]["context_sweep"][ctx]["token_stage_sums"]]
        mean_sum = float(np.mean(sums))
        seg_nonneg = all(t[s] >= -0.01 for t in
                         arms["probed"]["context_sweep"][ctx]["token_stage_sums"]
                         for s in SEGMENTS)
        ok = mean_sum <= pr and seg_nonneg
        if ctx == str(262100):
            ok = ok and mean_sum >= S8_MOE_TERM_262K / 2
        gates[f"G_S14_S1_ctx{ctx}"] = {
            "requirement": "segment sum <= probed token p50, segments >= 0"
                           + (", >= half the S8 MoE term" if ctx == str(262100) else ""),
            "mean_layer_sum_ms": mean_sum, "probed_p50_ms": pr,
            "segments_nonnegative": seg_nonneg, "pass": ok}

    payload = {
        "kind": "lightningstream_nemotron_s14_moe_layer_timeline",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "S14_MOE_LAYER_TIMELINE",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src" / "moe_lab" /
                                      "lightningstream_nemotron" / "runtime.py"),
        "config": {"capacity_per_layer": args.capacity, "max_ctx": args.max_ctx,
                   "contexts": [int(c) for c in contexts],
                   "embed_on_host": True, "backbone_fp8_kv": True,
                   "samples_per_context": 16, "varied_seed": 11,
                   "cache_mode": "up_only",
                   "arm_order": ["base0", "probed", "base1"]},
        "cache_gib": cache_bytes / GIB,
        "generation_identical": c1_pass,
        "arms": arms,
        "gates": gates,
        "gpu": gpu_state(),
        "claim_boundary": (
            "CUDA-event timeline of the real decode loop on this GPU at "
            "capacity 72. Events timestamp stream progress; no host sync is "
            "added inside a token. Stream segments are wall time on the stream "
            "and contain both work and event-waits by design; miss_copy_batch "
            "is measured on the copy stream and is NOT part of the layer "
            "totals. The 262100 context uses the established pos-jump protocol "
            "(64 real tokens, pos set, 32 warm, 16 measured). Attribution only: "
            "nothing is built, nothing is converted to tokens per second, no "
            "quality claim, no statement about other hardware."),
    }
    out = OUT_DIR / "s14_moe_layer_timeline.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for name, g in payload["gates"].items():
        verdict = g.get("pass", g.get("conclusive"))
        print(f"  {name}: {'PASS' if verdict else 'FAIL'}")
    return 0 if c1_pass else 2


if __name__ == "__main__":
    sys.exit(main())
