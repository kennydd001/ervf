"""Y1/Y2: what is the host round-trip worth, and does cutting bytes buy time?

Preregistered in Y1Y2_READBACK_AND_BYTES_PREREGISTRATION_2026-08-15.md.

Y1  The routed path pays a device->host sync per MoE layer because the expert ids
    must reach the host to index the pinned bank. S14 measured 4.7 ms of GPU idle
    plus a 3.5 ms launch-bound router per token. Arm B runs the real loop, still
    computes the router on device, and takes the ids from a capture of the very
    same run instead of reading them back. Same routes by construction, so the
    output must be bit-identical and the delta is the value of the sync.

Y2  ExactFlow A/B/C/D all rest on "fewer bytes per weight -> less time". This
    times the real NVFP4 GEMV on a real expert record at 100/75/50/25% of its
    bytes, structure unchanged. It is a cost oracle, not semantics.

runtime.py is untouched; Y1's probe lives in a subclass here.
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
GIB = 1024 ** 3

PROMPTS = ["The capital of France is", "The history of computing began when"]
GEN_TOKENS = 64
S14_HOST_GAP_PLUS_ROUTE_MS = {"0": 8.641, "262100": 8.175}
BYTE_FRACTIONS = [1.0, 0.75, 0.5, 0.25]
GATE_Y2_HALF = 0.40


class ReplayRuntime(LightningRuntime):
    """Real loop; optionally takes route ids from a capture instead of a readback."""

    capture = None          # dict: layer -> list of (idx, w)
    replay = None           # same shape; when set, no device->host readback
    step_idx = 0

    def _moe_cached(self, i, out):
        if self.replay is None:
            idx, w = super()._moe_cached(i, out)
            if self.capture is not None:
                self.capture.setdefault(i, []).append(
                    (np.asarray(idx, dtype=np.int64).copy(),
                     np.asarray(w, dtype=np.float64).copy()))
            return idx, w
        return self._moe_replayed(i, out)

    def _moe_replayed(self, i, out):
        """_moe_cached with the readback removed, everything else identical."""
        cp, d = self.cp, self.layer[i]

        self._route_device(i)                 # device work stays; result unread
        out.fill(0)
        self.fused.gemv_into(self.act[:self.shared_inter], d["sh_up_c"], d["sh_up_s"],
                             self.normed, d["sh_up_g"], self.shared_inter, self.hidden,
                             apply_relu2=True)
        self.fused.gemv_into(out, d["sh_dn_c"], d["sh_dn_s"],
                             self.act[:self.shared_inter], d["sh_dn_g"],
                             self.hidden, self.shared_inter)

        idx, w = self.replay[i][self.step_idx]

        bank, c = self.bank[i], self.cache[i]
        cmap, cap = c["map"], c["cap"]
        slots, needs_wait = [], []
        with self.copy_stream:
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

        order = [s for s in range(len(idx)) if not needs_wait[s]]
        order += [s for s in range(len(idx)) if needs_wait[s]]
        for s in order:
            e = int(idx[s])
            if needs_wait[s]:
                cp.cuda.get_current_stream().wait_event(self.evt[s])
            sl = slots[s]
            self.fused.gemv_into(self.act[:self.moe_inter],
                                 c["codes"][sl * UP_CODE:(sl + 1) * UP_CODE],
                                 c["scales"][sl * UP_SCALE:(sl + 1) * UP_SCALE],
                                 self.normed, float(bank["globals"][e, 1]),
                                 self.moe_inter, self.hidden, apply_relu2=True)
            self.fused.down_masked_into(
                self.tmp, bank["down_base_ptr"] + e * DOWN_PANEL_BYTES,
                self.act[:self.moe_inter], self.mstate,
                float(bank["globals"][e, 0]), self.hidden, self.moe_inter)
            self.fused.accumulate_into(out, self.tmp, float(w[s]), self.hidden)
        return np.asarray(idx), np.asarray(w)

    def step(self, token_id: int, capture_routes=None) -> int:
        out = super().step(token_id, capture_routes)
        self.step_idx += 1
        return out


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


def generate(rt, cp, tokenizer, replay_map=None):
    out = []
    for text in PROMPTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        rt.reset()
        rt.step_idx = 0
        rt.replay = replay_map.pop(0) if replay_map is not None else None
        nxt = None
        for t in ids:
            nxt = rt.step(t)
        gen = [int(nxt)]
        for _ in range(GEN_TOKENS - 1):
            gen.append(int(rt.step(gen[-1])))
        cp.cuda.Device(0).synchronize()
        out.append(gen)
    rt.replay = None
    return out


def capture_generation(rt, cp, tokenizer):
    maps = []
    for text in PROMPTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        rt.reset()
        rt.step_idx = 0
        rt.capture = {}
        nxt = None
        for t in ids:
            nxt = rt.step(t)
        cur = int(nxt)
        for _ in range(GEN_TOKENS - 1):
            cur = int(rt.step(cur))
        cp.cuda.Device(0).synchronize()
        maps.append(rt.capture)
        rt.capture = None
    return maps


def sweep_arm(rt, cp, target, max_ctx, replay=None, capture=False):
    """n7b protocol: prime, jump, warm, then 16 timed steps."""
    rng = np.random.default_rng(11)
    varied = [int(v) for v in rng.integers(1000, 60000, size=4096)]
    rt.reset()
    rt.step_idx = 0
    rt.capture = {} if capture else None
    rt.replay = None
    for j in range(min(target, 64)):
        rt.step(varied[j % len(varied)])
    rt.pos = target
    for j in range(32):
        rt.step(varied[(j + 64) % len(varied)])
    cp.cuda.Device(0).synchronize()
    if replay is not None:
        rt.replay = replay
        rt.step_idx = 0
    base = rt.step_idx
    samples = []
    for j in range(16):
        t0 = time.perf_counter_ns()
        rt.step(varied[(j + 96) % len(varied)])
        cp.cuda.Device(0).synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    cap = rt.capture
    rt.capture, rt.replay = None, None
    if capture:
        # keep only the 16 timed steps, re-indexed from 0
        return samples, {i: v[base:base + 16] for i, v in cap.items()}
    return samples, None


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=72)
    ap.add_argument("--max-ctx", type=int, default=262144)
    ap.add_argument("--contexts", type=int, nargs="*", default=[0, 262100])
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    free0, total = cp.cuda.runtime.memGetInfo()
    rt = ReplayRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                       embed_on_host=True, fp8_kv=True)
    rt.enable_cache(args.capacity)
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    # ------------------------------------------------------------------- Y2
    print("\nY2: time as a function of record bytes", loop_flush := True, flush=True)
    layer0 = rt.moe_layers[0]
    bank = rt.bank[layer0]
    hidden, inter = rt.hidden, rt.moe_inter
    full_codes = cp.asarray(bank["up_codes"][:UP_CODE])
    full_scales = cp.asarray(bank["up_scales"][:UP_SCALE])
    x = cp.asarray(np.random.default_rng(3).standard_normal(hidden).astype(np.float32))
    outv = cp.zeros(inter, dtype=cp.float32)
    gs = float(bank["globals"][0, 1])
    y2 = {}
    for frac in BYTE_FRACTIONS:
        cols = int(hidden * frac) // 16 * 16
        codes = cp.ascontiguousarray(
            full_codes.reshape(inter, hidden // 2)[:, :cols // 2]).reshape(-1)
        scales = cp.ascontiguousarray(
            full_scales.reshape(inter, hidden // 16)[:, :cols // 16]).reshape(-1)
        xs = cp.ascontiguousarray(x[:cols])
        for _ in range(5):
            rt.fused.gemv_into(outv, codes, scales, xs, gs, inter, cols)
        cp.cuda.Device(0).synchronize()
        s = []
        for _ in range(200):
            t0 = time.perf_counter_ns()
            rt.fused.gemv_into(outv, codes, scales, xs, gs, inter, cols)
            cp.cuda.Device(0).synchronize()
            s.append((time.perf_counter_ns() - t0) / 1e3)      # microseconds
        nbytes = int(codes.nbytes + scales.nbytes)
        st = pctl(s)
        y2[f"{frac:.2f}"] = {"fraction": frac, "cols": cols, "bytes": nbytes,
                             "us": st, "gb_s": nbytes / (st["p50"] * 1e-6) / 1e9}
        print(f"  {frac:>5.0%} cols={cols:>5} bytes={nbytes:>9,} "
              f"p50={st['p50']:8.2f} us  {y2[f'{frac:.2f}']['gb_s']:6.1f} GB/s",
              flush=True)

    full_us = y2["1.00"]["us"]["p50"]
    half_us = y2["0.50"]["us"]["p50"]
    y2_saving = 1.0 - half_us / full_us
    print(f"  halving the bytes saves {y2_saving * 100:.1f}% "
          f"(gate {GATE_Y2_HALF * 100:.0f}%)", flush=True)

    # ------------------------------------------------------------------- Y1
    print("\nY1: identity of the replayed-route path", flush=True)
    ref_gen = generate(rt, cp, tokenizer)
    cap_maps = capture_generation(rt, cp, tokenizer)
    rep_gen = generate(rt, cp, tokenizer, replay_map=list(cap_maps))
    identical = ref_gen == rep_gen
    print(f"  bit-identical: {identical}  {tokenizer.decode(ref_gen[0][:10])!r}",
          flush=True)

    y1 = {}
    if identical:
        for ctx in args.contexts:
            if ctx >= args.max_ctx - 8:
                continue
            _, routes = sweep_arm(rt, cp, ctx, args.max_ctx, capture=True)
            a1, b_, a2 = [], [], []
            for _ in range(args.rounds):
                s, _ = sweep_arm(rt, cp, ctx, args.max_ctx)
                a1 += s
                s, _ = sweep_arm(rt, cp, ctx, args.max_ctx, replay=routes)
                b_ += s
                s, _ = sweep_arm(rt, cp, ctx, args.max_ctx)
                a2 += s
            sa1, sb, sa2 = pctl(a1), pctl(b_), pctl(a2)
            base = 0.5 * (sa1["p50"] + sa2["p50"])
            drift = abs(sa2["p50"] - sa1["p50"])
            gain = base - sb["p50"]
            y1[str(ctx)] = {
                "context": ctx, "base_p50_ms": base, "replay_p50_ms": sb["p50"],
                "local_drift_ms": drift, "gain_ms": gain,
                "gain_relative": gain / base,
                "conclusive": bool(abs(gain) > drift),
                "s14_host_gap_plus_route_ms": S14_HOST_GAP_PLUS_ROUTE_MS.get(str(ctx)),
                "within_s14_bound": bool(
                    gain <= S14_HOST_GAP_PLUS_ROUTE_MS.get(str(ctx), 1e9) + drift),
                "arms": {"base1": sa1, "replay": sb, "base2": sa2},
            }
            print(f"  ctx {ctx:>6}: base {base:7.3f} ms | no-readback "
                  f"{sb['p50']:7.3f} ms | gain {gain:+.3f} ms "
                  f"({gain / base * 100:+.1f}%) | drift {drift:.3f}", flush=True)

    gates = {
        "G_Y1_C1_identity": {"required": "replayed-route generation bit-identical",
                             "passed": bool(identical)},
        "G_Y1_P1_gain": {c: {"gain_ms": v["gain_ms"], "conclusive": v["conclusive"]}
                         for c, v in y1.items()},
        "G_Y1_S1_within_s14_bound": {c: v["within_s14_bound"] for c, v in y1.items()},
        "G_Y2_1_halving_saves_40pct": {
            "required": GATE_Y2_HALF, "measured": y2_saving,
            "passed": bool(y2_saving >= GATE_Y2_HALF)},
        "G_Y2_2_bandwidth": {k: v["gb_s"] for k, v in y2.items()},
    }

    payload = {
        "kind": "lightningstream_nemotron_y1y2_readback_and_bytes",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "Y1_Y2_READBACK_AND_BYTES",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "contexts": args.contexts, "rounds": args.rounds,
                   "gen_tokens": GEN_TOKENS, "prompts": PROMPTS,
                   "byte_fractions": BYTE_FRACTIONS},
        "y2_bytes_vs_time": y2,
        "y2_halving_saving": y2_saving,
        "y1_identity": {"bit_identical": bool(identical),
                        "reference_generation": ref_gen,
                        "replayed_generation": rep_gen},
        "y1_contexts": y1,
        "gates": gates,
        "claim_boundary": (
            "Y1 is an ORACLE, not an implementation: arm B still does the router's "
            "device work and still does the LRU bookkeeping on the host, but takes "
            "the expert ids from a capture of the same run instead of reading them "
            "back, so what it measures is the value of removing the per-layer "
            "device->host synchronisation. It is not a claim that a device-side "
            "router would deliver exactly this. Y2 varies the RECORD SIZE of a real "
            "NVFP4 up_proj with the structure held fixed; it is a cost oracle for "
            "byte-reduction schemes and is not a quantizer, not semantics, and not "
            "a quality claim. Y1 figures are end-to-end token times on this GPU at "
            "capacity 72; Y2 figures are single-kernel microbenchmarks and are not "
            "converted to tokens per second."),
    }
    (OUT_DIR / "y1y2_readback_and_bytes.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\ngates")
    print(f"  G-Y1-C1 identity: {identical}")
    for c, v in y1.items():
        print(f"  G-Y1-P1 ctx {c}: {v['gain_ms']:+.3f} ms, conclusive={v['conclusive']}, "
              f"within S14 bound={v['within_s14_bound']}")
    print(f"  G-Y2-1 halving saves {y2_saving * 100:.1f}% "
          f"(gate {GATE_Y2_HALF * 100:.0f}%): {gates['G_Y2_1_halving_saves_40pct']['passed']}")
    print("\nwritten y1y2_readback_and_bytes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
