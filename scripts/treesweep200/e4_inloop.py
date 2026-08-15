"""E4 in-loop adoption of the v4 attention kernel (gate G-E4-T1).

Continues the open item in HANDOFF_E4_EN_VERDER_2026-08-15.md. E4 measured v4 in
isolation: bitwise identical to v1, 2.304 vs 2.803 ms/layer at t=262144. What was
never measured is what that does inside the real token loop.

Method follows s14_moe_layer_timeline.py: CUDA events timestamp stream progress
around each attention layer, so no host sync is added and the loop's overlap
stays intact. The kernel swap is a one-line monkeypatch because the wrappers have
identical signatures.

Three arms, v1 / v4 / v1, so the repeat of v1 bounds drift.
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

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "treesweep200"
ANCHOR = REPO_ROOT / "reports/lightningstream_nemotron/s5_baseline_generation.json"

GATE_T1_MS = 6.0
GATE_T1_STRETCH_MS = 4.8
GEN_TOKENS = 64
E4_ISOLATED = {"v1_ms_per_layer_262144": 2.803, "v4_ms_per_layer_262144": 2.304}


class TimedRuntime(LightningRuntime):
    """The real loop, with events bracketing each attention layer."""

    timing = False

    def alloc_events(self):
        cp = self.cp
        self._ev = [(cp.cuda.Event(), cp.cuda.Event()) for _ in self.attn_layers]
        self._ev_index = {layer: k for k, layer in enumerate(self.attn_layers)}
        self.last_attn_ms = None

    def _attention(self, i, out):
        if not self.timing:
            return super()._attention(i, out)
        e0, e1 = self._ev[self._ev_index[i]]
        e0.record()
        r = super()._attention(i, out)
        e1.record()
        return r

    def read_attn_ms(self):
        cp = self.cp
        total = 0.0
        per = []
        for e0, e1 in self._ev:
            ms = float(cp.cuda.get_elapsed_time(e0, e1))
            per.append(ms)
            total += ms
        self.last_attn_ms = total
        return total, per


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


def generate(rt, cp, tokenizer, prompts, n_tokens):
    out = []
    rt.timing = False
    for text in prompts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = rt.step(t)
        gen = [int(nxt)]
        for _ in range(n_tokens - 1):
            gen.append(int(rt.step(gen[-1])))
        cp.cuda.Device(0).synchronize()
        out.append({"prompt": text, "prompt_ids": [int(v) for v in ids],
                    "generated_ids": gen})
    return out


def sweep(rt, cp, contexts, max_ctx, samples=16):
    rng = np.random.default_rng(11)
    varied = [int(v) for v in rng.integers(1000, 60000, size=8192)]
    rows = {}
    for target in contexts:
        if target >= max_ctx - 8:
            continue
        rt.timing = False
        rt.reset()
        for j in range(min(target, 64)):
            rt.step(varied[j % len(varied)])
        rt.pos = target
        for j in range(32):
            rt.step(varied[(j + 64) % len(varied)])
        cp.cuda.Device(0).synchronize()
        rt.timing = True
        tok_ms, attn_ms, per_layer = [], [], []
        for j in range(samples):
            t0 = time.perf_counter_ns()
            rt.step(varied[(j + 96) % len(varied)])
            cp.cuda.Device(0).synchronize()
            tok_ms.append((time.perf_counter_ns() - t0) / 1e6)
            tot, per = rt.read_attn_ms()
            attn_ms.append(tot)
            per_layer.append(per)
        rt.timing = False
        rows[str(target)] = {
            "context": target,
            "token_ms": pctl(tok_ms), "attn_ms": pctl(attn_ms),
            "raw_token_ms": tok_ms, "raw_attn_ms": attn_ms,
            "per_layer_p50": [float(np.percentile([p[k] for p in per_layer], 50))
                              for k in range(len(per_layer[0]))],
        }
        print(f"    ctx {target:>6}: token p50 {rows[str(target)]['token_ms']['p50']:7.3f} ms | "
              f"attention {rows[str(target)]['attn_ms']['p50']:7.3f} ms", flush=True)
    return rows


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=72)
    ap.add_argument("--max-ctx", type=int, default=262144)
    ap.add_argument("--contexts", type=int, nargs="*", default=[0, 131072, 262100])
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    anchor = json.loads(ANCHOR.read_text(encoding="utf-8"))
    prompts = [p["prompt"] for p in anchor["prompts"]]

    rt = TimedRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                      embed_on_host=True, fp8_kv=True)
    rt.enable_cache(args.capacity)
    rt.alloc_events()
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    v1_fn = rt.k.attention_fp8_gqa
    v4_fn = rt.k.attention_fp8_gqa4
    print(f"kernels: v1={v1_fn.__name__} v4={v4_fn.__name__}", flush=True)

    arms, ref_gen = {}, None
    for name, fn in (("v1_a", v1_fn), ("v4", v4_fn), ("v1_b", v1_fn)):
        rt.k.attention_fp8_gqa = fn
        print(f"\narm {name}", flush=True)
        gen = generate(rt, cp, tokenizer, prompts, GEN_TOKENS)
        if ref_gen is None:
            ref_gen = gen
        same = [g["generated_ids"] == r["generated_ids"] for g, r in zip(gen, ref_gen)]
        print(f"  parity vs v1_a: {all(same)}", flush=True)
        per_round = [sweep(rt, cp, args.contexts, args.max_ctx)
                     for _ in range(args.rounds)]
        merged = {}
        for ctx in per_round[0]:
            tok = [v for r in per_round for v in r[ctx]["raw_token_ms"]]
            att = [v for r in per_round for v in r[ctx]["raw_attn_ms"]]
            merged[ctx] = {"context": int(ctx), "token_ms": pctl(tok),
                           "attn_ms": pctl(att), "raw_token_ms": tok,
                           "raw_attn_ms": att,
                           "per_layer_p50": per_round[0][ctx]["per_layer_p50"]}
        arms[name] = {"arm": name, "kernel": fn.__name__,
                      "generation": gen, "parity_vs_v1_a": bool(all(same)),
                      "sweep": merged}
    rt.k.attention_fp8_gqa = v1_fn

    # anchor check: the first 32 generated ids must match the frozen S5 anchor
    anchor_ok = []
    for g, a in zip(arms["v4"]["generation"], anchor["prompts"]):
        n = len(a["generated_ids"])
        anchor_ok.append(g["generated_ids"][:n] == [int(v) for v in a["generated_ids"]])
    print(f"\nanchor parity (first {len(anchor['prompts'][0]['generated_ids'])} "
          f"tokens vs s5): {all(anchor_ok)}", flush=True)

    contexts = [str(c) for c in args.contexts if c < args.max_ctx - 8]
    per_ctx = {}
    for ctx in contexts:
        a1 = arms["v1_a"]["sweep"][ctx]
        a2 = arms["v1_b"]["sweep"][ctx]
        b = arms["v4"]["sweep"][ctx]
        base_att = 0.5 * (a1["attn_ms"]["p50"] + a2["attn_ms"]["p50"])
        base_tok = 0.5 * (a1["token_ms"]["p50"] + a2["token_ms"]["p50"])
        per_ctx[ctx] = {
            "context": int(ctx),
            "attn_v1_ms": base_att, "attn_v4_ms": b["attn_ms"]["p50"],
            "attn_gain_ms": base_att - b["attn_ms"]["p50"],
            "attn_drift_ms": abs(a2["attn_ms"]["p50"] - a1["attn_ms"]["p50"]),
            "token_v1_ms": base_tok, "token_v4_ms": b["token_ms"]["p50"],
            "token_gain_ms": base_tok - b["token_ms"]["p50"],
            "token_drift_ms": abs(a2["token_ms"]["p50"] - a1["token_ms"]["p50"]),
        }
        r = per_ctx[ctx]
        print(f"  ctx {ctx:>6}: attention {base_att:6.3f} -> {r['attn_v4_ms']:6.3f} ms "
              f"({r['attn_gain_ms']:+.3f}, drift {r['attn_drift_ms']:.3f}) | "
              f"token {base_tok:7.3f} -> {r['token_v4_ms']:7.3f} ms "
              f"({r['token_gain_ms']:+.3f}, drift {r['token_drift_ms']:.3f})", flush=True)

    deep = contexts[-1]
    parity = all(a["parity_vs_v1_a"] for a in arms.values()) and all(anchor_ok)
    t1 = per_ctx[deep]["attn_v4_ms"] <= GATE_T1_MS
    gates = {
        "G_E4_T1_inloop": {
            "required_ms": GATE_T1_MS, "stretch_ms": GATE_T1_STRETCH_MS,
            "deep_context": int(deep),
            "measured_attn_ms": per_ctx[deep]["attn_v4_ms"],
            "token_parity": bool(parity),
            "passed": bool(t1 and parity),
            "stretch_passed": bool(per_ctx[deep]["attn_v4_ms"] <= GATE_T1_STRETCH_MS
                                   and parity)},
        "G_E4_T1_parity": {"arms_identical": {k: v["parity_vs_v1_a"]
                                              for k, v in arms.items()},
                           "anchor_first_n": bool(all(anchor_ok)),
                           "passed": bool(parity)},
        "drift_conclusive": {c: bool(abs(v["attn_gain_ms"]) > v["attn_drift_ms"])
                             for c, v in per_ctx.items()},
    }

    payload = {
        "kind": "treesweep200_e4_inloop_adoption",
        "registry": "TREESWEEP200",
        "phase": "E4_INLOOP",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "kernels_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/gpu_kernels.py"),
        "anchor_sha256": sha256_path(ANCHOR),
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "contexts": args.contexts, "rounds": args.rounds,
                   "gen_tokens": GEN_TOKENS, "prompts": prompts,
                   "attn_layers": len(rt.attn_layers)},
        "e4_isolated_reference": E4_ISOLATED,
        "arms": arms,
        "per_context": per_ctx,
        "anchor_parity": anchor_ok,
        "gates": gates,
        "claim_boundary": (
            "In-loop measurement: CUDA events timestamp stream progress around "
            "each of the six attention layers inside the real decode loop, so no "
            "host synchronisation is added and the loop's overlap is preserved. "
            "The attention figure is a COMPONENT of the token and is not a "
            "throughput result; the token figure beside it is end-to-end wall "
            "time on this GPU at capacity 72. v4 is bitwise identical to v1 by "
            "construction (E4 G-E4-C1), and the parity check here confirms that "
            "property survives inside the loop over 2 x 64 generated tokens and "
            "against the frozen S5 anchor. Three arms v1/v4/v1 bracket drift; a "
            "gain is only conclusive where it exceeds its own drift."),
    }
    (OUT_DIR / "E4_INLOOP_RESULTS.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\ngates")
    print(f"  parity            : {parity}")
    print(f"  G-E4-T1 (<= {GATE_T1_MS} ms): {gates['G_E4_T1_inloop']['passed']} "
          f"(measured {per_ctx[deep]['attn_v4_ms']:.3f} ms)")
    print("\nwritten E4_INLOOP_RESULTS.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
