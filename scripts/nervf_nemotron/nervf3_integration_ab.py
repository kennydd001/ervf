"""NERVF-3: ERVF inside the real runtime. A/B on one variable.

Preregistered gates (frozen here, before the run):
  G-NERVF-3C  generation bit-identical to the frozen V35 anchor and between arms
  G-NERVF-3P  primary: >= 1.35x on the MoE up-projection component in the loop
  G-NERVF-3T  token-level gain must exceed its own bracketed drift to be reported

Three arms base/ervf/base against one model load. The only variable is
`rt.fused.use_ervf`; every kernel argument, route, cache state and stream is
unchanged. ERVF replaces gemv_nvfp4_rows wherever it is called: the routed
expert up-projection, both shared-expert projections, the Mamba NVFP4
projections if present, and the LM head.
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
OUT_DIR = REPO_ROOT / "reports" / "nervf_nemotron"
ANCHOR = REPO_ROOT / "reports/treesweep200/V35_GENERATION_ANCHOR.json"
GATE_PRIMARY = 1.35
GEN_TOKENS = 64


class TimedRuntime(LightningRuntime):
    """Events around the routed-expert up-projection block of every MoE layer."""

    timing = False

    def alloc_events(self):
        cp = self.cp
        self._ev = [(cp.cuda.Event(), cp.cuda.Event()) for _ in self.moe_layers]
        self._ix = {l: k for k, l in enumerate(self.moe_layers)}

    def _moe_cached(self, i, out):
        if not self.timing:
            return super()._moe_cached(i, out)
        e0, e1 = self._ev[self._ix[i]]
        e0.record()
        r = super()._moe_cached(i, out)
        e1.record()
        return r

    def read_moe_ms(self):
        cp = self.cp
        return sum(float(cp.cuda.get_elapsed_time(a, b)) for a, b in self._ev)


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def pctl(v):
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}


def generate(rt, cp, tokenizer, prompts, n):
    rt.timing = False
    out = []
    for text in prompts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = rt.step(t)
        gen = [int(nxt)]
        for _ in range(n - 1):
            gen.append(int(rt.step(gen[-1])))
        cp.cuda.Device(0).synchronize()
        out.append({"prompt": text, "generated_ids": gen})
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
        tok, moe = [], []
        for j in range(samples):
            t0 = time.perf_counter_ns()
            rt.step(varied[(j + 96) % len(varied)])
            cp.cuda.Device(0).synchronize()
            tok.append((time.perf_counter_ns() - t0) / 1e6)
            moe.append(rt.read_moe_ms())
        rt.timing = False
        rows[str(target)] = {"context": target, "token_ms": pctl(tok),
                             "moe_ms": pctl(moe), "raw_token_ms": tok,
                             "raw_moe_ms": moe}
        print(f"    ctx {target:>6}: token {pctl(tok)['p50']:7.3f} ms | "
              f"MoE {pctl(moe)['p50']:7.3f} ms", flush=True)
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

    arms, ref = {}, None
    for name, use in (("base_a", False), ("ervf", True), ("base_b", False)):
        rt.fused.use_ervf = use
        print(f"\narm {name}: use_ervf={use}", flush=True)
        gen = generate(rt, cp, tokenizer, prompts, GEN_TOKENS)
        if ref is None:
            ref = gen
        same_arm = all(g["generated_ids"] == r["generated_ids"]
                       for g, r in zip(gen, ref))
        same_anchor = all(g["generated_ids"] == [int(v) for v in a["generated_ids"]]
                          for g, a in zip(gen, anchor["prompts"]))
        print(f"  parity: arm={same_arm} anchor={same_anchor}", flush=True)
        per_round = [sweep(rt, cp, args.contexts, args.max_ctx)
                     for _ in range(args.rounds)]
        merged = {}
        for ctx in per_round[0]:
            tk = [v for r in per_round for v in r[ctx]["raw_token_ms"]]
            mo = [v for r in per_round for v in r[ctx]["raw_moe_ms"]]
            merged[ctx] = {"context": int(ctx), "token_ms": pctl(tk),
                           "moe_ms": pctl(mo), "raw_token_ms": tk, "raw_moe_ms": mo}
        arms[name] = {"arm": name, "use_ervf": use, "generation": gen,
                      "parity_arm": bool(same_arm), "parity_anchor": bool(same_anchor),
                      "sweep": merged}
    rt.fused.use_ervf = False

    contexts = [str(c) for c in args.contexts if c < args.max_ctx - 8]
    per_ctx = {}
    for ctx in contexts:
        a1 = arms["base_a"]["sweep"][ctx]
        a2 = arms["base_b"]["sweep"][ctx]
        b = arms["ervf"]["sweep"][ctx]
        bm = 0.5 * (a1["moe_ms"]["p50"] + a2["moe_ms"]["p50"])
        bt = 0.5 * (a1["token_ms"]["p50"] + a2["token_ms"]["p50"])
        per_ctx[ctx] = {
            "context": int(ctx),
            "moe_base_ms": bm, "moe_ervf_ms": b["moe_ms"]["p50"],
            "moe_gain_ms": bm - b["moe_ms"]["p50"],
            "moe_speedup": bm / b["moe_ms"]["p50"],
            "moe_drift_ms": abs(a2["moe_ms"]["p50"] - a1["moe_ms"]["p50"]),
            "token_base_ms": bt, "token_ervf_ms": b["token_ms"]["p50"],
            "token_gain_ms": bt - b["token_ms"]["p50"],
            "token_drift_ms": abs(a2["token_ms"]["p50"] - a1["token_ms"]["p50"]),
        }
        r = per_ctx[ctx]
        print(f"  ctx {ctx:>6}: MoE {bm:7.3f} -> {r['moe_ervf_ms']:7.3f} ms "
              f"({r['moe_speedup']:.3f}x, drift {r['moe_drift_ms']:.3f}) | "
              f"token {bt:7.3f} -> {r['token_ervf_ms']:7.3f} "
              f"({r['token_gain_ms']:+.3f}, drift {r['token_drift_ms']:.3f})", flush=True)

    parity = all(a["parity_arm"] and a["parity_anchor"] for a in arms.values())
    deep = contexts[-1]
    best_sp = max(v["moe_speedup"] for v in per_ctx.values())
    gates = {
        "G_NERVF_3C_exact": {"arms": {k: v["parity_arm"] for k, v in arms.items()},
                             "anchor": {k: v["parity_anchor"] for k, v in arms.items()},
                             "passed": bool(parity)},
        "G_NERVF_3P_primary": {"required": GATE_PRIMARY,
                               "measured_best_moe_speedup": best_sp,
                               "per_context": {c: v["moe_speedup"]
                                               for c, v in per_ctx.items()},
                               "passed": bool(parity and best_sp >= GATE_PRIMARY)},
        "G_NERVF_3T_token": {c: {"gain_ms": v["token_gain_ms"],
                                 "conclusive": bool(abs(v["token_gain_ms"])
                                                    > v["token_drift_ms"])}
                             for c, v in per_ctx.items()},
    }

    payload = {
        "kind": "nervf_nemotron_integration_ab",
        "namespace": "NERVF_NEMOTRON", "phase": "NERVF_3",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "fused_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py"),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "anchor_sha256": sha256_path(ANCHOR),
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "contexts": args.contexts, "rounds": args.rounds,
                   "gen_tokens": GEN_TOKENS, "ervf_width": 16,
                   "moe_layers": len(rt.moe_layers)},
        "arms": arms, "per_context": per_ctx, "gates": gates,
        "claim_boundary": (
            "In-loop A/B on this GPU at capacity 72. CUDA events bracket the "
            "whole routed-expert block of every MoE layer, so the MoE figure is "
            "a COMPONENT that also contains the router, the shared expert and "
            "the down path -- ERVF only replaces the NVFP4 row-GEMV inside it, "
            "so the component speedup is diluted by design and is NOT the "
            "projection-plane speedup of NERVF-2. The token figure is end-to-end "
            "wall time. Exactness is a hard gate: generation must be identical "
            "both between arms and against the frozen V35 anchor. Three arms "
            "bracket drift; a token gain counts only where it exceeds its own "
            "drift. Not to be added to attention-v4, graph or gatherless gains."),
    }
    (OUT_DIR / "nervf3_integration_ab.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n  G-NERVF-3C exact : {parity}")
    print(f"  G-NERVF-3P (>={GATE_PRIMARY}x): "
          f"{gates['G_NERVF_3P_primary']['passed']} (best {best_sp:.3f}x)")
    print("\nwritten nervf3_integration_ab.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
