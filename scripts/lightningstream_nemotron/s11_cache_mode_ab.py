"""S11: up-only cache at full capacity vs. full-record cache at half capacity.

Preregistered in S11_FULL_RECORD_CACHE_PREREGISTRATION_2026-08-15.md.

Both arms get exactly the same number of cache BYTES -- a full-record slot is
exactly twice an up-only slot, so capacity 36 and capacity 72 allocate the same
4.328 GiB. The only variable is what a slot holds.

Three arms in one process, A1 / B / A2, so that the repeat of A bounds drift:
anything that moves between A1 and A2 is noise, not the effect of B.
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

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GIB = 1024 ** 3

# Same two prompts the S5 identity gate used.
PROMPTS = ["The capital of France is", "The history of computing began when"]
GEN_TOKENS = 64
ADOPT_MARGIN = 0.03          # G-S11-P1: 3% at ctx 262100
ARMS = [("A1", "up_only", 72), ("B", "full", 36), ("A2", "up_only", 72)]


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
                            "--query-gpu=memory.used,memory.free,temperature.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        u, f, t = [x.strip() for x in o.stdout.strip().split(",")]
        return {"used_mib": int(u), "free_mib": int(f), "temp_c": int(t)}
    except Exception as e:                                   # pragma: no cover
        return {"error": str(e)}


def generate(rt, cp, tokenizer):
    """Greedy continuations, as token ids, for the identity gate."""
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
    """Identical warm-up and sampling protocol to n7b_cached_decode.py."""
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
                             "tok_s_p50": 1000.0 / s["p50"], "gpu": gpu_state()}
        print(f"    ctx {target:>6}: p50 {s['p50']:7.2f} ms -> "
              f"{1000 / s['p50']:6.3f} tok/s", flush=True)
    return rows


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ctx", type=int, default=262144)
    ap.add_argument("--contexts", type=int, nargs="*", default=[0, 131072, 262100])
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
    rt = LightningRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                          embed_on_host=True, fp8_kv=True)
    free_shell, _ = cp.cuda.runtime.memGetInfo()
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    print(f"shell {(free0 - free_shell) / GIB:.3f} GiB", flush=True)

    arms, reference_gen = {}, None
    for name, mode, cap in ARMS:
        rt.cache = None
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cache_bytes = rt.enable_cache(cap, mode)
        free_now, _ = cp.cuda.runtime.memGetInfo()
        print(f"\narm {name}: mode={mode} capacity={cap} "
              f"cache {cache_bytes / GIB:.3f} GiB free {free_now / GIB:.3f} GiB",
              flush=True)

        rt.cache_stats = {"hits": 0, "misses": 0}
        gen = generate(rt, cp, tokenizer)
        if reference_gen is None:
            reference_gen = gen
        identical = gen == reference_gen
        gen_hits, gen_misses = rt.cache_stats["hits"], rt.cache_stats["misses"]
        print(f"  identity vs A1: {identical}  "
              f"first: {tokenizer.decode(gen[0][:12])!r}", flush=True)

        rt.cache_stats = {"hits": 0, "misses": 0}
        rows = context_sweep(rt, cp, args.contexts, args.max_ctx)
        hits, misses = rt.cache_stats["hits"], rt.cache_stats["misses"]

        arms[name] = {
            "arm": name, "mode": mode, "capacity": cap,
            "cache_bytes": int(cache_bytes), "cache_gib": cache_bytes / GIB,
            "device_free_bytes": int(free_now),
            "generation_token_ids": gen,
            "generation_text": [tokenizer.decode(g) for g in gen],
            "identical_to_A1": bool(identical),
            "generation_cache": {"hits": gen_hits, "misses": gen_misses,
                                 "hit_rate": gen_hits / max(1, gen_hits + gen_misses)},
            "sweep_cache": {"hits": hits, "misses": misses,
                            "hit_rate": hits / max(1, hits + misses)},
            "context_sweep": rows,
        }

    def p50(arm, ctx):
        return arms[arm]["context_sweep"][str(ctx)]["ms"]["p50"]

    deep = str(args.contexts[-1])
    a1, b, a2 = (1000.0 / p50("A1", deep), 1000.0 / p50("B", deep),
                 1000.0 / p50("A2", deep))
    drift = abs(a2 - a1)
    effect = b - a1
    ctx0_ok = (1000.0 / p50("B", args.contexts[0])
               >= 1000.0 / p50("A1", args.contexts[0]))
    identity_ok = all(arms[n]["identical_to_A1"] for n, _, _ in ARMS)

    gates = {
        "G_S11_C1_bit_identical": {
            "required": "arm B generation identical to arm A over 2 x 64 tokens",
            "passed": bool(arms["B"]["identical_to_A1"]),
            "all_arms_identical": bool(identity_ok)},
        "G_S11_P1_adopt": {
            "required_relative_gain_at_deep": ADOPT_MARGIN,
            "deep_context": int(deep),
            "A1_tok_s": a1, "B_tok_s": b,
            "relative_gain": (b - a1) / a1,
            "no_regression_at_ctx0": bool(ctx0_ok),
            "passed": bool((b - a1) / a1 >= ADOPT_MARGIN and ctx0_ok)},
        "G_S11_D1_drift": {
            "required": "|A2 - A1| < |B - A1| at the deepest context",
            "A2_tok_s": a2, "drift_tok_s": drift, "effect_tok_s": effect,
            "conclusive": bool(drift < abs(effect))},
    }

    result = {
        "kind": "lightningstream_nemotron_s11_cache_mode_ab",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "S11_FULL_RECORD_CACHE",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"max_ctx": args.max_ctx, "contexts": args.contexts,
                   "embed_on_host": True, "fp8_kv": True,
                   "gen_tokens": GEN_TOKENS, "prompts": PROMPTS,
                   "shell_bytes": int(free0 - free_shell),
                   "device_total_bytes": int(total)},
        "arms": arms,
        "gates": gates,
        "claim_boundary": (
            "Measured batch-1 single-stream decode on this GPU, three arms in one "
            "process against one model load, identical warm-up and sampling "
            "protocol to n7b_cached_decode.py. The two cache arms hold exactly "
            "the same number of bytes; capacity differs because a full-record "
            "slot is exactly twice an up-only slot. NOT a quality result, not a "
            "benchmark score, and not a claim about other hardware, batch sizes "
            "or capacities -- no capacity sweep was run, deliberately, because "
            "it would add a second variable."),
    }
    (OUT_DIR / "s11_cache_mode_ab.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\ndeep ctx {deep}:  A1 {a1:.3f}  B {b:.3f}  A2 {a2:.3f} tok/s")
    print(f"  effect B-A1 {effect:+.3f}  drift |A2-A1| {drift:.3f}  "
          f"conclusive={gates['G_S11_D1_drift']['conclusive']}")
    print(f"  G-S11-C1 identity: {gates['G_S11_C1_bit_identical']['passed']}")
    print(f"  G-S11-P1 adopt   : {gates['G_S11_P1_adopt']['passed']} "
          f"(gain {(b - a1) / a1 * 100:+.2f}%, need +{ADOPT_MARGIN * 100:.0f}%)")
    print("\nwritten s11_cache_mode_ab.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
