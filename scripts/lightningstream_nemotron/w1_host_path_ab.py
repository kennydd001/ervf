"""W1: does making the host path cheaper actually move the token?

Preregistered in W1_HOST_PATH_PREREGISTRATION_2026-08-15.md.

Three arms in one process against one model load -- base / fast / base -- so the
repeat of base bounds drift. The only variable is `rt.fast_host`, which swaps the
per-expert Python work for precomputed views and floats. Same kernels, same
arguments, same order, so the generation must be bit-identical.
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
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GIB = 1024 ** 3

PROMPTS = ["The capital of France is", "The history of computing began when"]
GEN_TOKENS = 64
ADOPT_GAIN_MS = 1.0
S14_HOST_GAP_MS = {"0": 5.058, "262100": 4.672}
ARMS = [("base1", False), ("fast", True), ("base2", False)]


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
        s = pctl(samples)
        rows[str(target)] = {"context": target, "ms": s, "raw_ms": samples}
        print(f"    ctx {target:>6}: p50 {s['p50']:7.3f} ms", flush=True)
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
    free0, total = cp.cuda.runtime.memGetInfo()
    rt = LightningRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                          embed_on_host=True, fp8_kv=True)
    free_shell, _ = cp.cuda.runtime.memGetInfo()
    cache_bytes = rt.enable_cache(args.capacity)
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    print(f"shell {(free0 - free_shell) / GIB:.3f} GiB | "
          f"cache {cache_bytes / GIB:.3f} GiB", flush=True)

    arms, ref = {}, None
    for name, fast in ARMS:
        rt.fast_host = fast
        print(f"\narm {name}: fast_host={fast}", flush=True)
        gen = generate(rt, cp, tokenizer)
        if ref is None:
            ref = gen
        identical = gen == ref
        print(f"  identity vs base1: {identical}", flush=True)
        per_round = []
        for _ in range(args.rounds):
            per_round.append(context_sweep(rt, cp, args.contexts, args.max_ctx))
        merged = {}
        for ctx in per_round[0]:
            raw = [v for r in per_round for v in r[ctx]["raw_ms"]]
            merged[ctx] = {"context": int(ctx), "ms": pctl(raw), "raw_ms": raw}
        arms[name] = {"arm": name, "fast_host": fast,
                      "identical_to_base1": bool(identical),
                      "generation_token_ids": gen,
                      "context_sweep": merged}
    rt.fast_host = False

    contexts = [str(c) for c in args.contexts if c < args.max_ctx - 8]
    result_ctx = {}
    for ctx in contexts:
        b1 = arms["base1"]["context_sweep"][ctx]["ms"]["p50"]
        b2 = arms["base2"]["context_sweep"][ctx]["ms"]["p50"]
        f = arms["fast"]["context_sweep"][ctx]["ms"]["p50"]
        base = 0.5 * (b1 + b2)
        drift = abs(b2 - b1)
        gain = base - f
        result_ctx[ctx] = {
            "context": int(ctx), "base_p50_ms": base, "fast_p50_ms": f,
            "local_drift_ms": drift, "gain_ms": gain, "gain_relative": gain / base,
            "conclusive": bool(abs(gain) > drift),
            "s14_host_gap_ms": S14_HOST_GAP_MS.get(ctx),
            "within_s14_bound": bool(
                gain <= S14_HOST_GAP_MS.get(ctx, 1e9) + drift),
        }
        print(f"  ctx {ctx:>6}: base {base:7.3f} | fast {f:7.3f} | "
              f"gain {gain:+.3f} ms ({gain / base * 100:+.1f}%) | drift {drift:.3f}",
              flush=True)

    deep = contexts[-1]
    shallow = contexts[0]
    identity = arms["fast"]["identical_to_base1"]
    adopt = (identity and result_ctx[deep]["gain_ms"] >= ADOPT_GAIN_MS
             and result_ctx[shallow]["gain_ms"] >= 0.0)
    gates = {
        "G_W1_C1_identity": {"required": "bit-identical over 2 x 64 tokens",
                             "passed": bool(identity),
                             "all_arms": bool(all(a["identical_to_base1"]
                                                  for a in arms.values()))},
        "G_W1_P1_adopt": {"required_gain_ms_at_deep": ADOPT_GAIN_MS,
                          "deep_context": int(deep),
                          "measured_gain_ms": result_ctx[deep]["gain_ms"],
                          "no_regression_at_ctx0": result_ctx[shallow]["gain_ms"] >= 0.0,
                          "passed": bool(adopt)},
        "G_W1_D1_drift": {c: result_ctx[c]["conclusive"] for c in contexts},
        "G_W1_S1_within_host_gap": {c: result_ctx[c]["within_s14_bound"]
                                    for c in contexts},
    }

    payload = {
        "kind": "lightningstream_nemotron_w1_host_path_ab",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "W1_HOST_PATH",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "contexts": args.contexts, "rounds": args.rounds,
                   "gen_tokens": GEN_TOKENS, "prompts": PROMPTS,
                   "cache_bytes": int(cache_bytes)},
        "arms": arms,
        "per_context": result_ctx,
        "gates": gates,
        "claim_boundary": (
            "Measured batch-1 single-stream decode on this GPU, three arms in one "
            "process against one model load, identical warm-up and sampling "
            "protocol to n7b_cached_decode.py. The only variable is whether the "
            "per-expert host work is precomputed; the kernels, their arguments "
            "and their order are unchanged, which is what the bit-identity gate "
            "checks. The gain is bounded by S14's measured host_gap and cannot "
            "move the ceilings measured in Z1 or Y2-R1. Not a quality result, "
            "not a claim about other hardware or capacities."),
    }
    (OUT_DIR / "w1_host_path_ab.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\ngates")
    print(f"  G-W1-C1 identity : {identity}")
    print(f"  G-W1-P1 adopt    : {adopt} "
          f"(gain {result_ctx[deep]['gain_ms']:+.3f} ms at ctx {deep}, "
          f"need +{ADOPT_GAIN_MS})")
    print("\nwritten w1_host_path_ab.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
