"""W1-R2: the third design -- short alternating blocks, averaged per block.

Preregistered in W1R2_C1R1_O1_PREREGISTRATION_2026-08-15.md.

W1 averaged 32 samples per arm but put the arms minutes apart (drift 4.5 ms).
W1-R1 put the arms next to each other but used one sample each (drift 6.8 ms).
This does both: blocks of 8 samples per arm, alternating base/fast immediately,
so each block median is tight AND neighbouring blocks share thermal state.
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

PROMPTS = ["The capital of France is", "The history of computing began when"]
GEN_TOKENS = 64
BLOCK = 8
BLOCKS_PER_ARM = 12
GATE_DRIFT_MS = 1.0
GATE_ADOPT_MS = 1.0
GATE_SIGN = 0.60


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


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


def block_sweep(rt, cp, target, max_ctx, blocks_per_arm):
    rng = np.random.default_rng(11)
    varied = [int(v) for v in rng.integers(1000, 60000, size=16384)]
    rt.fast_host = False
    rt.reset()
    for j in range(min(target, 64)):
        rt.step(varied[j % len(varied)])
    rt.pos = target
    for j in range(32):
        rt.step(varied[(j + 64) % len(varied)])
    cp.cuda.Device(0).synchronize()

    blocks, k = [], 96
    for b in range(2 * blocks_per_arm):
        fast = bool(b % 2)
        rt.fast_host = fast
        samples = []
        for _ in range(BLOCK):
            t0 = time.perf_counter_ns()
            rt.step(varied[k % len(varied)])
            cp.cuda.Device(0).synchronize()
            samples.append((time.perf_counter_ns() - t0) / 1e6)
            k += 1
        blocks.append({"index": b, "fast": fast, "samples": samples,
                       "p50": float(np.percentile(samples, 50))})
    rt.fast_host = False
    return blocks


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=72)
    ap.add_argument("--max-ctx", type=int, default=262144)
    ap.add_argument("--contexts", type=int, nargs="*", default=[0, 262100])
    ap.add_argument("--blocks", type=int, default=BLOCKS_PER_ARM)
    args = ap.parse_args()

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    rt = LightningRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                          embed_on_host=True, fp8_kv=True)
    rt.enable_cache(args.capacity)
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    rt.fast_host = False
    ref = generate(rt, cp, tokenizer)
    rt.fast_host = True
    alt = generate(rt, cp, tokenizer)
    rt.fast_host = False
    identical = ref == alt
    print(f"generation bit-identical: {identical}", flush=True)

    per_ctx = {}
    for ctx in args.contexts:
        if ctx >= args.max_ctx - 8:
            continue
        blocks = block_sweep(rt, cp, ctx, args.max_ctx, args.blocks)
        base = [b for b in blocks if not b["fast"]]
        fast = [b for b in blocks if b["fast"]]
        # effect: each fast block against the mean of its two base neighbours
        effects = []
        for i, fb in enumerate(fast):
            lo = base[i]["p50"]
            hi = base[i + 1]["p50"] if i + 1 < len(base) else base[i]["p50"]
            effects.append(0.5 * (lo + hi) - fb["p50"])
        drifts = [abs(base[i + 1]["p50"] - base[i]["p50"]) for i in range(len(base) - 1)]
        pos = sum(1 for e in effects if e > 0)
        per_ctx[str(ctx)] = {
            "context": ctx, "blocks": blocks,
            "effect_p50_ms": float(np.percentile(effects, 50)),
            "effect_mean_ms": float(np.mean(effects)),
            "effects": effects,
            "drift_p50_ms": float(np.percentile(drifts, 50)),
            "drifts": drifts,
            "sign_fraction": pos / len(effects),
            "resolution_ok": bool(float(np.percentile(drifts, 50)) < GATE_DRIFT_MS),
            "sign_ok": bool(float(np.percentile(effects, 50)) > 0
                            and pos / len(effects) >= GATE_SIGN),
        }
        r = per_ctx[str(ctx)]
        print(f"  ctx {ctx:>6}: effect p50 {r['effect_p50_ms']:+.3f} ms | "
              f"block drift p50 {r['drift_p50_ms']:.3f} ms | "
              f"sign {r['sign_fraction']:.2f} | resolution_ok={r['resolution_ok']}",
              flush=True)

    deep = str(args.contexts[-1])
    res_ok = per_ctx[deep]["resolution_ok"]
    adopt = bool(identical and res_ok
                 and per_ctx[deep]["effect_p50_ms"] >= GATE_ADOPT_MS
                 and per_ctx[str(args.contexts[0])]["effect_p50_ms"] >= 0.0)
    gates = {
        "G_W1R2_C1_identity": {"passed": bool(identical)},
        "G_W1R2_R1_resolution": {"required_drift_below_ms": GATE_DRIFT_MS,
                                 "per_context": {c: v["resolution_ok"]
                                                 for c, v in per_ctx.items()},
                                 "passed": bool(res_ok)},
        "G_W1R2_E1_sign": {c: v["sign_ok"] for c, v in per_ctx.items()},
        "G_W1_P1_unchanged": {"required_gain_ms": GATE_ADOPT_MS,
                              "measured_effect_p50_ms": per_ctx[deep]["effect_p50_ms"],
                              "conditional_on_resolution": res_ok,
                              "passed": adopt},
    }

    payload = {
        "kind": "lightningstream_nemotron_w1r2_block_paired",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "W1_R2_BLOCK_PAIRED",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "contexts": args.contexts, "block_samples": BLOCK,
                   "blocks_per_arm": args.blocks},
        "prior_designs": {"w1_arm_drift_ms": 4.520, "w1r1_triplet_drift_ms": 6.779},
        "identity": {"bit_identical": bool(identical), "reference": ref, "fast": alt},
        "per_context": per_ctx,
        "gates": gates,
        "claim_boundary": (
            "Measured batch-1 single-stream decode on this GPU. Blocks of 8 timed "
            "steps alternate between the two host paths inside one warm state, so "
            "each block median averages 8 samples and neighbouring blocks are "
            "seconds rather than minutes apart. The adoption gate G-W1-P1 is "
            "unchanged at 1.0 ms and, per the preregistration, only applies if the "
            "resolution gate passes. Not a quality result and not a claim about "
            "other hardware."),
    }
    (OUT_DIR / "w1r2_block_paired.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\ngates")
    print(f"  G-W1R2-C1 identity  : {identical}")
    print(f"  G-W1R2-R1 resolution: {res_ok} "
          f"(drift p50 {per_ctx[deep]['drift_p50_ms']:.3f} vs W1 4.520 / W1-R1 6.779)")
    print(f"  G-W1-P1 (unchanged) : {adopt}")
    print("\nwritten w1r2_block_paired.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
