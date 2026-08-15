"""N7-B: measured decode with a per-layer LRU expert cache.

Correctness is gated first against the frozen N6-A result; only then is
throughput measured.  Cache capacity is chosen from N7-A's MEASURED locality and
the free VRAM N5 measured, not tuned against the result.
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

import os as _os
MODEL_DIR = REPO_ROOT / "models" / _os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GIB = 1024 ** 3


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def gpu_state():
    try:
        o = subprocess.run(["nvidia-smi",
                            "--query-gpu=memory.used,memory.free,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        u, f, t, p = [x.strip() for x in o.stdout.strip().split(",")]
        return {"used_mib": int(u), "free_mib": int(f), "temp_c": int(t), "power_w": float(p)}
    except Exception as e:
        return {"error": str(e)}


def pct(v):
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "min": float(a.min())}


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=32)
    ap.add_argument("--embed-on-host", action="store_true")
    ap.add_argument("--fp32-kv", action="store_true")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--contexts", type=int, nargs="*", default=[0, 1024, 4032])
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
    free0, total = cp.cuda.runtime.memGetInfo()
    rt = LightningRuntime(MODEL_DIR, contexts_max=args.max_ctx, embed_on_host=args.embed_on_host, fp8_kv=not args.fp32_kv)
    free_shell, _ = cp.cuda.runtime.memGetInfo()
    cache_bytes = rt.enable_cache(args.capacity)
    free_cache, _ = cp.cuda.runtime.memGetInfo()
    rt.load_routed_bank()
    print(f"shell {(free0-free_shell)/GIB:.3f} GiB | cache {cache_bytes/GIB:.3f} GiB | "
          f"free {free_cache/GIB:.3f} GiB", flush=True)

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    # ------------------------------------------------- correctness gate
    ids = tok.encode("The capital of France is", add_special_tokens=False)
    rt.reset()
    nxt = None
    for t in ids:
        nxt = rt.step(t)
    cp.cuda.Device(0).synchronize()
    first = tok.decode([nxt])
    coherent = "paris" in first.strip().lower()
    print(f"correctness: -> {first!r} coherent={coherent}", flush=True)
    if not coherent:
        print("STOP: cached path incoherent.")
        return 3

    # ------------------------------------------------- steady-state decode
    rt.reset()
    rt.cache_stats = {"hits": 0, "misses": 0}
    prompt2 = tok.encode("The history of computing began when", add_special_tokens=False)
    cur, per_token, gen = prompt2[0], [], []
    for i in range(len(prompt2) + args.tokens):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter_ns()
        nxt = rt.step(cur)
        cp.cuda.Device(0).synchronize()
        ms = (time.perf_counter_ns() - t0) / 1e6
        if i >= len(prompt2) - 1:
            per_token.append(ms)
            gen.append(nxt)
            cur = nxt
        else:
            cur = prompt2[i + 1]

    # Warm-cache window: skip the first 32 tokens where the LRU is filling.
    warm = per_token[32:]
    cold = per_token[:32]
    s_all, s_warm = pct(per_token), pct(warm)
    hits, misses = rt.cache_stats["hits"], rt.cache_stats["misses"]
    hit_rate = hits / (hits + misses)
    print(f"generated: {tok.decode(gen)[:90]!r}", flush=True)
    print(f"hit rate {hit_rate*100:.1f}%  cold p50 {pct(cold)['p50']:.2f} ms  "
          f"warm p50 {s_warm['p50']:.2f} ms -> {1000/s_warm['p50']:.2f} tok/s", flush=True)

    # ------------------------------------------------- context sweep (warm)
    ctx_rows = {}
    # Varied token ids, not a repeated one: feeding the same token makes the
    # router pick the same experts every step, which drives the cache to a
    # near-100% hit rate and inflates throughput. Real decode has varying routes.
    rng = np.random.default_rng(11)
    varied = [int(v) for v in rng.integers(1000, 60000, size=4096)]
    for target in args.contexts:
        if target >= args.max_ctx - 8:
            continue
        rt.reset()
        for j in range(min(target, 64)):
            rt.step(varied[j % len(varied)])
        rt.pos = target
        for j in range(32):          # warm the cache at this depth
            rt.step(varied[(j + 64) % len(varied)])
        cp.cuda.Device(0).synchronize()
        samples = []
        for j in range(16):
            t0 = time.perf_counter_ns()
            rt.step(varied[(j + 96) % len(varied)])
            cp.cuda.Device(0).synchronize()
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        s = pct(samples)
        ctx_rows[str(target)] = {"context": target, "ms": s, "raw_ms": samples,
                                 "tok_s_p50": 1000.0 / s["p50"], "gpu": gpu_state()}
        print(f"  ctx {target:>6}: p50 {s['p50']:7.2f} ms -> {1000/s['p50']:6.3f} tok/s",
              flush=True)

    free_end, _ = cp.cuda.runtime.memGetInfo()
    result = {
        "kind": "lightningstream_nemotron_n7b_cached_decode",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N7_B_CACHED_DECODE",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "capacity_per_layer": args.capacity,
        "cache_bytes": cache_bytes,
        "cache_gib": cache_bytes / GIB,
        "device_total_bytes": int(total),
        "device_used_bytes": int(free0 - free_end),
        "device_free_end_bytes": int(free_end),
        "igpu_used": False,
        "correctness": {"top1": first, "coherent": coherent,
                        "n6a_expected": " Paris", "generated": tok.decode(gen)},
        "cache": {"hits": hits, "misses": misses, "hit_rate": hit_rate,
                  "n7a_simulated_hit_rate_at_32": 0.650},
        "decode_all_tokens": s_all,
        "decode_warm_cache": s_warm,
        "decode_cold_first_32": pct(cold),
        "tok_s_warm_p50": 1000.0 / s_warm["p50"],
        "raw_per_token_ms": per_token,
        "context_sweep_warm": ctx_rows,
        "baseline_uncached": {"ctx0_tok_s": 15.885, "ctx4096_tok_s": 15.193,
                              "token_ms_ctx0": 62.95},
        "claim_boundary": (
            "Measured batch-1 single-stream decode on this specific GPU with a "
            "per-layer LRU expert cache. Warm-cache figures exclude the first 32 "
            "tokens while the LRU fills, and that window is reported separately. "
            "NOT a quality result, benchmark score, or claim about other "
            "hardware, batch sizes, or prompts. Projections from N7-A are not "
            "measurements and are not restated here as such."),
    }
    (OUT_DIR / "n7b_cached_decode.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten n7b_cached_decode.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())



