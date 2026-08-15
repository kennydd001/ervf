"""S5 gated measurement: masked column-selective down_proj (design A2).

Methodology identical to n7b_cached_decode.py (capacity 31, FP8 KV, warm
windows, same context targets) so the ONLY variable vs the reproduced N8
baseline is the masked column-selective down path.

Gates (preregistered, S5 preregistration 2026-08-14 + R1 addendum):
  C1  greedy generation reproduces s5_baseline_generation.json ids exactly
  P3  ctx 0     p50 >= 21 tok/s   (no regression; reproduction gave 22,062)
  P1  ctx262100 p50 >= 15 tok/s   (minimum; reproduction gave 13,143)
  P2  ctx262100 p50 >= 18 tok/s   (primary)
C2 (transpose exactness) lives in s5_transpose_check.py; C3 (per-call rel_l2)
in the independent verifier.
"""

from __future__ import annotations

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

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GIB = 1024 ** 3

CONTEXTS = [0, 32768, 131072, 262100]
MAX_CTX = 262144
CAPACITY = 31
TOKENS = 160

GATES = {"P3_ctx0_min": 21.0, "P1_ctx262100_min": 15.0, "P2_ctx262100_primary": 18.0}


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def pct(v):
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

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
    baseline = json.loads((OUT_DIR / "s5_baseline_generation.json").read_text())

    free0, total = cp.cuda.runtime.memGetInfo()
    rt = LightningRuntime(MODEL_DIR, contexts_max=MAX_CTX, verbose=False)
    cache_bytes = rt.enable_cache(CAPACITY)
    rt.load_routed_bank()
    free1, _ = cp.cuda.runtime.memGetInfo()
    print(f"cache {cache_bytes/GIB:.3f} GiB | device used "
          f"{(free0-free1)/GIB:.3f} GiB | free {free1/GIB:.3f} GiB", flush=True)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    # ------------------------------------------------ G-S5-C1 correctness
    gen_report = []
    c1_ok = True
    for bp in baseline["prompts"]:
        ids = bp["prompt_ids"]
        rt.reset()
        cur, gen = ids[0], []
        for s in range(len(ids) + 32):
            nxt = rt.step(cur)
            if s >= len(ids) - 1:
                gen.append(nxt)
                cur = nxt
            else:
                cur = ids[s + 1]
        cp.cuda.Device(0).synchronize()
        same = gen == bp["generated_ids"]
        c1_ok &= same
        gen_report.append({"prompt": bp["prompt"], "identical_32_tokens": same,
                           "generated_text": tok.decode(gen)})
        print(f"C1 {bp['prompt']!r}: identical={same} {tok.decode(gen)[:60]!r}",
              flush=True)
    if not c1_ok:
        print("STOP: G-S5-C1 failed, no timing claims allowed.")
        # still record; gates evaluated below will reflect the failure

    # ------------------------------------------------ steady-state decode
    rt.reset()
    rt.cache_stats = {"hits": 0, "misses": 0}
    prompt2 = tok.encode("The history of computing began when", add_special_tokens=False)
    cur, per_token = prompt2[0], []
    for i in range(len(prompt2) + TOKENS):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter_ns()
        nxt = rt.step(cur)
        cp.cuda.Device(0).synchronize()
        ms = (time.perf_counter_ns() - t0) / 1e6
        if i >= len(prompt2) - 1:
            per_token.append(ms)
            cur = nxt
        else:
            cur = prompt2[i + 1]
    warm = pct(per_token[32:])
    hits, misses = rt.cache_stats["hits"], rt.cache_stats["misses"]
    hit_rate = hits / (hits + misses)
    print(f"hit rate {hit_rate*100:.1f}%  warm p50 {warm['p50']:.2f} ms -> "
          f"{1000/warm['p50']:.2f} tok/s", flush=True)

    # ------------------------------------------------ context sweep
    ctx_rows = {}
    rng = np.random.default_rng(11)
    varied = [int(v) for v in rng.integers(1000, 60000, size=4096)]
    for target in CONTEXTS:
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
        ctx_rows[str(target)] = {"ms": s, "raw_ms": samples,
                                 "tok_s_p50": 1000.0 / s["p50"]}
        print(f"  ctx {target:>6}: p50 {s['p50']:7.2f} ms -> "
              f"{1000/s['p50']:6.3f} tok/s", flush=True)

    gates = {
        "G-S5-C1_identical_generation": c1_ok,
        "G-S5-P3_no_regression_ctx0":
            ctx_rows["0"]["tok_s_p50"] >= GATES["P3_ctx0_min"],
        "G-S5-P1_262k_minimum":
            ctx_rows["262100"]["tok_s_p50"] >= GATES["P1_ctx262100_min"],
        "G-S5-P2_262k_primary":
            ctx_rows["262100"]["tok_s_p50"] >= GATES["P2_ctx262100_primary"],
    }
    result = {
        "kind": "lightningstream_nemotron_s5_masked_decode",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "S5_COLUMN_SELECTIVE_DOWN_PROJ",
        "design": "A2_sm_side_wide_gather",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(
            REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "fused_sha256": sha256_path(
            REPO_ROOT / "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py"),
        "baseline_sha256": baseline and sha256_path(OUT_DIR / "s5_baseline_generation.json"),
        "capacity_per_layer": CAPACITY,
        "cache_gib": cache_bytes / GIB,
        "device_used_gib": (free0 - free1) / GIB,
        "generation_checks": gen_report,
        "cache": {"hits": hits, "misses": misses, "hit_rate": hit_rate},
        "decode_warm_cache": warm,
        "tok_s_warm_p50": 1000.0 / warm["p50"],
        "context_sweep_warm": ctx_rows,
        "gates": gates,
        "reproduced_baseline_tok_s": {"0": 22.062, "32768": 20.147,
                                      "131072": 16.686, "262100": 13.143},
        "claim_boundary": (
            "Measured batch-1 decode of the masked column-selective runtime on "
            "this GPU at the measured contexts, generation gate checked against "
            "the frozen pre-S5 baseline. NOT a quality result, benchmark, or a "
            "claim about other hardware, batch sizes or prompts."),
    }
    (OUT_DIR / "s5_masked_decode.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("gates:", json.dumps(gates), "\nwritten s5_masked_decode.json")
    return 0 if all(gates.values()) else 3


if __name__ == "__main__":
    sys.exit(main())
