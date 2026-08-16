"""Bench: current Lightning (Nemotron 3.5) decode speed in tok/s.

Measures single-stream greedy decode with the adopted fast path
(device_cache + CUDA graph replay, `step_graph`). Optional `--no-graph`
times the plain `step()` path. Correctness spot-check: the produced token
ids are printed so runs can be compared; this is a speed harness, not a
parity gate.

Usage (from repo root):
    .venv-nemotron/Scripts/python.exe scripts/lightningstream_nemotron/bench_current_toks.py
    .venv-nemotron/Scripts/python.exe scripts/lightningstream_nemotron/bench_current_toks.py --tokens 400 --out bench.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL = REPO / "models" / "nemotron_3_5_lightning_v35"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=200, help="timed decode tokens")
    ap.add_argument("--prompt", type=int, default=8, help="prompt tokens staged before timing")
    ap.add_argument("--warmup", type=int, default=16, help="untimed decode tokens")
    ap.add_argument("--capacity", type=int, default=72, help="expert cache slots per layer")
    ap.add_argument("--cache-mode", choices=["up_only", "full"], default="up_only")
    ap.add_argument("--no-graph", action="store_true", help="use plain step() path")
    ap.add_argument("--seed-token", type=int, default=1, help="first prompt token id")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rt = LightningRuntime(str(MODEL), verbose=False)
    rt.device_cache = not args.no_graph
    cache_bytes = rt.enable_cache(args.capacity, mode=args.cache_mode)
    rt.load_routed_bank()
    graph_extra = 0
    if not args.no_graph:
        rt.setup_graph()
        graph_extra = rt.graph_extra_vram_bytes

    # Prompt: stage fixed ids (speed test, no tokenizer needed).
    ids = []
    t0 = time.perf_counter()
    if args.no_graph:
        tok = args.seed_token
        for _ in range(args.prompt):
            tok = rt.step(tok)
        for _ in range(args.warmup):
            tok = rt.step(tok)
        rt.cp.cuda.Device(0).synchronize()
        t1 = time.perf_counter()
        for _ in range(args.tokens):
            tok = rt.step(tok)
            ids.append(tok)
        rt.cp.cuda.Device(0).synchronize()
        t2 = time.perf_counter()
    else:
        # Stage the seed id as every prompt token (speed test; content is
        # irrelevant, only timing matters).
        for _ in range(args.prompt):
            rt.step_graph(args.seed_token)
        rt.ring_harvest(rt._ring_i - args.prompt, args.prompt)
        for _ in range(args.warmup):
            rt.step_graph()
        rt.ring_harvest(rt._ring_i - args.warmup, args.warmup)
        t1 = time.perf_counter()
        for _ in range(args.tokens):
            rt.step_graph()
        ids = rt.ring_harvest(rt._ring_i - args.tokens, args.tokens)
        t2 = time.perf_counter()

    dt = t2 - t1
    ms_per_token = 1000.0 * dt / args.tokens
    result = {
        "kind": "lightningstream_current_speed_bench",
        "model": str(MODEL),
        "path": "step_graph" if not args.no_graph else "step",
        "cache_mode": args.cache_mode,
        "capacity_per_layer": args.capacity,
        "prompt_tokens": args.prompt,
        "warmup_tokens": args.warmup,
        "timed_tokens": args.tokens,
        "prompt_plus_warmup_s": round(t1 - t0, 3),
        "timed_s": round(dt, 3),
        "ms_per_token": round(ms_per_token, 4),
        "tok_per_s": round(1000.0 / ms_per_token, 2),
        "cache_vram_bytes": int(cache_bytes),
        "graph_extra_vram_bytes": int(graph_extra),
        "first_ids": ids[:16],
        "last_ids": ids[-8:],
    }
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
