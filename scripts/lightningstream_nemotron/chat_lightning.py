"""Interactive chat CLI for the local Nemotron 3.5 Lightning runtime.

Greedy decode on the adopted fast path (device LRU expert cache + CUDA
graph replay). Prints tokens as they are generated and reports tok/s.

Usage (from repo root):
    .venv-nemotron/Scripts/python.exe scripts/lightningstream_nemotron/chat_lightning.py

Commands during chat:
    /reset          clear conversation and runtime state
    /bench N        time N decode tokens without prompt (speed check)
    /quit           exit
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
EOS_IDS = {2, 11}  # generation_config.json


def build_runtime(capacity: int, cache_mode: str) -> LightningRuntime:
    print("Loading runtime (first load stages weights; please wait)...", flush=True)
    t0 = time.perf_counter()
    rt = LightningRuntime(str(MODEL), verbose=False)
    rt.device_cache = True
    rt.enable_cache(capacity, mode=cache_mode)
    rt.load_routed_bank()
    rt.setup_graph()
    print(f"Runtime ready in {time.perf_counter() - t0:.1f}s "
          f"(graph extra VRAM {rt.graph_extra_vram_bytes / 2**20:.0f} MiB).", flush=True)
    return rt


def generate(rt, tok, prompt_ids, max_new: int):
    """Stage prompt, stream greedy decode. Returns (text, ids, tok/s)."""
    for tid in prompt_ids:
        rt.step_graph(int(tid))
    rt.ring_harvest(rt._ring_i - len(prompt_ids), len(prompt_ids))  # sync once

    out_ids, t0 = [], time.perf_counter()
    pieces = []
    for _ in range(max_new):
        rt.step_graph()
        tid = rt.ring_harvest(rt._ring_i - 1, 1)[0]
        if tid in EOS_IDS:
            break
        out_ids.append(tid)
        pieces.append(tok.decode([tid], skip_special_tokens=True))
        print(pieces[-1], end="", flush=True)
    dt = time.perf_counter() - t0
    print()
    tps = len(out_ids) / dt if dt > 0 else 0.0
    return "".join(pieces), out_ids, tps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=72)
    ap.add_argument("--cache-mode", choices=["up_only", "full"], default="up_only")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--system", type=str, default="You are a helpful assistant.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    rt = build_runtime(args.capacity, args.cache_mode)

    messages = [{"role": "system", "content": args.system}] if args.system else []
    print("Chat ready. /reset, /bench N, /quit", flush=True)
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            messages = messages[:1] if args.system else []
            rt.reset()
            print("(conversation and runtime state cleared)")
            continue
        if user.startswith("/bench"):
            n = int(user.split()[1]) if len(user.split()) > 1 else 200
            rt.reset()
            for _ in range(min(16, n)):
                rt.step_graph()
            rt.ring_harvest(rt._ring_i - 1, 1)
            t0 = time.perf_counter()
            for _ in range(n):
                rt.step_graph()
            rt.ring_harvest(rt._ring_i - 1, 1)
            dt = time.perf_counter() - t0
            print(f"bench: {n} tokens in {dt:.2f}s -> {n / dt:.1f} tok/s "
                  f"({1000.0 * dt / n:.2f} ms/token)")
            messages = messages[:1] if args.system else []
            continue

        messages.append({"role": "user", "content": user})
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True)
        if len(prompt_ids) + args.max_new > rt.max_ctx:
            print("(context full; use /reset)")
            messages.pop()
            continue
        print("Assistant: ", end="", flush=True)
        text, out_ids, tps = generate(rt, tok, prompt_ids, args.max_new)
        print(f"[{len(out_ids)} tokens, {tps:.1f} tok/s]")
        messages.append({"role": "assistant", "content": text})
    return 0


if __name__ == "__main__":
    sys.exit(main())
