"""E1 fase 2.1 A/B: device-resident routing + device-LRU cache (eager).

Preregistration:
reports/treesweep200/E1F21_DEVICE_ROUTING_PREREGISTRATION_2026-08-15.md

Arms (one runtime, flags toggled between arms; parity is judged against the
EXTERNAL frozen A1 ids, so same-process sequencing is not a blinder):
  BASE  default stack (device_cache off) -- sanity parity + timing reference
  DEV   device_cache=True, capacity 72   -- G-E1F21-C1 + timing
  INV   device_cache=True, capacity 56   -- G-E1F21-INV (must equal DEV tokens)
  CTL   device_cache=True, bad_pick=1    -- G-E1F21-CTL (must DIFFER from A1)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL_DIR = REPO / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
TS200 = REPO / "reports" / "treesweep200"
OUT = TS200 / "E1F21_DEVICE_ROUTING_AB.json"


def gpu_free() -> bool:
    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                        "--format=csv,noheader"], capture_output=True, text=True,
                       timeout=30)
    return not o.stdout.strip()


def run_gen(rt, prompt_ids, n):
    rt.reset()
    nxt = None
    for t in prompt_ids:
        nxt = rt.step(int(t))
    gen = [int(nxt)]
    for _ in range(n - 1):
        gen.append(int(rt.step(gen[-1])))
    return gen


def run_timed(rt, prompt_ids, n):
    rt.reset()
    nxt = None
    for t in prompt_ids:
        nxt = rt.step(int(t))
    import cupy as cp
    cp.cuda.Device(0).synchronize()
    samples = []
    cur = int(nxt)
    for _ in range(n - 1):
        t0 = time.perf_counter_ns()
        cur = int(rt.step(cur))
        cp.cuda.Device(0).synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return samples


def main() -> int:
    if not gpu_free():
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    anchor = json.loads((TS200 / "V36_DETERMINISTIC_ANCHOR.json").read_text())
    a1 = json.loads((TS200 / "A1_ADOPTION_PRECONDITION.json").read_text())
    expected = a1["gates"]["G_A2_ANCHOR_informative"]["produced_ids"]
    prompts = anchor["prompts"]
    n = int(anchor["gen_tokens"])

    import cupy as cp  # noqa: F401
    rt = LightningRuntime(MODEL_DIR, contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(int(anchor["capacity"]))
    rt.load_routed_bank()
    print("runtime loaded", flush=True)

    results: dict = {"prompts": [p["prompt"] for p in prompts], "arms": {}}

    def parity(ids_by_prompt):
        return {p: bool(expected.get(p) == g)
                for p, g in ids_by_prompt.items()}

    # ---- BASE: default stack ----
    base_ids, base_ms = {}, []
    for p in prompts:
        base_ids[p["prompt"]] = run_gen(rt, p["prompt_ids"], n)
        base_ms.extend(run_timed(rt, p["prompt_ids"], n))
    results["arms"]["BASE"] = {
        "parity_a1": parity(base_ids),
        "token_ms_p50": float(np.percentile(base_ms, 50)),
        "token_ms_mean": float(np.mean(base_ms)),
    }
    print("BASE done:", results["arms"]["BASE"], flush=True)

    # ---- DEV: device_cache, capacity 72 ----
    rt.device_cache = True
    dev_ids, dev_ms = {}, []
    for p in prompts:
        dev_ids[p["prompt"]] = run_gen(rt, p["prompt_ids"], n)
        dev_ms.extend(run_timed(rt, p["prompt_ids"], n))
    results["arms"]["DEV"] = {
        "parity_a1": parity(dev_ids),
        "token_ms_p50": float(np.percentile(dev_ms, 50)),
        "token_ms_mean": float(np.mean(dev_ms)),
    }
    print("DEV done:", results["arms"]["DEV"], flush=True)

    # ---- INV: device_cache, capacity 56 ----
    rt.enable_cache(56)
    inv_ids = {p["prompt"]: run_gen(rt, p["prompt_ids"], n) for p in prompts}
    results["arms"]["INV"] = {
        "same_as_dev72": {p["prompt"]: bool(inv_ids[p["prompt"]] == dev_ids[p["prompt"]])
                          for p in prompts},
    }
    print("INV done:", results["arms"]["INV"], flush=True)

    # ---- CTL: bad_pick must FAIL parity ----
    rt.enable_cache(int(anchor["capacity"]))
    rt._bad_pick = 1
    ctl_ids = {p["prompt"]: run_gen(rt, p["prompt_ids"], n) for p in prompts}
    ctl_match = parity(ctl_ids)
    results["arms"]["CTL"] = {
        "parity_a1": ctl_match,
        "must_fail": not any(ctl_match.values()),
    }
    print("CTL done:", results["arms"]["CTL"], flush=True)

    rt._bad_pick = 0
    rt.device_cache = False

    OUT.write_text(json.dumps(results, indent=2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
