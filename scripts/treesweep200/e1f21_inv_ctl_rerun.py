"""E1 fase 2.1 re-run of the INV and CTL arms after the enable_cache fix.

The first A/B (E1F21_DEVICE_ROUTING_AB.json) ran INV and CTL with a stale
device-LRU: enable_cache rebuilt the host cache but not ``_dev_cache``, so
both arms ran new capacity semantics over dirty slot state. INV failed for
that reason and CTL's failure could not be attributed to bad_pick alone.

runtime.enable_cache now resets ``_dev_cache``. This script re-runs ONLY the
affected arms, recomputing DEV-72 ids in-process as the INV reference. BASE
and DEV from the original A/B are untouched (both ran with fresh state).

Gates (frozen in E1F21_DEVICE_ROUTING_PREREGISTRATION_2026-08-15.md):
  G-E1F21-INV : tokens at capacity 56 == tokens at capacity 72
  G-E1F21-CTL : bad_pick=1 must BREAK parity with the frozen A1 ids
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL_DIR = REPO / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
TS200 = REPO / "reports" / "treesweep200"
OUT = TS200 / "E1F21_INV_CTL_RERUN.json"


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


def main() -> int:
    if not gpu_free():
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    anchor = json.loads((TS200 / "V36_DETERMINISTIC_ANCHOR.json").read_text())
    a1 = json.loads((TS200 / "A1_ADOPTION_PRECONDITION.json").read_text())
    expected = a1["gates"]["G_A2_ANCHOR_informative"]["produced_ids"]
    prompts = anchor["prompts"]
    n = int(anchor["gen_tokens"])

    rt = LightningRuntime(MODEL_DIR, contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(int(anchor["capacity"]))
    rt.load_routed_bank()
    rt.device_cache = True
    print("runtime loaded", flush=True)

    results: dict = {"prompts": [p["prompt"] for p in prompts], "arms": {}}

    # Reference: DEV at capacity 72, fresh state, untimed.
    dev_ids = {p["prompt"]: run_gen(rt, p["prompt_ids"], n) for p in prompts}
    results["arms"]["DEV72_ref"] = {
        "parity_a1": {p: bool(expected.get(p) == g) for p, g in dev_ids.items()},
    }
    print("DEV72_ref done:", results["arms"]["DEV72_ref"], flush=True)

    # INV: capacity 56 must produce identical tokens (hit/miss is not allowed
    # to change numerics).
    rt.enable_cache(56)
    inv_ids = {p["prompt"]: run_gen(rt, p["prompt_ids"], n) for p in prompts}
    results["arms"]["INV"] = {
        "same_as_dev72": {p["prompt"]: bool(inv_ids[p["prompt"]] == dev_ids[p["prompt"]])
                          for p in prompts},
    }
    print("INV done:", results["arms"]["INV"], flush=True)

    # CTL: bad_pick sabotage must break parity with the frozen A1 ids.
    rt.enable_cache(int(anchor["capacity"]))
    rt._bad_pick = 1
    ctl_ids = {p["prompt"]: run_gen(rt, p["prompt_ids"], n) for p in prompts}
    ctl_match = {p: bool(expected.get(p) == g) for p, g in ctl_ids.items()}
    results["arms"]["CTL"] = {
        "parity_a1": ctl_match,
        "must_fail": not any(ctl_match.values()),
    }
    print("CTL done:", results["arms"]["CTL"], flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
