"""Read-only diagnostic: is the -20/+30 capacity split (layer_capacity.py)
near the top of what per-layer reallocation can offer, or is there more on
the table with a more aggressive split? Sweeps a few candidate
(reduce_delta, boost_delta) pairs at constant total budget, hit-rate only.

Each candidate runs as a separate subprocess (--one=reduce,boost), same fix
as diag_v6_component_breakdown.py needed: building several full 30B
runtimes sequentially in one process exhausts pinned host memory even with
explicit cleanup.

Not a gated PRO experiment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_gpu_free, require_model_dir, utc_now, write_json_atomic

UP_CODE = 2_494_464
UP_SCALE = 311_808
REDUCE_LAYERS = [38, 10, 40, 20, 43, 13]
BOOST_LAYERS = [1, 3, 51, 6]
BASELINE_CAP = 72

CANDIDATES = [
    (0, 0),      # uniform baseline
    (-20, 30),   # the shipped split
    (-30, 45),
    (-40, 60),
    (-50, 75),
    (-60, 90),
]


def _reallocate_layer(rt, layer: int, new_cap: int):
    cp = rt.cp
    entry = {
        "codes": cp.zeros(new_cap * UP_CODE, dtype=cp.uint8),
        "scales": cp.zeros(new_cap * UP_SCALE, dtype=cp.uint8),
        "map": OrderedDict(),
        "cap": new_cap,
    }
    entry["slot_codes"] = [entry["codes"][k * UP_CODE:(k + 1) * UP_CODE] for k in range(new_cap)]
    entry["slot_scales"] = [entry["scales"][k * UP_SCALE:(k + 1) * UP_SCALE] for k in range(new_cap)]
    rt.cache[layer] = entry


def _out_path(idx: int) -> Path:
    return REPO / "pro_research" / f"diag_capacity_sweep_arm_{idx}.json"


def run_one(idx: int) -> int:
    require_gpu_free()
    reduce_delta, boost_delta = CANDIDATES[idx]
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(BASELINE_CAP)
    rt.load_routed_bank()
    rt.device_cache = True
    rt.deterministic_accum = True

    if reduce_delta != 0 or boost_delta != 0:
        for layer in REDUCE_LAYERS:
            _reallocate_layer(rt, layer, BASELINE_CAP + reduce_delta)
        for layer in BOOST_LAYERS:
            _reallocate_layer(rt, layer, BASELINE_CAP + boost_delta)
        rt._dev_cache = {}

    prompt = "The history of computing began when"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    ids = tok.encode(prompt, add_special_tokens=False)

    import cupy as cp
    rt.reset()
    n = 256
    nxt = None
    for t in ids:
        nxt = int(rt.step(int(t)))
    cur = nxt
    for _ in range(n - 1):
        cur = int(rt.step(cur))
    cp.cuda.Device(0).synchronize()

    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]
    total_hits = total_misses = 0
    for i in moe_layers:
        dc = getattr(rt, "_dev_cache", {}).get(i)
        if dc is None:
            continue
        h, m = [int(x) for x in cp.asnumpy(dc["stats2"])]
        total_hits += h
        total_misses += m

    result = {
        "reduce_delta": reduce_delta, "boost_delta": boost_delta,
        "budget_neutral_check": 6 * reduce_delta + 4 * boost_delta,
        "hits": total_hits, "misses": total_misses,
        "hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) else None,
    }
    write_json_atomic(_out_path(idx), result, archive=False)
    print(result)
    return 0


def drive() -> int:
    for idx in range(len(CANDIDATES)):
        print(f"=== candidate {idx}: {CANDIDATES[idx]} ===", flush=True)
        rc = subprocess.run([sys.executable, __file__, "--one", str(idx)]).returncode
        if rc != 0:
            print(f"candidate {idx} failed with exit code {rc}")
            return rc

    import json
    results = []
    for idx in range(len(CANDIDATES)):
        results.append(json.loads(_out_path(idx).read_text(encoding="utf-8")))

    payload = {
        "kind": "diag_capacity_sweep",
        "created_utc": utc_now(),
        "note": "read-only diagnostic, hit-rate only, sweeps reallocation aggressiveness at constant total budget, one subprocess per candidate",
        "results": results,
    }
    out = REPO / "pro_research" / "diag_capacity_sweep.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", type=int)
    ap.add_argument("--drive", action="store_true")
    args = ap.parse_args()
    if args.one is not None:
        return run_one(args.one)
    return drive()


if __name__ == "__main__":
    raise SystemExit(main())
