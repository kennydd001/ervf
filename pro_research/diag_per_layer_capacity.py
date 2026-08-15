"""Read-only diagnostic: does moving cache budget from low-miss layers to
high-miss layers reduce total misses at constant total capacity?

diag_hitrate_v4.json (2026-08-16) measured per-layer miss rates at uniform
capacity=72 and found them strongly non-uniform: layers 1/3/6/51 miss
25-42%, most others 6-15%. This tests a non-uniform reallocation: -20
capacity on the 6 lowest-miss layers (38,10,40,20,43,13; 72->52 each,
-120 total slots) and +30 on the 4 highest-miss layers (1,3,51,6; 72->102
each, +120 total slots) -- budget-neutral.

Reallocates specific layers' cache entries by mirroring enable_cache's own
allocation code (runtime.py:324-378) exactly for mode="up_only" -- no edit
to runtime.py, just calling the same allocation pattern for a subset of
layers after the normal enable_cache(72) call. _moe_dev already reads
c["cap"] per-layer dynamically, so heterogeneous per-layer capacity is
already structurally supported; only enable_cache's convenience API is
uniform.

Not a gated PRO experiment -- hit-rate only, no timing claim.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, require_gpu_free, require_model_dir, utc_now, write_json_atomic

UP_CODE = 2_494_464
UP_SCALE = 311_808

REDUCE_LAYERS = [38, 10, 40, 20, 43, 13]
REDUCE_DELTA = -20
BOOST_LAYERS = [1, 3, 51, 6]
BOOST_DELTA = 30
BASELINE_CAP = 72


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


def run_one(nonuniform: bool, n: int = 256):
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(BASELINE_CAP)
    rt.load_routed_bank()
    rt.device_cache = True
    rt.deterministic_accum = True

    total_cap = sum(rt.cache[layer]["cap"] for layer in rt.cache)
    if nonuniform:
        for layer in REDUCE_LAYERS:
            _reallocate_layer(rt, layer, BASELINE_CAP + REDUCE_DELTA)
        for layer in BOOST_LAYERS:
            _reallocate_layer(rt, layer, BASELINE_CAP + BOOST_DELTA)
        rt._dev_cache = {}
    total_cap_after = sum(rt.cache[layer]["cap"] for layer in rt.cache)

    prompt = "The history of computing began when"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    ids = tok.encode(prompt, add_special_tokens=False)

    import cupy as cp
    rt.reset()
    nxt = None
    for t in ids:
        nxt = int(rt.step(int(t)))
    cur = nxt
    for _ in range(n - 1):
        cur = int(rt.step(cur))
    cp.cuda.Device(0).synchronize()

    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]
    total_hits = total_misses = 0
    per_layer = {}
    for i in moe_layers:
        dc = getattr(rt, "_dev_cache", {}).get(i)
        if dc is None:
            continue
        h, m = [int(x) for x in cp.asnumpy(dc["stats2"])]
        total_hits += h
        total_misses += m
        per_layer[str(i)] = {"cap": int(rt.cache[i]["cap"]), "hits": h, "misses": m}

    result = {
        "nonuniform": nonuniform,
        "total_cap_before_realloc": total_cap,
        "total_cap_after_realloc": total_cap_after,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) else None,
        "per_layer": per_layer,
    }

    del rt
    import gc
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return result


def main() -> int:
    require_gpu_free()
    uniform = run_one(nonuniform=False)
    print("uniform:", {"hits": uniform["total_hits"], "misses": uniform["total_misses"], "hit_rate": uniform["hit_rate"]}, flush=True)
    nonuniform = run_one(nonuniform=True)
    print("nonuniform:", {"hits": nonuniform["total_hits"], "misses": nonuniform["total_misses"], "hit_rate": nonuniform["hit_rate"]}, flush=True)

    payload = {
        "kind": "diag_per_layer_capacity",
        "created_utc": utc_now(),
        "note": "read-only diagnostic, hit-rate only, not a gated PRO experiment; total cache budget held constant",
        "config": {
            "baseline_cap": BASELINE_CAP,
            "reduce_layers": REDUCE_LAYERS, "reduce_delta": REDUCE_DELTA,
            "boost_layers": BOOST_LAYERS, "boost_delta": BOOST_DELTA,
        },
        "uniform": uniform,
        "nonuniform": nonuniform,
        "misses_delta": nonuniform["total_misses"] - uniform["total_misses"],
        "improvement": nonuniform["total_misses"] < uniform["total_misses"],
    }
    out = REPO / "pro_research" / "diag_per_layer_capacity.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
