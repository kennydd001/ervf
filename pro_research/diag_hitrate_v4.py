"""Read-only diagnostic: MoE device-cache hit rate under the V4 configuration.

Not a gated PRO experiment -- no preregistration, no claim boundary, writes
nothing but a small JSON for the notebook. Purpose: find out whether the
24.3 ms/token V4 result is still dominated by PCIe cache-miss traffic or by
something else, so the next optimization target is chosen from evidence
instead of a guess.

`_moe_dev` (the device_cache path V4 uses) accumulates per-layer hit/miss
counts into `dev["stats2"]`, a device int32[2] buffer written by the
`cache_assign` kernel. That buffer is not read back anywhere in the existing
runner scripts. This script sums it across all MoE layers after a rollout.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, require_gpu_free, require_model_dir, utc_now, write_json_atomic


def main() -> int:
    require_gpu_free()
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    import cupy as cp

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.device_cache = True
    rt.deterministic_accum = True

    prompt = "The history of computing began when"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    ids = tok.encode(prompt, add_special_tokens=False)

    rt.reset()
    nxt = None
    for t in ids:
        nxt = int(rt.step(int(t)))
    cur = nxt
    n = 256
    for _ in range(n - 1):
        cur = int(rt.step(cur))
    cp.cuda.Device(0).synchronize()

    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]
    total_hits = total_misses = 0
    per_layer = []
    for i in moe_layers:
        dc = getattr(rt, "_dev_cache", {}).get(i)
        if dc is None:
            continue
        h, m = [int(x) for x in cp.asnumpy(dc["stats2"])]
        total_hits += h
        total_misses += m
        per_layer.append({"layer": i, "hits": h, "misses": m})

    calls = total_hits + total_misses
    hit_rate = total_hits / calls if calls else None
    payload = {
        "kind": "diag_hitrate_v4",
        "created_utc": utc_now(),
        "note": "read-only diagnostic, not a gated PRO experiment",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "prompt": prompt,
        "prompt_tokens": len(ids),
        "generated_tokens": n,
        "top_k": int(rt.top_k),
        "capacity_per_layer": 72,
        "moe_layer_count": len(moe_layers),
        "total_hits": total_hits,
        "total_misses": total_misses,
        "hit_rate": hit_rate,
        "misses_per_token": total_misses / n,
        "misses_per_token_per_layer": (total_misses / n) / len(moe_layers) if moe_layers else None,
        "per_layer": per_layer,
    }
    out = REPO / "pro_research" / "diag_hitrate_v4.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
