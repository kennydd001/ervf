"""Read-only diagnostic: per-component GPU time inside _moe_dev.

Not a gated PRO experiment. Wraps rt.fused.gemv_ervf_indirect (up-proj ERVF
GEMV, reads from the device cache) and rt.fused.down_masked_into_indirect
(down-proj masked/sparse gather, reads from the host-mapped bank on every
call regardless of hit/miss) with cp.cuda.Event pairs, in EAGER mode so each
call is real Python -- unlike inside a captured graph, timings here are
per-call ground truth, not a one-shot capture-time snapshot.

Purpose: the diagnostic in diag_hitrate_v4.py showed 85.6% up-proj hit rate
but could not establish how many milliseconds/token the down-proj masked
gather (which never benefits from that cache) actually costs. Earlier
reasoning from static byte-size constants was corrected twice already in this
session (DOWN_PANEL_BYTES is ~2.68 MB/expert panel, not ~1 KB) -- this script
replaces further arithmetic with a direct measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, percentiles, require_gpu_free, require_model_dir, utc_now, write_json_atomic


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

    fused = rt.fused
    orig_up = fused.gemv_ervf_indirect
    orig_down = fused.down_masked_into_indirect

    up_events: list[tuple] = []
    down_events: list[tuple] = []

    def timed_up(*args, **kwargs):
        e0 = cp.cuda.Event()
        e1 = cp.cuda.Event()
        e0.record()
        r = orig_up(*args, **kwargs)
        e1.record()
        up_events.append((e0, e1))
        return r

    def timed_down(*args, **kwargs):
        e0 = cp.cuda.Event()
        e1 = cp.cuda.Event()
        e0.record()
        r = orig_down(*args, **kwargs)
        e1.record()
        down_events.append((e0, e1))
        return r

    fused.gemv_ervf_indirect = timed_up
    fused.down_masked_into_indirect = timed_down

    prompt = "The history of computing began when"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    ids = tok.encode(prompt, add_special_tokens=False)

    rt.reset()
    nxt = None
    for t in ids:
        nxt = int(rt.step(int(t)))
    cp.cuda.Device(0).synchronize()

    # Discard prompt-phase events (warmup/compile noise), time only decode.
    up_events.clear()
    down_events.clear()

    n = 128
    cur = nxt
    token_ms: list[float] = []
    import time
    for _ in range(n):
        t0 = time.perf_counter_ns()
        cur = int(rt.step(cur))
        cp.cuda.Device(0).synchronize()
        token_ms.append((time.perf_counter_ns() - t0) / 1e6)

    fused.gemv_ervf_indirect = orig_up
    fused.down_masked_into_indirect = orig_down

    up_ms = [cp.cuda.get_elapsed_time(a, b) for a, b in up_events]
    down_ms = [cp.cuda.get_elapsed_time(a, b) for a, b in down_events]

    calls_per_token = len(up_events) / n if n else 0
    payload = {
        "kind": "diag_component_timing_v4",
        "created_utc": utc_now(),
        "note": "read-only diagnostic, not a gated PRO experiment; eager device_cache path",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "fused_nvfp4.py",)),
        "prompt": prompt,
        "generated_tokens": n,
        "up_gemv_calls": len(up_events),
        "down_gather_calls": len(down_events),
        "calls_per_token": calls_per_token,
        "token_ms": percentiles(token_ms),
        "up_gemv_ms_per_call": percentiles(up_ms),
        "down_gather_ms_per_call": percentiles(down_ms),
        "up_gemv_ms_per_token_total": sum(up_ms) / n,
        "down_gather_ms_per_token_total": sum(down_ms) / n,
        "up_plus_down_fraction_of_token": (sum(up_ms) + sum(down_ms)) / sum(token_ms) if sum(token_ms) else None,
    }
    out = REPO / "pro_research" / "diag_component_timing_v4.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
