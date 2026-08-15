"""S9-A: is the MoE term launch-bound? Measure before building anything.

S8 put MoE at 39.523 ms/token (23 layers x 1.718 ms) against a ~1.5 ms
bandwidth roofline. The stated suspicion was launch overhead: roughly 4 kernels
per expert x 138 experts ~= 550 launches per token. This measures the actual
per-launch cost and the per-expert kernel count, so H-S9 (batch six experts into
one launch) is either justified or dropped without writing a kernel.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL = REPO / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning_v35")
OUT = REPO / "reports" / "lightningstream_nemotron" / "s9a_launch_probe.json"


def main() -> int:
    import cupy as cp

    sync = cp.cuda.Device(0).synchronize
    probe = cp.RawKernel(r"""
extern "C" __global__ void nop(float* o){ if (threadIdx.x == 1<<20) o[0] = 1.0f; }
""", "nop")
    o = cp.zeros(1, dtype=cp.float32)

    # --- bare launch cost, no memory traffic
    for _ in range(200):
        probe((1,), (128,), (o,))
    sync()
    N = 2000
    t0 = time.perf_counter_ns()
    for _ in range(N):
        probe((1,), (128,), (o,))
    sync()
    launch_us = (time.perf_counter_ns() - t0) / 1e3 / N

    # --- same, with a realistic grid so it is not a degenerate case
    for _ in range(100):
        probe((1856,), (256,), (o,))
    sync()
    t0 = time.perf_counter_ns()
    for _ in range(N):
        probe((1856,), (256,), (o,))
    sync()
    launch_big_us = (time.perf_counter_ns() - t0) / 1e3 / N

    # --- count kernels actually issued per expert call
    rt = LightningRuntime(MODEL, contexts_max=4096, embed_on_host=True)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rng = np.random.default_rng(3)
    varied = [int(v) for v in rng.integers(1000, 60000, size=64)]
    rt.reset()
    for j in range(24):
        rt.step(varied[j % len(varied)])
    sync()

    counter = {"n": 0}
    orig_expert = rt.fused.expert
    orig_acc = rt.fused.accumulate_into
    orig_gemv = rt.fused.gemv_into

    def wrap(f):
        def g(*a, **kw):
            counter["n"] += 1
            return f(*a, **kw)
        return g

    rt.fused.expert = wrap(orig_expert)
    rt.fused.accumulate_into = wrap(orig_acc)
    rt.fused.gemv_into = wrap(orig_gemv)
    moe_i = rt.moe_layers[0]
    counter["n"] = 0
    rt._moe(moe_i, rt.acc)
    sync()
    calls_per_layer = counter["n"]
    rt.fused.expert, rt.fused.accumulate_into, rt.fused.gemv_into = (
        orig_expert, orig_acc, orig_gemv)

    # fused.expert itself issues 2 gemv launches; accumulate/gemv_into issue 1
    est_launches_layer = calls_per_layer * 2
    est_launches_token = est_launches_layer * 23
    moe_measured_ms = 39.523
    launch_share_ms = est_launches_token * launch_big_us / 1e3

    res = {
        "kind": "lightningstream_nemotron_s9a_launch_probe",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "bare_launch_us": launch_us,
        "launch_grid1856_us": launch_big_us,
        "fused_calls_per_moe_layer": calls_per_layer,
        "estimated_launches_per_moe_layer": est_launches_layer,
        "estimated_launches_per_token_moe": est_launches_token,
        "moe_measured_ms_at_262k": moe_measured_ms,
        "launch_overhead_ms_estimate": launch_share_ms,
        "launch_share_of_moe": launch_share_ms / moe_measured_ms,
        "verdict": ("launch-bound: batching justified"
                    if launch_share_ms / moe_measured_ms > 0.30
                    else "NOT launch-bound: H-S9 dropped, cause is elsewhere"),
        "claim_boundary": ("Launch cost measured with an empty kernel on this "
                           "GPU; the per-token estimate multiplies that by the "
                           "counted launches. It is an estimate of a bound, not "
                           "a measurement of the MoE term."),
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for k, v in res.items():
        if k not in ("kind", "completed_utc", "claim_boundary"):
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
