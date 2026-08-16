"""What does one cross-stream event op cost in eager mode on this machine?

V14 (B3 overlap) came out bit-exact but **+3.65 ms/token slower**. The
mechanism it adds per expert slot is: one `record` on the gather stream, one
`wait_event` on the gather stream, one `wait_event` on the main stream, one
`record` on the main stream. With 23 MoE layers x 6 slots that is roughly
23 x 6 x 3 = 414 event operations per token that the serial version does not do.

If an event op costs ~8-9 us in eager mode on Windows/WDDM, 414 of them is
~3.6 ms -- which would account for the entire regression, and would mean the
overlap idea is sound but cannot be paid for outside a captured CUDA graph,
where fork/join become static graph edges with no runtime API cost.

That is a specific, falsifiable claim about a number nobody here has measured.
This measures it instead of asserting it.

Arms (all with a trivial kernel so the measurement is the API, not the work):
  baseline_launches   N kernel launches on one stream, no events
  record_only         N launches + one event record each
  full_pattern        N launches + the exact record/wait/wait/record pattern
                      V14 issues per slot

Read-only, no model load.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_gpu_free, utc_now, write_json_atomic

N_OPS = 414          # 23 MoE layers x 6 slots x 3 event ops, one token's worth
REPEATS = 50

SRC = r"""
extern "C" __global__ void tiny(float* p) { if (threadIdx.x == 0) p[0] += 1.0f; }
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    mod = cp.RawModule(code=SRC, options=("-std=c++14",))
    tiny = mod.get_function("tiny")
    buf = cp.zeros(1, dtype=cp.float32)

    main_s = cp.cuda.get_current_stream()
    side = cp.cuda.Stream(non_blocking=True)
    evs_a = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(N_OPS)]
    evs_b = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(N_OPS)]

    def timed(fn):
        fn()
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(REPEATS):
            fn()
        cp.cuda.Device(0).synchronize()
        return (time.perf_counter_ns() - t0) / 1e6 / REPEATS

    def baseline():
        for _ in range(N_OPS):
            tiny((1,), (32,), (buf,))

    def record_only():
        for i in range(N_OPS):
            tiny((1,), (32,), (buf,))
            evs_a[i].record(main_s)

    def full_pattern():
        for i in range(N_OPS):
            with side:
                side.wait_event(evs_b[i - 1] if i else evs_b[0])
                tiny((1,), (32,), (buf,))
                evs_a[i].record(side)
            main_s.wait_event(evs_a[i])
            tiny((1,), (32,), (buf,))
            evs_b[i].record(main_s)

    ms_base = timed(baseline)
    ms_rec = timed(record_only)
    ms_full = timed(full_pattern)

    per_record = (ms_rec - ms_base) / N_OPS * 1000.0
    # full_pattern issues 2 launches + 2 records + 2 waits per iteration
    per_iter_extra = (ms_full - 2 * ms_base) / N_OPS * 1000.0

    payload = {
        "kind": "diag_event_op_cost",
        "created_utc": utc_now(),
        "note": "prices the cross-stream event machinery V14 adds, to test whether the +3.65 ms/token regression is the API cost rather than a failure of the overlap idea",
        "n_ops_per_arm": N_OPS, "repeats": REPEATS,
        "arms": {
            "baseline_launches_ms": ms_base,
            "record_only_ms": ms_rec,
            "full_pattern_ms": ms_full,
        },
        "derived": {
            "us_per_event_record": per_record,
            "us_per_full_slot_pattern": per_iter_extra,
            "v14_regression_ms_measured": 3.6518,
            "predicted_regression_ms_from_event_cost": per_iter_extra * N_OPS / 3.0 / 1000.0,
        },
    }
    write_json_atomic(REPO / "pro_research" / "diag_event_op_cost.json", payload,
                      archive=False)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
