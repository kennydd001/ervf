"""Read-only diagnostic: does down_proj still cost what eager measured, once
it's inside V4's captured CUDA graph?

diag_component_timing_v4.py and diag_down_subkernels_v4.py measured the
down_proj pipeline in EAGER mode (9.57-11.39 ms/token, framed panel_scan +
reduce_partials as launch-overhead-bound: 552 small host-dispatched kernel
launches per token). But V4 runs this exact pipeline inside a captured CUDA
graph, which is specifically designed to remove host-side launch overhead --
the eager number may substantially overstate what's recoverable by batching.

cp.cuda.Event.record() is itself a capturable stream operation: if the event
wrapper is installed before rt.setup_graph() captures, the event-record nodes
become part of the graph and re-fire on every replay. This measures the
REAL in-graph cost, not a re-derivation from eager timing.

setup_graph() runs the token body twice: once as an uncaptured warmup (which
also passes through the wrapper, but its events are not part of the graph),
then once inside begin_capture()/end_capture(). Down_masked_into_indirect is
called exactly 138 times per token body (23 MoE layers x top_k 6, confirmed
by diag_component_timing_v4.py's down_gather_calls count) -- so the LAST 138
recorded event pairs are the ones actually baked into the graph; the first
138 are warmup noise, discarded.

Not a gated PRO experiment.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, percentiles, require_gpu_free, require_model_dir, utc_now, write_json_atomic
from ervf_dense import DenseERVF
from selective_ervf_v3 import _install_selective

CALLS_PER_TOKEN = 138  # 23 MoE layers x top_k 6, matches diag_component_timing_v4.py


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

    dense = DenseERVF()
    restore, _counters = _install_selective(rt, dense)  # match V4 exactly

    fused = rt.fused
    orig_down = fused.down_masked_into_indirect
    orig_up = fused.gemv_ervf_indirect
    # Bracket with ONE pair of events per pass (first-call start, last-call
    # end) instead of one pair per call -- fewer captured event nodes, lower
    # chance of hitting a WDDM/graph event-timing edge case.
    state = {"down_calls": 0, "up_calls": 0}
    down_start = cp.cuda.Event()
    down_end = cp.cuda.Event()
    up_start = cp.cuda.Event()
    up_end = cp.cuda.Event()

    def wrap_down(*args, **kwargs):
        state["down_calls"] += 1
        is_first = state["down_calls"] % CALLS_PER_TOKEN == 1
        if is_first:
            down_start.record()
        r = orig_down(*args, **kwargs)
        if state["down_calls"] % CALLS_PER_TOKEN == 0:
            down_end.record()
        return r

    def wrap_up(*args, **kwargs):
        state["up_calls"] += 1
        is_first = state["up_calls"] % CALLS_PER_TOKEN == 1
        if is_first:
            up_start.record()
        r = orig_up(*args, **kwargs)
        if state["up_calls"] % CALLS_PER_TOKEN == 0:
            up_end.record()
        return r

    fused.down_masked_into_indirect = wrap_down
    fused.gemv_ervf_indirect = wrap_up

    rt.setup_graph()  # warmup (uncaptured, calls 1..138) + capture (calls 139..276)

    fused.down_masked_into_indirect = orig_down
    fused.gemv_ervf_indirect = orig_up
    restore()

    assert state["down_calls"] == 2 * CALLS_PER_TOKEN, f"expected {2*CALLS_PER_TOKEN} down calls, got {state['down_calls']}"
    assert state["up_calls"] == 2 * CALLS_PER_TOKEN, f"expected {2*CALLS_PER_TOKEN} up calls, got {state['up_calls']}"
    captured_down = [(down_start, down_end)]
    captured_up = [(up_start, up_end)]

    # Warm the graph a few times before measuring (first replay pays one-time costs).
    for _ in range(5):
        rt.step_graph(None)
    rt._graph_stream.synchronize()

    rounds = 20
    down_ms_per_round: list[float] = []
    up_ms_per_round: list[float] = []
    token_ms: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        rt.step_graph(None)
        rt._graph_stream.synchronize()
        token_ms.append((time.perf_counter_ns() - t0) / 1e6)
        down_ms_per_round.append(sum(cp.cuda.get_elapsed_time(a, b) for a, b in captured_down))
        up_ms_per_round.append(sum(cp.cuda.get_elapsed_time(a, b) for a, b in captured_up))

    payload = {
        "kind": "diag_down_ingraph_timing",
        "created_utc": utc_now(),
        "note": "read-only diagnostic; measures down_proj/up_proj cost INSIDE V4's captured graph via captured event-record nodes, not eager re-derivation",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "rounds": rounds,
        "token_ms": percentiles(token_ms),
        "down_pipeline_ms_per_token_ingraph": percentiles(down_ms_per_round),
        "up_gemv_ms_per_token_ingraph": percentiles(up_ms_per_round),
        "down_fraction_of_token_ingraph": (sum(down_ms_per_round) / sum(token_ms)) if sum(token_ms) else None,
        "eager_reference_for_comparison": {
            "note": "from diag_component_timing_v4.json / diag_down_subkernels_v4.json, same GPU, same model, eager device_cache path (no graph)",
            "down_pipeline_ms_per_token_eager": 9.573,
            "down_pipeline_ms_per_token_eager_subkernel_split": 11.390,
            "up_gemv_ms_per_token_eager": 5.003,
            "eager_token_ms_p50": 33.46,
        },
    }
    out = REPO / "pro_research" / "diag_down_ingraph_timing.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
