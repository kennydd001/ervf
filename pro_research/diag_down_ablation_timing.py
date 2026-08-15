"""Read-only diagnostic: upper-bound the in-graph cost of down_proj by
ablation, since cp.cuda.Event.get_elapsed_time() on events captured inside a
CUDA graph raises cudaErrorInvalidValue on this stack (tried in
diag_down_ingraph_timing.py -- a real technical limitation, not a bug in
that script; same class of finding as G2's cudaGraphLaunch restriction).

Method: build V4's exact graph twice.
  - REAL: unmodified, exactly what V4 measured (41.13 tok/s, 24.3152 ms/token).
  - STUB: fused.down_masked_into_indirect replaced with a cheap no-op (writes
    zeros to `out`, skips the gather/scan/compute/reduce entirely) before
    setup_graph() captures. Produces WRONG tokens -- this arm is never token-
    compared and must never be cited as a correctness result. Timing-only.

token_time(REAL) - token_time(STUB) is an UPPER BOUND on what any down_proj
optimization (batching, PCIe restructuring, or both) could possibly recover
in-graph, since a true no-op is strictly cheaper than any correct
replacement. This decides whether V5 (PRO_V5_PREREGISTRATION.md) is worth
building before writing any new kernel.

Not a gated PRO experiment. No claim about correctness or a speed win --
purely bounds the opportunity.
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


def _build_and_time(stub_down: bool, rounds: int = 200):
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.device_cache = True
    rt.deterministic_accum = True

    dense = DenseERVF()
    restore, _counters = _install_selective(rt, dense)  # exactly V4's setup

    fused = rt.fused
    orig_down = fused.down_masked_into_indirect
    if stub_down:
        def stub(out, *a, **kw):
            out.fill(0)
        fused.down_masked_into_indirect = stub

    rt.setup_graph()

    if stub_down:
        fused.down_masked_into_indirect = orig_down
    restore()

    for _ in range(10):
        rt.step_graph(None)
    rt._graph_stream.synchronize()

    token_ms = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        rt.step_graph(None)
        rt._graph_stream.synchronize()
        token_ms.append((time.perf_counter_ns() - t0) / 1e6)

    del rt, dense
    import gc
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return token_ms


def main() -> int:
    require_gpu_free()

    real_ms = _build_and_time(stub_down=False)
    stub_ms = _build_and_time(stub_down=True)

    real_p = percentiles(real_ms)
    stub_p = percentiles(stub_ms)
    gap = real_p["p50"] - stub_p["p50"] if real_p["p50"] and stub_p["p50"] else None

    payload = {
        "kind": "diag_down_ablation_timing",
        "created_utc": utc_now(),
        "note": (
            "read-only, timing-only diagnostic; STUB arm produces WRONG tokens "
            "by design (down_masked_into_indirect replaced with out.fill(0)) "
            "and must never be read as a correctness result"
        ),
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "rounds_per_arm": 200,
        "real_token_ms": real_p,
        "stub_no_down_proj_token_ms": stub_p,
        "upper_bound_down_proj_ingraph_ms_per_token": gap,
        "upper_bound_fraction_of_real_token": (gap / real_p["p50"]) if gap and real_p["p50"] else None,
        "v4_reference_full_result_ms_per_token": 24.3152,
        "eager_reference_for_comparison": {
            "note": "diag_component_timing_v4.json / diag_down_subkernels_v4.json, same GPU/model, eager (no graph)",
            "down_pipeline_ms_per_token_eager": 9.573,
            "down_pipeline_ms_per_token_eager_subkernel_split": 11.390,
        },
    }
    out = REPO / "pro_research" / "diag_down_ablation_timing.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
