"""Read-only diagnostic: full component breakdown of V6's ~22.6 ms/token,
using the same ablation technique diag_down_ablation_timing.py validated
(cp.cuda.get_elapsed_time on graph-captured events raises
cudaErrorInvalidValue on this stack, so wall-clock ablation is the reliable
method).

Builds V6's exact stack (device routing + graph-safe + selective ERVF +
batched down_proj) once per arm and times it: REAL (unmodified) and four
STUB variants, each with one whole sub-block replaced by a no-op before
setup_graph() captures. STUB arms produce WRONG tokens by design and are
timing-only, never a correctness claim.

  STUB_ATTN:  rt._attention -> out.fill(0)   (6 attention layers)
  STUB_MAMBA: rt._mamba -> out.fill(0)       (23 Mamba layers)
  STUB_MOE:   rt._moe -> out.fill(0), (None, None)   (23 MoE layers, shared + routed)
  STUB_GEMV_INTO: fused.gemv_into stubbed globally -- this hits BOTH the
    lm_head GEMV (1x/token) AND the shared-expert up/down GEMVs inside
    _moe_dev (2x/layer x 23 layers = 46x/token), since both call the same
    method. Labelled "lmhead_plus_shared_expert", not "lmhead" alone -- a
    real measurement, just a coarser one than originally intended. This
    bound OVERLAPS with the "moe" bound (both include shared-expert cost) --
    they must not be summed as if disjoint.

Each arm runs as a SEPARATE PROCESS (invoked with --arm=X), because building
five full 30B runtimes sequentially in one process exhausts pinned host
memory (load_routed_bank's pinned allocation is not being fully released
between cupy memory-pool frees + gc.collect() in-process; a fresh process
guarantees the OS reclaims it). --aggregate reads all five per-arm JSONs and
computes the bounds. Not a gated PRO experiment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, percentiles, require_gpu_free, require_model_dir, utc_now, write_json_atomic

ARMS = ("real", "attn", "mamba", "moe", "lmhead")
STUB_LABELS = {"attn": "attn", "mamba": "mamba", "moe": "moe", "lmhead": "lmhead_plus_shared_expert"}


def _arm_out_path(arm: str) -> Path:
    return REPO / "pro_research" / f"diag_v6_component_breakdown_arm_{arm}.json"


def run_one_arm(arm: str) -> int:
    require_gpu_free()
    import cupy as cp
    from down_proj_batch_kernels import DownProjBatchKernels
    from ervf_dense import DenseERVF
    from moe_dev_batched import install_batched_moe_dev
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    from selective_ervf_v3 import _install_selective

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.device_cache = True
    rt.deterministic_accum = True

    dense = DenseERVF()
    restore_sel, _ = _install_selective(rt, dense)
    batch_kernels = DownProjBatchKernels()
    restore_moe = install_batched_moe_dev(rt, batch_kernels)

    if arm == "attn":
        def stub_attn(self, i, out):
            out.fill(0)
        rt._attention = types.MethodType(stub_attn, rt)
    elif arm == "mamba":
        def stub_mamba(self, i, out):
            out.fill(0)
        rt._mamba = types.MethodType(stub_mamba, rt)
    elif arm == "moe":
        def stub_moe(self, i, out):
            out.fill(0)
            return None, None
        rt._moe = types.MethodType(stub_moe, rt)
    elif arm == "lmhead":
        def stub_gemv_into(out, *a, **kw):
            pass
        rt.fused.gemv_into = stub_gemv_into

    rt.setup_graph()
    restore_sel()
    restore_moe()

    for _ in range(10):
        rt.step_graph(None)
    rt._graph_stream.synchronize()

    rounds = 150
    token_ms = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        rt.step_graph(None)
        rt._graph_stream.synchronize()
        token_ms.append((time.perf_counter_ns() - t0) / 1e6)

    payload = {
        "kind": "diag_v6_component_breakdown_arm",
        "arm": arm,
        "created_utc": utc_now(),
        "rounds": rounds,
        "token_ms": percentiles(token_ms),
    }
    write_json_atomic(_arm_out_path(arm), payload, archive=False)
    print(payload)
    return 0


def aggregate() -> int:
    import json

    arms = {}
    for arm in ARMS:
        p = _arm_out_path(arm)
        if not p.exists():
            print(f"missing arm result: {arm} ({p})")
            return 1
        d = json.loads(p.read_text(encoding="utf-8"))
        label = "real" if arm == "real" else STUB_LABELS[arm]
        arms[label] = d["token_ms"]

    real_p50 = arms["real"]["p50"]
    bounds = {}
    for label in ("attn", "mamba", "moe", "lmhead_plus_shared_expert"):
        stub_p50 = arms[label]["p50"]
        gap = real_p50 - stub_p50
        bounds[label] = {"upper_bound_ms_per_token": gap, "fraction_of_real_token": gap / real_p50 if real_p50 else None}

    known_down_proj_ms = 6.5058  # diag_down_ablation_timing.py, same in-graph ablation method
    payload = {
        "kind": "diag_v6_component_breakdown",
        "created_utc": utc_now(),
        "note": "read-only, timing-only; STUB arms produce wrong tokens by design, never a correctness claim; moe and lmhead_plus_shared_expert bounds OVERLAP (both include shared-expert cost) and must not be summed as disjoint",
        "arms_token_ms": arms,
        "upper_bounds": bounds,
        "cross_reference": {
            "down_proj_in_graph_upper_bound_ms_from_diag_down_ablation_timing": known_down_proj_ms,
            "moe_bound_minus_down_proj_bound_approx_shared_plus_up_proj_ms": (
                bounds["moe"]["upper_bound_ms_per_token"] - known_down_proj_ms
            ),
        },
    }
    out = REPO / "pro_research" / "diag_v6_component_breakdown.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--drive", action="store_true", help="spawn one subprocess per arm, then aggregate")
    args = ap.parse_args()

    if args.drive:
        for arm in ARMS:
            print(f"=== running arm: {arm} ===", flush=True)
            rc = subprocess.run([sys.executable, __file__, "--arm", arm]).returncode
            if rc != 0:
                print(f"arm {arm} failed with exit code {rc}")
                return rc
        return aggregate()
    if args.aggregate:
        return aggregate()
    if args.arm:
        return run_one_arm(args.arm)
    ap.error("pass --arm=X, --aggregate, or --drive")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
