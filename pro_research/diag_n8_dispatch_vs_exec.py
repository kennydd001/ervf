"""Direct empirical test of the explanation diag_n8_cache_hitrate.py's
result pointed to (agents/RESEARCH_NOTEBOOK.md 2026-08-16, "Vervolg... het
mysterie eindelijk op"): that proto_multi_seq_full_model_n8.py's collapse
(0.253x, 4x slower than solo) is Python-orchestration/kernel-launch
overhead scaling with N, not a cache effect (hit rate was HIGHER at N=8,
ruling that out directly).

CUDA kernel launches are asynchronous: cp.cuda.Device(0).synchronize() is
what actually waits for GPU execution to finish. This measures, for one
real N=8 decode step using the SAME verified state-swap mechanism: (a) pure
Python dispatch time -- wall clock from issuing every kernel call in the
step to the LAST call returning, WITHOUT any synchronize() in between (CUDA
queues launches near-instantly if the queue isn't full; this approximates
the CPU-side cost of state-swapping + launching hundreds of kernels), then
(b) one synchronize() at the very end to get true total wall time. If (a)
is a large fraction of (b), that directly confirms Python/launch overhead
-- not GPU compute or PCIe transfer -- dominates at this N.

Not a gated PRO experiment -- a root-cause diagnostic, read-only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, require_gpu_free, require_model_dir, utc_now, write_json_atomic

PROMPTS = [
    "The history of computing began when",
    "Write a correct Python function that computes the longest increasing subsequence length in O(n log n), then explain its invariant.\n",
    "The recipe calls for two cups of flour, a pinch of salt, and",
    "In the quiet village, the old fisherman noticed something strange about the tide",
    "The quarterly earnings report showed a significant increase in revenue driven by",
    "Photosynthesis is the process by which plants convert light energy into",
    "The defendant's attorney argued that the evidence presented by the prosecution was",
    "To configure the network firewall, first navigate to the settings panel and",
]

STATE_ATTRS = [
    "ssm", "conv", "kc", "vc", "kv_dim", "pos",
    "h", "tmp", "acc", "normed", "act", "_act_moe", "_act_shared",
    "proj", "convo", "dt", "y", "gn", "qv", "kv_", "vv", "ctx",
    "logits", "rlog", "route_pack",
    "stage_c", "stage_s", "mstate", "contrib",
    "copy_stream", "evt", "part_acc", "part_ml",
]

DECODE_STEPS_WARMUP = 5
MEASURE_STEPS = 30  # matches proto_multi_seq_full_model_n8.py's DECODE_STEPS
                    # exactly (queue depth matters -- see the docstring note
                    # added 2026-08-16 about reconciling this against that
                    # script's cp.cuda.Event()-based 0.253x collapse finding)


def snapshot_state(rt):
    rt._alloc_state()
    return {name: getattr(rt, name) for name in STATE_ATTRS}


def use_state(rt, state):
    for name, value in state.items():
        setattr(rt, name, value)


def save_state(rt, state):
    state["pos"] = rt.pos


def run_dispatch_vs_exec(rt, tok, N):
    cp = rt.cp
    ids_by_seq = [tok.encode(PROMPTS[s], add_special_tokens=False) for s in range(N)]
    rt.enable_cache(72)
    state = [snapshot_state(rt) for _ in range(N)]
    cur = [None] * N
    for s in range(N):
        use_state(rt, state[s])
        rt.pos = 0
        nxt = None
        for t in ids_by_seq[s]:
            nxt = int(rt.step(int(t)))
        cur[s] = nxt
        save_state(rt, state[s])
    cp.cuda.Device(0).synchronize()

    # warmup (not measured) -- lets any first-call lazy allocation happen
    # outside the timed region.
    for _ in range(DECODE_STEPS_WARMUP):
        for s in range(N):
            use_state(rt, state[s])
            cur[s] = int(rt.step(cur[s]))
            save_state(rt, state[s])
    cp.cuda.Device(0).synchronize()

    # (a) pure dispatch: issue MEASURE_STEPS steps worth of calls back to
    # back with NO synchronize() until the very end -- approximates
    # CPU-side Python + launch-issuance cost, since CUDA queues launches
    # without blocking the host as long as the queue has room.
    t_dispatch_start = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        for s in range(N):
            use_state(rt, state[s])
            cur[s] = int(rt.step(cur[s]))
            save_state(rt, state[s])
    t_dispatch_end = time.perf_counter()
    dispatch_only_ms = (t_dispatch_end - t_dispatch_start) * 1000.0

    # (b) full wall time including the final synchronize() -- true total,
    # continuing from where dispatch left off (already includes the
    # dispatch time measured above; this synchronize() is what makes it
    # "real" wall time as opposed to just queueing time).
    cp.cuda.Device(0).synchronize()
    t_total_end = time.perf_counter()
    total_wall_ms = (t_total_end - t_dispatch_start) * 1000.0

    return {
        "n_sequences": N,
        "measure_steps": MEASURE_STEPS,
        "dispatch_only_ms": dispatch_only_ms,
        "total_wall_ms_incl_sync": total_wall_ms,
        "gpu_exec_tail_ms": total_wall_ms - dispatch_only_ms,
        "dispatch_fraction_of_total": dispatch_only_ms / total_wall_ms if total_wall_ms else None,
        "ms_per_step_total": total_wall_ms / MEASURE_STEPS,
    }


def main() -> int:
    require_gpu_free()
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    rt.device_cache = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    result_n1 = run_dispatch_vs_exec(rt, tok, 1)
    print(f"N=1: {result_n1}", flush=True)
    result_n8 = run_dispatch_vs_exec(rt, tok, 8)
    print(f"N=8: {result_n8}", flush=True)

    payload = {
        "kind": "diag_n8_dispatch_vs_exec",
        "created_utc": utc_now(),
        "note": "empirically tests whether Python-dispatch/kernel-launch-issuance time (measured WITHOUT synchronize() until the end, so it captures CPU-side queueing cost, not GPU execution) is a large and/or N-scaling fraction of total wall time -- the concrete follow-up to diag_n8_cache_hitrate.py's finding that ruled out caching as the cause of proto_multi_seq_full_model_n8.py's 0.253x collapse",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "result_n1": result_n1,
        "result_n8": result_n8,
        "dispatch_fraction_grew_with_n": (
            result_n8["dispatch_fraction_of_total"] is not None
            and result_n1["dispatch_fraction_of_total"] is not None
            and result_n8["dispatch_fraction_of_total"] > result_n1["dispatch_fraction_of_total"] + 0.1
        ),
        "ms_per_step_ratio_n8_over_n1": (
            result_n8["ms_per_step_total"] / result_n1["ms_per_step_total"]
            if result_n1["ms_per_step_total"] else None
        ),
    }
    out = REPO / "pro_research" / "diag_n8_dispatch_vs_exec.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
