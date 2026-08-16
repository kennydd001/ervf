"""Resolves a real discrepancy between two measurements of the SAME
computation before trusting either: proto_multi_seq_full_model_n8.py
(cp.cuda.Event()-based) found N=8 collapses to 0.253x aggregate vs solo (a
4x slowdown); diag_n8_dispatch_vs_exec.py (time.perf_counter()-based, same
state-swap mechanism, same N=8) found N=8 scales almost perfectly linearly
with N (7.9-8.1x cost for 8x work) -- no collapse at all. These cannot both
be describing the same reality; agents/RESEARCH_NOTEBOOK.md's own
methodology requires resolving disagreeing measurements, not picking
whichever is more convenient.

This measures BOTH cp.cuda.Event() and time.perf_counter()+synchronize()
on the IDENTICAL loop iterations, back to back in the same run, eliminating
cross-script/cross-run confounds (GPU throttle state, warmup differences,
allocator state) as possible explanations for the disagreement.

Not a gated PRO experiment -- a methodology-reconciliation diagnostic,
read-only.
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

N = 8
DECODE_STEPS = 30


def snapshot_state(rt):
    rt._alloc_state()
    return {name: getattr(rt, name) for name in STATE_ATTRS}


def use_state(rt, state):
    for name, value in state.items():
        setattr(rt, name, value)


def save_state(rt, state):
    state["pos"] = rt.pos


def main() -> int:
    require_gpu_free()
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    rt.device_cache = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    ids_by_seq = [tok.encode(p, add_special_tokens=False) for p in PROMPTS[:N]]

    # ---- replicate proto_multi_seq_full_model_n8.py's Phase 2 (correctness
    # gate) workload VERBATIM before the timed measurement: this is the one
    # structural difference the earlier, simpler reconciliation test did NOT
    # have, and that script's Phase 2 does substantial prior GPU work (N=8
    # ground-truth generation via rt.reset(), then a full N=8 swapped-path
    # generation) building and discarding several full state sets before its
    # own Phase 3b timing -- testing directly whether THIS prior workload
    # pattern (not just a large static allocation, already ruled out by
    # diag_alloc_pressure.py) is what the earlier reconciliation test missed.
    ground_truth_tokens = [[] for _ in range(N)]
    for s in range(N):
        rt.reset()
        nxt = None
        for t in ids_by_seq[s]:
            nxt = int(rt.step(int(t)))
        ground_truth_tokens[s].append(nxt)
        for _ in range(DECODE_STEPS - 1):
            nxt = int(rt.step(nxt))
            ground_truth_tokens[s].append(nxt)
    cp.cuda.Device(0).synchronize()

    rt.enable_cache(72)
    phase2_state = [snapshot_state(rt) for _ in range(N)]
    phase2_cur = [None] * N
    for s in range(N):
        use_state(rt, phase2_state[s])
        rt.pos = 0
        nxt = None
        for t in ids_by_seq[s]:
            nxt = int(rt.step(int(t)))
        phase2_cur[s] = nxt
        save_state(rt, phase2_state[s])
    swapped_tokens = [[] for _ in range(N)]
    for s in range(N):
        swapped_tokens[s].append(phase2_cur[s])
    for _ in range(DECODE_STEPS - 1):
        for s in range(N):
            use_state(rt, phase2_state[s])
            phase2_cur[s] = int(rt.step(phase2_cur[s]))
            save_state(rt, phase2_state[s])
            swapped_tokens[s].append(phase2_cur[s])
    cp.cuda.Device(0).synchronize()
    print(f"Phase-2-replica equivalence: {ground_truth_tokens == swapped_tokens}", flush=True)
    del phase2_state, phase2_cur, ground_truth_tokens, swapped_tokens
    # ---- end Phase 2 replica -------------------------------------------

    # ---- Phase 3a replica: solo (N=1) Event()-timed measurement,
    # immediately before the N=8 measurement, matching the original
    # script's exact ordering (Phase 3a runs right before Phase 3b in the
    # SAME script, SAME rt object -- the one remaining structural piece
    # not yet reproduced by this reconciliation attempt).
    rt.enable_cache(72)
    ids_solo = tok.encode(PROMPTS[0], add_special_tokens=False)
    state_solo = snapshot_state(rt)
    use_state(rt, state_solo)
    rt.pos = 0
    nxt_solo = None
    for t in ids_solo:
        nxt_solo = int(rt.step(int(t)))
    save_state(rt, state_solo)
    cp.cuda.Device(0).synchronize()

    e0s, e1s = cp.cuda.Event(), cp.cuda.Event()
    e0s.record()
    for _ in range(DECODE_STEPS):
        use_state(rt, state_solo)
        nxt_solo = int(rt.step(nxt_solo))
        save_state(rt, state_solo)
    e1s.record()
    e1s.synchronize()
    solo_ms = cp.cuda.get_elapsed_time(e0s, e1s)
    solo_tok_s = DECODE_STEPS / (solo_ms / 1000.0)
    print(f"Phase-3a-replica solo: {solo_tok_s:.3f} tok/s", flush=True)
    del state_solo
    # ---- end Phase 3a replica --------------------------------------------

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

    # warmup, not measured
    for _ in range(5):
        for s in range(N):
            use_state(rt, state[s])
            cur[s] = int(rt.step(cur[s]))
            save_state(rt, state[s])
    cp.cuda.Device(0).synchronize()

    # ---- Method A: cp.cuda.Event(), exactly matching
    # proto_multi_seq_full_model_n8.py's own timing code verbatim -- INCLUDING
    # the tokens_by_seq[s].append(cur[s]) call inside the timed loop, which
    # this reconciliation script did NOT have until now (the one remaining
    # line-level difference found by re-reading the original script closely).
    tokens_by_seq = [[] for _ in range(N)]
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    for _ in range(DECODE_STEPS):
        for s in range(N):
            use_state(rt, state[s])
            cur[s] = int(rt.step(cur[s]))
            save_state(rt, state[s])
            tokens_by_seq[s].append(cur[s])
    e1.record()
    e1.synchronize()
    event_ms = cp.cuda.get_elapsed_time(e0, e1)

    cp.cuda.Device(0).synchronize()

    # ---- Method B: time.perf_counter(), same loop, immediately after
    # (same GPU/throttle state as Method A just finished, minimal gap).
    t0 = time.perf_counter()
    for _ in range(DECODE_STEPS):
        for s in range(N):
            use_state(rt, state[s])
            cur[s] = int(rt.step(cur[s]))
            save_state(rt, state[s])
    cp.cuda.Device(0).synchronize()
    t1 = time.perf_counter()
    perfcounter_ms = (t1 - t0) * 1000.0

    aggregate_tok_s_event = (N * DECODE_STEPS) / (event_ms / 1000.0)
    aggregate_tok_s_perfcounter = (N * DECODE_STEPS) / (perfcounter_ms / 1000.0)

    payload = {
        "kind": "diag_n8_timing_reconcile",
        "created_utc": utc_now(),
        "note": "measures cp.cuda.Event() (Method A, matching proto_multi_seq_full_model_n8.py exactly) and time.perf_counter()+synchronize() (Method B, matching diag_n8_dispatch_vs_exec.py) on the IDENTICAL loop, back to back in the same run, to resolve why the two scripts disagreed by ~4x on N=8's aggregate tok/s",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "n_sequences": N,
        "decode_steps": DECODE_STEPS,
        "phase3a_replica_solo_tok_s": solo_tok_s,
        "method_a_event_ms": event_ms,
        "method_a_aggregate_tok_s": aggregate_tok_s_event,
        "method_a_speedup_vs_solo": aggregate_tok_s_event / solo_tok_s if solo_tok_s else None,
        "method_b_perfcounter_ms": perfcounter_ms,
        "method_b_aggregate_tok_s": aggregate_tok_s_perfcounter,
        "ratio_b_over_a": perfcounter_ms / event_ms if event_ms else None,
        "methods_agree_within_20pct": (
            abs(event_ms - perfcounter_ms) / max(event_ms, perfcounter_ms) < 0.2
            if event_ms and perfcounter_ms else None
        ),
    }
    out = REPO / "pro_research" / "diag_n8_timing_reconcile.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
