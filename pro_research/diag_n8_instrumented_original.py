"""N=8 extension of the verified naive baseline mechanism
(proto_multi_seq_full_model.py, proto_multi_seq_full_model_n4.py). N=4's
own result was a genuine surprise -- the incidental warm-cache-only benefit
did NOT grow with N as isolated diagnostics suggested (+4.7% at N=4 vs
+5.4% at N=2, later both revised down to +2.05%/lower at a robust 40-step
horizon), and a follow-up found that simply enlarging the cache capacity to
match N made things WORSE, not better (0.706x, real regression, cause
verified NOT to be cache_assign's eviction scan). This tests whether the
flat-to-declining trend continues, worsens, or reverses at N=8 -- a genuinely
new data point on an already-verified, currently-best-performing mechanism
(this naive approach beats every "smart" explicit-sharing variant attempted
so far in absolute tok/s).

Same mechanism, same correctness discipline as the N=2/N=4 versions (state-
swap gated by a bit-exact interleaved equivalence check against independent
single-sequence ground truth before any timing claim).

Not a gated PRO experiment -- a scoped extension of an already-verified prototype.
"""

from __future__ import annotations

import sys
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
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    rt.device_cache = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    ids_by_seq_gt = [tok.encode(p, add_special_tokens=False) for p in PROMPTS[:N]]

    # ================= Phase 2: correctness gate (same discipline as N=2/N=4) =================
    ground_truth_tokens = [[] for _ in range(N)]
    for s in range(N):
        rt.reset()
        nxt = None
        for t in ids_by_seq_gt[s]:
            nxt = int(rt.step(int(t)))
        ground_truth_tokens[s].append(nxt)
        for _ in range(DECODE_STEPS - 1):
            nxt = int(rt.step(nxt))
            ground_truth_tokens[s].append(nxt)
    cp.cuda.Device(0).synchronize()

    rt.enable_cache(72)
    state = [snapshot_state(rt) for _ in range(N)]
    cur = [None] * N
    for s in range(N):
        use_state(rt, state[s])
        rt.pos = 0
        nxt = None
        for t in ids_by_seq_gt[s]:
            nxt = int(rt.step(int(t)))
        cur[s] = nxt
        save_state(rt, state[s])

    swapped_tokens = [[] for _ in range(N)]
    for s in range(N):
        swapped_tokens[s].append(cur[s])
    for _ in range(DECODE_STEPS - 1):
        for s in range(N):
            use_state(rt, state[s])
            cur[s] = int(rt.step(cur[s]))
            save_state(rt, state[s])
            swapped_tokens[s].append(cur[s])
    cp.cuda.Device(0).synchronize()

    equivalence_pass = (ground_truth_tokens == swapped_tokens)
    if not equivalence_pass:
        payload = {
            "kind": "proto_multi_seq_full_model_n8",
            "created_utc": utc_now(),
            "phase_reached": "phase2_correctness_gate_FAILED",
            "ground_truth_tokens": ground_truth_tokens,
            "swapped_tokens": swapped_tokens,
            "note": "state-swap mechanism did NOT reproduce independent single-sequence runs bit-exact under interleaved swapping at N=8.",
        }
        out = REPO / "pro_research" / "proto_multi_seq_full_model_n8.json"
        write_json_atomic(out, payload, archive=False)
        print(payload)
        return 1

    print(f"Phase 2 PASS at N=8: swapped-state run bit-exact matches ground truth ({DECODE_STEPS} tokens x {N} sequences)")

    # ================= Phase 3a: N=1 control, same config =================
    rt.enable_cache(72)
    ids_solo = tok.encode(PROMPTS[0], add_special_tokens=False)
    state_solo = snapshot_state(rt)
    use_state(rt, state_solo)
    rt.pos = 0
    nxt = None
    for t in ids_solo:
        nxt = int(rt.step(int(t)))
    save_state(rt, state_solo)
    cp.cuda.Device(0).synchronize()

    e0s, e1s = cp.cuda.Event(), cp.cuda.Event()
    e0s.record()
    for _ in range(DECODE_STEPS):
        use_state(rt, state_solo)
        nxt = int(rt.step(nxt))
        save_state(rt, state_solo)
    e1s.record()
    e1s.synchronize()
    solo_total_ms = cp.cuda.get_elapsed_time(e0s, e1s)
    solo_ms_per_token = solo_total_ms / DECODE_STEPS
    solo_tok_s = 1000.0 / solo_ms_per_token

    # ================= Phase 3b: real N=8 naive aggregate baseline =================
    rt.enable_cache(72)
    ids_by_seq = [tok.encode(p, add_special_tokens=False) for p in PROMPTS[:N]]
    state = [snapshot_state(rt) for _ in range(N)]
    cur_token = [None] * N
    for s in range(N):
        use_state(rt, state[s])
        rt.pos = 0
        nxt = None
        for t in ids_by_seq[s]:
            nxt = int(rt.step(int(t)))
        cur_token[s] = nxt
        save_state(rt, state[s])
    cp.cuda.Device(0).synchronize()

    import time as _time
    _t_wall_start = _time.perf_counter()
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    tokens_by_seq = [[] for _ in range(N)]
    for _ in range(DECODE_STEPS):
        for s in range(N):
            use_state(rt, state[s])
            cur_token[s] = int(rt.step(cur_token[s]))
            save_state(rt, state[s])
            tokens_by_seq[s].append(cur_token[s])
    e1.record()
    e1.synchronize()
    total_ms = cp.cuda.get_elapsed_time(e0, e1)
    _t_wall_end = _time.perf_counter()
    _wall_ms = (_t_wall_end - _t_wall_start) * 1000.0
    print(f"[INSTRUMENTED] cp.cuda.Event() total_ms={total_ms:.3f}  wall-clock perf_counter ms={_wall_ms:.3f}  ratio(wall/event)={_wall_ms/total_ms:.4f}", flush=True)

    total_real_tokens = N * DECODE_STEPS
    ms_per_token_aggregate = total_ms / total_real_tokens
    aggregate_tok_s = 1000.0 / ms_per_token_aggregate

    payload = {
        "kind": "proto_multi_seq_full_model_n8",
        "created_utc": utc_now(),
        "phase_reached": "phase3_naive_baseline_measured",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "n_sequences": N,
        "decode_steps_per_sequence": DECODE_STEPS,
        "phase2_equivalence_pass": equivalence_pass,
        "note": "N=8 extension of the verified naive baseline (N=2: +2.05% robust; N=4: +4.7% at 15 steps, flat vs N=2) -- tests whether the flat-to-declining trend continues, worsens, or reverses at N=8; the naive (no explicit sharing) mechanism remains the best-performing batch>1 approach measured this session in absolute tok/s",
        "solo_n1_same_config_ms_per_token": solo_ms_per_token,
        "solo_n1_same_config_tok_s": solo_tok_s,
        "total_wall_ms_for_all_real_tokens": total_ms,
        "total_real_tokens_across_all_sequences": total_real_tokens,
        "ms_per_real_token_aggregate": ms_per_token_aggregate,
        "aggregate_tok_s": aggregate_tok_s,
        "aggregate_speedup_vs_solo": aggregate_tok_s / solo_tok_s if solo_tok_s else None,
        "tokens_by_sequence": tokens_by_seq,
    }
    out = REPO / "pro_research" / "proto_multi_seq_full_model_n8.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
