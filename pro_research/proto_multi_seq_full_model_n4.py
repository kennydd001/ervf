"""N=4 extension of proto_multi_seq_full_model.py's verified naive baseline
(bit-exact state-swap mechanism, N=2 result: +5.4% aggregate from incidental
warm-cache reuse alone). Earlier diagnostics this session
(diag_batch_warm_cache.py, diag_cross_sequence_union.py) found warm-cache/
union-overlap benefits GROW with N (more sequences -> more chance of
expert overlap) -- this checks whether that trend shows up in the same real,
end-to-end, verified measurement as the N=2 result, not just in isolated
routing-overlap counts.

Same mechanism, same correctness discipline as proto_multi_seq_full_model.py
(reused verbatim, not reimplemented): state-swap gated by a bit-exact
interleaved equivalence check against independent single-sequence ground
truth before any timing claim.

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
]

STATE_ATTRS = [
    "ssm", "conv", "kc", "vc", "kv_dim", "pos",
    "h", "tmp", "acc", "normed", "act", "_act_moe", "_act_shared",
    "proj", "convo", "dt", "y", "gn", "qv", "kv_", "vv", "ctx",
    "logits", "rlog", "route_pack",
    "stage_c", "stage_s", "mstate", "contrib",
    "copy_stream", "evt", "part_acc", "part_ml",
]

N = 4
DECODE_STEPS = 15


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

    # ================= Phase 2: correctness gate (same discipline as N=2) =================
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
            "kind": "proto_multi_seq_full_model_n4",
            "created_utc": utc_now(),
            "phase_reached": "phase2_correctness_gate_FAILED",
            "ground_truth_tokens": ground_truth_tokens,
            "swapped_tokens": swapped_tokens,
            "note": "state-swap mechanism did NOT reproduce independent single-sequence runs bit-exact under interleaved swapping at N=4.",
        }
        out = REPO / "pro_research" / "proto_multi_seq_full_model_n4.json"
        write_json_atomic(out, payload, archive=False)
        print(payload)
        return 1

    print(f"Phase 2 PASS at N=4: swapped-state run bit-exact matches ground truth ({DECODE_STEPS} tokens x {N} sequences)")

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

    # ================= Phase 3b: real N=4 naive aggregate baseline =================
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

    total_real_tokens = N * DECODE_STEPS
    ms_per_token_aggregate = total_ms / total_real_tokens
    aggregate_tok_s = 1000.0 / ms_per_token_aggregate

    payload = {
        "kind": "proto_multi_seq_full_model_n4",
        "created_utc": utc_now(),
        "phase_reached": "phase3_naive_baseline_measured",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "n_sequences": N,
        "decode_steps_per_sequence": DECODE_STEPS,
        "phase2_equivalence_pass": equivalence_pass,
        "note": "N=4 extension of proto_multi_seq_full_model.py's N=2 naive baseline (+5.4% aggregate) -- same mechanism, checks whether the incidental warm-cache benefit grows with N as isolated diagnostics (diag_batch_warm_cache.py, diag_cross_sequence_union.py) suggested it should",
        "solo_n1_same_config_ms_per_token": solo_ms_per_token,
        "solo_n1_same_config_tok_s": solo_tok_s,
        "total_wall_ms_for_all_real_tokens": total_ms,
        "total_real_tokens_across_all_sequences": total_real_tokens,
        "ms_per_real_token_aggregate": ms_per_token_aggregate,
        "aggregate_tok_s": aggregate_tok_s,
        "aggregate_speedup_vs_solo": aggregate_tok_s / solo_tok_s if solo_tok_s else None,
        "tokens_by_sequence": tokens_by_seq,
    }
    out = REPO / "pro_research" / "proto_multi_seq_full_model_n4.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
