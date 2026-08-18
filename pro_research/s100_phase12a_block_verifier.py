"""S100 phase 12A: perfect-draft block verifier (preregistered).

Builds a B-token block graph by unrolling B decode bodies with the EXISTING
bit-exact M=1 kernels (no weight sharing yet -- that is phase 12C). Draft
tokens are the runtime's own greedy continuation (acceptance is 100% by
construction). This isolates two things:

1. correctness: a block forward must reproduce B sequential decode steps
   exactly -- argmax identity at every position, identical Mamba/conv state,
   identical KV bytes, identical final logits;
2. cost: full cycle wall time (draft upload + block graph launch + harvest
   copy + sync) for B in {2,4,8}, against the preregistered break-even gates
   B=2 <=18 ms, B=4 <=28 ms, B=8 <=40 ms.

State semantics: every body updates Mamba/conv state and KV in place, in the
same order as sequential decode. That is exact iff acceptance is total; the
harness verifies acceptance==100% and treats any mismatch as a correctness
failure (shadow-state rollback for partial acceptance is phase 12D scope).

The comparator is B ordinary sequential step_graph launches with the same
sync cadence (one sync per B tokens), so the measurement shows exactly what
block-graph amortisation buys before ERVF-M exists.

Run: .venv-nemotron/Scripts/python.exe pro_research/s100_phase12a_block_verifier.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pro_research"))
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "pro_research" / "results" / "s100_phase12" / "S100_PHASE12A_BLOCK_VERIFIER.json"
BLOCKS = (2, 4, 8)
GATES_MS = {2: 18.0, 4: 28.0, 8: 40.0}
N_CORRECT_CYCLES = 8
N_TIMED_CYCLES = 32
PROMPTS_USED = 2


def setup_block_graph(rt, B: int) -> None:
    """Capture one graph holding B unrolled decode bodies. Body 0 embeds
    _tok_dev (the last committed token); body j>=1 embeds draft d_{j-1} from
    _blk_draft. Every body's argmax lands in _blk_out[j]; the last body also
    updates _tok_dev so the next cycle starts from the accepted tip."""
    import cupy as cp

    cp_ = rt.cp
    feed = max(B - 1, 1)
    rt._blk_B = B
    rt._blk_draft = cp_.zeros(feed, dtype=cp_.int32)
    rt._blk_out = cp_.zeros(B, dtype=cp_.int32)
    rt._blk_draft_st = cp.cuda.alloc_pinned_memory(4 * feed)
    rt._blk_draft_np = np.frombuffer(rt._blk_draft_st, dtype=np.int32)
    rt._blk_out_st = cp.cuda.alloc_pinned_memory(4 * B)
    rt._blk_out_np = np.frombuffer(rt._blk_out_st, dtype=np.int32)

    k = rt.k

    def body(j: int) -> None:
        src = rt._tok_dev if j == 0 else rt._blk_draft[j - 1 : j]
        dst = rt._blk_out[j : j + 1]
        k.embed_gather(rt.h, rt._embed_tbl_ptr, src, rt.hidden)
        for i, ch in enumerate(rt.pattern):
            d = rt.layer[i]
            k.norm(rt.normed, rt.h, d["norm"], rt.hidden, rt.eps)
            if ch == "M":
                rt._mamba(i, rt.acc)
            elif ch == "*":
                rt._attention(i, rt.acc)
            else:
                rt._moe(i, rt.acc)
            k.add_(rt.h, rt.acc, rt.hidden)
        k.norm(rt.normed, rt.h, rt.norm_f, rt.hidden, rt.eps)
        if rt.lm_head_kind == "nvfp4":
            rt.fused.gemv_into(rt.logits, rt.lm_head_codes, rt.lm_head_scales,
                               rt.normed, rt.lm_head_g, rt.vocab, rt.hidden)
        else:
            k.mv_bf16(rt.logits, rt.lm_head, rt.normed, rt.vocab, rt.hidden)
        k.argmax_logits(dst, rt.logits, rt.vocab, rt._am_max, rt._am_idx)
        k.pos_increment(rt._pos_dev)

    def all_bodies() -> None:
        for j in range(B):
            body(j)
        cp_.copyto(rt._tok_dev, rt._blk_out[B - 1 : B])

    s = rt._graph_stream
    with s:
        all_bodies()  # warmup: compile everything; state is reset afterwards
    s.synchronize()
    s.begin_capture()
    with s:
        all_bodies()
    rt._blk_graph = s.end_capture()
    s.synchronize()


def step_block(rt, draft_ids, sync: bool = True):
    """One verification cycle: upload B-1 feed ids, launch the block graph,
    copy the B argmax ids to the pinned out-slot. Returns the B ids."""
    s = rt._graph_stream
    rtc = rt.cp.cuda.runtime
    B = rt._blk_B
    rt._blk_draft_np[: B - 1] = draft_ids[: B - 1]
    rtc.memcpyAsync(rt._blk_draft.data.ptr, rt._blk_draft_st.ptr,
                    4 * (B - 1), rtc.memcpyHostToDevice, s.ptr)
    rt._blk_graph.launch(s)
    rtc.memcpyAsync(rt._blk_out_st.ptr, rt._blk_out.data.ptr, 4 * B,
                    rtc.memcpyDeviceToHost, s.ptr)
    if sync:
        s.synchronize()
        return [int(x) for x in rt._blk_out_np[:B]]
    return None


def harvest_block(rt):
    rt._graph_stream.synchronize()
    B = rt._blk_B
    return [int(x) for x in rt._blk_out_np[:B]]


def state_fingerprint(rt) -> str:
    """sha256 over all Mamba ssm/conv arrays, the used KV region and pos.

    KV layout is [n_kv, max_ctx, head_dim] (see kv_append_fp8_dp), and exactly
    positions [0, pos) are written; unwritten tail rows hold nondeterministic
    warmup garbage (reset() does not clear KV) and must not be hashed."""
    h = hashlib.sha256()
    for i in sorted(rt.ssm):
        h.update(rt.ssm[i].get().tobytes())
        h.update(rt.conv[i].get().tobytes())
    used = int(rt._pos_dev.get()[0])
    for i in sorted(rt.kc):
        h.update(rt.kc[i].reshape(rt.n_kv, rt.max_ctx, rt.head_dim)[:, :used].get().tobytes())
        h.update(rt.vc[i].reshape(rt.n_kv, rt.max_ctx, rt.head_dim)[:, :used].get().tobytes())
    h.update(np.int32(used).tobytes())
    return h.hexdigest()


def main() -> int:
    p: dict = {"kind": "s100_phase12a_block_verifier", "status": "started",
               "blocks": BLOCKS, "gates_ms": GATES_MS}
    try:
        import cupy as cp  # noqa: F401
        from common import environment_snapshot
        from diag_component_marginals_graph import _prefill, _reset_exact_state
        from s100_phase10a_runtime import build
        from s100_phase9_trace import load_prompts

        prompts = load_prompts(REPO)
        bundle = build()  # production map: identical to the phase-10 base arm
        rt = bundle.rt

        per_B: dict[str, dict] = {}
        for B in BLOCKS:
            t_setup0 = time.perf_counter_ns()
            setup_block_graph(rt, B)
            _reset_exact_state(rt)
            setup_ms = (time.perf_counter_ns() - t_setup0) / 1e6

            n_ids = (N_CORRECT_CYCLES + 4 + N_TIMED_CYCLES) * B
            correctness: dict = {}
            cycle_ms: list[float] = []
            seq_ms: list[float] = []
            for pi in range(PROMPTS_USED):
                prompt_ids = prompts[pi]["prompt_ids"]

                # Baseline: sequential decode, same production graph. The
                # correctness snapshot is taken exactly at the end of the
                # correctness window, then decoding continues to supply the
                # drafts for the warmup and timed cycles.
                _reset_exact_state(rt)
                _prefill(rt, prompt_ids)
                ids_base = []
                for _ in range(n_ids):
                    slot = int(rt._ring_i)
                    rt.step_graph(None)
                    ids_base.append(int(rt.ring_harvest(slot, 1)[0]))
                    if len(ids_base) == N_CORRECT_CYCLES * B:
                        fp_base = state_fingerprint(rt)
                        logits_base = rt.logits.get().copy()

                # Sequential comparator timing: B launches, one sync per B
                # tokens, measured here while the state is on-trajectory.
                s = rt._graph_stream
                for t in range(N_TIMED_CYCLES):
                    e0 = cp.cuda.Event(); e1 = cp.cuda.Event()
                    e0.record(s)
                    for _ in range(B):
                        rt.step_graph(None)
                    e1.record(s)
                    s.synchronize()
                    seq_ms.append(cp.cuda.get_elapsed_time(e0, e1))

                # Block verifier: drafts are the baseline continuation, run
                # from a fresh prefill so the state trajectory is identical.
                _reset_exact_state(rt)
                _prefill(rt, prompt_ids)
                accepted = 0
                checked = 0
                for c in range(N_CORRECT_CYCLES):
                    draft = ids_base[c * B : (c + 1) * B]
                    got = step_block(rt, draft)
                    checked += B
                    accepted += sum(1 for a, b in zip(got, draft) if a == b)
                fp_blk = state_fingerprint(rt)
                logits_blk = rt.logits.get().copy()

                ok_acc = accepted == checked
                ok_state = fp_blk == fp_base
                ok_logits = bool(np.array_equal(logits_blk, logits_base))
                if pi == 0:
                    correctness = {
                        "positions_checked": checked,
                        "positions_accepted": accepted,
                        "acceptance": accepted / checked,
                        "argmax_identity": ok_acc,
                        "state_fingerprint_equal": ok_state,
                        "final_logits_bitexact": ok_logits,
                    }
                if not (ok_acc and ok_state and ok_logits):
                    raise RuntimeError(
                        f"block verifier correctness failed at B={B} "
                        f"prompt={pi}: acc={accepted}/{checked} "
                        f"state={ok_state} logits={ok_logits}")

                # Timing: warmup cycles keep the state advancing on the same
                # golden trajectory, then timed block cycles.
                drafts = [ids_base[(N_CORRECT_CYCLES + 4 + t) * B :
                                   (N_CORRECT_CYCLES + 5 + t) * B]
                          for t in range(N_TIMED_CYCLES)]
                for t in range(4):
                    step_block(rt, ids_base[(N_CORRECT_CYCLES + t) * B :
                                            (N_CORRECT_CYCLES + t + 1) * B])
                for t in range(N_TIMED_CYCLES):
                    e0 = cp.cuda.Event(); e1 = cp.cuda.Event()
                    e0.record(s)
                    step_block(rt, drafts[t], sync=False)
                    e1.record(s)
                    s.synchronize()
                    cycle_ms.append(cp.cuda.get_elapsed_time(e0, e1))

            cyc = float(np.median(cycle_ms))
            seq = float(np.median(seq_ms))
            per_B[str(B)] = {
                "setup_ms": setup_ms,
                "correctness": correctness,
                "cycle_ms_median": cyc,
                "cycle_ms_p10": float(np.percentile(cycle_ms, 10)),
                "cycle_ms_p90": float(np.percentile(cycle_ms, 90)),
                "sequential_B_tokens_ms_median": seq,
                "block_vs_sequential_ratio": cyc / seq,
                "useful_tok_s_perfect_draft": 1000.0 * B / cyc,
                "gate_ms": GATES_MS[B],
                "gate_pass": bool(cyc <= GATES_MS[B]),
            }

        p.update({
            "status": "measured",
            "environment": environment_snapshot(),
            "n_correct_cycles": N_CORRECT_CYCLES,
            "n_timed_cycles": N_TIMED_CYCLES,
            "prompts_used": PROMPTS_USED,
            "note": "ordinary M=1 kernels, no weight sharing (phase 12C scope); "
                    "perfect draft, acceptance verified 100%; gates are the "
                    "preregistered break-even targets for the FINAL verifier",
            "per_B": per_B,
            "any_gate_pass": any(v["gate_pass"] for v in per_B.values()),
        })
        bundle.restore_combined()
        bundle.restore_sel()
    except Exception as e:  # noqa: BLE001
        p.update({"status": "technical_failure",
                  "error": {"type": type(e).__name__, "message": str(e),
                            "traceback": traceback.format_exc()}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(p, indent=2, allow_nan=False))
    return 0 if p.get("status") == "measured" else 2


if __name__ == "__main__":
    sys.exit(main())
