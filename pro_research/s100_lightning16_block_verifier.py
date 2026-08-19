"""Fresh Lightning perfect-draft block verifier with ordinary parent kernels."""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from common import REPO, write_json_atomic, utc_now
from s100_lightning16_common import RESULTS, assert_lightning, ensure_results

OUT = RESULTS / "S100_LIGHTNING16_BLOCK_VERIFIER.json"
BLOCKS = (2, 4, 8)
N_CORRECT_CYCLES = 6
N_TIMED_CYCLES = 24
PROMPTS_USED = 2

def setup_block_graph(rt, block):
    import cupy as cp

    feed = max(block - 1, 1)
    rt._l16_block = block
    rt._l16_draft = cp.zeros(feed, cp.int32)
    rt._l16_out = cp.zeros(block, cp.int32)
    rt._l16_draft_st = cp.cuda.alloc_pinned_memory(4 * feed)
    rt._l16_draft_np = np.frombuffer(
        rt._l16_draft_st, np.int32
    )
    rt._l16_out_st = cp.cuda.alloc_pinned_memory(4 * block)
    rt._l16_out_np = np.frombuffer(rt._l16_out_st, np.int32)
    k = rt.k

    def body(index):
        source = (
            rt._tok_dev
            if index == 0
            else rt._l16_draft[index - 1:index]
        )
        destination = rt._l16_out[index:index + 1]
        k.embed_gather(
            rt.h, rt._embed_tbl_ptr, source, rt.hidden
        )
        for layer, kind in enumerate(rt.pattern):
            data = rt.layer[layer]
            k.norm(
                rt.normed, rt.h, data["norm"],
                rt.hidden, rt.eps,
            )
            if kind == "M":
                rt._mamba(layer, rt.acc)
            elif kind == "*":
                rt._attention(layer, rt.acc)
            else:
                rt._moe(layer, rt.acc)
            k.add_(rt.h, rt.acc, rt.hidden)
        k.norm(
            rt.normed, rt.h, rt.norm_f,
            rt.hidden, rt.eps,
        )
        if rt.lm_head_kind == "nvfp4":
            rt.fused.gemv_into(
                rt.logits,
                rt.lm_head_codes,
                rt.lm_head_scales,
                rt.normed,
                rt.lm_head_g,
                rt.vocab,
                rt.hidden,
            )
        else:
            k.mv_bf16(
                rt.logits, rt.lm_head, rt.normed,
                rt.vocab, rt.hidden,
            )
        k.argmax_logits(
            destination, rt.logits, rt.vocab,
            rt._am_max, rt._am_idx,
        )
        k.pos_increment(rt._pos_dev)

    def all_bodies():
        for index in range(block):
            body(index)
        cp.copyto(
            rt._tok_dev,
            rt._l16_out[block - 1:block],
        )

    stream = rt._graph_stream
    with stream:
        all_bodies()
    stream.synchronize()
    stream.begin_capture()
    with stream:
        all_bodies()
    rt._l16_graph = stream.end_capture()
    stream.synchronize()

def step_block(rt, draft, sync=True):
    runtime = rt.cp.cuda.runtime
    stream = rt._graph_stream
    block = rt._l16_block
    rt._l16_draft_np[:block - 1] = draft[:block - 1]
    runtime.memcpyAsync(
        rt._l16_draft.data.ptr,
        rt._l16_draft_st.ptr,
        4 * (block - 1),
        runtime.memcpyHostToDevice,
        stream.ptr,
    )
    rt._l16_graph.launch(stream)
    runtime.memcpyAsync(
        rt._l16_out_st.ptr,
        rt._l16_out.data.ptr,
        4 * block,
        runtime.memcpyDeviceToHost,
        stream.ptr,
    )
    if sync:
        stream.synchronize()
        return [int(x) for x in rt._l16_out_np[:block]]
    return None

def state_fingerprint(rt):
    digest = hashlib.sha256()
    for layer in sorted(rt.ssm):
        digest.update(rt.ssm[layer].get().tobytes())
        digest.update(rt.conv[layer].get().tobytes())
    used = int(rt._pos_dev.get()[0])
    for layer in sorted(rt.kc):
        digest.update(
            rt.kc[layer].reshape(
                rt.n_kv, rt.max_ctx, rt.head_dim
            )[:, :used].get().tobytes()
        )
        digest.update(
            rt.vc[layer].reshape(
                rt.n_kv, rt.max_ctx, rt.head_dim
            )[:, :used].get().tobytes()
        )
    digest.update(np.int32(used).tobytes())
    return digest.hexdigest()

def main():
    ensure_results()
    payload = {
        "kind": "s100_lightning16_block_verifier",
        "status": "started",
        "blocks": list(BLOCKS),
        "started_utc": utc_now(),
    }
    try:
        import cupy as cp
        from diag_component_marginals_graph import (
            _prefill, _reset_exact_state
        )
        from s100_phase10a_runtime import build
        from s100_phase9_trace import load_prompts

        ident = assert_lightning()
        prompts = load_prompts(REPO)
        bundle = build()
        rt = bundle.rt
        per_block = {}

        for block in BLOCKS:
            setup_start = time.perf_counter_ns()
            setup_block_graph(rt, block)
            _reset_exact_state(rt)
            setup_ms = (
                time.perf_counter_ns() - setup_start
            ) / 1e6

            needed = (
                N_CORRECT_CYCLES + 4 + N_TIMED_CYCLES
            ) * block
            cycles = []
            sequential = []
            correctness = None

            for prompt_index in range(PROMPTS_USED):
                prompt_ids = prompts[prompt_index]["prompt_ids"]
                _reset_exact_state(rt)
                _prefill(rt, prompt_ids)
                golden = []
                for _ in range(needed):
                    slot = int(rt._ring_i)
                    rt.step_graph(None)
                    golden.append(
                        int(rt.ring_harvest(slot, 1)[0])
                    )
                    if len(golden) == N_CORRECT_CYCLES * block:
                        base_state = state_fingerprint(rt)
                        base_logits = rt.logits.get().copy()

                stream = rt._graph_stream
                for _ in range(N_TIMED_CYCLES):
                    begin = cp.cuda.Event()
                    end = cp.cuda.Event()
                    begin.record(stream)
                    for _ in range(block):
                        rt.step_graph(None)
                    end.record(stream)
                    stream.synchronize()
                    sequential.append(
                        cp.cuda.get_elapsed_time(begin, end)
                    )

                _reset_exact_state(rt)
                _prefill(rt, prompt_ids)
                accepted = checked = 0
                for cycle in range(N_CORRECT_CYCLES):
                    draft = golden[
                        cycle * block:(cycle + 1) * block
                    ]
                    got = step_block(rt, draft)
                    checked += block
                    accepted += sum(
                        int(left == right)
                        for left, right in zip(got, draft)
                    )
                block_state = state_fingerprint(rt)
                block_logits = rt.logits.get().copy()
                result = {
                    "positions_checked": checked,
                    "positions_accepted": accepted,
                    "acceptance": accepted / checked,
                    "argmax_identity": accepted == checked,
                    "state_fingerprint_equal": (
                        block_state == base_state
                    ),
                    "final_logits_bitexact": bool(
                        np.array_equal(block_logits, base_logits)
                    ),
                }
                if not all((
                    result["argmax_identity"],
                    result["state_fingerprint_equal"],
                    result["final_logits_bitexact"],
                )):
                    raise RuntimeError(
                        f"Lightning block correctness failed "
                        f"B={block} prompt={prompt_index}: {result}"
                    )
                if correctness is None:
                    correctness = result

                for cycle in range(4):
                    offset = (N_CORRECT_CYCLES + cycle) * block
                    step_block(
                        rt, golden[offset:offset + block]
                    )
                for cycle in range(N_TIMED_CYCLES):
                    offset = (
                        N_CORRECT_CYCLES + 4 + cycle
                    ) * block
                    begin = cp.cuda.Event()
                    end = cp.cuda.Event()
                    begin.record(stream)
                    step_block(
                        rt,
                        golden[offset:offset + block],
                        sync=False,
                    )
                    end.record(stream)
                    stream.synchronize()
                    cycles.append(
                        cp.cuda.get_elapsed_time(begin, end)
                    )

            cycle_ms = float(np.median(cycles))
            sequential_ms = float(np.median(sequential))
            per_block[str(block)] = {
                "setup_ms": setup_ms,
                "correctness": correctness,
                "cycle_ms_median": cycle_ms,
                "cycle_ms_p10": float(np.percentile(cycles, 10)),
                "cycle_ms_p90": float(np.percentile(cycles, 90)),
                "sequential_B_tokens_ms_median": sequential_ms,
                "block_vs_sequential_ratio": (
                    cycle_ms / sequential_ms
                ),
                "useful_tok_s_perfect_draft": (
                    1000.0 * block / cycle_ms
                ),
                "perfect_draft_s100_open": bool(
                    1000.0 * block / cycle_ms >= 100.0
                ),
            }

        payload.update({
            "status": "measured",
            "identity": ident,
            "per_B": per_block,
            "LIGHTNING_PERFECT_DRAFT_S100_OPEN": any(
                row["perfect_draft_s100_open"]
                for row in per_block.values()
            ),
            "completed_utc": utc_now(),
        })
        bundle.restore_combined()
        bundle.restore_sel()
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "per_B": {
            key: {
                "cycle_ms": row.get("cycle_ms_median"),
                "tok_s": row.get("useful_tok_s_perfect_draft"),
                "correctness": row.get("correctness"),
            }
            for key, row in payload.get("per_B", {}).items()
        },
        "LIGHTNING_PERFECT_DRAFT_S100_OPEN": payload.get(
            "LIGHTNING_PERFECT_DRAFT_S100_OPEN"
        ),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
