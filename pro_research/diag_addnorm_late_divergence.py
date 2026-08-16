"""Layer-by-layer real-activation differential probe for PV2-10 AddNorm.

No timing claim. The actual model state always continues with the production
add+RMSNorm outputs; the fused candidate runs only on scratch copies.
"""
from __future__ import annotations

import argparse
import gc
import json
import traceback
from typing import Any

import cupy as cp
import numpy as np

from common import REPO, environment_snapshot, first_divergence, require_gpu_free, utc_now
from graph_e1f22 import _load_prompt_set
from queue_stream_v12 import _build_v6, _reset_exact_state, _run_sync

OUT = REPO / "pro_research" / "results" / "v12_async" / "DIAG_ADDNORM_LATE_DIVERGENCE.json"
PLAN = REPO / "pro_research" / "ADDNORM_LATE_DIVERGENCE_DIAGNOSTIC.md"

CUDA = r"""
__device__ __forceinline__ float diag_bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
extern "C" __global__ void diag_add_rmsnorm_bf16w(
    float* __restrict__ h, const float* __restrict__ residual,
    const unsigned short* __restrict__ w, float* __restrict__ out,
    const int n, const float eps)
{
    extern __shared__ float red[];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = h[i] + residual[i];
        h[i] = v;
        acc = fmaf(v, v, acc);
    }
    for (int o = warpSize >> 1; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        const int nw = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nw; ++i) s += red[i];
        red[31] = rsqrtf(s / (float)n + eps);
    }
    __syncthreads();
    const float scale = red[31];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = h[i] * scale * diag_bf16_to_f32(w[i]);
}
"""


class AddNormCandidate:
    def __init__(self):
        self.mod = cp.RawModule(code=CUDA, options=("-std=c++14",))
        self.kernel = self.mod.get_function("diag_add_rmsnorm_bf16w")

    def apply(self, h, residual, w, out, n: int, eps: float) -> None:
        self.kernel((1,), (256,),
                    (h, residual, w, out, np.int32(n), np.float32(eps)),
                    shared_mem=32 * 4)


def _first_bit_diff(a, b) -> tuple[int | None, int]:
    bits_a = a.view(cp.uint32)
    bits_b = b.view(cp.uint32)
    mask = bits_a != bits_b
    count = int(cp.asnumpy(cp.count_nonzero(mask)))
    if count == 0:
        return None, 0
    idx = int(cp.asnumpy(cp.flatnonzero(mask)[0]))
    return idx, count


def _manual_token(rt, cand: AddNormCandidate, token_id: int, scratch: dict[str, Any],
                  phase: str, token_index: int) -> tuple[int, dict[str, Any] | None]:
    k = rt.k
    rt._tok_dev.fill(np.int32(token_id))
    k.embed_gather(rt.h, rt._embed_tbl_ptr, rt._tok_dev, rt.hidden)
    k.norm(rt.normed, rt.h, rt.layer[0]["norm"], rt.hidden, rt.eps)

    for i, ch in enumerate(rt.pattern):
        if ch == "M":
            rt._mamba(i, rt.acc)
        elif ch == "*":
            rt._attention(i, rt.acc)
        else:
            rt._moe(i, rt.acc)

        cp.copyto(scratch["hb"], rt.h)
        cp.copyto(scratch["hc"], rt.h)
        next_w = rt.layer[i + 1]["norm"] if i + 1 < len(rt.pattern) else rt.norm_f
        k.add_(scratch["hb"], rt.acc, rt.hidden)
        k.norm(scratch["nb"], scratch["hb"], next_w, rt.hidden, rt.eps)
        cand.apply(scratch["hc"], rt.acc, next_w, scratch["nc"], rt.hidden, rt.eps)
        cp.cuda.Device(0).synchronize()

        h_idx, h_count = _first_bit_diff(scratch["hb"], scratch["hc"])
        n_idx, n_count = _first_bit_diff(scratch["nb"], scratch["nc"])
        if h_count or n_count:
            rec: dict[str, Any] = {
                "phase": phase,
                "token_index": token_index,
                "input_token_id": int(token_id),
                "layer": i,
                "layer_kind": ch,
                "hidden_mismatch_count": h_count,
                "hidden_first_index": h_idx,
                "normed_mismatch_count": n_count,
                "normed_first_index": n_idx,
            }
            if h_idx is not None:
                rec["hidden_ref_bits"] = int(cp.asnumpy(scratch["hb"].view(cp.uint32)[h_idx]))
                rec["hidden_candidate_bits"] = int(cp.asnumpy(scratch["hc"].view(cp.uint32)[h_idx]))
            if n_idx is not None:
                rec["normed_ref_bits"] = int(cp.asnumpy(scratch["nb"].view(cp.uint32)[n_idx]))
                rec["normed_candidate_bits"] = int(cp.asnumpy(scratch["nc"].view(cp.uint32)[n_idx]))
            return -1, rec

        # Continue the real causal state using production results only.
        cp.copyto(rt.h, scratch["hb"])
        cp.copyto(rt.normed, scratch["nb"])

    if rt.lm_head_kind == "nvfp4":
        rt.fused.gemv_into(rt.logits, rt.lm_head_codes, rt.lm_head_scales,
                           rt.normed, rt.lm_head_g, rt.vocab, rt.hidden)
    else:
        k.mv_bf16(rt.logits, rt.lm_head, rt.normed, rt.vocab, rt.hidden)
    k.argmax_logits(rt._tok_dev, rt.logits, rt.vocab, rt._am_max, rt._am_idx)
    k.pos_increment(rt._pos_dev)
    cp.cuda.Device(0).synchronize()
    return int(cp.asnumpy(rt._tok_dev)[0]), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=160)
    args = ap.parse_args()
    n = max(130, int(args.tokens))
    payload: dict[str, Any] = {
        "kind": "diag_addnorm_late_divergence",
        "status": "started",
        "started_utc": utc_now(),
        "generated_tokens": n,
        "plan": str(PLAN.relative_to(REPO)),
        "timing_claim": False,
    }
    bundle = None
    try:
        require_gpu_free()
        prompts, _expected, _n, capacity = _load_prompt_set("full")
        selected = next(p for p in prompts if p["prompt"].startswith("The history of computing"))
        payload["prompt"] = selected["prompt"]
        payload["environment"] = environment_snapshot((
            REPO / "pro_research" / "diag_addnorm_late_divergence.py",
            REPO / "pro_research" / "ADDNORM_LATE_DIVERGENCE_DIAGNOSTIC.md",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "gpu_kernels.py",
        ))

        rt, dense, down, up, restore_sel, restore_moe, sel_counts = _build_v6(capacity)
        bundle = (rt, dense, down, up, restore_sel, restore_moe)
        cand = AddNormCandidate()
        scratch = {name: cp.empty(rt.hidden, dtype=cp.float32)
                   for name in ("hb", "hc", "nb", "nc")}

        graph_ids, _ = _run_sync(rt, selected["prompt_ids"], n)
        _reset_exact_state(rt)

        first_mismatch = None
        nxt = None
        prompt_outputs: list[int] = []
        for pi, tok in enumerate(selected["prompt_ids"]):
            nxt, first_mismatch = _manual_token(rt, cand, int(tok), scratch, "prompt", pi)
            if first_mismatch is not None:
                break
            prompt_outputs.append(int(nxt))

        manual_ids: list[int] = []
        if first_mismatch is None:
            if nxt is None:
                raise RuntimeError("empty prompt")
            cur = int(nxt)
            manual_ids.append(cur)
            for gi in range(1, n):
                cur, first_mismatch = _manual_token(rt, cand, cur, scratch, "generated", gi)
                if first_mismatch is not None:
                    break
                manual_ids.append(int(cur))

        graph_div = first_divergence(graph_ids[:len(manual_ids)], manual_ids)
        if first_mismatch is not None:
            conclusion = "direct_addnorm_bit_mismatch_found"
        elif graph_div is not None or len(manual_ids) != len(graph_ids):
            conclusion = "addnorm_direct_exact_but_manual_reference_differs_from_graph"
        else:
            conclusion = "direct_addnorm_exact_through_window_graph_reference_matches"

        payload.update({
            "first_addnorm_bit_mismatch": first_mismatch,
            "manual_generated_ids": manual_ids,
            "graph_reference_ids": graph_ids,
            "manual_vs_graph_first_divergence": graph_div,
            "manual_vs_graph_same_length": len(manual_ids) == len(graph_ids),
            "conclusion": conclusion,
            "status": "measured",
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({"status": "technical_failure", "completed_utc": utc_now(),
                        "error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()}})
    finally:
        if bundle is not None:
            rt, dense, down, up, restore_sel, restore_moe = bundle
            try: restore_sel(); restore_moe()
            except Exception: pass
            del rt, dense, down, up
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(json.dumps({"status": payload.get("status"),
                      "conclusion": payload.get("conclusion"),
                      "first_addnorm_bit_mismatch": payload.get("first_addnorm_bit_mismatch"),
                      "manual_vs_graph_first_divergence": payload.get("manual_vs_graph_first_divergence"),
                      "output": str(OUT)}, indent=2))
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
