"""PRO V13: thermally balanced remeasurement of the exact PV2 QKV fusion."""
from __future__ import annotations

import argparse
import gc
import json
import types
import traceback
from typing import Any

import cupy as cp
import numpy as np

from common import REPO, environment_snapshot, first_divergence, percentiles, require_gpu_free, utc_now
from graph_e1f22 import _load_prompt_set
from layer_capacity import apply_nonuniform_capacity
from queue_stream_v12 import _build_v6, _preheat, _run_sync

RESULT_DIR = REPO / "pro_research" / "results" / "v12_async"
OUT = RESULT_DIR / "PRO_V13_QKV_STEADY.json"
PREREG = REPO / "pro_research" / "V13_QKV_STEADY_PREREGISTRATION.md"

CUDA = r"""
__device__ __forceinline__ float v13_bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
#define V13_W 16
#define V13_V 16

__device__ __forceinline__ float v13_reduce_q(float acc[V13_V]) {
    const int lane = threadIdx.x & 15;
    float s[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        float v = acc[g * 2] + acc[g * 2 + 1];
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o, 16);
        s[g] = v;
    }
    if (lane == 0) {
        const float a0 = s[0] + s[4];
        const float a1 = s[1] + s[5];
        const float a2 = s[2] + s[6];
        const float a3 = s[3] + s[7];
        return (a0 + a2) + (a1 + a3);
    }
    return 0.0f;
}

extern "C" __global__ void v13_qkv_mixed_fused(
    const unsigned short* __restrict__ Wq,
    const unsigned short* __restrict__ Wk,
    const unsigned short* __restrict__ Wv,
    const float* __restrict__ x,
    float* __restrict__ qout,
    float* __restrict__ kout,
    float* __restrict__ vout,
    const int qrows, const int kvrows, const int cols)
{
    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __syncthreads();

    const int qblocks = (qrows + 15) >> 4;
    if ((int)blockIdx.x < qblocks) {
        const int sub = threadIdx.x >> 4;
        const int lane = threadIdx.x & 15;
        const int row = (int)blockIdx.x * 16 + sub;
        const bool valid = row < qrows;
        const unsigned short* w = Wq + (size_t)(valid ? row : 0) * cols;
        float part[16];
        #pragma unroll
        for (int vi = 0; vi < 16; ++vi) part[vi] = 0.0f;
        #pragma unroll
        for (int vi = 0; vi < 16; ++vi) {
            const int tid = lane + 16 * vi;
            if (valid)
                for (int k = tid; k < cols; k += 256)
                    part[vi] = fmaf(v13_bf16_to_f32(w[k]), sx[k], part[vi]);
        }
        const float y = v13_reduce_q(part);
        if (lane == 0 && valid) qout[row] = y;
        return;
    }

    int b = (int)blockIdx.x - qblocks;
    const bool is_v = b >= kvrows;
    if (is_v) b -= kvrows;
    if (b >= kvrows) return;
    const unsigned short* W = is_v ? Wv : Wk;
    float* out = is_v ? vout : kout;
    const unsigned short* w = W + (size_t)b * cols;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < cols; k += 256)
        acc = fmaf(v13_bf16_to_f32(w[k]), sx[k], acc);
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float ws[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) ws[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float y = (lane < 8) ? ws[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            y += __shfl_down_sync(0xffffffffu, y, o);
        if (lane == 0) out[b] = y;
    }
}
"""


class QKVCandidate:
    def __init__(self):
        self.mod = cp.RawModule(code=CUDA, options=("-std=c++14", "--use_fast_math"))
        self.kernel = self.mod.get_function("v13_qkv_mixed_fused")

    def apply(self, q, k, v, Wq, Wk, Wv, x, qrows: int, kvrows: int, cols: int) -> None:
        blocks = (qrows + 15) // 16 + 2 * kvrows
        self.kernel((blocks,), (256,),
                    (Wq, Wk, Wv, x, q, k, v,
                     np.int32(qrows), np.int32(kvrows), np.int32(cols)),
                    shared_mem=cols * 4)


def install(rt, candidate: QKVCandidate):
    original = rt._attention
    def attention(self, i, out):
        if not (self.graph_mode and self.fp8_kv):
            return original(i, out)
        k, d = self.k, self.layer[i]
        qrows = self.n_heads * self.head_dim
        kvrows = self.n_kv * self.head_dim
        candidate.apply(self.qv, self.kv_, self.vv,
                        d["q_proj"], d["k_proj"], d["v_proj"], self.normed,
                        qrows, kvrows, self.hidden)
        scale = 1.0 / float(np.sqrt(self.head_dim))
        k.kv_write_fp8_dp(self.kc[i], self.kv_, self._pos_dev,
                          self.n_kv, self.head_dim, self.max_ctx)
        k.kv_write_fp8_dp(self.vc[i], self.vv, self._pos_dev,
                          self.n_kv, self.head_dim, self.max_ctx)
        k.attention_fp8_gqa4_dp(self.ctx, self.qv, self.kc[i], self.vc[i],
                                self._pos_dev, self.n_heads, self.head_dim,
                                self.groups, self.max_ctx, scale,
                                self.part_acc, self.part_ml)
        k.mv_bf16(out, d["o_proj"], self.ctx, self.hidden, qrows)
    rt._attention = types.MethodType(attention, rt)
    def restore():
        rt._attention = original
    return restore


def _drop_graph(rt) -> None:
    if hasattr(rt, "_graph_stream"):
        try: rt._graph_stream.synchronize()
        except Exception: pass
    rt._graph = None
    rt.graph_mode = False
    for name in ("_tok_dev", "_pos_dev", "_am_max", "_am_idx", "_embed_pinned",
                 "_stage_mem", "_stage_np", "_ring_mem", "_ring_np", "_graph_stream",
                 "_ring_i", "_ring_size"):
        if hasattr(rt, name):
            try: delattr(rt, name)
            except Exception: pass
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _capture(rt, capacity: int) -> int:
    _drop_graph(rt)
    rt.enable_cache(capacity)
    apply_nonuniform_capacity(rt)
    rt.device_cache = True
    rt.deterministic_accum = True
    free0 = int(cp.cuda.Device(0).mem_info[0])
    rt.setup_graph()
    free1 = int(cp.cuda.Device(0).mem_info[0])
    return int(getattr(rt, "graph_extra_vram_bytes", free0 - free1))


def _dot(rt) -> str:
    try:
        x = rt._graph.debug_dot_str()
        return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x)
    except Exception:
        return ""


def _bits_equal(a, b) -> bool:
    return bool(cp.asnumpy(cp.all(a.view(cp.uint32) == b.view(cp.uint32))))


def _micro(rt, cand: QKVCandidate) -> dict[str, Any]:
    i = rt.attn_layers[0]
    d = rt.layer[i]
    qrows, kvrows, cols = rt.n_heads * rt.head_dim, rt.n_kv * rt.head_dim, rt.hidden
    x = cp.random.RandomState(20260816).standard_normal(cols, dtype=cp.float32)
    qb, kb, vb = cp.empty(qrows, cp.float32), cp.empty(kvrows, cp.float32), cp.empty(kvrows, cp.float32)
    qc, kc, vc = cp.empty_like(qb), cp.empty_like(kb), cp.empty_like(vb)
    rt.k.mv_bf16(qb, d["q_proj"], x, qrows, cols)
    rt.k.mv_bf16(kb, d["k_proj"], x, kvrows, cols)
    rt.k.mv_bf16(vb, d["v_proj"], x, kvrows, cols)
    cand.apply(qc, kc, vc, d["q_proj"], d["k_proj"], d["v_proj"], x, qrows, kvrows, cols)
    cp.cuda.Device(0).synchronize()
    return {"q_bitexact": _bits_equal(qb, qc), "k_bitexact": _bits_equal(kb, kc),
            "v_bitexact": _bits_equal(vb, vc),
            "shape": {"qrows": qrows, "kvrows": kvrows, "cols": cols}}


def _run_block(rt, prompts: list[dict[str, Any]], n: int) -> dict[str, Any]:
    ids: dict[str, list[int]] = {}
    samples: list[float] = []
    for p in prompts:
        out, ms = _run_sync(rt, p["prompt_ids"], n)
        ids[p["prompt"]] = out
        samples.extend(ms)
    return {"ids": ids, "timing_ms": percentiles(samples), "raw_timing_ms": samples}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload: dict[str, Any] = {"kind": "pro_v13_qkv_steady", "status": "started",
                               "mode": args.mode, "started_utc": utc_now(),
                               "preregistration": str(PREREG.relative_to(REPO))}
    bundle = None
    restore_qkv = None
    try:
        require_gpu_free()
        prompts, _expected, _n, capacity = _load_prompt_set(args.mode)
        n = 32 if args.mode == "smoke" else 128
        preheat_n = 48 if args.mode == "smoke" else 96
        schedule = ["BASE", "QKV", "QKV", "BASE", "QKV", "BASE", "BASE", "QKV"]
        payload["config"] = {"tokens_per_prompt_per_block": n, "preheat_tokens": preheat_n,
                             "schedule": schedule, "capacity": capacity}
        payload["environment"] = environment_snapshot((
            REPO / "pro_research" / "v13_qkv_steady.py",
            REPO / "pro_research" / "V13_QKV_STEADY_PREREGISTRATION.md",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "gpu_kernels.py",
        ))
        rt, dense, down, up, restore_sel, restore_moe, sel_counts = _build_v6(capacity)
        bundle = (rt, dense, down, up, restore_sel, restore_moe)
        cand = QKVCandidate()
        payload["micro"] = _micro(rt, cand)
        blocks: list[dict[str, Any]] = []
        qkv_active = False
        qkv_dot_all = True

        for bi, treatment in enumerate(schedule):
            if treatment == "QKV" and not qkv_active:
                restore_qkv = install(rt, cand); qkv_active = True
            elif treatment == "BASE" and qkv_active:
                assert restore_qkv is not None
                restore_qkv(); restore_qkv = None; qkv_active = False

            extra = _capture(rt, capacity)
            dot = _dot(rt).lower()
            if treatment == "QKV":
                qkv_dot_all = qkv_dot_all and "v13_qkv_mixed_fused" in dot
            _preheat(rt, prompts[0]["prompt_ids"], preheat_n)
            rec = _run_block(rt, prompts, n)
            rec.update({"index": bi, "treatment": treatment,
                        "extra_vram_bytes": extra,
                        "candidate_name_present": "v13_qkv_mixed_fused" in dot})
            blocks.append(rec)

        if qkv_active and restore_qkv is not None:
            restore_qkv(); restore_qkv = None; qkv_active = False

        canonical = next(b for b in blocks if b["treatment"] == "BASE")["ids"]
        parity: list[dict[str, Any]] = []
        all_ids_exact = True
        for b in blocks:
            divs = {p: first_divergence(canonical[p], b["ids"].get(p, [])) for p in canonical}
            exact = all(v is None for v in divs.values())
            all_ids_exact = all_ids_exact and exact
            parity.append({"index": b["index"], "treatment": b["treatment"],
                           "exact": exact, "first_divergence": divs})

        base_p50 = [float(b["timing_ms"]["p50"]) for b in blocks if b["treatment"] == "BASE"]
        cand_p50 = [float(b["timing_ms"]["p50"]) for b in blocks if b["treatment"] == "QKV"]
        base_median = float(np.median(np.asarray(base_p50, dtype=np.float64)))
        cand_median = float(np.median(np.asarray(cand_p50, dtype=np.float64)))
        base_range = max(base_p50) - min(base_p50)
        cand_range = max(cand_p50) - min(cand_p50)
        gain = base_median - cand_median
        required_gain = max(0.10, 0.005 * base_median)
        base_samples = sum(int(b["timing_ms"]["count"]) for b in blocks if b["treatment"] == "BASE")
        cand_samples = sum(int(b["timing_ms"]["count"]) for b in blocks if b["treatment"] == "QKV")
        gates = {
            "micro_qkv_bitexact": all(bool(payload["micro"][k]) for k in ("q_bitexact","k_bitexact","v_bitexact")),
            "all_block_token_parity": all_ids_exact,
            "all_qkv_graphs_contain_candidate": qkv_dot_all,
            "base_block_p50_range_le_1ms": base_range <= 1.0,
            "qkv_block_p50_range_le_1ms": cand_range <= 1.0,
            "no_material_regression": cand_median <= base_median * 1.002,
            "positive_gain": gain >= required_gain,
            "full_samples_ge_1000_each": (base_samples >= 1000 and cand_samples >= 1000) if args.mode == "full" else None,
        }
        correctness = gates["micro_qkv_bitexact"] and gates["all_block_token_parity"] and gates["all_qkv_graphs_contain_candidate"]
        stable = gates["base_block_p50_range_le_1ms"] and gates["qkv_block_p50_range_le_1ms"]
        if not correctness:
            status = "correctness_failed"
        elif args.mode == "full" and not stable:
            status = "unresolved_unstable"
        elif args.mode == "full" and gates["positive_gain"] and gates["no_material_regression"] and gates["full_samples_ge_1000_each"]:
            status = "positive_recommend_compose"
        elif args.mode == "full":
            status = "negative_stable" if stable else "unresolved_unstable"
        else:
            status = "smoke_pass"
        payload.update({"blocks": blocks, "parity": parity, "gates": gates,
                        "summary": {"base_block_p50_ms": base_p50, "qkv_block_p50_ms": cand_p50,
                                    "base_median_p50_ms": base_median,
                                    "qkv_median_p50_ms": cand_median,
                                    "base_range_ms": base_range, "qkv_range_ms": cand_range,
                                    "gain_ms": gain, "required_gain_ms": required_gain,
                                    "base_samples": base_samples, "qkv_samples": cand_samples,
                                    "qkv_tok_s_from_block_median": 1000.0 / cand_median},
                        "status": status, "completed_utc": utc_now()})
    except Exception as exc:
        payload.update({"status": "technical_failure", "completed_utc": utc_now(),
                        "error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()}})
    finally:
        if restore_qkv is not None:
            try: restore_qkv()
            except Exception: pass
        if bundle is not None:
            rt, dense, down, up, restore_sel, restore_moe = bundle
            try: restore_sel(); restore_moe()
            except Exception: pass
            del rt, dense, down, up
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp"); tmp.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8"); tmp.replace(OUT)
    print(json.dumps({"status": payload.get("status"), "summary": payload.get("summary"),
                      "gates": payload.get("gates"), "output": str(OUT)}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
