"""PV2-12: fuse exact NVFP4 LM-head ERVF with hierarchical greedy top-1."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shared import (
    capture_v6, compare_arms, cuda_time, environment, graph_dot,
    new_v6_bundle, percentiles, pointer, prompt_set, result_path, run_arm,
    same_bits, status_from_gates, utc_now, write_json,
)

OUT = result_path("PV2_12_LMHEAD_ARGMAX.json")

CUDA = r"""
#define PV2_WIDTH 16
#define PV2_VIRTUAL 16
#define PV2_RPB 16

extern "C" __global__ void pv2_lmhead_ervf_block_argmax(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ e2m1_lut,
    const float* __restrict__ e4m3_lut,
    const float* __restrict__ x,
    float* __restrict__ debug_logits,
    float* __restrict__ block_max,
    int* __restrict__ block_idx,
    const float global_scale,
    const int rows, const int cols, const int write_logits)
{
    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __shared__ float s_e2m1[16];
    __shared__ float vals[16];
    __shared__ int ids[16];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    __syncthreads();

    const int lane = threadIdx.x & 15;
    const int sub = threadIdx.x >> 4;
    const int row = (int)blockIdx.x * 16 + sub;
    const bool valid = row < rows;
    const int n_bytes = cols >> 1;
    const int n_vec = n_bytes >> 2;
    const unsigned char* crow = codes + (size_t)(valid ? row : 0) * n_bytes;
    const unsigned char* srow = scales + (size_t)(valid ? row : 0) * (cols >> 4);
    const uchar4* crow4 = reinterpret_cast<const uchar4*>(crow);

    float part[16];
    #pragma unroll
    for (int vi = 0; vi < 16; ++vi) part[vi] = 0.0f;
    #pragma unroll
    for (int vi = 0; vi < 16; ++vi) {
        const int tid = lane + 16 * vi;
        float acc = 0.0f;
        if (valid) {
            for (int v = tid; v < n_vec; v += 256) {
                const uchar4 q = crow4[v];
                const int b = v << 2;
                const float s = e4m3_lut[srow[b >> 3]] * global_scale;
                const int k = b << 1;
                acc = fmaf(s_e2m1[q.x & 15] * s, sx[k], acc);
                acc = fmaf(s_e2m1[q.x >> 4] * s, sx[k + 1], acc);
                acc = fmaf(s_e2m1[q.y & 15] * s, sx[k + 2], acc);
                acc = fmaf(s_e2m1[q.y >> 4] * s, sx[k + 3], acc);
                acc = fmaf(s_e2m1[q.z & 15] * s, sx[k + 4], acc);
                acc = fmaf(s_e2m1[q.z >> 4] * s, sx[k + 5], acc);
                acc = fmaf(s_e2m1[q.w & 15] * s, sx[k + 6], acc);
                acc = fmaf(s_e2m1[q.w >> 4] * s, sx[k + 7], acc);
            }
            for (int b = (n_vec << 2) + tid; b < n_bytes; b += 256) {
                const unsigned char q = crow[b];
                const float s = e4m3_lut[srow[b >> 3]] * global_scale;
                const int k = b << 1;
                acc = fmaf(s_e2m1[q & 15] * s, sx[k], acc);
                acc = fmaf(s_e2m1[q >> 4] * s, sx[k + 1], acc);
            }
        }
        part[vi] = acc;
    }

    float s8[8];
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        float v = part[w * 2] + part[w * 2 + 1];
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off, 16);
        s8[w] = v;
    }
    float y = -3.0e38f;
    if (lane == 0 && valid) {
        const float t0 = s8[0] + s8[4];
        const float t1 = s8[1] + s8[5];
        const float t2 = s8[2] + s8[6];
        const float t3 = s8[3] + s8[7];
        const float u0 = t0 + t2;
        const float u1 = t1 + t3;
        y = u0 + u1;
        if (write_logits) debug_logits[row] = y;
    }
    if (lane == 0) {
        vals[sub] = y;
        ids[sub] = valid ? row : 0x7fffffff;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float best = vals[0];
        int bi = ids[0];
        #pragma unroll
        for (int j = 1; j < 16; ++j) {
            const float v = vals[j];
            const int ii = ids[j];
            if (v > best || (v == best && ii < bi)) { best = v; bi = ii; }
        }
        block_max[blockIdx.x] = best;
        block_idx[blockIdx.x] = bi;
    }
}

extern "C" __global__ void pv2_argmax_final_serial(
    const float* __restrict__ block_max,
    const int* __restrict__ block_idx,
    int* __restrict__ tok,
    const int nblocks)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float best = block_max[0];
    int bi = block_idx[0];
    for (int i = 1; i < nblocks; ++i) {
        const float v = block_max[i];
        const int ii = block_idx[i];
        if (v > best || (v == best && ii < bi)) { best = v; bi = ii; }
    }
    tok[0] = bi;
}
"""


class LMArgmax:
    def __init__(self, rows: int):
        import cupy as cp
        self.cp = cp
        self.rows = int(rows)
        self.nblocks = (self.rows + 15) // 16
        self.mod = cp.RawModule(code=CUDA, options=("-std=c++14",))
        self.main = self.mod.get_function("pv2_lmhead_ervf_block_argmax")
        self.final = self.mod.get_function("pv2_argmax_final_serial")
        self.block_max = cp.empty(self.nblocks, dtype=cp.float32)
        self.block_idx = cp.empty(self.nblocks, dtype=cp.int32)

    def apply(self, tok, logits, codes, scales, e2m1, e4m3, x,
              global_scale: float, rows: int, cols: int, write_logits: bool):
        self.main((self.nblocks,), (256,),
                  (codes, scales, e2m1, e4m3, x, logits,
                   self.block_max, self.block_idx, np.float32(global_scale),
                   np.int32(rows), np.int32(cols),
                   np.int32(1 if write_logits else 0)),
                  shared_mem=cols * 4)
        self.final((1,), (256,),
                   (self.block_max, self.block_idx, tok, np.int32(self.nblocks)))


def install(rt, candidate: LMArgmax):
    orig_gemv = rt.fused.gemv_into
    orig_argmax = rt.k.argmax_logits
    lm_codes_ptr = pointer(rt.lm_head_codes)
    logits_ptr = pointer(rt.logits)

    def gemv(out, codes, scales, x, global_scale, rows, cols,
             apply_relu2=False, out_scale=1.0):
        if (pointer(out) == logits_ptr and pointer(codes) == lm_codes_ptr
                and int(rows) == int(rt.vocab) and int(cols) == int(rt.hidden)
                and not apply_relu2 and float(out_scale) == 1.0):
            candidate.apply(rt._tok_dev, rt.logits, codes, scales,
                            rt.fused.e2m1, rt.fused.e4m3, x,
                            float(global_scale), int(rows), int(cols), False)
            return None
        return orig_gemv(out, codes, scales, x, global_scale, rows, cols,
                         apply_relu2=apply_relu2, out_scale=out_scale)

    def argmax(tok, logits, vocab, maxbuf, idxbuf):
        if pointer(logits) == logits_ptr and pointer(tok) == pointer(rt._tok_dev):
            return None
        return orig_argmax(tok, logits, vocab, maxbuf, idxbuf)

    rt.fused.gemv_into = gemv
    rt.k.argmax_logits = argmax

    def restore():
        rt.fused.gemv_into = orig_gemv
        rt.k.argmax_logits = orig_argmax
    return restore


def micro(rt, candidate: LMArgmax, full: bool) -> dict[str, Any]:
    import cupy as cp
    rows, cols = int(rt.vocab), int(rt.hidden)
    x = cp.random.RandomState(20260816).standard_normal(cols, dtype=cp.float32)
    lr = cp.empty(rows, cp.float32)
    lc = cp.empty(rows, cp.float32)
    tr = cp.zeros(1, cp.int32)
    tc = cp.zeros(1, cp.int32)
    amx = cp.empty(256, cp.float32)
    ami = cp.empty(256, cp.int32)

    rt.fused.gemv_into(lr, rt.lm_head_codes, rt.lm_head_scales, x,
                       rt.lm_head_g, rows, cols)
    rt.k.argmax_logits(tr, lr, rows, amx, ami)
    candidate.apply(tc, lc, rt.lm_head_codes, rt.lm_head_scales,
                    rt.fused.e2m1, rt.fused.e4m3, x, rt.lm_head_g,
                    rows, cols, True)
    cp.cuda.Device(0).synchronize()
    exact_logits = same_bits(lr, lc)
    ref_tok, cand_tok = int(cp.asnumpy(tr)[0]), int(cp.asnumpy(tc)[0])

    def base_fn():
        rt.fused.gemv_into(lr, rt.lm_head_codes, rt.lm_head_scales, x,
                           rt.lm_head_g, rows, cols)
        rt.k.argmax_logits(tr, lr, rows, amx, ami)

    def cand_fn():
        candidate.apply(tc, lc, rt.lm_head_codes, rt.lm_head_scales,
                        rt.fused.e2m1, rt.fused.e4m3, x, rt.lm_head_g,
                        rows, cols, False)

    repeats = 120 if full else 30
    bt, ct = cuda_time(base_fn, 5, repeats), cuda_time(cand_fn, 5, repeats)
    bp, cpct = percentiles(bt), percentiles(ct)
    return {
        "all_logits_bitexact": exact_logits,
        "reference_token": ref_tok, "candidate_token": cand_tok,
        "top1_exact": ref_tok == cand_tok,
        "base_ms": bp, "candidate_ms": cpct,
        "speedup_p50": float(bp["p50"] / cpct["p50"]),
        "rows": rows, "cols": cols, "nblocks": candidate.nblocks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload: dict[str, Any] = {"kind": "pv2_lmhead_argmax", "status": "started",
                               "mode": args.mode, "started_utc": utc_now(),
                               "preregistration": "PREREGISTRATION.md",
                               "claim_boundary": "greedy top-1 decode only; logits-requesting APIs retain production path"}
    bundle = None
    try:
        from shared import require_gpu_free
        require_gpu_free()
        prompts, _expected, n, _capacity = prompt_set(args.mode)
        payload["environment"] = environment((Path(__file__), HERE / "PREREGISTRATION.md"))
        bundle = new_v6_bundle(); rt = bundle.rt
        candidate = LMArgmax(rt.vocab)
        payload["micro"] = micro(rt, candidate, args.mode == "full")

        capture_v6(bundle); base_a = run_arm(rt, prompts, n)
        restore = install(rt, candidate)
        extra = capture_v6(bundle); dot = graph_dot(rt).lower(); cand = run_arm(rt, prompts, n)
        restore()
        capture_v6(bundle); base_b = run_arm(rt, prompts, n)

        par_a, par_b = compare_arms(base_a, cand), compare_arms(base_b, cand)
        pa, pc, pb = float(base_a["timing_ms"]["p50"]), float(cand["timing_ms"]["p50"]), float(base_b["timing_ms"]["p50"])
        mid, drift = (pa + pb) / 2.0, abs(pa - pb)
        gates = {
            "micro_all_logits_bitexact": bool(payload["micro"]["all_logits_bitexact"]),
            "micro_top1_exact": bool(payload["micro"]["top1_exact"]),
            "micro_speedup_ge_1_02": payload["micro"]["speedup_p50"] >= 1.02,
            "graph_contains_main": "pv2_lmhead_ervf_block_argmax" in dot,
            "graph_contains_final": "pv2_argmax_final_serial" in dot,
            "causal_parity": all(x["identical"] for x in par_a.values()) and all(x["identical"] for x in par_b.values()),
            "base_drift_le_1ms": drift <= 1.0,
            "no_regression_gt_0_2pct": pc <= mid * 1.002,
            "samples_ge_500": int(cand["timing_ms"]["count"]) >= 500 if args.mode == "full" else None,
            "extra_vram_lt_64MiB": extra < 64 * 1024 * 1024,
        }
        required = ("micro_all_logits_bitexact", "micro_top1_exact",
                    "micro_speedup_ge_1_02", "graph_contains_main",
                    "graph_contains_final", "causal_parity",
                    "base_drift_le_1ms", "no_regression_gt_0_2pct",
                    "extra_vram_lt_64MiB")
        if args.mode == "full": required += ("samples_ge_500",)
        payload.update({
            "graph": {"dot_length": len(dot), "candidate_name_present": "pv2_lmhead_ervf_block_argmax" in dot, "extra_vram_bytes": extra},
            "arms": {"BASE_A": base_a, "LMHEAD_ARGMAX": cand, "BASE_B": base_b},
            "parity": {"candidate_vs_base_a": par_a, "candidate_vs_base_b": par_b},
            "gates": gates, "adopt": all(bool(gates[k]) for k in required),
            "summary": {"base_a_p50_ms": pa, "candidate_p50_ms": pc,
                        "base_b_p50_ms": pb, "baseline_mid_p50_ms": mid,
                        "gain_ms": mid - pc, "candidate_tok_s": 1000.0 / pc,
                        "base_drift_ms": drift},
            "status": status_from_gates(gates, required), "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update(status="technical_failure", completed_utc=utc_now(),
                       error={"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()})
    finally:
        if bundle is not None: bundle.close()
    write_json(OUT, payload)
    print(json.dumps({"status": payload.get("status"), "adopt": payload.get("adopt"),
                      "summary": payload.get("summary"), "output": str(OUT)}, indent=2))
    return 0 if payload.get("status") in {"pass", "gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
