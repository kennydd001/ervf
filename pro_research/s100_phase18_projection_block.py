from __future__ import annotations

import json
import statistics
import traceback
from pathlib import Path

import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now

OUT = REPO / "pro_research" / "results" / "s100_phase18_projection_block.json"
HORIZONS = (2, 4, 8)
LAYERS_TO_SAMPLE = 3
CORR_TOL = 1e-4


def kernel_source(T: int) -> str:
    return f"""
extern "C" __global__ void batched_nvfp4_t{T}(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ e2m1,
    const float* __restrict__ e4m3,
    const float* __restrict__ x,
    float* __restrict__ out,
    const float global_scale,
    const int rows,
    const int cols)
{{
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = e2m1[threadIdx.x];
    __shared__ float warp_sums[{T}][32];
    float acc[{T}];
    #pragma unroll
    for (int t = 0; t < {T}; ++t) acc[t] = 0.0f;
    __syncthreads();
    const int n_bytes = cols >> 1;
    const int n_vec = n_bytes >> 2;
    const int n_scales = cols >> 4;
    const unsigned char* crow = codes + (size_t)row * n_bytes;
    const unsigned char* srow = scales + (size_t)row * n_scales;
    const uchar4* crow4 = reinterpret_cast<const uchar4*>(crow);
    for (int v = threadIdx.x; v < n_vec; v += blockDim.x) {{
        const uchar4 q = crow4[v];
        const int b = v << 2;
        const float s = e4m3[srow[b >> 3]] * global_scale;
        const int k = b << 1;
        const float w0 = lut[q.x & 0x0F] * s;
        const float w1 = lut[q.x >> 4] * s;
        const float w2 = lut[q.y & 0x0F] * s;
        const float w3 = lut[q.y >> 4] * s;
        const float w4 = lut[q.z & 0x0F] * s;
        const float w5 = lut[q.z >> 4] * s;
        const float w6 = lut[q.w & 0x0F] * s;
        const float w7 = lut[q.w >> 4] * s;
        #pragma unroll
        for (int t = 0; t < {T}; ++t) {{
            const float* xt = x + (size_t)t * cols + k;
            acc[t] = fmaf(w0, xt[0], acc[t]);
            acc[t] = fmaf(w1, xt[1], acc[t]);
            acc[t] = fmaf(w2, xt[2], acc[t]);
            acc[t] = fmaf(w3, xt[3], acc[t]);
            acc[t] = fmaf(w4, xt[4], acc[t]);
            acc[t] = fmaf(w5, xt[5], acc[t]);
            acc[t] = fmaf(w6, xt[6], acc[t]);
            acc[t] = fmaf(w7, xt[7], acc[t]);
        }}
    }}
    for (int b = (n_vec << 2) + threadIdx.x; b < n_bytes; b += blockDim.x) {{
        const unsigned char q = crow[b];
        const float s = e4m3[srow[b >> 3]] * global_scale;
        const int k = b << 1;
        #pragma unroll
        for (int t = 0; t < {T}; ++t) {{
            const float* xt = x + (size_t)t * cols + k;
            acc[t] = fmaf(lut[q & 0x0F] * s, xt[0], acc[t]);
            acc[t] = fmaf(lut[q >> 4] * s, xt[1], acc[t]);
        }}
    }}
    #pragma unroll
    for (int t = 0; t < {T}; ++t) {{
        for (int off = 16; off > 0; off >>= 1)
            acc[t] += __shfl_down_sync(0xffffffffu, acc[t], off);
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) warp_sums[t][warp] = acc[t];
    }}
    __syncthreads();
    if ((threadIdx.x >> 5) == 0) {{
        const int lane = threadIdx.x & 31;
        const int nw = (blockDim.x + 31) >> 5;
        #pragma unroll
        for (int t = 0; t < {T}; ++t) {{
            float v = lane < nw ? warp_sums[t][lane] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) out[(size_t)t * rows + row] = v;
        }}
    }}
}}
"""


def timed(cp, fn, reps=16):
    for _ in range(3):
        fn()
    cp.cuda.get_current_stream().synchronize()
    vals = []
    for _ in range(reps):
        a, b = cp.cuda.Event(), cp.cuda.Event()
        a.record()
        fn()
        b.record()
        b.synchronize()
        vals.append(float(cp.cuda.get_elapsed_time(a, b)))
    return {
        "median_ms": statistics.median(vals),
        "p10_ms": float(np.percentile(vals, 10)),
        "p90_ms": float(np.percentile(vals, 90)),
    }


def project_current(rt, d, out, x, side):
    if side == "in":
        rows, cols = int(rt.proj.size), int(rt.hidden)
        kind = d["in_k"]
        codes, scales, g = d.get("in_codes"), d.get("in_scales"), d.get("in_g")
    else:
        rows, cols = int(rt.hidden), int(rt.d_inner)
        kind = d["out_k"]
        codes, scales, g = d.get("out_codes"), d.get("out_scales"), d.get("out_g")
    if kind == "nvfp4":
        rt.fused.gemv_into(out, codes, scales, x, g, rows, cols)
    elif kind == "fp8_tensor":
        w = d["in_w8"] if side == "in" else d["out_w8"]
        s = d["in_s"] if side == "in" else d["out_s"]
        rt.k.mv_fp8_tensor(out, w, x, s, rows, cols)
    else:
        w = d["in_w"] if side == "in" else d["out_w"]
        rt.k.mv_bf16(out, w, x, rows, cols)
    return rows, cols, kind


class ProjectionBlock:
    def __init__(self, rt):
        import torch
        import cupy as cp

        self.rt = rt
        self.torch = torch
        self.cp = cp
        self.modules = {}
        self.weight_t = {}

    def _module(self, T):
        mod = self.modules.get(T)
        if mod is None:
            mod = self.cp.RawModule(
                code=kernel_source(T), options=("-std=c++14",)
            )
            self.modules[T] = mod.get_function(f"batched_nvfp4_t{T}")
        return self.modules[T]

    def _bf16_weight(self, W, rows, cols):
        key = (int(W.data.ptr), rows, cols)
        wt = self.weight_t.get(key)
        if wt is None:
            raw = (
                self.torch.utils.dlpack.from_dlpack(W)
                .view(self.torch.bfloat16)
                .reshape(rows, cols)
                .clone()
            )
            wt = raw.t().contiguous()
            self.weight_t[key] = wt
        return wt

    def apply(self, d, x, side, out, bf16_mode="bf16_out"):
        cp = self.cp
        T, cols = x.shape
        if side == "in":
            rows, kind = int(self.rt.proj.size), d["in_k"]
            codes, scales, g = d.get("in_codes"), d.get("in_scales"), d.get("in_g")
            W = d.get("in_w")
        else:
            rows, kind = int(self.rt.hidden), d["out_k"]
            codes, scales, g = d.get("out_codes"), d.get("out_scales"), d.get("out_g")
            W = d.get("out_w")
        if kind == "nvfp4":
            self._module(T)(
                (rows,), (256,),
                (codes, scales, self.rt.fused.e2m1, self.rt.fused.e4m3,
                 x, out, np.float32(g), np.int32(rows), np.int32(cols)),
            )
        elif kind == "bf16":
            xt = self.torch.utils.dlpack.from_dlpack(x)
            xt = xt.to(self.torch.bfloat16)
            wt = self._bf16_weight(W, rows, cols)
            if bf16_mode == "fp32_out":
                y = self.torch.mm(xt, wt, out_dtype=self.torch.float32)
            else:
                y = self.torch.mm(xt, wt).float()
            self.torch.utils.dlpack.from_dlpack(out).copy_(y)
        else:
            raise RuntimeError(f"unsupported projection format: {kind}")
        return rows, cols, kind


def main():
    payload = {
        "kind": "s100_phase18_projection_block",
        "status": "started",
        "horizons": list(HORIZONS),
        "claim_boundary": "format-preserving projection block benchmark; no full-layer claim",
        "started_utc": utc_now(),
    }
    try:
        import torch
        import cupy as cp
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

        rt = LightningRuntime(
            require_model_dir(), contexts_max=512,
            embed_on_host=True, fp8_kv=True, verbose=False,
        )
        rt.load_routed_bank()
        rt.deterministic_accum = True
        block = ProjectionBlock(rt)
        layers = [int(x) for x in rt.mamba_layers]
        chosen = sorted({layers[0], layers[len(layers) // 2], layers[-1]})
        results = []

        for layer in chosen:
            d = rt.layer[layer]
            for side in ("in", "out"):
                kind = d["in_k"] if side == "in" else d["out_k"]
                if kind not in ("nvfp4", "bf16"):
                    continue
                cols = int(rt.hidden) if side == "in" else int(rt.d_inner)
                rows = int(rt.proj.size) if side == "in" else int(rt.hidden)
                for T in HORIZONS:
                    rng = np.random.default_rng(1700 + layer * 10 + T)
                    x = cp.asarray(
                        rng.standard_normal((T, cols), dtype=np.float32)
                    )
                    base = cp.empty((T, rows), cp.float32)
                    cand = cp.empty_like(base)

                    def baseline():
                        for t in range(T):
                            project_current(rt, d, base[t], x[t], side)

                    def candidate():
                        block.apply(d, x, side, cand)

                    baseline()
                    candidate()
                    cp.cuda.get_current_stream().synchronize()
                    base_np = cp.asnumpy(base)
                    cand_np = cp.asnumpy(cand)
                    diff = cand_np - base_np
                    corr = {
                        "nrmse": float(
                            np.linalg.norm(diff)
                            / max(np.linalg.norm(base_np), 1e-30)
                        ),
                        "max_abs": float(np.max(np.abs(diff))),
                        "finite": bool(np.isfinite(cand_np).all()),
                    }
                    corr["pass"] = corr["nrmse"] <= CORR_TOL and corr["finite"]
                    bt = timed(cp, baseline)
                    ct = timed(cp, candidate)
                    rec = {
                        "layer": layer, "side": side, "format": kind,
                        "H": T, "rows": rows, "cols": cols,
                        "correctness": corr,
                        "baseline": bt, "candidate": ct,
                        "speedup": bt["median_ms"] / ct["median_ms"],
                    }
                    print(
                        f"P18 layer={layer} {side} {kind} H={T}: "
                        f"speed={rec['speedup']:.3f}x "
                        f"nrmse={corr['nrmse']:.2e} pass={corr['pass']}",
                        flush=True,
                    )
                    results.append(rec)
                    if kind == "bf16":
                        fp32_cand = cp.empty_like(base)

                        def candidate_fp32():
                            block.apply(d, x, side, fp32_cand, bf16_mode="fp32_out")

                        candidate_fp32()
                        cp.cuda.get_current_stream().synchronize()
                        fp32_np = cp.asnumpy(fp32_cand)
                        fp32_diff = fp32_np - base_np
                        fp32_corr = {
                            "nrmse": float(
                                np.linalg.norm(fp32_diff)
                                / max(np.linalg.norm(base_np), 1e-30)
                            ),
                            "max_abs": float(np.max(np.abs(fp32_diff))),
                            "finite": bool(np.isfinite(fp32_np).all()),
                        }
                        fp32_corr["pass"] = (
                            fp32_corr["nrmse"] <= CORR_TOL
                            and fp32_corr["finite"]
                        )
                        fp32_t = timed(cp, candidate_fp32)
                        results.append({
                            "layer": layer, "side": side,
                            "format": kind, "variant": "fp32_out", "H": T,
                            "rows": rows, "cols": cols,
                            "correctness": fp32_corr,
                            "candidate": fp32_t,
                            "speedup_vs_serial_baseline": (
                                bt["median_ms"] / fp32_t["median_ms"]
                            ),
                        })
                        print(
                            f"P18 layer={layer} {side} bf16 fp32_out H={T}: "
                            f"speed={bt['median_ms'] / fp32_t['median_ms']:.3f}x "
                            f"nrmse={fp32_corr['nrmse']:.2e} pass={fp32_corr['pass']}",
                            flush=True,
                        )
                        del fp32_cand
                    del x, base, cand
                    cp.get_default_memory_pool().free_all_blocks()

        payload.update({
            "status": "measured",
            "sampled_layers": chosen,
            "results": results,
            "all_correct": all(r["correctness"]["pass"] for r in results),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
