from __future__ import annotations

import json
import statistics
import traceback
from pathlib import Path

import numpy as np

from common import REPO, require_model_dir, utc_now, write_json_atomic
from s100_phase18_projection_block import (
    HORIZONS,
    ProjectionBlock,
    project_current,
    timed,
)

OUT = REPO / "pro_research" / "results" / "s100_phase19_residual_projection.json"
CORR_TOL = 1e-4
LAYERS_TO_SAMPLE = 3


def fp8_block_source(T):
    return f"""
__device__ __forceinline__ float e4m3_decode(unsigned char x) {{
    const int s = x >> 7, E = (x >> 3) & 0xF, m = x & 7;
    const float v = (E == 0) ? ((float)m * 1.953125e-3f)
                             : ((float)(8 + m) * exp2f((float)(E - 10)));
    return s ? -v : v;
}}
extern "C" __global__ void batched_fp8_t{T}(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ out, const float wscale,
    const int rows, const int cols) {{
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float lut[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = e4m3_decode((unsigned char)i);
    __syncthreads();
    const unsigned char* w = W + (size_t)row * cols;
    float acc[{T}];
    #pragma unroll
    for (int t = 0; t < {T}; ++t) acc[t] = 0.0f;
    for (int k = threadIdx.x; k < cols; k += blockDim.x) {{
        const float wv = lut[w[k]] * wscale;
        #pragma unroll
        for (int t = 0; t < {T}; ++t)
            acc[t] = fmaf(wv, x[(size_t)t * cols + k], acc[t]);
    }}
    #pragma unroll
    for (int t = 0; t < {T}; ++t) {{
        for (int o = 16; o > 0; o >>= 1)
            acc[t] += __shfl_down_sync(0xffffffffu, acc[t], o);
        __shared__ float ws[{T}][32];
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        if (lane == 0) ws[t][warp] = acc[t];
        __syncthreads();
        if (warp == 0) {{
            const int nw = (blockDim.x + 31) >> 5;
            float v = lane < nw ? ws[t][lane] : 0.0f;
            for (int o = 16; o > 0; o >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0) out[(size_t)t * rows + row] = v;
        }}
        __syncthreads();
    }}
}}
"""


def fp8_residual2_source(T):
    return f"""
__device__ __forceinline__ float e4m3_decode(unsigned char x) {{
    const int s = x >> 7, E = (x >> 3) & 0xF, m = x & 7;
    const float v = (E == 0) ? ((float)m * 1.953125e-3f)
                             : ((float)(8 + m) * exp2f((float)(E - 10)));
    return s ? -v : v;
}}
__device__ __forceinline__ float bf16_round(float v) {{
    unsigned int u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xFFFF0000u);
}}
extern "C" __global__ void batched_fp8_residual2_t{T}(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ out, const float wscale,
    const int rows, const int cols) {{
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float lut[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = e4m3_decode((unsigned char)i);
    __syncthreads();
    const unsigned char* w = W + (size_t)row * cols;
    float acc[{T}];
    #pragma unroll
    for (int t = 0; t < {T}; ++t) acc[t] = 0.0f;
    for (int k = threadIdx.x; k < cols; k += blockDim.x) {{
        const float wv = lut[w[k]] * wscale;
        #pragma unroll
        for (int t = 0; t < {T}; ++t) {{
            const float q0 = bf16_round(x[(size_t)t * cols + k]);
            const float q1 = bf16_round(x[(size_t)t * cols + k] - q0);
            acc[t] = fmaf(wv, q0, acc[t]);
            acc[t] = fmaf(wv, q1, acc[t]);
        }}
    }}
    #pragma unroll
    for (int t = 0; t < {T}; ++t) {{
        for (int o = 16; o > 0; o >>= 1)
            acc[t] += __shfl_down_sync(0xffffffffu, acc[t], o);
        __shared__ float ws[{T}][32];
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        if (lane == 0) ws[t][warp] = acc[t];
        __syncthreads();
        if (warp == 0) {{
            const int nw = (blockDim.x + 31) >> 5;
            float v = lane < nw ? ws[t][lane] : 0.0f;
            for (int o = 16; o > 0; o >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0) out[(size_t)t * rows + row] = v;
        }}
        __syncthreads();
    }}
}}
"""


class FP8ProjectionBlock:
    def __init__(self, cp):
        self.cp = cp
        self.modules = {}

    def _module(self, T):
        mod = self.modules.get(T)
        if mod is None:
            mod = self.cp.RawModule(
                code=fp8_block_source(T), options=("-std=c++14", "--use_fast_math")
            )
            self.modules[T] = mod.get_function(f"batched_fp8_t{T}")
        return self.modules[T]

    def apply(self, W, x, out, wscale):
        T, cols = x.shape
        rows = out.shape[1]
        self._module(T)((rows,), (256,),
                        (W, x, out, np.float32(wscale),
                         np.int32(rows), np.int32(cols)))

    def apply_residual2(self, W, x, out, wscale):
        T, cols = x.shape
        rows = out.shape[1]
        key = ("residual2", T)
        mod = self.modules.get(key)
        if mod is None:
            raw = self.cp.RawModule(
                code=fp8_residual2_source(T),
                options=("-std=c++14", "--use_fast_math"),
            )
            mod = raw.get_function(f"batched_fp8_residual2_t{T}")
            self.modules[key] = mod
        mod((rows,), (256,),
            (W, x, out, np.float32(wscale),
             np.int32(rows), np.int32(cols)))


def torch_bf16_terms(torch, x_t, terms):
    terms_out = []
    residual = x_t
    for _ in range(terms):
        q = residual.to(torch.bfloat16)
        terms_out.append(q)
        residual = residual - q.float()
    return terms_out


def corr(a, b):
    d = b - a
    nrmse = float(np.linalg.norm(d) / max(np.linalg.norm(a), 1e-30))
    return {
        "nrmse": nrmse,
        "max_abs": float(np.max(np.abs(d))),
        "finite": bool(np.isfinite(b).all()),
        "pass": bool(nrmse <= CORR_TOL and np.isfinite(b).all()),
    }


def main():
    payload = {
        "kind": "s100_phase19_residual_projection",
        "status": "started",
        "model_id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        "horizons": list(HORIZONS),
        "residual_terms": [1, 2, 3],
        "correctness_gate_nrmse": CORR_TOL,
        "claim_boundary": (
            "residual projection screen; no full-layer claim until all sampled "
            "projection arms pass"
        ),
        "started_utc": utc_now(),
    }
    try:
        import torch
        import cupy as cp
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

        model_dir = require_model_dir()
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        if cfg.get("architectures") != ["NemotronHForCausalLM"]:
            raise RuntimeError(f"unexpected architectures: {cfg.get('architectures')}")
        rt = LightningRuntime(
            model_dir, contexts_max=512,
            embed_on_host=True, fp8_kv=True, verbose=False,
        )
        rt.load_routed_bank()
        rt.deterministic_accum = True
        block = ProjectionBlock(rt)
        fp8_block = FP8ProjectionBlock(cp)
        layers = [int(x) for x in rt.mamba_layers]
        chosen = sorted({layers[0], layers[len(layers) // 2], layers[-1]})
        results = []
        payload["available_projection_formats"] = {
            str(layer): {
                "in": rt.layer[layer]["in_k"],
                "out": rt.layer[layer]["out_k"],
            }
            for layer in chosen
        }

        for layer in chosen:
            d = rt.layer[layer]
            for side in ("in", "out"):
                kind = d["in_k"] if side == "in" else d["out_k"]
                if kind not in ("nvfp4", "bf16", "fp8_tensor"):
                    continue
                cols = int(rt.hidden) if side == "in" else int(rt.d_inner)
                rows = int(rt.proj.size) if side == "in" else int(rt.hidden)
                for H in HORIZONS:
                    rng = np.random.default_rng(1900 + layer * 10 + H)
                    x = cp.asarray(rng.standard_normal((H, cols), dtype=np.float32))
                    base = cp.empty((H, rows), cp.float32)

                    def baseline():
                        for t in range(H):
                            project_current(rt, d, base[t], x[t], side)

                    baseline()
                    cp.cuda.get_current_stream().synchronize()
                    base_np = cp.asnumpy(base)
                    bt = timed(cp, baseline)

                    for terms in (1, 2, 3):
                        cand = cp.empty_like(base)
                        if kind == "bf16":
                            xt = torch.utils.dlpack.from_dlpack(x)
                            qterms = torch_bf16_terms(torch, xt, terms)
                            wt = block._bf16_weight(
                                d["in_w"] if side == "in" else d["out_w"],
                                rows, cols,
                            )

                            def candidate():
                                ys = [
                                    torch.mm(q, wt, out_dtype=torch.float32)
                                    for q in qterms
                                ]
                                torch.utils.dlpack.from_dlpack(cand).copy_(_sum(ys, torch))

                        elif kind == "fp8_tensor":
                            xt = torch.utils.dlpack.from_dlpack(x)
                            qterms = torch_bf16_terms(torch, xt, terms)
                            xcp = [cp.asarray(q.float()) for q in qterms]
                            stacked_x = cp.concatenate(xcp, axis=0)
                            raw = cp.empty((terms * H, rows), cp.float32)
                            W = d["in_w8"] if side == "in" else d["out_w8"]
                            scale = d["in_s"] if side == "in" else d["out_s"]

                            def candidate():
                                fp8_block.apply(W, stacked_x, raw, scale)
                                cand[...] = raw.reshape(terms, H, rows).sum(axis=0)

                        else:
                            xt = torch.utils.dlpack.from_dlpack(x)
                            qterms = torch_bf16_terms(torch, xt, terms)
                            xcp = [cp.asarray(q.float()) for q in qterms]

                            def candidate():
                                for t in range(H):
                                    project_current(rt, d, cand[t], xcp[0][t], side)
                                    for q in xcp[1:]:
                                        tmp = cp.empty((rows,), cp.float32)
                                        project_current(rt, d, tmp, q[t], side)
                                        cand[t] += tmp

                        candidate()
                        cp.cuda.get_current_stream().synchronize()
                        got = cp.asnumpy(cand)
                        c = corr(base_np, got)
                        ct = timed(cp, candidate)
                        rec = {
                            "layer": layer, "side": side, "format": kind,
                            "H": H, "residual_terms": terms,
                            "rows": rows, "cols": cols,
                            "correctness": c,
                            "baseline": bt, "candidate": ct,
                            "speedup": bt["median_ms"] / ct["median_ms"],
                            "route_note": (
                                "existing exact fused GEMV per residual term; "
                                "not a scaled_mm Tensor-Core implementation"
                                if kind == "nvfp4" else
                                "existing exact FP8 tensor-vector primitive per "
                                "residual term; batched FP8 candidate is tested"
                                if kind == "fp8_tensor" else None
                            ),
                        }
                        print(
                            f"P19 layer={layer} {side} {kind} H={H} "
                            f"terms={terms}: speed={rec['speedup']:.3f}x "
                            f"nrmse={c['nrmse']:.2e} pass={c['pass']}",
                            flush=True,
                        )
                        results.append(rec)
                        del cand
                    del x, base
                    cp.get_default_memory_pool().free_all_blocks()

        payload.update({
            "status": "measured",
            "model_dir": str(model_dir),
            "config_identity": {
                "architectures": cfg.get("architectures"),
                "model_type": cfg.get("model_type"),
                "num_hidden_layers": cfg.get("num_hidden_layers"),
                "hidden_size": cfg.get("hidden_size"),
            },
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


def _sum(values, torch):
    out = values[0]
    for value in values[1:]:
        out = out + value
    return out


if __name__ == "__main__":
    raise SystemExit(main())
