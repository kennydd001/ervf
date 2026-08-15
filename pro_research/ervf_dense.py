"""PRO-G1: generalise ERVF to resident BF16, FP8-tensor and FP32 GEMVs.

NERVF proved the exact reduction remapping for NVFP4. The resident shell still
uses the old one-row/256-thread reduction geometry for attention projections,
Mamba FP8 projections, and MoE routers. This file implements the same logical
reduction tree with 16 physical lanes per output row and 16 rows per block.

The experiment is fail-closed: full-model integration is skipped unless every
registered real checkpoint shape is bit-identical to the production kernel.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import (
    REPO,
    environment_snapshot,
    first_divergence,
    geometric_mean,
    load_json,
    percentiles,
    require_gpu_free,
    require_model_dir,
    result_path,
    utc_now,
    write_json_atomic,
)

OUT = result_path("PRO_G1_DENSE_ERVF.json")
TS200 = REPO / "reports" / "treesweep200"

CUDA_SOURCE = r"""
__device__ __forceinline__ float pro_bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

__device__ __forceinline__ float pro_e4m3_decode(unsigned char x) {
    const int s = x >> 7, E = (x >> 3) & 0xF, m = x & 7;
    const float v = (E == 0) ? ((float)m * 1.953125e-3f)
                             : ((float)(8 + m) * exp2f((float)(E - 10)));
    return s ? -v : v;
}

#define PRO_WIDTH 16
#define PRO_VIRTUAL (256 / PRO_WIDTH)
#define PRO_ROWS_PER_BLOCK (256 / PRO_WIDTH)

// Reconstruct the production kernel's exact two-level reduction tree.
// Each physical lane owns the reference accumulators for virtual tids
// lane + 16*vi. The reference offset-16 operation is lane-local; offsets
// 8/4/2/1 remain 16-wide shuffles. Lane zero then reproduces the reference
// second-warp reduction over its eight warp sums.
__device__ __forceinline__ float pro_reduce_exact(float acc[PRO_VIRTUAL]) {
    const int lane = threadIdx.x & (PRO_WIDTH - 1);
    float s[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        float v = acc[g * 2] + acc[g * 2 + 1];
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o, PRO_WIDTH);
        s[g] = v;
    }
    if (lane == 0) {
        const float a0 = s[0] + s[4];
        const float a1 = s[1] + s[5];
        const float a2 = s[2] + s[6];
        const float a3 = s[3] + s[7];
        const float b0 = a0 + a2;
        const float b1 = a1 + a3;
        return b0 + b1;
    }
    return 0.0f;
}

extern "C" __global__ void pro_gemv_bf16_ervf16(
    const unsigned short* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ out,
    const int rows, const int cols)
{
    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __syncthreads();

    const int sub = threadIdx.x / PRO_WIDTH;
    const int lane = threadIdx.x & (PRO_WIDTH - 1);
    const int row = blockIdx.x * PRO_ROWS_PER_BLOCK + sub;
    const bool valid = row < rows;
    const unsigned short* w = W + (size_t)(valid ? row : 0) * cols;

    float acc[PRO_VIRTUAL];
    #pragma unroll
    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) acc[vi] = 0.0f;
    #pragma unroll
    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) {
        const int tid = lane + PRO_WIDTH * vi;
        if (valid)
            for (int k = tid; k < cols; k += 256)
                acc[vi] = fmaf(pro_bf16_to_f32(w[k]), sx[k], acc[vi]);
    }
    const float v = pro_reduce_exact(acc);
    if (lane == 0 && valid) out[row] = v;
}

extern "C" __global__ void pro_gemv_f32_ervf16(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ out,
    const int rows, const int cols)
{
    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __syncthreads();

    const int sub = threadIdx.x / PRO_WIDTH;
    const int lane = threadIdx.x & (PRO_WIDTH - 1);
    const int row = blockIdx.x * PRO_ROWS_PER_BLOCK + sub;
    const bool valid = row < rows;
    const float* w = W + (size_t)(valid ? row : 0) * cols;

    float acc[PRO_VIRTUAL];
    #pragma unroll
    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) acc[vi] = 0.0f;
    #pragma unroll
    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) {
        const int tid = lane + PRO_WIDTH * vi;
        if (valid)
            for (int k = tid; k < cols; k += 256)
                acc[vi] = fmaf(w[k], sx[k], acc[vi]);
    }
    const float v = pro_reduce_exact(acc);
    if (lane == 0 && valid) out[row] = v;
}

extern "C" __global__ void pro_gemv_fp8_tensor_ervf16(
    const unsigned char* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ out,
    const float wscale,
    const int rows, const int cols)
{
    extern __shared__ float smem[];
    float* sx = smem;
    float* lut = smem + cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = pro_e4m3_decode((unsigned char)i);
    __syncthreads();

    const int sub = threadIdx.x / PRO_WIDTH;
    const int lane = threadIdx.x & (PRO_WIDTH - 1);
    const int row = blockIdx.x * PRO_ROWS_PER_BLOCK + sub;
    const bool valid = row < rows;
    const unsigned char* w = W + (size_t)(valid ? row : 0) * cols;

    float acc[PRO_VIRTUAL];
    #pragma unroll
    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) acc[vi] = 0.0f;
    const int nvec = cols >> 2;
    const uchar4* w4 = reinterpret_cast<const uchar4*>(w);
    #pragma unroll
    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) {
        const int tid = lane + PRO_WIDTH * vi;
        if (valid) {
            for (int qidx = tid; qidx < nvec; qidx += 256) {
                const uchar4 q = w4[qidx];
                const int k = qidx << 2;
                acc[vi] = fmaf(lut[q.x], sx[k],     acc[vi]);
                acc[vi] = fmaf(lut[q.y], sx[k + 1], acc[vi]);
                acc[vi] = fmaf(lut[q.z], sx[k + 2], acc[vi]);
                acc[vi] = fmaf(lut[q.w], sx[k + 3], acc[vi]);
            }
            for (int b = (nvec << 2) + tid; b < cols; b += 256)
                acc[vi] = fmaf(lut[w[b]], sx[b], acc[vi]);
        }
    }
    const float v = pro_reduce_exact(acc);
    if (lane == 0 && valid) out[row] = v * wscale;
}
"""


@dataclass
class Case:
    name: str
    kind: str
    W: Any
    rows: int
    cols: int
    scale: float = 1.0
    calls_per_token: int = 1


class DenseERVF:
    def __init__(self):
        import cupy as cp

        self.cp = cp
        self.mod = cp.RawModule(
            code=CUDA_SOURCE, options=("-std=c++14", "--use_fast_math")
        )
        self.bf16 = self.mod.get_function("pro_gemv_bf16_ervf16")
        self.f32 = self.mod.get_function("pro_gemv_f32_ervf16")
        self.fp8 = self.mod.get_function("pro_gemv_fp8_tensor_ervf16")
        self.block = 256
        self.rows_per_block = 16

    def mv_bf16(self, out, W, x, rows: int, cols: int):
        self.bf16(
            ((rows + self.rows_per_block - 1) // self.rows_per_block,),
            (self.block,),
            (W, x, out, np.int32(rows), np.int32(cols)),
            shared_mem=cols * 4,
        )

    def mv_f32(self, out, W, x, rows: int, cols: int):
        self.f32(
            ((rows + self.rows_per_block - 1) // self.rows_per_block,),
            (self.block,),
            (W, x, out, np.int32(rows), np.int32(cols)),
            shared_mem=cols * 4,
        )

    def mv_fp8_tensor(self, out, W, x, wscale: float, rows: int, cols: int):
        self.fp8(
            ((rows + self.rows_per_block - 1) // self.rows_per_block,),
            (self.block,),
            (W, x, out, np.float32(wscale), np.int32(rows), np.int32(cols)),
            shared_mem=(cols + 256) * 4,
        )


def _fadd(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(np.float64(a) + np.float64(b))


def _reference_reduce(vals: np.ndarray) -> np.float32:
    a = np.asarray(vals, dtype=np.float32).copy()
    if a.shape != (256,):
        raise ValueError("expected 256 virtual accumulators")
    warp_sums = np.zeros(8, dtype=np.float32)
    for warp in range(8):
        w = a[warp * 32:(warp + 1) * 32].copy()
        for offset in (16, 8, 4, 2, 1):
            old = w.copy()
            for lane in range(32 - offset):
                w[lane] = _fadd(old[lane], old[lane + offset])
        warp_sums[warp] = w[0]
    lanes = np.zeros(32, dtype=np.float32)
    lanes[:8] = warp_sums
    for offset in (16, 8, 4, 2, 1):
        old = lanes.copy()
        for lane in range(32 - offset):
            lanes[lane] = _fadd(old[lane], old[lane + offset])
    return lanes[0]


def _ervf_reduce(vals: np.ndarray) -> np.float32:
    a = np.asarray(vals, dtype=np.float32)
    lane_warp = np.zeros((16, 8), dtype=np.float32)
    for lane in range(16):
        for g in range(8):
            lane_warp[lane, g] = _fadd(a[lane + 32 * g], a[lane + 16 + 32 * g])
    for g in range(8):
        w = lane_warp[:, g].copy()
        for offset in (8, 4, 2, 1):
            old = w.copy()
            for lane in range(16 - offset):
                w[lane] = _fadd(old[lane], old[lane + offset])
        lane_warp[:, g] = w
    s = lane_warp[0]
    a0 = _fadd(s[0], s[4])
    a1 = _fadd(s[1], s[5])
    a2 = _fadd(s[2], s[6])
    a3 = _fadd(s[3], s[7])
    return _fadd(_fadd(a0, a2), _fadd(a1, a3))


def cpu_selftest() -> dict[str, Any]:
    rng = np.random.default_rng(20260815)
    mismatches = 0
    examples = []
    for seed in range(200):
        vals = (rng.standard_normal(256) * (10.0 ** rng.uniform(-4, 4))).astype(np.float32)
        r = _reference_reduce(vals)
        e = _ervf_reduce(vals)
        same = r.view(np.uint32) == e.view(np.uint32)
        if not same:
            mismatches += 1
            if len(examples) < 3:
                examples.append({"seed": seed, "reference_bits": int(r.view(np.uint32)), "ervf_bits": int(e.view(np.uint32))})
    return {"trials": 200, "mismatches": mismatches, "examples": examples, "passed": mismatches == 0}


def _new_runtime():
    sys.path.insert(0, str(REPO / "src"))
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    return LightningRuntime(
        require_model_dir(), contexts_max=4096, embed_on_host=True,
        fp8_kv=True, verbose=False,
    )


def _collect_cases(rt) -> list[Case]:
    cases: list[Case] = []
    # One real layer per distinct projection family/shape is enough for the
    # exact geometry test; calls_per_token records full-runtime importance.
    if rt.attn_layers:
        i = rt.attn_layers[0]
        d = rt.layer[i]
        hq = rt.n_heads * rt.head_dim
        hkv = rt.n_kv * rt.head_dim
        cases.extend([
            Case("attn_q_bf16", "bf16", d["q_proj"], hq, rt.hidden, calls_per_token=len(rt.attn_layers)),
            Case("attn_k_bf16", "bf16", d["k_proj"], hkv, rt.hidden, calls_per_token=len(rt.attn_layers)),
            Case("attn_v_bf16", "bf16", d["v_proj"], hkv, rt.hidden, calls_per_token=len(rt.attn_layers)),
            Case("attn_o_bf16", "bf16", d["o_proj"], rt.hidden, hq, calls_per_token=len(rt.attn_layers)),
        ])
    if rt.moe_layers:
        i = rt.moe_layers[0]
        d = rt.layer[i]
        cases.append(Case("router_f32", "f32", d["gate_w"], rt.n_experts, rt.hidden, calls_per_token=len(rt.moe_layers)))
    if rt.mamba_layers:
        i = rt.mamba_layers[0]
        d = rt.layer[i]
        if d["in_k"] == "fp8_tensor":
            cases.append(Case("mamba_in_fp8", "fp8", d["in_w8"], int(rt.proj.size), rt.hidden, float(d["in_s"]), len(rt.mamba_layers)))
        elif d["in_k"] == "bf16":
            cases.append(Case("mamba_in_bf16", "bf16", d["in_w"], int(rt.proj.size), rt.hidden, calls_per_token=len(rt.mamba_layers)))
        if d["out_k"] == "fp8_tensor":
            cases.append(Case("mamba_out_fp8", "fp8", d["out_w8"], rt.hidden, rt.d_inner, float(d["out_s"]), len(rt.mamba_layers)))
        elif d["out_k"] == "bf16":
            cases.append(Case("mamba_out_bf16", "bf16", d["out_w"], rt.hidden, rt.d_inner, calls_per_token=len(rt.mamba_layers)))
    return cases


def _time_cuda(call: Callable[[], None], *, repeats: int, rounds: int = 5) -> list[float]:
    import cupy as cp

    samples: list[float] = []
    for _ in range(rounds):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        for _ in range(repeats):
            call()
        end.record()
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, end)) / repeats)
    return samples


def _bench_case(rt, pro: DenseERVF, case: Case, repeats: int) -> dict[str, Any]:
    import cupy as cp

    rng = np.random.default_rng(abs(hash(case.name)) & 0xFFFFFFFF)
    x = cp.asarray(rng.standard_normal(case.cols).astype(np.float32))
    ref = cp.zeros(case.rows, dtype=cp.float32)
    out = cp.zeros(case.rows, dtype=cp.float32)

    if case.kind == "bf16":
        ref_call = lambda: rt.k.mv_bf16(ref, case.W, x, case.rows, case.cols)
        pro_call = lambda: pro.mv_bf16(out, case.W, x, case.rows, case.cols)
        weight_bytes = case.rows * case.cols * 2
    elif case.kind == "f32":
        ref_call = lambda: rt.k.mv_f32(ref, case.W, x, case.rows, case.cols)
        pro_call = lambda: pro.mv_f32(out, case.W, x, case.rows, case.cols)
        weight_bytes = case.rows * case.cols * 4
    elif case.kind == "fp8":
        ref_call = lambda: rt.k.mv_fp8_tensor(ref, case.W, x, case.scale, case.rows, case.cols)
        pro_call = lambda: pro.mv_fp8_tensor(out, case.W, x, case.scale, case.rows, case.cols)
        weight_bytes = case.rows * case.cols
    else:
        raise ValueError(case.kind)

    ref_call(); pro_call(); cp.cuda.Device(0).synchronize()
    r = cp.asnumpy(ref)
    p = cp.asnumpy(out)
    bit_equal = bool(np.array_equal(r.view(np.uint32), p.view(np.uint32)))
    diff = np.abs(r.astype(np.float64) - p.astype(np.float64))

    # ABBA timing reduces a persistent first-arm bias.
    ref_samples = _time_cuda(ref_call, repeats=repeats)
    pro_samples = _time_cuda(pro_call, repeats=repeats)
    pro_samples += _time_cuda(pro_call, repeats=repeats)
    ref_samples += _time_cuda(ref_call, repeats=repeats)
    ref_ms = float(np.median(ref_samples))
    pro_ms = float(np.median(pro_samples))
    speedup = ref_ms / pro_ms if pro_ms > 0 else math.inf
    return {
        "name": case.name,
        "kind": case.kind,
        "rows": case.rows,
        "cols": case.cols,
        "calls_per_token": case.calls_per_token,
        "bit_equal": bit_equal,
        "mismatch_count": int(np.count_nonzero(r.view(np.uint32) != p.view(np.uint32))),
        "max_abs": float(diff.max(initial=0.0)),
        "reference_ms": ref_ms,
        "ervf_ms": pro_ms,
        "speedup": float(speedup),
        "reference_gb_s_weight_only": float(weight_bytes / (ref_ms * 1e6)),
        "ervf_gb_s_weight_only": float(weight_bytes / (pro_ms * 1e6)),
        "raw_reference_samples_ms": ref_samples,
        "raw_ervf_samples_ms": pro_samples,
    }


def _run_model_arm(rt, prompts: list[dict[str, Any]], n: int) -> tuple[dict[str, list[int]], list[float]]:
    import cupy as cp

    ids_by_prompt: dict[str, list[int]] = {}
    samples: list[float] = []
    for prompt in prompts:
        rt.reset()
        nxt = None
        for token in prompt["prompt_ids"]:
            nxt = int(rt.step(int(token)))
        cp.cuda.Device(0).synchronize()
        cur = int(nxt)
        ids = [cur]
        for _ in range(n - 1):
            t0 = time.perf_counter_ns()
            cur = int(rt.step(cur))
            cp.cuda.Device(0).synchronize()
            samples.append((time.perf_counter_ns() - t0) / 1e6)
            ids.append(cur)
        ids_by_prompt[prompt["prompt"]] = ids
    return ids_by_prompt, samples


def _anchor_prompts() -> list[dict[str, Any]]:
    anchor = load_json(TS200 / "V36_DETERMINISTIC_ANCHOR.json")
    return [{"prompt": p["prompt"], "prompt_ids": [int(x) for x in p["prompt_ids"]]} for p in anchor["prompts"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="full")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--micro-only", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        result = cpu_selftest()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1

    payload: dict[str, Any] = {
        "kind": "pro_g1_generalized_ervf",
        "status": "started",
        "started_utc": utc_now(),
        "mode": args.mode,
        "claim_boundary": (
            "Exact reduction-geometry experiment on resident BF16/FP8/FP32 GEMVs. "
            "Only an integrated A/B may be interpreted as token-level speed."
        ),
        "cpu_reduction_selftest": cpu_selftest(),
    }

    try:
        if not payload["cpu_reduction_selftest"]["passed"]:
            raise RuntimeError("CPU virtual reduction mapping selftest failed")
        require_gpu_free()
        runtime_file = REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py"
        kernels_file = REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "gpu_kernels.py"
        payload["environment"] = environment_snapshot((runtime_file, kernels_file, Path(__file__)))

        rt = _new_runtime()
        pro = DenseERVF()
        cases = _collect_cases(rt)
        if not cases:
            raise RuntimeError("No supported resident GEMV cases found in this checkpoint")
        repeats = 5 if args.mode == "smoke" else 20
        micro = [_bench_case(rt, pro, case, repeats) for case in cases]
        exact = all(x["bit_equal"] for x in micro)
        gmean = geometric_mean(x["speedup"] for x in micro)
        no_regression = all(x["speedup"] >= 0.95 for x in micro)
        weighted_ref = sum(x["reference_ms"] * x["calls_per_token"] for x in micro)
        weighted_pro = sum(x["ervf_ms"] * x["calls_per_token"] for x in micro)
        weighted_speedup = weighted_ref / weighted_pro if weighted_pro > 0 else None
        payload["microbench"] = {
            "cases": micro,
            "gates": {
                "all_bit_exact": exact,
                "geometric_mean_speedup_ge_1_25": bool(gmean is not None and gmean >= 1.25),
                "no_case_regression_gt_5pct": no_regression,
            },
            "geometric_mean_speedup": gmean,
            "weighted_reference_ms_model": weighted_ref,
            "weighted_ervf_ms_model": weighted_pro,
            "weighted_speedup_model": weighted_speedup,
        }

        can_integrate = exact and no_regression and gmean is not None and gmean >= 1.25
        payload["integration_opened"] = bool(can_integrate and not args.micro_only)
        if can_integrate and not args.micro_only:
            rt.enable_cache(72)
            rt.device_cache = True
            rt.deterministic_accum = True
            rt.load_routed_bank()
            prompts = _anchor_prompts()
            n = 32 if args.mode == "smoke" else 256

            original = {
                "mv_bf16": rt.k.mv_bf16,
                "mv_f32": rt.k.mv_f32,
                "mv_fp8_tensor": rt.k.mv_fp8_tensor,
            }

            # Base A.
            base_a_ids, base_a_ms = _run_model_arm(rt, prompts, n)

            # Candidate; rebuild cache so residency history cannot favor it.
            rt.enable_cache(72)
            rt.device_cache = True
            rt.k.mv_bf16 = pro.mv_bf16
            rt.k.mv_f32 = pro.mv_f32
            rt.k.mv_fp8_tensor = pro.mv_fp8_tensor
            pro_ids, pro_ms = _run_model_arm(rt, prompts, n)

            # Base B drift arm.
            rt.enable_cache(72)
            rt.device_cache = True
            rt.k.mv_bf16 = original["mv_bf16"]
            rt.k.mv_f32 = original["mv_f32"]
            rt.k.mv_fp8_tensor = original["mv_fp8_tensor"]
            base_b_ids, base_b_ms = _run_model_arm(rt, prompts, n)

            ba = percentiles(base_a_ms)
            pp = percentiles(pro_ms)
            bb = percentiles(base_b_ms)
            baseline_mid = (float(ba["p50"]) + float(bb["p50"])) / 2.0
            gain_ms = baseline_mid - float(pp["p50"])
            exact_prompts = {
                p["prompt"]: {
                    "base_a_vs_pro": base_a_ids[p["prompt"]] == pro_ids[p["prompt"]],
                    "base_b_vs_pro": base_b_ids[p["prompt"]] == pro_ids[p["prompt"]],
                    "base_a_vs_base_b": base_a_ids[p["prompt"]] == base_b_ids[p["prompt"]],
                    "first_divergence_a_pro": first_divergence(base_a_ids[p["prompt"]], pro_ids[p["prompt"]]),
                }
                for p in prompts
            }
            payload["integration"] = {
                "tokens_per_prompt": n,
                "arms": {
                    "BASE_A": {"timing_ms": ba, "raw_timing_ms": base_a_ms, "ids": base_a_ids},
                    "PRO_ERVF": {"timing_ms": pp, "raw_timing_ms": pro_ms, "ids": pro_ids},
                    "BASE_B": {"timing_ms": bb, "raw_timing_ms": base_b_ms, "ids": base_b_ids},
                },
                "parity": exact_prompts,
                "baseline_mid_p50_ms": baseline_mid,
                "gain_ms": gain_ms,
                "gain_fraction": gain_ms / baseline_mid if baseline_mid > 0 else None,
                "gates": {
                    "all_rollouts_identical": all(all(v for k, v in x.items() if k.startswith("base_")) for x in exact_prompts.values()),
                    "integrated_gain_ge_1_5_ms_or_5pct": bool(gain_ms >= 1.5 or gain_ms / baseline_mid >= 0.05),
                    "base_drift_lte_1_ms": abs(float(ba["p50"]) - float(bb["p50"])) <= 1.0,
                },
                "tok_s": {
                    "baseline_mid": 1000.0 / baseline_mid,
                    "pro": 1000.0 / float(pp["p50"]),
                },
            }

        micro_gates = payload["microbench"]["gates"]
        if not all(micro_gates.values()):
            payload["status"] = "micro_gate_failed"
        elif args.micro_only:
            payload["status"] = "micro_pass"
        elif "integration" not in payload:
            payload["status"] = "integration_not_opened"
        elif all(payload["integration"]["gates"].values()):
            payload["status"] = "pass"
        else:
            payload["status"] = "integration_gate_failed"
        payload["completed_utc"] = utc_now()

        del rt, pro
        gc.collect()
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    except Exception as exc:
        payload["status"] = "technical_failure"
        payload["completed_utc"] = utc_now()
        payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    write_json_atomic(OUT, payload)
    print(json.dumps({"status": payload["status"], "output": str(OUT)}, indent=2))
    return 0 if payload["status"] in {"pass", "micro_pass", "micro_gate_failed", "integration_gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
