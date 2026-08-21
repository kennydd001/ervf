from __future__ import annotations

import numpy as np


CUDA_SOURCE = r"""
__device__ __forceinline__ float p69_bf16(const unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}

__device__ __forceinline__ float p69_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void ornith_rmsnorm_h4(
    const float* __restrict__ input4,
    const unsigned short* __restrict__ weight,
    float* __restrict__ output4)
{
    const int token = (int)blockIdx.x;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const float* input = input4 + (size_t)token * 2048;
    float* output = output4 + (size_t)token * 2048;
    float square_sum = 0.0f;
    for (int dim = (int)threadIdx.x; dim < 2048; dim += (int)blockDim.x) {
        const float value = input[dim];
        square_sum = fmaf(value, value, square_sum);
    }
    square_sum = p69_warp_sum(square_sum);
    __shared__ float partial[8];
    __shared__ float inverse_rms;
    if (lane == 0) partial[warp] = square_sum;
    __syncthreads();
    if ((int)threadIdx.x == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int index = 0; index < 8; ++index) total += partial[index];
        inverse_rms = rsqrtf(total * (1.0f / 2048.0f) + 1.0e-6f);
    }
    __syncthreads();
    for (int dim = (int)threadIdx.x; dim < 2048; dim += (int)blockDim.x) {
        output[dim] = input[dim] * inverse_rms * (1.0f + p69_bf16(weight[dim]));
    }
}

extern "C" __global__ void ornith_add_rmsnorm_h4(
    float* __restrict__ residual4,
    const float* __restrict__ branch4,
    const unsigned short* __restrict__ weight,
    float* __restrict__ output4)
{
    const int token = (int)blockIdx.x;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    float* residual = residual4 + (size_t)token * 2048;
    const float* branch = branch4 + (size_t)token * 2048;
    float* output = output4 + (size_t)token * 2048;
    float square_sum = 0.0f;
    for (int dim = (int)threadIdx.x; dim < 2048; dim += (int)blockDim.x) {
        const float value = residual[dim] + branch[dim];
        residual[dim] = value;
        square_sum = fmaf(value, value, square_sum);
    }
    square_sum = p69_warp_sum(square_sum);
    __shared__ float partial[8];
    __shared__ float inverse_rms;
    if (lane == 0) partial[warp] = square_sum;
    __syncthreads();
    if ((int)threadIdx.x == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int index = 0; index < 8; ++index) total += partial[index];
        inverse_rms = rsqrtf(total * (1.0f / 2048.0f) + 1.0e-6f);
    }
    __syncthreads();
    for (int dim = (int)threadIdx.x; dim < 2048; dim += (int)blockDim.x) {
        output[dim] = residual[dim] * inverse_rms * (1.0f + p69_bf16(weight[dim]));
    }
}

// 256 router rows plus one shared-gate row. Every BF16 weight load updates
// all four speculative positions without staging 32 KiB of activations.
extern "C" __global__ void ornith_router_shared_h4(
    const unsigned short* __restrict__ router_weight,
    const unsigned short* __restrict__ shared_weight,
    const float* __restrict__ input4,
    float* __restrict__ router_logits4,
    float* __restrict__ shared_logits4)
{
    const int row = (int)blockIdx.x;
    const unsigned short* weight = row < 256
        ? router_weight + (size_t)row * 2048
        : shared_weight;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int dim = (int)threadIdx.x; dim < 2048; dim += (int)blockDim.x) {
        const float w = p69_bf16(weight[dim]);
        a0 = fmaf(w, input4[dim], a0);
        a1 = fmaf(w, input4[2048 + dim], a1);
        a2 = fmaf(w, input4[4096 + dim], a2);
        a3 = fmaf(w, input4[6144 + dim], a3);
    }
    a0 = p69_warp_sum(a0); a1 = p69_warp_sum(a1);
    a2 = p69_warp_sum(a2); a3 = p69_warp_sum(a3);
    __shared__ float partial[4][8];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    if (lane == 0) {
        partial[0][warp] = a0; partial[1][warp] = a1;
        partial[2][warp] = a2; partial[3][warp] = a3;
    }
    __syncthreads();
    if ((int)threadIdx.x < 4) {
        float total = 0.0f;
        #pragma unroll
        for (int index = 0; index < 8; ++index) total += partial[threadIdx.x][index];
        if (row < 256) router_logits4[(size_t)threadIdx.x * 256 + row] = total;
        else shared_logits4[threadIdx.x] = total;
    }
}

// Four independent serial selectors. At n=256,k=8, avoiding a full-block
// bitonic network saves 36 barriers. Low expert ID wins an exact logit tie.
extern "C" __global__ void ornith_top8_cache_h4(
    const float* __restrict__ router_logits4,
    const int* __restrict__ slot_of,
    int* __restrict__ ids4,
    float* __restrict__ weights4,
    int* __restrict__ slots4,
    int* __restrict__ need4)
{
    if ((int)threadIdx.x != 0) return;
    const int token = (int)blockIdx.x;
    const float* logits = router_logits4 + (size_t)token * 256;
    int chosen[8];
    float selected[8];
    #pragma unroll
    for (int pick = 0; pick < 8; ++pick) {
        int best_id = -1;
        float best_value = -3.0e38f;
        for (int expert = 0; expert < 256; ++expert) {
            bool used = false;
            #pragma unroll
            for (int prior = 0; prior < 8; ++prior) {
                if (prior < pick && chosen[prior] == expert) used = true;
            }
            const float value = logits[expert];
            if (!used && value > best_value) {
                best_value = value;
                best_id = expert;
            }
        }
        chosen[pick] = best_id;
        selected[pick] = best_value;
    }
    float exponential[8];
    float denominator = 0.0f;
    #pragma unroll
    for (int pick = 0; pick < 8; ++pick) {
        exponential[pick] = expf(selected[pick] - selected[0]);
        denominator += exponential[pick];
    }
    #pragma unroll
    for (int pick = 0; pick < 8; ++pick) {
        const int route = token * 8 + pick;
        const int expert = chosen[pick];
        const int slot = slot_of[expert];
        ids4[route] = expert;
        weights4[route] = exponential[pick] / denominator;
        slots4[route] = slot;
        need4[route] = slot < 0 ? 1 : 0;
    }
}

extern "C" __global__ void ornith_moe_combine_rmsnorm_h4(
    float* __restrict__ residual4,
    const float* __restrict__ expert_outputs32,
    const float* __restrict__ route_weights32,
    const float* __restrict__ shared_output4,
    const float* __restrict__ shared_logits4,
    const unsigned short* __restrict__ next_norm_weight,
    float* __restrict__ next_normed4)
{
    const int token = (int)blockIdx.x;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    float* residual = residual4 + (size_t)token * 2048;
    float* next_normed = next_normed4 + (size_t)token * 2048;
    const float shared_scale = 1.0f / (1.0f + expf(-shared_logits4[token]));
    float square_sum = 0.0f;
    for (int dim = (int)threadIdx.x; dim < 2048; dim += (int)blockDim.x) {
        float routed = 0.0f;
        #pragma unroll
        for (int route = 0; route < 8; ++route) {
            routed = fmaf(
                expert_outputs32[((size_t)token * 8 + route) * 2048 + dim],
                route_weights32[token * 8 + route], routed
            );
        }
        const float combined = residual[dim] + routed
            + shared_output4[(size_t)token * 2048 + dim] * shared_scale;
        residual[dim] = combined;
        square_sum = fmaf(combined, combined, square_sum);
    }
    square_sum = p69_warp_sum(square_sum);
    __shared__ float partial[8];
    __shared__ float inverse_rms;
    if (lane == 0) partial[warp] = square_sum;
    __syncthreads();
    if ((int)threadIdx.x == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int index = 0; index < 8; ++index) total += partial[index];
        inverse_rms = rsqrtf(total * (1.0f / 2048.0f) + 1.0e-6f);
    }
    __syncthreads();
    for (int dim = (int)threadIdx.x; dim < 2048; dim += (int)blockDim.x) {
        next_normed[dim] = residual[dim] * inverse_rms
            * (1.0f + p69_bf16(next_norm_weight[dim]));
    }
}
"""


class OrnithSupportH4Kernels:
    NAMES = (
        "ornith_rmsnorm_h4",
        "ornith_add_rmsnorm_h4",
        "ornith_router_shared_h4",
        "ornith_top8_cache_h4",
        "ornith_moe_combine_rmsnorm_h4",
    )

    def __init__(self):
        import cupy as cp

        self.module = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=self.NAMES,
        )
        self.functions = {name: self.module.get_function(name) for name in self.NAMES}

    def norm(self, input4, weight, output4) -> None:
        self.functions["ornith_rmsnorm_h4"]((4,), (256,), (input4, weight, output4))

    def add_norm(self, residual4, branch4, weight, output4) -> None:
        self.functions["ornith_add_rmsnorm_h4"](
            (4,), (256,), (residual4, branch4, weight, output4)
        )

    def router_shared(self, router_weight, shared_weight, input4, logits4, shared4) -> None:
        self.functions["ornith_router_shared_h4"](
            (257,), (256,), (router_weight, shared_weight, input4, logits4, shared4)
        )

    def top8_cache(self, logits4, slot_of, ids4, weights4, slots4, need4) -> None:
        self.functions["ornith_top8_cache_h4"](
            (4,), (32,), (logits4, slot_of, ids4, weights4, slots4, need4)
        )

    def combine_norm(
        self, residual4, expert_outputs32, route_weights32, shared_output4,
        shared_logits4, next_norm_weight, next_normed4,
    ) -> None:
        self.functions["ornith_moe_combine_rmsnorm_h4"](
            (4,), (256,), (
                residual4, expert_outputs32, route_weights32, shared_output4,
                shared_logits4, next_norm_weight, next_normed4,
            )
        )

    def resource_audit(self) -> dict[str, dict[str, int | None]]:
        result = {}
        for name, function in self.functions.items():
            function.compile()
            attributes = getattr(function, "attributes", {}) or {}
            result[name] = {
                "num_regs": attributes.get("num_regs"),
                "shared_size_bytes_static": attributes.get("shared_size_bytes"),
                "local_size_bytes": attributes.get("local_size_bytes"),
                "max_threads_per_block": attributes.get("max_threads_per_block"),
            }
        return result
