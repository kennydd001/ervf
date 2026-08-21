from __future__ import annotations

import numpy as np


_PREFIX = r"""
__device__ __forceinline__ float p68_bf16(const unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}

__device__ __forceinline__ float p68_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Grid (18, 4): 16 Q heads followed by 2 KV heads for every H4 row.
// Qwen3.5 Q projection rows are [query256, gate256] per head. The gate stays
// in q_gate4 and is consumed after attention.
extern "C" __global__ void ornith_qkv_prepare_h4(
    const float* __restrict__ q_gate4,
    const float* __restrict__ key4,
    const float* __restrict__ value4,
    const unsigned short* __restrict__ q_norm_weight,
    const unsigned short* __restrict__ k_norm_weight,
    const float* __restrict__ cos4,
    const float* __restrict__ sin4,
    float* __restrict__ prepared_q4,
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    const int base_context,
    const int max_context)
{
    const int owner = (int)blockIdx.x;
    const int token = (int)blockIdx.y;
    const int dim = (int)threadIdx.x;
    const int lane = dim & 31;
    const int warp = dim >> 5;
    __shared__ float normalized[256];
    __shared__ float partial[8];
    __shared__ float inverse_rms;

    if (owner < 16) {
        const size_t source =
            (size_t)token * 8192 + (size_t)owner * 512 + dim;
        const float raw = q_gate4[source];
        float square = p68_warp_sum(raw * raw);
        if (lane == 0) partial[warp] = square;
        __syncthreads();
        if (dim == 0) {
            float total = 0.0f;
            #pragma unroll
            for (int index = 0; index < 8; ++index) total += partial[index];
            inverse_rms = rsqrtf(total * (1.0f / 256.0f) + 1.0e-6f);
        }
        __syncthreads();
        normalized[dim] = raw * inverse_rms * (1.0f + p68_bf16(q_norm_weight[dim]));
        __syncthreads();
        float output = normalized[dim];
        if (dim < 32) {
            output = normalized[dim] * cos4[(size_t)token * 64 + dim]
                   - normalized[dim + 32] * sin4[(size_t)token * 64 + dim];
        } else if (dim < 64) {
            output = normalized[dim] * cos4[(size_t)token * 64 + dim]
                   + normalized[dim - 32] * sin4[(size_t)token * 64 + dim];
        }
        prepared_q4[((size_t)token * 16 + owner) * 256 + dim] = output;
        return;
    }

    const int kv_head = owner - 16;
    const size_t source = ((size_t)token * 2 + kv_head) * 256 + dim;
    const float raw = key4[source];
    float square = p68_warp_sum(raw * raw);
    if (lane == 0) partial[warp] = square;
    __syncthreads();
    if (dim == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int index = 0; index < 8; ++index) total += partial[index];
        inverse_rms = rsqrtf(total * (1.0f / 256.0f) + 1.0e-6f);
    }
    __syncthreads();
    normalized[dim] = raw * inverse_rms * (1.0f + p68_bf16(k_norm_weight[dim]));
    __syncthreads();
    float output = normalized[dim];
    if (dim < 32) {
        output = normalized[dim] * cos4[(size_t)token * 64 + dim]
               - normalized[dim + 32] * sin4[(size_t)token * 64 + dim];
    } else if (dim < 64) {
        output = normalized[dim] * cos4[(size_t)token * 64 + dim]
               + normalized[dim - 32] * sin4[(size_t)token * 64 + dim];
    }
    const size_t cache =
        ((size_t)kv_head * max_context + base_context + token) * 256 + dim;
    key_cache[cache] = output;
    value_cache[cache] = value4[source];
}
"""


_ATTENTION_TEMPLATE = r"""
extern "C" __global__ void __KERNEL_NAME__(
    const float* __restrict__ prepared_q4,
    const float* __restrict__ q_gate4,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    float* __restrict__ output4,
    const int base_context,
    const int max_context)
{
    const int linear = (int)blockIdx.x;
    const int subgroup = linear % __SUBGROUPS__;
    const int kv_head = (linear / __SUBGROUPS__) % 2;
    const int token = linear / (__SUBGROUPS__ * 2);
    const int dim = (int)threadIdx.x;
    const int lane = dim & 31;
    const int warp = dim >> 5;
    const int first_q_head = kv_head * 8 + subgroup * __GROUP__;
    const int context = base_context + token + 1;

    float query[__GROUP__];
    float accumulator[__GROUP__];
    #pragma unroll
    for (int local_head = 0; local_head < __GROUP__; ++local_head) {
        const int q_head = first_q_head + local_head;
        query[local_head] =
            prepared_q4[((size_t)token * 16 + q_head) * 256 + dim];
        accumulator[local_head] = 0.0f;
    }

    __shared__ float dot_partial[__GROUP__][8];
    __shared__ float running_max[__GROUP__];
    __shared__ float running_sum[__GROUP__];
    __shared__ float correction[__GROUP__];
    __shared__ float probability[__GROUP__];
    if (dim < __GROUP__) {
        running_max[dim] = -3.0e38f;
        running_sum[dim] = 0.0f;
    }
    __syncthreads();

    const float* key = key_cache + (size_t)kv_head * max_context * 256;
    const float* value = value_cache + (size_t)kv_head * max_context * 256;
    for (int position = 0; position < context; ++position) {
        const float key_value = key[(size_t)position * 256 + dim];
        #pragma unroll
        for (int local_head = 0; local_head < __GROUP__; ++local_head) {
            float dot = p68_warp_sum(query[local_head] * key_value);
            if (lane == 0) dot_partial[local_head][warp] = dot;
        }
        __syncthreads();
        if (dim < __GROUP__) {
            float dot = 0.0f;
            #pragma unroll
            for (int index = 0; index < 8; ++index) dot += dot_partial[dim][index];
            const float score = dot * 0.0625f;
            const float next_max = fmaxf(running_max[dim], score);
            const float corr = expf(running_max[dim] - next_max);
            const float prob = expf(score - next_max);
            running_sum[dim] = running_sum[dim] * corr + prob;
            running_max[dim] = next_max;
            correction[dim] = corr;
            probability[dim] = prob;
        }
        __syncthreads();
        const float value_item = value[(size_t)position * 256 + dim];
        #pragma unroll
        for (int local_head = 0; local_head < __GROUP__; ++local_head) {
            accumulator[local_head] =
                accumulator[local_head] * correction[local_head]
                + probability[local_head] * value_item;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int local_head = 0; local_head < __GROUP__; ++local_head) {
        const int q_head = first_q_head + local_head;
        const float gate =
            q_gate4[(size_t)token * 8192 + (size_t)q_head * 512 + 256 + dim];
        const float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
        output4[((size_t)token * 16 + q_head) * 256 + dim] =
            (accumulator[local_head] / running_sum[local_head]) * sigmoid_gate;
    }
}
"""


def _attention_source(name: str, group: int) -> str:
    if 8 % group:
        raise ValueError(group)
    return (
        _ATTENTION_TEMPLATE.replace("__KERNEL_NAME__", name)
        .replace("__SUBGROUPS__", str(8 // group))
        .replace("__GROUP__", str(group))
    )


CUDA_SOURCE = _PREFIX + "\n".join(
    (
        _attention_source("ornith_attention_g1_h4", 1),
        _attention_source("ornith_attention_g4_h4", 4),
        _attention_source("ornith_attention_g8_h4", 8),
    )
)


class OrnithFullAttentionH4Kernels:
    ARMS = {"g1": ("ornith_attention_g1_h4", 64),
            "g4": ("ornith_attention_g4_h4", 16),
            "g8": ("ornith_attention_g8_h4", 8)}

    def __init__(self):
        import cupy as cp

        names = ("ornith_qkv_prepare_h4",) + tuple(row[0] for row in self.ARMS.values())
        self.module = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=names,
        )
        self.functions = {name: self.module.get_function(name) for name in names}

    def prepare(
        self, q_gate4, key4, value4, q_norm, k_norm, cos4, sin4,
        prepared_q4, key_cache, value_cache, base_context: int, max_context: int,
    ) -> None:
        self.functions["ornith_qkv_prepare_h4"](
            (18, 4),
            (256,),
            (
                q_gate4, key4, value4, q_norm, k_norm, cos4, sin4,
                prepared_q4, key_cache, value_cache,
                np.int32(base_context), np.int32(max_context),
            ),
        )

    def attention(
        self, arm: str, prepared_q4, q_gate4, key_cache, value_cache,
        output4, base_context: int, max_context: int,
    ) -> None:
        name, blocks = self.ARMS[arm]
        self.functions[name](
            (blocks,),
            (256,),
            (
                prepared_q4, q_gate4, key_cache, value_cache, output4,
                np.int32(base_context), np.int32(max_context),
            ),
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
