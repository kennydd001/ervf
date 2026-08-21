from __future__ import annotations

import numpy as np


CUDA_SOURCE = r"""
__device__ __forceinline__ float p67_bf16(const unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}

__device__ __forceinline__ float p67_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float p67_silu(float value) {
    return value / (1.0f + expf(-value));
}

__device__ __forceinline__ float p67_softplus(float value) {
    return value > 20.0f ? value : log1pf(expf(value));
}

// Fuse both 32x2048 BF16 gate projections across the four speculative rows.
// One CTA owns one value head, so every A/B weight is reused four times.
extern "C" __global__ void ornith_ab_gate_h4(
    const unsigned short* __restrict__ weight_a,
    const unsigned short* __restrict__ weight_b,
    const float* __restrict__ hidden4,
    const unsigned short* __restrict__ a_log,
    const unsigned short* __restrict__ dt_bias,
    float* __restrict__ beta4,
    float* __restrict__ g4,
    const int cols)
{
    const int head = (int)blockIdx.x;
    if (head >= 32) return;
    const unsigned short* wa = weight_a + (size_t)head * cols;
    const unsigned short* wb = weight_b + (size_t)head * cols;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += (int)blockDim.x) {
        const float av = p67_bf16(wa[col]);
        const float bv = p67_bf16(wb[col]);
        a0 = fmaf(av, hidden4[col], a0);
        a1 = fmaf(av, hidden4[(size_t)cols + col], a1);
        a2 = fmaf(av, hidden4[(size_t)2 * cols + col], a2);
        a3 = fmaf(av, hidden4[(size_t)3 * cols + col], a3);
        b0 = fmaf(bv, hidden4[col], b0);
        b1 = fmaf(bv, hidden4[(size_t)cols + col], b1);
        b2 = fmaf(bv, hidden4[(size_t)2 * cols + col], b2);
        b3 = fmaf(bv, hidden4[(size_t)3 * cols + col], b3);
    }
    a0 = p67_warp_sum(a0); a1 = p67_warp_sum(a1);
    a2 = p67_warp_sum(a2); a3 = p67_warp_sum(a3);
    b0 = p67_warp_sum(b0); b1 = p67_warp_sum(b1);
    b2 = p67_warp_sum(b2); b3 = p67_warp_sum(b3);

    __shared__ float partial[8][8];
    __shared__ float total[8];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    if (lane == 0) {
        partial[0][warp] = a0; partial[1][warp] = a1;
        partial[2][warp] = a2; partial[3][warp] = a3;
        partial[4][warp] = b0; partial[5][warp] = b1;
        partial[6][warp] = b2; partial[7][warp] = b3;
    }
    __syncthreads();
    if ((int)threadIdx.x < 8) {
        float sum = 0.0f;
        #pragma unroll
        for (int index = 0; index < 8; ++index) sum += partial[threadIdx.x][index];
        total[threadIdx.x] = sum;
    }
    __syncthreads();
    if ((int)threadIdx.x < 4) {
        const int token = (int)threadIdx.x;
        const float beta_input = total[4 + token];
        const float a_input = total[token] + p67_bf16(dt_bias[head]);
        beta4[(size_t)token * 32 + head] = 1.0f / (1.0f + expf(-beta_input));
        g4[(size_t)token * 32 + head] =
            -expf(p67_bf16(a_log[head])) * p67_softplus(a_input);
    }
}

// Kernel width equals H4. After one launch the rolling state is exactly the
// current four inputs, while each output sees the correct causal window.
extern "C" __global__ void ornith_conv_silu_h4(
    const float* __restrict__ mixed4,
    const unsigned short* __restrict__ weight,
    float* __restrict__ conv_state,
    float* __restrict__ convolved4,
    const int channels)
{
    const int channel = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (channel >= channels) return;
    float* state = conv_state + (size_t)channel * 4;
    const float s0 = state[0], s1 = state[1], s2 = state[2], s3 = state[3];
    const float x0 = mixed4[channel];
    const float x1 = mixed4[(size_t)channels + channel];
    const float x2 = mixed4[(size_t)2 * channels + channel];
    const float x3 = mixed4[(size_t)3 * channels + channel];
    const unsigned short* w = weight + (size_t)channel * 4;
    const float w0 = p67_bf16(w[0]), w1 = p67_bf16(w[1]);
    const float w2 = p67_bf16(w[2]), w3 = p67_bf16(w[3]);

    float y0 = fmaf(w0, s1, fmaf(w1, s2, fmaf(w2, s3, w3 * x0)));
    float y1 = fmaf(w0, s2, fmaf(w1, s3, fmaf(w2, x0, w3 * x1)));
    float y2 = fmaf(w0, s3, fmaf(w1, x0, fmaf(w2, x1, w3 * x2)));
    float y3 = fmaf(w0, x0, fmaf(w1, x1, fmaf(w2, x2, w3 * x3)));
    convolved4[channel] = p67_silu(y0);
    convolved4[(size_t)channels + channel] = p67_silu(y1);
    convolved4[(size_t)2 * channels + channel] = p67_silu(y2);
    convolved4[(size_t)3 * channels + channel] = p67_silu(y3);
    state[0] = x0; state[1] = x1; state[2] = x2; state[3] = x3;
}

// One 128-thread CTA owns a value head and its 128x128 FP32 state. Four
// recurrent updates are executed in order. The final RMSNorm and SiLU(z) gate
// are fused while the core output is still in registers.
extern "C" __global__ void ornith_delta_norm_h4(
    const float* __restrict__ convolved4,
    const float* __restrict__ z4,
    const float* __restrict__ beta4,
    const float* __restrict__ g4,
    const unsigned short* __restrict__ norm_weight,
    float* __restrict__ recurrent_state,
    float* __restrict__ output4)
{
    const int head = (int)blockIdx.x;
    const int dim = (int)threadIdx.x;
    if (head >= 32 || dim >= 128) return;
    const int key_head = head >> 1;
    const int lane = dim & 31;
    const int warp = dim >> 5;
    const size_t state_base = (size_t)head * 128 * 128;

    __shared__ float q_shared[128];
    __shared__ float k_shared[128];
    __shared__ float q_partial[4];
    __shared__ float k_partial[4];
    __shared__ float norm_partial[4];
    __shared__ float inv_q;
    __shared__ float inv_k;
    __shared__ float inv_rms;

    #pragma unroll
    for (int token = 0; token < 4; ++token) {
        const size_t row = (size_t)token * 8192;
        const float q_raw = convolved4[row + (size_t)key_head * 128 + dim];
        const float k_raw = convolved4[row + 2048 + (size_t)key_head * 128 + dim];
        float q_sum = p67_warp_sum(q_raw * q_raw);
        float k_sum = p67_warp_sum(k_raw * k_raw);
        if (lane == 0) {
            q_partial[warp] = q_sum;
            k_partial[warp] = k_sum;
        }
        __syncthreads();
        if (dim == 0) {
            const float qs = q_partial[0] + q_partial[1] + q_partial[2] + q_partial[3];
            const float ks = k_partial[0] + k_partial[1] + k_partial[2] + k_partial[3];
            inv_q = rsqrtf(qs + 1.0e-6f) * 0.08838834764831845f;
            inv_k = rsqrtf(ks + 1.0e-6f);
        }
        __syncthreads();
        q_shared[dim] = q_raw * inv_q;
        k_shared[dim] = k_raw * inv_k;
        __syncthreads();

        const float decay = expf(g4[(size_t)token * 32 + head]);
        float kv_memory = 0.0f;
        #pragma unroll 4
        for (int key_dim = 0; key_dim < 128; ++key_dim) {
            const size_t index = state_base + (size_t)key_dim * 128 + dim;
            const float state_value = recurrent_state[index] * decay;
            recurrent_state[index] = state_value;
            kv_memory = fmaf(state_value, k_shared[key_dim], kv_memory);
        }
        const float value = convolved4[row + 4096 + (size_t)head * 128 + dim];
        const float delta = (value - kv_memory) * beta4[(size_t)token * 32 + head];
        float core_output = 0.0f;
        #pragma unroll 4
        for (int key_dim = 0; key_dim < 128; ++key_dim) {
            const size_t index = state_base + (size_t)key_dim * 128 + dim;
            const float state_value = fmaf(k_shared[key_dim], delta, recurrent_state[index]);
            recurrent_state[index] = state_value;
            core_output = fmaf(state_value, q_shared[key_dim], core_output);
        }

        float square_sum = p67_warp_sum(core_output * core_output);
        if (lane == 0) norm_partial[warp] = square_sum;
        __syncthreads();
        if (dim == 0) {
            const float total =
                norm_partial[0] + norm_partial[1] + norm_partial[2] + norm_partial[3];
            inv_rms = rsqrtf(total * (1.0f / 128.0f) + 1.0e-6f);
        }
        __syncthreads();
        const float gate = z4[(size_t)token * 4096 + (size_t)head * 128 + dim];
        output4[(size_t)token * 4096 + (size_t)head * 128 + dim] =
            core_output * inv_rms * p67_bf16(norm_weight[dim]) * p67_silu(gate);
        __syncthreads();
    }
}
"""


class OrnithLinearH4Kernels:
    def __init__(self):
        import cupy as cp

        names = (
            "ornith_ab_gate_h4",
            "ornith_conv_silu_h4",
            "ornith_delta_norm_h4",
        )
        self.module = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=names,
        )
        self.functions = {name: self.module.get_function(name) for name in names}

    def gates(self, weight_a, weight_b, hidden4, a_log, dt_bias, beta4, g4) -> None:
        if tuple(hidden4.shape) != (4, 2048):
            raise ValueError(f"hidden4 shape mismatch: {hidden4.shape}")
        self.functions["ornith_ab_gate_h4"](
            (32,),
            (256,),
            (
                weight_a,
                weight_b,
                hidden4,
                a_log,
                dt_bias,
                beta4,
                g4,
                np.int32(2048),
            ),
        )

    def convolution(self, mixed4, weight, conv_state, convolved4) -> None:
        if tuple(mixed4.shape) != (4, 8192):
            raise ValueError(f"mixed4 shape mismatch: {mixed4.shape}")
        self.functions["ornith_conv_silu_h4"](
            (32,),
            (256,),
            (mixed4, weight, conv_state, convolved4, np.int32(8192)),
        )

    def delta_norm(self, convolved4, z4, beta4, g4, norm_weight, state, output4) -> None:
        if tuple(output4.shape) != (4, 4096):
            raise ValueError(f"output4 shape mismatch: {output4.shape}")
        self.functions["ornith_delta_norm_h4"](
            (32,),
            (128,),
            (convolved4, z4, beta4, g4, norm_weight, state, output4),
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
