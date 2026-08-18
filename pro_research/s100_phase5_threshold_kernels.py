"""Fused relative-threshold panel scan for phase 5."""
from __future__ import annotations
import numpy as np

CUDA_SOURCE = r"""
extern "C" __global__ void panel_scan_threshold_batched(
    const float* __restrict__ act,
    const int inter,
    const float alpha,
    unsigned int* __restrict__ panel_masks,
    int* __restrict__ panel_list,
    int* __restrict__ panel_count,
    int* __restrict__ nz_list,
    int* __restrict__ nz_count,
    float* __restrict__ max_act)
{
    const int s = blockIdx.x;
    const int npanel = inter >> 4;
    const float* __restrict__ act_s = act + (size_t)s * inter;
    unsigned int* __restrict__ masks_s = panel_masks + (size_t)s * npanel;
    int* __restrict__ list_s = panel_list + (size_t)s * npanel;
    int* __restrict__ nz_s = nz_list + (size_t)s * inter;
    __shared__ float red[256];

    float lm = 0.0f;
    for (int j = threadIdx.x; j < inter; j += blockDim.x)
        lm = fmaxf(lm, act_s[j]);
    red[threadIdx.x] = lm;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        __syncthreads();
    }
    const float cut = alpha > 0.0f ? alpha * red[0] : 0.0f;
    if (threadIdx.x == 0) max_act[s] = red[0];

    if (threadIdx.x < npanel) masks_s[threadIdx.x] = 0u;
    if (threadIdx.x == 0) { panel_count[s] = 0; nz_count[s] = 0; }
    __syncthreads();

    for (int j = threadIdx.x; j < inter; j += blockDim.x) {
        const float v = act_s[j];
        const bool keep = (v != 0.0f) && (alpha <= 0.0f || v >= cut);
        if (keep) atomicOr(&masks_s[j >> 4], 1u << (j & 15));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int n = 0, m = 0;
        for (int p = 0; p < npanel; p++) {
            const unsigned int mk = masks_s[p];
            if (mk) {
                list_s[n++] = p;
                for (int c = 0; c < 16; c++)
                    if (mk & (1u << c)) nz_s[m++] = (p << 4) + c;
            }
        }
        panel_count[s] = n;
        nz_count[s] = m;
    }
}
"""

class Phase5ThresholdKernels:
    def __init__(self):
        import cupy as cp
        self.cp=cp
        self.mod=cp.RawModule(code=CUDA_SOURCE,options=("-std=c++14",))
        self.panel_scan_threshold_batched=self.mod.get_function("panel_scan_threshold_batched")
