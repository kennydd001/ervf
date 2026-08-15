#!/usr/bin/env python3
"""Frozen CAP0 uint32 sentinel sources.  No imports and no side effects."""

INTEL_BUILD_OPTIONS = "-cl-std=CL2.0 -cl-fp32-correctly-rounded-divide-sqrt"
NVIDIA_NVRTC_OPTIONS = ("--std=c++14", "--fmad=false")

INTEL_SOURCE = r"""
__kernel void cap0_intel_bijection(__global uint *words, const uint count) {
    const uint i = get_global_id(0);
    if (i < count) {
        const uint v = words[i] ^ (uint)0xA5A5A5A5u;
        words[i] = rotate(v, (uint)7u) + (uint)0x3C6EF372u;
    }
}
""".strip() + "\n"

NVIDIA_SOURCE = r"""
extern "C" __global__ void cap0_nvidia_bijection(unsigned int *words,
                                                   const unsigned int count) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const unsigned int v = words[i] + 0x9E3779B9u;
        const unsigned int r = (v >> 11u) | (v << 21u);
        words[i] = r ^ 0xC3C3C3C3u;
    }
}
""".strip() + "\n"




