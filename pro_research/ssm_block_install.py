"""Install the block-per-(h,p) ssm_step on a live runtime.

Measured in isolation (diag_ssm_twophase.json, 23 distinct cold state buffers):
bit-exact on both `y` and the state, **x1.031** against the production fused
kernel -- 0.024 ms/token. Small alone, but V18 just showed that two bit-exact
mechanisms each under their own gate combined to roughly twice their sum, and
this one lives in a completely different part of the model (Mamba) from V18's
down_proj work, so there is no shared resource for them to fight over.

## The kernel

Production `ssm_decode_step` runs `(H,) x min(256, P)` = 64 blocks of 64
threads -- 128 warps on 26 SMs. This runs one block per (h, p) with N threads:
4,096 blocks x 128 threads = 16,384 warps, with the state update fully parallel
and coalesced inside the block, and thread 0 then walking the `acc` reduction
sequentially over shared memory. Traffic is identical to production (one read,
one write of the state); the two-phase alternative added 50% more traffic and
came out slower.

Thread 0 must do `fmaf(s, C, acc)` itself -- precomputing `s*C` in parallel and
summing afterwards rounds twice instead of once, which would break bit-exactness.

## Compile flags

`ssm_decode_step` lives in `gpu_kernels.py`, which builds with
**`--use_fast_math`**, and the kernel contains two `__expf` calls, so it is
highly sensitive to that flag. Building this replacement without it reproduces
PV2-10's failure exactly -- a token divergence around step 124 from a few ulps
in `decay`. This is measurement rule 4 in agents/TODO.md, and it is the reason
this file passes the flag explicitly rather than inheriting a default.
"""

from __future__ import annotations

import numpy as np

CUDA_SOURCE = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

extern "C" __global__ void ssm_block_hp(
    float* __restrict__ state, const float* __restrict__ x,
    const float* __restrict__ Bv, const float* __restrict__ Cv,
    const float* __restrict__ dt, const float* __restrict__ Alog,
    const unsigned short* __restrict__ Dv, float* __restrict__ y,
    const int H, const int P, const int N, const int heads_per_group)
{
    extern __shared__ float sh[];
    const int idx = blockIdx.x;
    const int h = idx / P, p = idx - h * P;
    const int g = h / heads_per_group;
    const float dth = dt[h];
    const float decay = __expf(-__expf(Alog[h]) * dth);
    const float xv = x[idx];
    const float dx = dth * xv;
    float* __restrict__ srow = state + (size_t)idx * N;
    const int n = threadIdx.x;
    if (n < N) {
        const float s = fmaf(decay, srow[n], dx * Bv[g * N + n]);
        srow[n] = s;
        sh[n] = s;
    }
    __syncthreads();
    if (n == 0) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) acc = fmaf(sh[j], Cv[g * N + j], acc);
        y[idx] = acc + bf16_to_f32(Dv[h]) * xv;
    }
}
"""


def install_ssm_block(rt):
    """Swap rt.k.ssm_step for the block-per-(h,p) launch. Returns restore()."""
    import cupy as cp

    # gpu_kernels.py builds with --use_fast_math and ssm_decode_step contains two
    # __expf calls, so omitting the flag here changes `decay` by a few ulps and
    # reproduces PV2-10's token-124 divergence. See measurement rule 4.
    mod = cp.RawModule(code=CUDA_SOURCE, options=("-std=c++14", "--use_fast_math"))
    k_block = mod.get_function("ssm_block_hp")
    orig = rt.k.ssm_step

    def ssm_step(y, state, x, Bv, Cv, dt, Alog, Dv, H, P, N, hpg):
        k_block((H * P,), (N,),
                (state, x, Bv, Cv, dt, Alog, Dv, y,
                 np.int32(H), np.int32(P), np.int32(N), np.int32(hpg)),
                shared_mem=N * 4)

    rt.k.ssm_step = ssm_step

    def restore():
        rt.k.ssm_step = orig

    return restore
