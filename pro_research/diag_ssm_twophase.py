"""Two-phase ssm_step: the last bit-exact route at the occupancy problem.

Where this stands. `ssm_step` costs 1.095 ms/token in-graph at 88.1 GB/s = 34%
of the kernel rate -- the worst of anything measured. The layout hypothesis was
built, verified bit-exact, and **refuted**: transposing to [h][n][p] is 46%
slower on cold state, because in [h][p][n] each thread already streams a
contiguous 512 B row.

What remains is occupancy. The launch is `(H,) x min(256, P)` = **64 blocks of
64 threads = 128 warps on 26 SMs, about 5 warps per SM** -- nowhere near enough
to hide DRAM latency. The n loop is elementwise independent (`s[n]` depends only
on `s[n]` and `Bv[n]`) **except for the `acc` reduction**, and parallelising
that changes the summation order and breaks bit-exactness.

## The split

  phase 1   fully parallel over (h, p, n): 524,288 elements instead of 4,096.
            Computes `s = fmaf(decay, state[idx], dx * Bv[g*N+n])` and stores it
            back to state -- the identical expression, so the identical value.
  phase 2   one thread per (h, p): the sequential `acc` over n, reading the s
            values phase 1 just wrote, then `y = acc + Dh*xv`.

`acc` still walks n = 0..N-1 in order in a single thread, so y is bit-identical.
That is the whole point: the reduction order is what must not move, and it does
not.

## The cost this trades against

Phase 2 re-reads the state, so the pair moves 6.29 MB per layer where the fused
kernel moves 4.19 -- **50% more traffic**. The bet is that phase 2's read is
served from L2 (2.10 MB per layer, written microseconds earlier into a 32 MiB
cache) while phase 1 gets 128x the parallelism. If that bet is wrong the extra
traffic shows up directly, which is why it is measured rather than argued.

## Harness

23 distinct state buffers, matching what the real loop touches (48.3 MB, past
L2). A single reused buffer would measure L2 and has already, twice today,
produced a number with the wrong sign -- most recently selling this very
kernel's layout transpose as a 48% win when it is a 46% regression.

Gate: y bit-identical to the production fused kernel, states bit-identical,
outputs finite. Timing is read only if that holds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_gpu_free, utc_now, write_json_atomic

H, P, N = 64, 64, 128
HPG = 8
LAYERS = 23
ROUNDS = 200

SRC = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

// PRODUCTION: verbatim ssm_decode_step.
extern "C" __global__ void ssm_fused(
    float* __restrict__ state, const float* __restrict__ x,
    const float* __restrict__ Bv, const float* __restrict__ Cv,
    const float* __restrict__ dt, const float* __restrict__ Alog,
    const unsigned short* __restrict__ Dv, float* __restrict__ y,
    const int H, const int P, const int N, const int heads_per_group)
{
    const int h = blockIdx.x;
    if (h >= H) return;
    const int g = h / heads_per_group;
    const float dth = dt[h];
    const float decay = __expf(-__expf(Alog[h]) * dth);
    const float Dh = bf16_to_f32(Dv[h]);
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        float* srow = state + ((size_t)h * P + p) * N;
        const float xv = x[h * P + p];
        const float dx = dth * xv;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const float s = fmaf(decay, srow[n], dx * Bv[g * N + n]);
            srow[n] = s;
            acc = fmaf(s, Cv[g * N + n], acc);
        }
        y[h * P + p] = acc + Dh * xv;
    }
}

// PHASE 1: one thread per (h, p, n). 524,288 elements instead of 4,096.
// Same expression, so the same value lands in state.
extern "C" __global__ void ssm_phase1(
    float* __restrict__ state, const float* __restrict__ x,
    const float* __restrict__ Bv, const float* __restrict__ dt,
    const float* __restrict__ Alog,
    const int H, const int P, const int N, const int heads_per_group)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)H * P * N;
    if (i >= total) return;
    const int n = (int)(i % (size_t)N);
    const size_t hp = i / (size_t)N;
    const int p = (int)(hp % (size_t)P);
    const int h = (int)(hp / (size_t)P);
    const int g = h / heads_per_group;
    const float dth = dt[h];
    const float decay = __expf(-__expf(Alog[h]) * dth);
    const float dx = dth * x[h * P + p];
    state[i] = fmaf(decay, state[i], dx * Bv[g * N + n]);
}

// BLOCK-PER-HP: one block per (h, p) with N threads. The state update is fully
// parallel and coalesced within the block (128 consecutive floats), and the acc
// is still walked sequentially by thread 0 over shared memory -- same order,
// same fmaf, so bit-exact. Crucially the traffic is IDENTICAL to the fused
// kernel (one read + one write of the state); unlike the two-phase split there
// is no extra re-read. Parallelism goes from 4,096 threads to 524,288.
// Note thread 0 must do `fmaf(s, C, acc)` itself: precomputing s*C in parallel
// and summing afterwards would round twice instead of once.
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

// PHASE 2: one thread per (h, p); the sequential acc over n, in order.
extern "C" __global__ void ssm_phase2(
    const float* __restrict__ state, const float* __restrict__ x,
    const float* __restrict__ Cv, const unsigned short* __restrict__ Dv,
    float* __restrict__ y,
    const int H, const int P, const int N, const int heads_per_group)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * P) return;
    const int h = idx / P, p = idx - h * P;
    const int g = h / heads_per_group;
    const float* __restrict__ srow = state + ((size_t)h * P + p) * N;
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) acc = fmaf(srow[n], Cv[g * N + n], acc);
    y[idx] = acc + bf16_to_f32(Dv[h]) * x[idx];
}
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    mod = cp.RawModule(code=SRC, options=("-std=c++14",))
    k_fused = mod.get_function("ssm_fused")
    k_p1 = mod.get_function("ssm_phase1")
    k_p2 = mod.get_function("ssm_phase2")
    k_bhp = mod.get_function("ssm_block_hp")

    rng = np.random.default_rng(20260816)
    st = (rng.standard_normal((H, P, N)) * 0.1).astype(np.float32)
    x = cp.asarray((rng.standard_normal(H * P) * 0.5).astype(np.float32))
    Bv = cp.asarray(rng.standard_normal(H // HPG * N).astype(np.float32))
    Cv = cp.asarray(rng.standard_normal(H // HPG * N).astype(np.float32))
    dt = cp.asarray((np.abs(rng.standard_normal(H)) * 0.05).astype(np.float32))
    Alog = cp.asarray(rng.standard_normal(H).astype(np.float32))
    Dv = cp.asarray(((rng.standard_normal(H) * 0.1).astype(np.float32)
                     .view(np.uint32) >> 16).astype(np.uint16))
    y_f = cp.zeros(H * P, dtype=cp.float32)
    y_t = cp.zeros(H * P, dtype=cp.float32)

    s_f = [cp.asarray(st.reshape(-1)) for _ in range(LAYERS)]
    s_t = [cp.asarray(st.reshape(-1)) for _ in range(LAYERS)]
    s_b = [cp.asarray(st.reshape(-1)) for _ in range(LAYERS)]
    y_b = cp.zeros(H * P, dtype=cp.float32)

    dims = (np.int32(H), np.int32(P), np.int32(N), np.int32(HPG))
    total = H * P * N
    b1 = (total + 255) // 256
    b2 = (H * P + 255) // 256
    thr = min(256, P)

    k_fused((H,), (thr,), (s_f[0], x, Bv, Cv, dt, Alog, Dv, y_f, *dims))
    k_p1((b1,), (256,), (s_t[0], x, Bv, dt, Alog, *dims))
    k_p2((b2,), (256,), (s_t[0], x, Cv, Dv, y_t, *dims))
    k_bhp((H * P,), (N,), (s_b[0], x, Bv, Cv, dt, Alog, Dv, y_b, *dims),
          shared_mem=N * 4)
    cp.cuda.Device(0).synchronize()
    y_exact = bool(np.array_equal(cp.asnumpy(y_f).view(np.uint32),
                                  cp.asnumpy(y_t).view(np.uint32)))
    y_exact_b = bool(np.array_equal(cp.asnumpy(y_f).view(np.uint32),
                                    cp.asnumpy(y_b).view(np.uint32)))
    st_exact_b = bool(np.array_equal(cp.asnumpy(s_f[0]).view(np.uint32),
                                     cp.asnumpy(s_b[0]).view(np.uint32)))
    st_exact = bool(np.array_equal(cp.asnumpy(s_f[0]).view(np.uint32),
                                   cp.asnumpy(s_t[0]).view(np.uint32)))
    finite = bool(np.isfinite(cp.asnumpy(y_f)).all())

    def timed(fn):
        fn()
        cp.cuda.Device(0).synchronize()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        for _ in range(ROUNDS):
            fn()
        e1.record()
        e1.synchronize()
        return cp.cuda.get_elapsed_time(e0, e1) / ROUNDS

    def run_fused():
        for li in range(LAYERS):
            k_fused((H,), (thr,), (s_f[li], x, Bv, Cv, dt, Alog, Dv, y_f, *dims))

    def run_two():
        for li in range(LAYERS):
            k_p1((b1,), (256,), (s_t[li], x, Bv, dt, Alog, *dims))
            k_p2((b2,), (256,), (s_t[li], x, Cv, Dv, y_t, *dims))

    def run_block():
        for li in range(LAYERS):
            k_bhp((H * P,), (N,), (s_b[li], x, Bv, Cv, dt, Alog, Dv, y_b, *dims),
                  shared_mem=N * 4)

    ms_f = timed(run_fused)
    ms_t = timed(run_two)
    ms_b = timed(run_block)
    rw_fused = 2 * H * P * N * 4 * LAYERS
    rw_two = 3 * H * P * N * 4 * LAYERS

    payload = {
        "kind": "diag_ssm_twophase",
        "created_utc": utc_now(),
        "note": "ssm_step is occupancy-limited: 64 blocks x 64 threads = ~5 warps/SM. Phase 1 gets 128x the parallelism; phase 2 keeps the sequential acc so the summation order -- and therefore bit-exactness -- is untouched. The trade is 50% more traffic (phase 2 re-reads the state), betting that read is L2-served. 23 distinct state buffers so nothing is L2-resident by accident; a single reused buffer has produced a wrong-sign result twice today.",
        "geometry": {"H": H, "P": P, "N": N, "layers": LAYERS,
                     "fused_blocks": H, "fused_threads": thr,
                     "phase1_blocks": b1, "phase2_blocks": b2,
                     "fused_warps_total": H * thr // 32,
                     "phase1_warps_total": b1 * 256 // 32,
                     "rw_bytes_fused": rw_fused, "rw_bytes_twophase": rw_two},
        "gates": {"y_bit_exact_twophase": y_exact, "state_bit_exact_twophase": st_exact,
                  "y_bit_exact_block_hp": y_exact_b,
                  "state_bit_exact_block_hp": st_exact_b, "finite": finite},
        "fused": {"ms_per_token": ms_f, "gb_s": rw_fused / (ms_f * 1e-3) / 1e9},
        "twophase": {"ms_per_token": ms_t, "gb_s": rw_two / (ms_t * 1e-3) / 1e9},
        "block_per_hp": {"ms_per_token": ms_b, "gb_s": rw_fused / (ms_b * 1e-3) / 1e9,
                         "blocks": H * P, "threads": N,
                         "warps_total": H * P * N // 32},
        "speedup_twophase": ms_f / ms_t if ms_t else None,
        "speedup_block_per_hp": ms_f / ms_b if ms_b else None,
        "saving_ms_per_token_twophase": ms_f - ms_t,
        "saving_ms_per_token_block_per_hp": ms_f - ms_b,
        "in_graph_reference": {"ssm_step_ms": 1.095, "headroom_ms": 0.724},
        "status": ("measured" if (y_exact and st_exact and y_exact_b
                                  and st_exact_b and finite)
                   else "correctness_failed"),
    }
    write_json_atomic(REPO / "pro_research" / "diag_ssm_twophase.json", payload,
                      archive=False)
    print(json.dumps({k: payload[k] for k in
                      ("gates", "fused", "twophase", "block_per_hp",
                       "speedup_twophase", "speedup_block_per_hp",
                       "saving_ms_per_token_block_per_hp", "status")}, indent=2))
    return 0 if payload["status"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
