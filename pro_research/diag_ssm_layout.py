"""ssm_step runs at 34% because its state layout fights its thread mapping.

Measured in-graph: `ssm_step` costs **1.095 ms/token at 88.1 GB/s = 34% of the
kernel rate**, the worst efficiency of anything measured today (attention 45.5%,
down_masked 60%, Mamba GEMVs 80-86%, shared_expert 90%), on 96.5 MB/token of
pure VRAM read-modify-write.

The kernel (gpu_kernels.py:153) explains it:

    const int h = blockIdx.x;                                  // H = 64 blocks
    for (int p = threadIdx.x; p < P; p += blockDim.x) {         // P = 64 threads
        float* srow = state + ((size_t)h * P + p) * N;          // N = 128 floats
        for (int n = 0; n < N; ++n) {
            const float s = fmaf(decay, srow[n], dx * Bv[g*N+n]);
            srow[n] = s;
            acc = fmaf(s, Cv[g*N+n], acc);
        }
    }

Thread `p` owns a contiguous row of N floats, so at inner step `n` threads p and
p+1 touch addresses **N*4 = 512 B apart**. A warp's 32 threads therefore span
16 KB and request 4 useful bytes from each of 32 separate sectors -- around
**12.5% of every transaction is used**. 88.1 / 0.125 would be well past the
device, so coalescing is not the only factor, but it is the obvious first one.

## The fix, and why it can be bit-exact

Transpose the state from `[h][p][n]` to `[h][n][p]`. Then thread `p` reads
`state[h*P*N + n*P + p]` and the 64 threads of a step touch **64 consecutive
floats** -- fully coalesced. Every arithmetic operation, and crucially the
sequential `acc` accumulation over n = 0..N-1, is untouched: same order, same
values, same rounding. Only the address changes.

That matters because the obvious alternative -- parallelising the `acc`
reduction across threads -- would change the summation order and break
bit-exactness. The transpose does not.

It is also self-contained: `ssm[i]` is written and read only by this kernel and
zeroed at reset, so the layout is private to it.

## Arms

  layout_pn   the production kernel and layout, [h][p][n]
  layout_np   identical arithmetic, state transposed to [h][n][p]

Gate: the y outputs must be **bit-identical** between the two, starting from
states that are transposes of each other. Timing is read only if that holds.
Real shapes: H=64 heads, P=64 head_dim, N=128 state, 23 layers per token.
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

// PRODUCTION: verbatim ssm_decode_step, state laid out [h][p][n].
extern "C" __global__ void ssm_pn(
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

// CANDIDATE: identical arithmetic and identical accumulation order; the state
// is transposed to [h][n][p] so the 64 threads of a step read 64 consecutive
// floats instead of 64 addresses 512 B apart.
extern "C" __global__ void ssm_np(
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
    float* __restrict__ sh = state + (size_t)h * P * N;
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        const float xv = x[h * P + p];
        const float dx = dth * xv;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const size_t idx = (size_t)n * P + p;          // <- only change
            const float s = fmaf(decay, sh[idx], dx * Bv[g * N + n]);
            sh[idx] = s;
            acc = fmaf(s, Cv[g * N + n], acc);
        }
        y[h * P + p] = acc + Dh * xv;
    }
}
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    mod = cp.RawModule(code=SRC, options=("-std=c++14",))
    k_pn = mod.get_function("ssm_pn")
    k_np = mod.get_function("ssm_np")

    rng = np.random.default_rng(20260816)
    st = (rng.standard_normal((H, P, N)) * 0.1).astype(np.float32)
    x = cp.asarray((rng.standard_normal(H * P) * 0.5).astype(np.float32))
    Bv = cp.asarray(rng.standard_normal(H // HPG * N).astype(np.float32))
    Cv = cp.asarray(rng.standard_normal(H // HPG * N).astype(np.float32))
    dt = cp.asarray((np.abs(rng.standard_normal(H)) * 0.05).astype(np.float32))
    Alog = cp.asarray(rng.standard_normal(H).astype(np.float32))
    Dv = cp.asarray(((rng.standard_normal(H) * 0.1).astype(np.float32)
                     .view(np.uint32) >> 16).astype(np.uint16))
    y_pn = cp.zeros(H * P, dtype=cp.float32)
    y_np = cp.zeros(H * P, dtype=cp.float32)

    # One state buffer per layer, not one reused 23 times: the real loop touches
    # 23 distinct 2.10 MB states (48.3 MB total) so each is cold in a 32 MiB L2.
    # Reusing a single buffer would keep it L2-resident and flatter BOTH arms --
    # the same artifact that inflated a GEMV measurement to 336 GB/s earlier today.
    s_pn_all = [cp.asarray(st.reshape(-1)) for _ in range(LAYERS)]
    s_np_all = [cp.asarray(st.transpose(0, 2, 1).copy().reshape(-1))
                for _ in range(LAYERS)]
    s_pn, s_np = s_pn_all[0], s_np_all[0]

    args = (x, Bv, Cv, dt, Alog, Dv)
    dims = (np.int32(H), np.int32(P), np.int32(N), np.int32(HPG))
    threads = min(256, P)

    k_pn((H,), (threads,), (s_pn, *args, y_pn, *dims))
    k_np((H,), (threads,), (s_np, *args, y_np, *dims))
    cp.cuda.Device(0).synchronize()
    y_exact = bool(np.array_equal(cp.asnumpy(y_pn).view(np.uint32),
                                  cp.asnumpy(y_np).view(np.uint32)))
    # the states must also match, modulo the transpose
    a = cp.asnumpy(s_pn).reshape(H, P, N)
    b = cp.asnumpy(s_np).reshape(H, N, P).transpose(0, 2, 1)
    state_exact = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    finite = bool(np.isfinite(cp.asnumpy(y_pn)).all())

    def timed(k, states):
        def run():
            for li in range(LAYERS):
                k((H,), (threads,), (states[li], *args, y_pn, *dims))
        run()
        cp.cuda.Device(0).synchronize()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        for _ in range(ROUNDS):
            run()
        e1.record()
        e1.synchronize()
        return cp.cuda.get_elapsed_time(e0, e1) / ROUNDS

    ms_pn = timed(k_pn, s_pn_all)
    ms_np = timed(k_np, s_np_all)
    rw_bytes = 2 * H * P * N * 4 * LAYERS

    payload = {
        "kind": "diag_ssm_layout",
        "created_utc": utc_now(),
        "note": "ssm_step's state is [h][p][n] while thread p owns a whole n-row, so adjacent threads are N*4 = 512 B apart and each warp instruction uses ~4 of every 32 bytes fetched. Transposing to [h][n][p] makes the 64 threads of a step read 64 consecutive floats. Arithmetic, accumulation order and rounding are untouched -- only the address changes -- which is what keeps it bit-exact, unlike parallelising the acc reduction.",
        "geometry": {"H": H, "P": P, "N": N, "heads_per_group": HPG,
                     "layers_per_token": LAYERS, "blocks": H, "threads": threads,
                     "state_bytes_per_layer": H * P * N * 4,
                     "distinct_state_buffers": LAYERS,
                     "total_state_bytes": H * P * N * 4 * LAYERS,
                     "rw_bytes_per_token": rw_bytes},
        "gates": {"y_bit_exact": y_exact, "state_bit_exact_modulo_transpose": state_exact,
                  "finite": finite},
        "layout_pn": {"ms_per_token": ms_pn, "gb_s": rw_bytes / (ms_pn * 1e-3) / 1e9},
        "layout_np": {"ms_per_token": ms_np, "gb_s": rw_bytes / (ms_np * 1e-3) / 1e9},
        "speedup": ms_pn / ms_np if ms_np else None,
        "saving_ms_per_token": ms_pn - ms_np,
        "in_graph_reference": {"ssm_step_ms": 1.095, "achieved_gb_s": 88.1,
                               "kernel_rate_gb_s": 260.0, "headroom_ms": 0.724},
        "status": "measured" if (y_exact and state_exact and finite) else "correctness_failed",
    }
    write_json_atomic(REPO / "pro_research" / "diag_ssm_layout.json", payload,
                      archive=False)
    print(json.dumps({k: payload[k] for k in
                      ("gates", "layout_pn", "layout_np", "speedup",
                       "saving_ms_per_token", "in_graph_reference", "status")},
                     indent=2))
    return 0 if payload["status"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
