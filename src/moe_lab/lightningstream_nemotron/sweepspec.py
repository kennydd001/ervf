"""X1: expert-major NVFP4 kernels for a block verifier.

The autoregressive path reads an expert record once per token. A block verifier
reads it once per BLOCK and multiplies it against every candidate position routed
to that expert. These kernels are the batched forms of the two the routed path
already uses, with the decode, the per-uchar4 FMA order and the panel walk left
exactly as they were, so B=1 reproduces the existing kernels bit for bit and B>1
only changes how many activation vectors ride along.

Node selection and result scatter go through index arrays, never through
per-expert device copies: a copy per (node, expert) pair would cost more than the
grouping saves and would be measuring the harness instead of the idea.

Nothing here touches runtime.py. This is an oracle, not a runtime path.
"""

from __future__ import annotations

import numpy as np

MAX_B = 8

_SOURCE = r"""
#define MAX_B 8

// Batched gemv_nvfp4_rows: one weight read, B activation vectors selected from
// `x` by `nodes`, results written to the slots named by `dst`.
extern "C" __global__ void gemm_nvfp4_rows_b(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float*         __restrict__ e2m1_lut,
    const float*         __restrict__ e4m3_lut,
    const float*         __restrict__ x,        // [n_all, cols]
    const int*           __restrict__ nodes,    // [B] rows of x to use
    float*               __restrict__ out,      // [B, rows]
    const float global_scale,
    const int rows, const int cols, const int B,
    const int apply_relu2)
{
    extern __shared__ float sx[];               // [B * cols]
    const int row = blockIdx.x;
    if (row >= rows) return;

    for (int i = threadIdx.x; i < B * cols; i += blockDim.x) {
        const int b = i / cols;
        const int j = i - b * cols;
        sx[i] = x[(size_t)nodes[b] * cols + j];
    }
    __shared__ float s_e2m1[16];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    __syncthreads();

    const int n_bytes  = cols >> 1;
    const int n_scales = cols >> 4;
    const unsigned char* __restrict__ crow = codes  + (size_t)row * n_bytes;
    const unsigned char* __restrict__ srow = scales + (size_t)row * n_scales;

    float acc[MAX_B];
    #pragma unroll
    for (int b = 0; b < MAX_B; ++b) acc[b] = 0.0f;

    const int n_vec = n_bytes >> 2;
    const uchar4* __restrict__ crow4 = reinterpret_cast<const uchar4*>(crow);

    for (int v = threadIdx.x; v < n_vec; v += blockDim.x) {
        const uchar4 q = crow4[v];
        const int bidx = v << 2;
        const float s = e4m3_lut[srow[bidx >> 3]] * global_scale;
        const int k = bidx << 1;
        const float w0 = s_e2m1[q.x & 0x0F] * s;
        const float w1 = s_e2m1[q.x >> 4]   * s;
        const float w2 = s_e2m1[q.y & 0x0F] * s;
        const float w3 = s_e2m1[q.y >> 4]   * s;
        const float w4 = s_e2m1[q.z & 0x0F] * s;
        const float w5 = s_e2m1[q.z >> 4]   * s;
        const float w6 = s_e2m1[q.w & 0x0F] * s;
        const float w7 = s_e2m1[q.w >> 4]   * s;
        #pragma unroll
        for (int b = 0; b < MAX_B; ++b) {
            if (b >= B) break;
            const float* sb = sx + (size_t)b * cols;
            float a = acc[b];
            a = fmaf(w0, sb[k],     a);
            a = fmaf(w1, sb[k + 1], a);
            a = fmaf(w2, sb[k + 2], a);
            a = fmaf(w3, sb[k + 3], a);
            a = fmaf(w4, sb[k + 4], a);
            a = fmaf(w5, sb[k + 5], a);
            a = fmaf(w6, sb[k + 6], a);
            a = fmaf(w7, sb[k + 7], a);
            acc[b] = a;
        }
    }
    for (int bidx = (n_vec << 2) + threadIdx.x; bidx < n_bytes; bidx += blockDim.x) {
        const unsigned char byte = crow[bidx];
        const float s = e4m3_lut[srow[bidx >> 3]] * global_scale;
        const int k = bidx << 1;
        const float w0 = s_e2m1[byte & 0x0F] * s;
        const float w1 = s_e2m1[byte >> 4]   * s;
        #pragma unroll
        for (int b = 0; b < MAX_B; ++b) {
            if (b >= B) break;
            const float* sb = sx + (size_t)b * cols;
            float a = acc[b];
            a = fmaf(w0, sb[k],     a);
            a = fmaf(w1, sb[k + 1], a);
            acc[b] = a;
        }
    }

    __shared__ float warp_sums[MAX_B][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int b = 0; b < MAX_B; ++b) {
        if (b >= B) break;
        float a = acc[b];
        for (int off = warpSize >> 1; off > 0; off >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0) warp_sums[b][warp] = a;
    }
    __syncthreads();
    if (warp == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        #pragma unroll
        for (int b = 0; b < MAX_B; ++b) {
            if (b >= B) break;
            float v = (lane < n_warps) ? warp_sums[b][lane] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) {
                if (apply_relu2) {
                    const float r = fmaxf(v, 0.0f);
                    out[(size_t)b * rows + row] = r * r;
                } else {
                    out[(size_t)b * rows + row] = v;
                }
            }
        }
    }
}

// Batched gemv_down_masked_partial over the UNION panel mask. A column where
// node b is zero contributes fmaf(w, 0, acc) = acc exactly, so widening a node's
// mask to the union cannot change its value.
extern "C" __global__ void gemm_down_masked_b(
    const unsigned char* __restrict__ bank,
    const float*         __restrict__ act,        // [B, inter]
    const int*           __restrict__ panel_list,
    const unsigned int*  __restrict__ panel_masks,
    const int*           __restrict__ panel_count,
    const float*         __restrict__ e2m1_lut,
    const float*         __restrict__ e4m3_lut,
    float*               __restrict__ partials,   // [B, nchunks, rows]
    const float global_scale,
    const int rows, const int inter, const int B)
{
    const int nchunks = gridDim.y;
    const int chunk = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_e2m1[16];
    __shared__ float s_e4m3[256];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    if (threadIdx.x < 256) s_e4m3[threadIdx.x] = e4m3_lut[threadIdx.x];
    __syncthreads();
    if (row >= rows) return;

    const int hb = row >> 1;
    const int hi = row & 1;
    const int rowhalf = rows >> 1;
    const int pcount = *panel_count;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;

    float acc[MAX_B];
    #pragma unroll
    for (int b = 0; b < MAX_B; ++b) acc[b] = 0.0f;

    for (int pi = chunk; pi < pcount; pi += nchunks) {
        const int p = panel_list[pi];
        const unsigned char* __restrict__ pbase = bank + (size_t)p * panel_stride;
        const float s = s_e4m3[pbase[row]] * global_scale;
        const unsigned char* __restrict__ pcodes = pbase + rows;
        unsigned int m = panel_masks[p];
        while (m) {
            const int c = __ffs(m) - 1;
            m &= m - 1;
            const unsigned char byte = pcodes[(size_t)c * rowhalf + hb];
            const float w = s_e2m1[hi ? (byte >> 4) : (byte & 15)] * s;
            const int col = (p << 4) + c;
            #pragma unroll
            for (int b = 0; b < MAX_B; ++b) {
                if (b >= B) break;
                acc[b] = fmaf(w, act[(size_t)b * inter + col], acc[b]);
            }
        }
    }
    #pragma unroll
    for (int b = 0; b < MAX_B; ++b) {
        if (b >= B) break;
        partials[((size_t)b * nchunks + chunk) * rows + row] = acc[b];
    }
}

// Reduce the chunk partials straight into each node's (node, slot) contribution
// buffer, so no device-to-device copy is needed to scatter the results.
extern "C" __global__ void reduce_partials_scatter(
    const float* __restrict__ partials,     // [B, nchunks, rows]
    const int*   __restrict__ dst,          // [B] destination slot per lane
    float*       __restrict__ contrib,      // [n_slots, rows]
    const int rows, const int nchunks, const int B)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    for (int b = 0; b < B; ++b) {
        float a = 0.0f;
        for (int c = 0; c < nchunks; ++c)
            a += partials[((size_t)b * nchunks + c) * rows + row];
        contrib[(size_t)dst[b] * rows + row] = a;
    }
}

// Sum each node's expert contributions in ROUTE-SLOT order, so the per-node
// accumulation order is identical to the token-major path.
extern "C" __global__ void reduce_slots(
    const float* __restrict__ contrib,      // [B, top_k, rows]
    const float* __restrict__ weights,      // [B, top_k]
    float*       __restrict__ out,          // [B, rows]
    const int rows, const int top_k, const int B)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    for (int b = 0; b < B; ++b) {
        float a = out[(size_t)b * rows + row];
        for (int s = 0; s < top_k; ++s)
            a = fmaf(weights[b * top_k + s],
                     contrib[(((size_t)b * top_k) + s) * rows + row], a);
        out[(size_t)b * rows + row] = a;
    }
}
"""


class SweepMoE:
    """Expert-major block execution of the routed experts."""

    def __init__(self, fused, kernels, block: int = 256):
        import cupy as cp

        self.cp = cp
        self.fused = fused
        self.k = kernels
        self.block = block
        self.mod = cp.RawModule(code=_SOURCE, options=("-std=c++14",))
        self.gemm_b = self.mod.get_function("gemm_nvfp4_rows_b")
        self.down_b = self.mod.get_function("gemm_down_masked_b")
        self.reduce_scatter = self.mod.get_function("reduce_partials_scatter")
        self.reduce_slots_k = self.mod.get_function("reduce_slots")
        # Staging B activation vectors of width `hidden` needs more than the
        # 48 KiB default. The opt-in cap counts this kernel's static shared
        # (the E2M1 table plus the per-b warp-sum scratch) as well, so ask for
        # the largest value the driver actually accepts and settle it here, not
        # inside a timed region.
        cap = int(cp.cuda.Device(0).attributes["MaxSharedMemoryPerBlockOptin"])
        self.max_shared = 0
        for want in range(cap, 32 * 1024, -2048):
            try:
                self.gemm_b.max_dynamic_shared_size_bytes = want
                self.max_shared = want
                break
            except Exception:
                continue
        if not self.max_shared:
            raise RuntimeError(f"no opt-in dynamic shared size accepted (cap {cap})")

    # ------------------------------------------------------------- planning
    @staticmethod
    def plan(routes, top_k: int):
        """Group (node, slot) pairs by expert, experts in first-seen order."""
        groups: dict[int, list[tuple[int, int]]] = {}
        for b, r in enumerate(routes):
            for s in range(top_k):
                groups.setdefault(int(r[s]), []).append((b, s))
        return groups

    def compile_plan(self, groups, top_k: int):
        """Device-side index arrays for one layer's plan, built once."""
        cp = self.cp
        out = []
        for e, pairs in groups.items():
            nodes = cp.asarray(np.array([b for b, _ in pairs], dtype=np.int32))
            dst = cp.asarray(np.array([b * top_k + s for b, s in pairs],
                                      dtype=np.int32))
            out.append((int(e), nodes, dst, len(pairs)))
        return out

    def alloc_state(self, hidden: int, inter: int, top_k: int, B: int):
        cp = self.cp
        npanel = inter // 16
        nch = self.fused.nchunks
        return {
            "acts": cp.zeros(B * inter, dtype=cp.float32),
            "act_sum": cp.zeros(inter, dtype=cp.float32),
            "partials": cp.zeros(B * nch * hidden, dtype=cp.float32),
            "contrib": cp.zeros(B * top_k * hidden, dtype=cp.float32),
            "weights": cp.zeros(B * top_k, dtype=cp.float32),
            "masks": cp.zeros(npanel, dtype=cp.uint32),
            "plist": cp.zeros(npanel, dtype=cp.int32),
            "pcount": cp.zeros(1, dtype=cp.int32),
            "nz": cp.zeros(inter, dtype=cp.int32),
            "nzc": cp.zeros(1, dtype=cp.int32),
            "mirror": cp.zeros(npanel * (hidden + 16 * (hidden // 2)), dtype=cp.uint8),
            "nchunks": nch, "npanel": npanel, "top_k": top_k, "B": B,
        }

    # -------------------------------------------------------------- kernels
    def gemm_into(self, out, codes, scales, x, nodes, global_scale: float,
                  rows: int, cols: int, n: int, apply_relu2: bool = False):
        shared = n * cols * 4
        if shared > self.max_shared:
            raise ValueError(f"n={n}, cols={cols} needs {shared} B shared, "
                             f"device allows {self.max_shared}")
        self.gemm_b((rows,), (self.block,),
                    (codes, scales, self.fused.e2m1, self.fused.e4m3, x, nodes,
                     out, np.float32(global_scale), np.int32(rows), np.int32(cols),
                     np.int32(n), np.int32(1 if apply_relu2 else 0)),
                    shared_mem=shared)

    def down_union(self, dst, bank_ptr: int, state, global_scale: float,
                   hidden: int, inter: int, n: int, gather_from_host: bool = True):
        cp = self.cp
        acts2d = state["acts"][:n * inter].reshape(n, inter)
        state["act_sum"][:] = acts2d.sum(axis=0)
        self.fused.panel_scan_k((1,), (256,),
                                (state["act_sum"], np.int32(inter), state["masks"],
                                 state["plist"], state["pcount"],
                                 state["nz"], state["nzc"]))
        if gather_from_host:
            max_warps = inter + state["npanel"]
            blocks = (max_warps * 32 + 255) // 256
            self.fused.gather_k((blocks,), (256,),
                                (np.uint64(bank_ptr), state["mirror"],
                                 state["plist"], state["pcount"],
                                 state["nz"], state["nzc"], np.int32(hidden)))
            src = state["mirror"].data.ptr
        else:
            src = bank_ptr
        nch = state["nchunks"]
        self.down_b(((hidden + 127) // 128, nch), (128,),
                    (np.uint64(src), state["acts"], state["plist"], state["masks"],
                     state["pcount"], self.fused.e2m1, self.fused.e4m3,
                     state["partials"], np.float32(global_scale),
                     np.int32(hidden), np.int32(inter), np.int32(n)))
        self.reduce_scatter(((hidden + 255) // 256,), (256,),
                            (state["partials"], dst, state["contrib"],
                             np.int32(hidden), np.int32(nch), np.int32(n)))

    def moe_block(self, out, rt, layer: int, x_all, compiled_plan, state, B: int):
        """Expert-major MoE for B nodes of one layer. `out` is [B, hidden]."""
        from .runtime import UP_CODE, UP_SCALE, DOWN_PANEL_BYTES

        hidden, inter, top_k = rt.hidden, rt.moe_inter, rt.top_k
        bank, cache = rt.bank[layer], rt.cache[layer]
        for e, nodes, dst, n in compiled_plan:
            slot = cache["map"][e]
            self.gemm_into(state["acts"],
                           cache["codes"][slot * UP_CODE:(slot + 1) * UP_CODE],
                           cache["scales"][slot * UP_SCALE:(slot + 1) * UP_SCALE],
                           x_all, nodes, float(bank["globals"][e, 1]),
                           inter, hidden, n, apply_relu2=True)
            self.down_union(dst, bank["down_base_ptr"] + e * DOWN_PANEL_BYTES,
                            state, float(bank["globals"][e, 0]), hidden, inter, n)
        self.reduce_slots_k(((hidden + 255) // 256,), (256,),
                            (state["contrib"], state["weights"], out,
                             np.int32(hidden), np.int32(top_k), np.int32(B)))
