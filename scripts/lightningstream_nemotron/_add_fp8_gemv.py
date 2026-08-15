"""Add the FP8-per-tensor GEMV that 3.5 Lightning's Mamba projections need.

3.5 Lightning stores backbone.layers.N.mixer.in_proj/out_proj as F8_E4M3, one
byte per weight (not packed), scaled by a single FP32 weight_scale. Nemotron 3
Nano had no such format, so no kernel existed. Decode goes through the same
shared 256-entry LUT the FP8 KV path uses -- computing E4M3 arithmetically made
that path compute-bound and slower (measured, N8).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
K = ROOT / "src" / "moe_lab" / "lightningstream_nemotron" / "gpu_kernels.py"

KERNEL = r'''
// FP8 per-tensor GEMV: W is F8_E4M3, one byte per weight, scaled by one scalar.
// Used by 3.5 Lightning's Mamba in_proj/out_proj.
extern "C" __global__ void gemv_fp8_tensor(
    const unsigned char* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ out,
    const float wscale,
    const int rows, const int cols)
{
    extern __shared__ float smem[];
    float* sx = smem;                 // cols floats
    float* lut = smem + cols;         // 256 floats

    const int row = blockIdx.x;
    if (row >= rows) return;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = e4m3_decode((unsigned char)i);
    __syncthreads();

    const uchar4* __restrict__ w4 =
        reinterpret_cast<const uchar4*>(W + (size_t)row * cols);
    const int nvec = cols >> 2;
    float acc = 0.0f;
    for (int v = threadIdx.x; v < nvec; v += blockDim.x) {
        const uchar4 q = w4[v];
        const int k = v << 2;
        acc = fmaf(lut[q.x], sx[k],     acc);
        acc = fmaf(lut[q.y], sx[k + 1], acc);
        acc = fmaf(lut[q.z], sx[k + 2], acc);
        acc = fmaf(lut[q.w], sx[k + 3], acc);
    }
    for (int b = (nvec << 2) + threadIdx.x; b < cols; b += blockDim.x)
        acc = fmaf(lut[W[(size_t)row * cols + b]], sx[b], acc);

    for (int o = warpSize >> 1; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float ws[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) ws[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        float v = (lane < nw) ? ws[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) out[row] = v * wscale;
    }
}
'''

WRAPPER = '''
    def mv_fp8_tensor(self, out, W, x, wscale, rows, cols):
        """FP8-per-tensor GEMV; shared holds the activation plus a 256-entry LUT."""
        self.gemv_fp8_tensor((rows,), (self.block,),
                             (W, x, out, np.float32(wscale),
                              np.int32(rows), np.int32(cols)),
                             shared_mem=(cols + 256) * 4)
'''


def main() -> int:
    src = K.read_text(encoding="utf-8")

    if "gemv_fp8_tensor" in src:
        print("already present")
        return 0

    anchor = '// Warp-per-position flash decoding over an FP8 KV cache.'
    if anchor not in src:
        print("anchor not found")
        return 3
    src = src.replace(anchor, KERNEL + "\n" + anchor, 1)

    src = src.replace('"attn_decode_warp_fp8_gqa")',
                      '"attn_decode_warp_fp8_gqa", "gemv_fp8_tensor")', 1)

    marker = "    def mv_bf16(self, out, W, x, rows, cols):"
    src = src.replace(marker, WRAPPER.lstrip("\n") + "\n" + marker, 1)

    K.write_text(src, encoding="utf-8")
    print("added gemv_fp8_tensor kernel, registration and wrapper")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
