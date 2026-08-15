#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CODE_BYTES 655360ULL

__device__ __forceinline__ unsigned load_u32(const unsigned char* p) {
    return (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 | (unsigned)p[3] << 24;
}
__device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float(((unsigned)value) << 16);
}
__device__ __forceinline__ unsigned short float_to_bf16(float value) {
    unsigned bits = __float_as_uint(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return (unsigned short)(bits >> 16);
}
__device__ __forceinline__ float rounded_bf16_float(float value) {
    return bf16_to_float(float_to_bf16(value));
}
__device__ __forceinline__ unsigned long long round_shift_even(unsigned long long number, int shift) {
    if (shift <= 0) return number << (-shift);
    if (shift >= 64) return 0ULL;
    unsigned long long quotient = number >> shift;
    unsigned long long mask = (1ULL << shift) - 1ULL;
    unsigned long long remainder = number & mask;
    unsigned long long halfway = 1ULL << (shift - 1);
    return quotient + (remainder > halfway || (remainder == halfway && (quotient & 1ULL)));
}
__device__ __forceinline__ unsigned short multiply_bf16_exact(unsigned short a, unsigned short b, unsigned* ok) {
    unsigned sign = ((unsigned)(a ^ b)) & 0x8000U;
    unsigned ae = ((unsigned)a >> 7) & 255U, be = ((unsigned)b >> 7) & 255U;
    unsigned af = ((unsigned)a) & 127U, bf = ((unsigned)b) & 127U;
    if (ae == 255U || be == 255U) { *ok = 0U; return (unsigned short)0xffffU; }
    if ((ae == 0U && af == 0U) || (be == 0U && bf == 0U)) return (unsigned short)sign;
    unsigned long long an = ae == 0U ? (unsigned long long)af : (unsigned long long)(128U + af);
    unsigned long long bn = be == 0U ? (unsigned long long)bf : (unsigned long long)(128U + bf);
    int ax = ae == 0U ? -133 : (int)ae - 134;
    int bx = be == 0U ? -133 : (int)be - 134;
    unsigned long long number = an * bn;
    int exponent = ax + bx;
    int highest = 63 - __clzll(number);
    int top = highest + exponent;
    if (top > 127) { *ok = 0U; return (unsigned short)0xffffU; }
    if (top >= -126) {
        int shift = highest - 7;
        unsigned long long significand = round_shift_even(number, shift);
        if (significand == 256ULL) { significand = 128ULL; shift += 1; }
        int unbiased = exponent + shift + 7;
        if (unbiased > 127) { *ok = 0U; return (unsigned short)0xffffU; }
        return (unsigned short)(sign | ((unsigned)(unbiased + 127) << 7) | ((unsigned)significand & 127U));
    }
    int shift = -133 - exponent;
    unsigned long long fraction = round_shift_even(number, shift);
    if (fraction == 0ULL) return (unsigned short)sign;
    if (fraction >= 128ULL) return (unsigned short)(sign | 0x0080U);
    return (unsigned short)(sign | (unsigned)fraction);
}

extern "C" __global__ void q5_linear(
    const unsigned char* record, const unsigned short* input,
    unsigned short* output, unsigned* counters) {
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<8>(block);
    int lane = (int)tile.thread_rank();
    int rows = (int)load_u32(record + 12), cols = (int)load_u32(record + 16);
    if (!((rows == 512 && cols == 2048) || (rows == 2048 && cols == 512))) return;
    int row = (int)blockIdx.x * 32 + (int)threadIdx.x / 8;
    if (row >= rows) return;
    const unsigned char* codes = record + 64;
    const unsigned short* scales = (const unsigned short*)(record + 64 + CODE_BYTES);
    int virtual_count = cols == 2048 ? 32 : 8;
    float partial[32];
    #pragma unroll
    for (int virtual_index = 0; virtual_index < 32; ++virtual_index) {
        if (virtual_index < virtual_count) {
            int pack = lane + 8 * virtual_index;
            int column = pack * 8;
            unsigned long long row_stride = cols == 2048 ? 1280ULL : 320ULL;
            const unsigned char* source = codes + (unsigned long long)row * row_stride + (unsigned long long)pack * 5ULL;
            unsigned long long fields = (unsigned long long)source[0] | (unsigned long long)source[1] << 8 |
                (unsigned long long)source[2] << 16 | (unsigned long long)source[3] << 24 | (unsigned long long)source[4] << 32;
            float accumulator = 0.0f;
            int groups_per_row = cols >> 7;
            float scale = bf16_to_float(scales[row * groups_per_row + (column >> 7)]);
            #pragma unroll
            for (int field = 0; field < 8; ++field) {
                int q = (int)((fields >> (5 * field)) & 31ULL) - 15;
                accumulator = fmaf(rounded_bf16_float((float)q * scale), bf16_to_float(input[column + field]), accumulator);
            }
            partial[virtual_index] = accumulator;
        }
    }
    for (int distance = cols == 2048 ? 16 : 4; distance >= 1; distance >>= 1) {
        #pragma unroll
        for (int index = 0; index < 16; ++index) {
            if (index < distance) partial[index] = __fadd_rn(partial[index], partial[index + distance]);
        }
    }
    float value = partial[0];
    #pragma unroll
    for (int distance = 4; distance >= 1; distance >>= 1) {
        float other = tile.shfl_down(value, distance);
        if (lane < distance) value = __fadd_rn(value, other);
    }
    if (lane == 0) { output[row] = float_to_bf16(value); atomicAdd(&counters[row], 1U); }
}

extern "C" __global__ void bf16_lut_activation(
    const unsigned short* gate, const unsigned short* up, const unsigned short* lut,
    unsigned short* silu, unsigned short* activation, unsigned* counters) {
    int row = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row >= 512) return;
    unsigned short gate_word = gate[row], up_word = up[row];
    if ((((unsigned)gate_word >> 7) & 255U) == 255U || (((unsigned)up_word >> 7) & 255U) == 255U) return;
    unsigned short silu_word = lut[(unsigned)gate_word];
    unsigned ok = 1U;
    unsigned short activation_word = multiply_bf16_exact(silu_word, up_word, &ok);
    if (!ok) return;
    silu[row] = silu_word;
    activation[row] = activation_word;
    atomicAdd(&counters[row], 1U);
}
