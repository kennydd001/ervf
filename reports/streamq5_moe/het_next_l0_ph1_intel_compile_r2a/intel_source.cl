
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define CODE_BYTES 655360UL

inline float bf16_to_float(ushort value) {
    return as_float(((uint)value) << 16);
}
inline ushort float_to_bf16(float value) {
    uint bits = as_uint(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return (ushort)(bits >> 16);
}
inline float rounded_bf16_float(float value) {
    return bf16_to_float(float_to_bf16(value));
}
inline ulong round_shift_even(ulong number, int shift) {
    if (shift <= 0) return number << (-shift);
    if (shift >= 64) return 0UL;
    ulong quotient = number >> shift;
    ulong mask = (1UL << shift) - 1UL;
    ulong remainder = number & mask;
    ulong halfway = 1UL << (shift - 1);
    return quotient + (remainder > halfway || (remainder == halfway && (quotient & 1UL)));
}
inline ushort multiply_bf16_exact(ushort a, ushort b, __private uint* ok) {
    uint sign = ((uint)(a ^ b)) & 0x8000U;
    uint ae = ((uint)a >> 7) & 255U, be = ((uint)b >> 7) & 255U;
    uint af = ((uint)a) & 127U, bf = ((uint)b) & 127U;
    if (ae == 255U || be == 255U) { *ok = 0U; return (ushort)0xffffU; }
    if ((ae == 0U && af == 0U) || (be == 0U && bf == 0U)) return (ushort)sign;
    ulong an = ae == 0U ? (ulong)af : (ulong)(128U + af);
    ulong bn = be == 0U ? (ulong)bf : (ulong)(128U + bf);
    int ax = ae == 0U ? -133 : (int)ae - 134;
    int bx = be == 0U ? -133 : (int)be - 134;
    ulong number = an * bn;
    int exponent = ax + bx;
    int highest = 63 - (int)clz(number);
    int top = highest + exponent;
    if (top > 127) { *ok = 0U; return (ushort)0xffffU; }
    if (top >= -126) {
        int shift = highest - 7;
        ulong significand = round_shift_even(number, shift);
        if (significand == 256UL) { significand = 128UL; shift += 1; }
        int unbiased = exponent + shift + 7;
        if (unbiased > 127) { *ok = 0U; return (ushort)0xffffU; }
        return (ushort)(sign | ((uint)(unbiased + 127) << 7) | ((uint)significand & 127U));
    }
    int shift = -133 - exponent;
    ulong fraction = round_shift_even(number, shift);
    if (fraction == 0UL) return (ushort)sign;
    if (fraction >= 128UL) return (ushort)(sign | 0x0080U);
    return (ushort)(sign | (uint)fraction);
}

inline void linear_2048(
    __global const uchar* record, __global const ushort* input,
    __global ushort* output, __global uint* counters) {
    int subgroup = (int)get_sub_group_id();
    int lane = (int)get_sub_group_local_id();
    int row = (int)get_group_id(0) * 32 + subgroup;
    if (row >= 512) return;
    __global const uchar* codes = record + 64;
    __global const ushort* scales = (__global const ushort*)(record + 64 + CODE_BYTES);
    float partial[32];
    #pragma unroll
    for (int virtual_index = 0; virtual_index < 32; ++virtual_index) {
        int pack = lane + 8 * virtual_index;
        int column = pack * 8;
        __global const uchar* source = codes + (ulong)row * 1280UL + (ulong)pack * 5UL;
        ulong fields = (ulong)source[0] | (ulong)source[1] << 8 | (ulong)source[2] << 16 |
                       (ulong)source[3] << 24 | (ulong)source[4] << 32;
        float accumulator = 0.0f;
        float scale = bf16_to_float(scales[row * 16 + (column >> 7)]);
        #pragma unroll
        for (int field = 0; field < 8; ++field) {
            int q = (int)((fields >> (5 * field)) & 31UL) - 15;
            float weight = rounded_bf16_float((float)q * scale);
            accumulator = fma(weight, bf16_to_float(input[column + field]), accumulator);
        }
        partial[virtual_index] = accumulator;
    }
    #pragma unroll
    for (int distance = 16; distance >= 1; distance >>= 1) {
        #pragma unroll
        for (int index = 0; index < distance; ++index) partial[index] = partial[index] + partial[index + distance];
    }
    float value = partial[0];
    #pragma unroll
    for (int distance = 4; distance >= 1; distance >>= 1) {
        float other = intel_sub_group_shuffle_down(value, value, (uint)distance);
        if (lane < distance) value = value + other;
    }
    if (lane == 0) {
        output[row] = float_to_bf16(value);
        atomic_inc((volatile __global unsigned int*)&counters[row]);
    }
}

inline void linear_512(
    __global const uchar* record, __global const ushort* input,
    __global ushort* output, __global uint* counters) {
    int subgroup = (int)get_sub_group_id();
    int lane = (int)get_sub_group_local_id();
    int row = (int)get_group_id(0) * 32 + subgroup;
    if (row >= 2048) return;
    __global const uchar* codes = record + 64;
    __global const ushort* scales = (__global const ushort*)(record + 64 + CODE_BYTES);
    float partial[8];
    #pragma unroll
    for (int virtual_index = 0; virtual_index < 8; ++virtual_index) {
        int pack = lane + 8 * virtual_index;
        int column = pack * 8;
        __global const uchar* source = codes + (ulong)row * 320UL + (ulong)pack * 5UL;
        ulong fields = (ulong)source[0] | (ulong)source[1] << 8 | (ulong)source[2] << 16 |
                       (ulong)source[3] << 24 | (ulong)source[4] << 32;
        float accumulator = 0.0f;
        float scale = bf16_to_float(scales[row * 4 + (column >> 7)]);
        #pragma unroll
        for (int field = 0; field < 8; ++field) {
            int q = (int)((fields >> (5 * field)) & 31UL) - 15;
            float weight = rounded_bf16_float((float)q * scale);
            accumulator = fma(weight, bf16_to_float(input[column + field]), accumulator);
        }
        partial[virtual_index] = accumulator;
    }
    #pragma unroll
    for (int distance = 4; distance >= 1; distance >>= 1) {
        #pragma unroll
        for (int index = 0; index < distance; ++index) partial[index] = partial[index] + partial[index + distance];
    }
    float value = partial[0];
    #pragma unroll
    for (int distance = 4; distance >= 1; distance >>= 1) {
        float other = intel_sub_group_shuffle_down(value, value, (uint)distance);
        if (lane < distance) value = value + other;
    }
    if (lane == 0) {
        output[row] = float_to_bf16(value);
        atomic_inc((volatile __global unsigned int*)&counters[row]);
    }
}

__kernel __attribute__((reqd_work_group_size(256,1,1))) __attribute__((intel_reqd_sub_group_size(8)))
void gate_linear(__global const uchar* record, __global const ushort* input,
                 __global ushort* output, __global uint* counters) {
    linear_2048(record, input, output, counters);
}
__kernel __attribute__((reqd_work_group_size(256,1,1))) __attribute__((intel_reqd_sub_group_size(8)))
void up_linear(__global const uchar* record, __global const ushort* input,
               __global ushort* output, __global uint* counters) {
    linear_2048(record, input, output, counters);
}
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void activation(__global const ushort* gate, __global const ushort* up,
                __global const ushort* lut, __global ushort* silu,
                __global ushort* activated, __global uint* counters) {
    int row = (int)get_global_id(0);
    if (row >= 512) return;
    ushort gate_word = gate[row], up_word = up[row];
    if ((((uint)gate_word >> 7) & 255U) == 255U || (((uint)up_word >> 7) & 255U) == 255U) return;
    ushort silu_word = lut[(uint)gate_word];
    uint ok = 1U;
    ushort activation_word = multiply_bf16_exact(silu_word, up_word, &ok);
    if (!ok) return;
    silu[row] = silu_word;
    activated[row] = activation_word;
    atomic_inc((volatile __global unsigned int*)&counters[row]);
}
__kernel __attribute__((reqd_work_group_size(256,1,1))) __attribute__((intel_reqd_sub_group_size(8)))
void down_linear(__global const uchar* record, __global const ushort* input,
                 __global ushort* output, __global uint* counters) {
    linear_512(record, input, output, counters);
}
