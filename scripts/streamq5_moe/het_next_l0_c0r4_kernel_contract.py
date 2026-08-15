#!/usr/bin/env python3
"""Frozen device-kernel contract for C0-R4. This module has strings only."""

INTEL_OPENCL_SOURCE = r'''
#pragma OPENCL FP_CONTRACT OFF
inline float bf16f(ushort x) { return as_float(((uint)x) << 16); }
inline ushort bf16bits(float x) {
  uint b=as_uint(x), l=(b>>16)&1U; b += 0x7fffU+l; return (ushort)(b>>16);
}
inline float bf16round(float x) { return bf16f(bf16bits(x)); }
inline int q5(__global const uchar *p, ulong i) {
  ulong pack=i>>3, slot=i&7, o=pack*5;
  ulong w=(ulong)p[o]|((ulong)p[o+1]<<8)|((ulong)p[o+2]<<16)|
          ((ulong)p[o+3]<<24)|((ulong)p[o+4]<<32);
  uint f=(uint)((w>>(5*slot))&31UL); return f==31U ? -2147483647 : (int)f-15;
}
__kernel void q5_linear(__global const uchar *codes,__global const ushort *scales,
 __global const ushort *x,__global ushort *y,uint rows,uint cols) {
 uint r=get_global_id(0); if(r>=rows)return; float sum=0.0f;
 for(uint c=0;c<cols;c++){int q=q5(codes,(ulong)r*cols+c); if(q==-2147483647){y[r]=0xffff;return;}
  float w=bf16round((float)q*bf16f(scales[r*(cols>>7)+(c>>7)]));
  sum=sum+w*bf16f(x[c]);}
 y[r]=bf16bits(sum);
}
// No host activation is accepted. gate/up buffers are device outputs and act is down input.
__kernel void swiglu_bf16(__global const ushort *gate,__global const ushort *up,
 __global ushort *act,uint n) {
 uint i=get_global_id(0); if(i>=n)return; float g=bf16f(gate[i]);
 // exp is the backend implementation under bitwise oracle adjudication; no tolerance is allowed.
 float s=g/(1.0f+exp(-g)); act[i]=bf16bits(bf16round(s)*bf16f(up[i]));
}
'''

NVIDIA_CUDA_SOURCE = r'''
#include <cuda_bf16.h>
__device__ __forceinline__ int q5(const unsigned char *p,unsigned long long i){
 unsigned long long pack=i>>3,slot=i&7,o=pack*5;
 unsigned long long w=(unsigned long long)p[o]|((unsigned long long)p[o+1]<<8)|
  ((unsigned long long)p[o+2]<<16)|((unsigned long long)p[o+3]<<24)|((unsigned long long)p[o+4]<<32);
 unsigned f=(w>>(5*slot))&31ULL; return f==31U ? -2147483647 : (int)f-15;
}
extern "C" __global__ void q5_linear(const unsigned char *codes,const __nv_bfloat16 *scales,
 const __nv_bfloat16 *x,__nv_bfloat16 *y,unsigned rows,unsigned cols){
 unsigned r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;float sum=0.0f;
 for(unsigned c=0;c<cols;c++){int q=q5(codes,(unsigned long long)r*cols+c);if(q==-2147483647){y[r]=__ushort_as_bfloat16(0xffff);return;}
  float w=__bfloat162float(__float2bfloat16_rn((float)q*__bfloat162float(scales[r*(cols>>7)+(c>>7)])));
  sum += w*__bfloat162float(x[c]);} y[r]=__float2bfloat16_rn(sum);
}
extern "C" __global__ void swiglu_bf16(const __nv_bfloat16 *gate,const __nv_bfloat16 *up,
 __nv_bfloat16 *act,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float g=__bfloat162float(gate[i]);float s=g/(1.0f+expf(-g));
 act[i]=__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(s))*__bfloat162float(up[i]));}
'''

KERNEL_CONTRACT = {
    "q5_field": "q_plus_15_in_0_30_field31_error_little_8_in_5",
    "pipeline": ["q5_gate", "q5_up", "device_swiglu_bf16", "q5_down", "bf16_output"],
    "forbidden": ["host_activation_input", "cpu_oracle_alias", "split_matrix"],
    "correctness": "every retained BF16 word bitwise equal to frozen CPU oracle or negative",
}
