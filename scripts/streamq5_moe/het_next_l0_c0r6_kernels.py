#!/usr/bin/env python3
"""Corrected C0-R6 width-8 ERGV sentinel sources, fixed flags and geometry."""
INTEL_FLAGS=("-cl-std=CL2.0","-cl-fp32-correctly-rounded-divide-sqrt")
NVIDIA_FLAGS=("--std=c++14","--fmad=false")
INTEL_SOURCE=r'''
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define WIDTH 8
#define VIRTUAL 32
#define ROWS_PER_BLOCK 32
inline ushort bf16bits(float v){uint b=as_uint(v),l=(b>>16)&1U;b+=0x7fffU+l;return (ushort)(b>>16);}
__kernel __attribute__((reqd_work_group_size(256,1,1))) __attribute__((intel_reqd_sub_group_size(8)))
void c0r6_ergv8_sentinel(__global const float*x,__global ushort*y,int rows,int cols){
 int row=(int)get_group_id(0)*ROWS_PER_BLOCK+(int)get_sub_group_id();int lane=(int)get_sub_group_local_id();if(row>=rows)return;float partial[VIRTUAL];
 #pragma unroll
 for(int vi=0;vi<VIRTUAL;vi++){int tid=lane+WIDTH*vi;float sum=0.0f;for(int c=tid;c<cols;c+=256)sum=fma((float)((row*17+c*13)%31-15),x[c],sum);partial[vi]=sum;}
 #pragma unroll
 for(int stride=128;stride>=WIDTH;stride>>=1){
  #pragma unroll
  for(int i=0;i<stride/WIDTH;i++)partial[i]=partial[i]+partial[i+stride/WIDTH];
 }
 float v=partial[0];
 #pragma unroll
 for(int off=4;off>0;off>>=1){float z=intel_sub_group_shuffle_down(v,v,(uint)off);if(lane<off)v=v+z;}
 if(lane==0)y[row]=bf16bits(v);
}
__kernel __attribute__((reqd_work_group_size(64,1,1))) void c0r6_usm_copyless(__global uchar*p,uint n){uint i=get_global_id(0);if(i<n)p[i]^=(uchar)0x5a;}
'''
NVIDIA_SOURCE=r'''
#include <cuda_bf16.h>
#define WIDTH 8
#define VIRTUAL 32
__device__ __forceinline__ unsigned short bf16bits(float v){unsigned b=__float_as_uint(v),l=(b>>16)&1U;b+=0x7fffU+l;return (unsigned short)(b>>16);}
extern "C" __global__ void c0r6_ergv8_sentinel(const float*x,unsigned short*y,int rows,int cols){
 int group=threadIdx.x/WIDTH,lane=threadIdx.x&(WIDTH-1),row=blockIdx.x*(256/WIDTH)+group;if(row>=rows)return;float partial[VIRTUAL];
 #pragma unroll
 for(int vi=0;vi<VIRTUAL;vi++){int tid=lane+WIDTH*vi;float sum=0.0f;for(int c=tid;c<cols;c+=256)sum=__fmaf_rn((float)((row*17+c*13)%31-15),x[c],sum);partial[vi]=sum;}
 #pragma unroll
 for(int stride=128;stride>=WIDTH;stride>>=1){
  #pragma unroll
  for(int i=0;i<stride/WIDTH;i++)partial[i]=partial[i]+partial[i+stride/WIDTH];
 }
 float v=partial[0];
 #pragma unroll
 for(int off=4;off>0;off>>=1)v+=__shfl_down_sync(0xffffffffU,v,off,WIDTH);
 if(lane==0)y[row]=bf16bits(v);
}
extern "C" __global__ void c0r6_cuda_sentinel(unsigned char*p,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]=(unsigned char)(p[i]+0x33U);}
'''
CONTRACT={'width':8,'threads':256,'virtual':32,'rows_per_block':32,'shapes':[(32,256)],'reduction':['lane+8*virtual','stride128','stride64','stride32','stride16','stride8','shuffle4','shuffle2','shuffle1'],'output':'BF16_RNE_words','intel_flags':INTEL_FLAGS,'nvidia_flags':NVIDIA_FLAGS}
