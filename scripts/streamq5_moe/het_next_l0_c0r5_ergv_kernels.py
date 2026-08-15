#!/usr/bin/env python3
"""Frozen C0-R5 capability and future width-8 ERGV kernel sources; no imports."""

INTEL_FLAGS=("-cl-std=CL2.0","-cl-fp32-correctly-rounded-divide-sqrt")
NVIDIA_FLAGS=("--std=c++14","--fmad=false")
WIDTH=8;WORKGROUP=256;VIRTUAL=32;ROWS_PER_BLOCK=32;GROUP=128

INTEL_CAPABILITY_SOURCE=r'''
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void c0r5_usm_sentinel(__global uchar *p,uint n){uint i=get_global_id(0);if(i<n)p[i]^=(uchar)0x5a;}
'''
NVIDIA_CAPABILITY_SOURCE=r'''
extern "C" __global__ void c0r5_cuda_sentinel(unsigned char *p,unsigned n){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]=(unsigned char)(p[i]+0x33U);}
'''

INTEL_ERVG_SOURCE=r'''
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define WIDTH 8
#define VIRTUAL 32
#define ROWS_PER_BLOCK 32
inline float bf16f(ushort v){return as_float(((uint)v)<<16);}
inline ushort bf16bits(float v){uint b=as_uint(v),l=(b>>16)&1U;b+=0x7fffU+l;return (ushort)(b>>16);}
inline float bf16round(float v){return bf16f(bf16bits(v));}
inline int code5(__global const uchar*p,ulong pack,int slot){ulong o=pack*5UL,w=(ulong)p[o]|((ulong)p[o+1]<<8)|((ulong)p[o+2]<<16)|((ulong)p[o+3]<<24)|((ulong)p[o+4]<<32);uint f=(uint)((w>>(5*slot))&31UL);return f==31U?-2147483647:(int)f-15;}
inline float row8(__global const uchar*codes,__global const ushort*scales,__global const ushort*x,int row,int cols,int lane,int *bad){
 int packs=cols>>3,groups=cols>>7;float partial[VIRTUAL];
 #pragma unroll
 for(int vi=0;vi<VIRTUAL;vi++){int pack=lane+WIDTH*vi;float sum=0.0f;if(pack<packs){int col=pack<<3;float s=bf16f(scales[row*groups+(col>>7)]);
  #pragma unroll
  for(int j=0;j<8;j++){int q=code5(codes,(ulong)row*packs+pack,j);if(q==-2147483647){*bad=1;q=0;}float w=bf16round((float)q*s);sum=fma(w,bf16f(x[col+j]),sum);}}partial[vi]=sum;}
 #pragma unroll
 for(int stride=128;stride>=WIDTH;stride>>=1){#pragma unroll for(int i=0;i<stride/WIDTH;i++)partial[i]=partial[i]+partial[i+stride/WIDTH];}
 float v=partial[0];#pragma unroll for(int off=WIDTH/2;off>0;off>>=1){float z=intel_sub_group_shuffle_down(v,v,(uint)off);if(lane<off)v=v+z;}return v;}
__kernel __attribute__((reqd_work_group_size(256,1,1))) __attribute__((intel_reqd_sub_group_size(8)))
void c0r5_q5_ervg8(__global const uchar*codes,__global const ushort*scales,__global const ushort*x,__global ushort*y,__global int*bad,int rows,int cols){int block=get_group_id(0),sub=get_sub_group_id(),lane=get_sub_group_local_id(),row=block*ROWS_PER_BLOCK+sub;if(row>=rows)return;int b=0;float v=row8(codes,scales,x,row,cols,lane,&b);if(lane==0){y[row]=bf16bits(v);if(b)atomic_or(bad,1);}}
'''

NVIDIA_ERVG_SOURCE=r'''
#include <cuda_bf16.h>
#define WIDTH 8
#define VIRTUAL 32
__device__ __forceinline__ float bf16f(unsigned short v){return __uint_as_float(((unsigned)v)<<16);}
__device__ __forceinline__ unsigned short bf16bits(float v){unsigned b=__float_as_uint(v),l=(b>>16)&1U;b+=0x7fffU+l;return (unsigned short)(b>>16);}
__device__ __forceinline__ float bf16round(float v){return bf16f(bf16bits(v));}
__device__ __forceinline__ int code5(const unsigned char*p,long long pack,int slot){long long o=pack*5LL;unsigned long long w=(unsigned long long)p[o]|((unsigned long long)p[o+1]<<8)|((unsigned long long)p[o+2]<<16)|((unsigned long long)p[o+3]<<24)|((unsigned long long)p[o+4]<<32);unsigned f=(w>>(5*slot))&31ULL;return f==31U?-2147483647:(int)f-15;}
__device__ __forceinline__ float row8(const unsigned char*codes,const unsigned short*scales,const unsigned short*x,int row,int cols,int lane,int*bad){int packs=cols>>3,groups=cols>>7;float partial[VIRTUAL];
 #pragma unroll
 for(int vi=0;vi<VIRTUAL;vi++){int pack=lane+WIDTH*vi;float sum=0.0f;if(pack<packs){int col=pack<<3;float s=bf16f(scales[row*groups+(col>>7)]);
  #pragma unroll
  for(int j=0;j<8;j++){int q=code5(codes,(long long)row*packs+pack,j);if(q==-2147483647){*bad=1;q=0;}float w=bf16round((float)q*s);sum=__fmaf_rn(w,bf16f(x[col+j]),sum);}}partial[vi]=sum;}
 #pragma unroll
 for(int stride=128;stride>=WIDTH;stride>>=1){#pragma unroll for(int i=0;i<stride/WIDTH;i++)partial[i]=partial[i]+partial[i+stride/WIDTH];}
 float v=partial[0];#pragma unroll for(int off=WIDTH/2;off>0;off>>=1)v+=__shfl_down_sync(0xffffffffU,v,off,WIDTH);return v;}
extern "C" __global__ void c0r5_q5_ervg8(const unsigned char*codes,const unsigned short*scales,const unsigned short*x,unsigned short*y,int*bad,int rows,int cols){int group=threadIdx.x/WIDTH,lane=threadIdx.x&(WIDTH-1),row=blockIdx.x*(256/WIDTH)+group;if(row>=rows)return;int b=0;float v=row8(codes,scales,x,row,cols,lane,&b);if(lane==0){y[row]=bf16bits(v);if(b)atomicOr(bad,1);}}
'''

LAUNCH_CONTRACT={
 'intel':{'local_size':256,'subgroup':8,'rows_per_workgroup':32,'global_workgroups':'ceil(rows/32)','input_dtype':'BF16 words','output_dtype':'BF16 words','flags':INTEL_FLAGS},
 'nvidia':{'block_threads':256,'subgroup':8,'rows_per_block':32,'grid_blocks':'ceil(rows/32)','input_dtype':'BF16 words','output_dtype':'BF16 words','flags':NVIDIA_FLAGS},
 'reduction':'32 virtual packs/lane; partial stride 128,64,32,16,8; subgroup offsets 4,2,1; FP32 FMA; one BF16 output cast',
 'shapes':[(512,2048),(512,2048),(2048,512)],'field31':'atomic error and invalid result','full_expert_pipeline':'gate+up ERVG -> device BF16 SiLU-times-up -> down ERVG; validation wrapper remains closed'
}
