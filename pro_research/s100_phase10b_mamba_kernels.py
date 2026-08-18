from __future__ import annotations
import numpy as np

HEADER=r"""
__device__ __forceinline__ float dec(unsigned char x){
 int s=x>>7,E=(x>>3)&15,m=x&7;
 float v=(E==0)?((float)m*1.953125e-3f):((float)(8+m)*exp2f((float)(E-10)));
 return s?-v:v;
}
__device__ __forceinline__ unsigned int ld_def(const unsigned int* p){return *p;}
__device__ __forceinline__ unsigned int ld_cg(const unsigned int* p){
 unsigned int v;asm volatile("ld.global.cg.u32 %0, [%1];":"=r"(v):"l"(p));return v;
}
__device__ __forceinline__ unsigned int ld_cs(const unsigned int* p){
 unsigned int v;asm volatile("ld.global.cs.u32 %0, [%1];":"=r"(v):"l"(p));return v;
}
"""

def red(width):
 if width==8:
  return """
 float s[8];
 #pragma unroll
 for(int g=0;g<8;++g){
  float a0=acc[g*4]+acc[g*4+2],a1=acc[g*4+1]+acc[g*4+3];
  float v=a0+a1;
  for(int o=4;o>0;o>>=1)v+=__shfl_down_sync(0xffffffffu,v,o,8);
  s[g]=v;
 }"""
 if width==16:
  return """
 float s[8];
 #pragma unroll
 for(int g=0;g<8;++g){
  float v=acc[g*2]+acc[g*2+1];
  for(int o=8;o>0;o>>=1)v+=__shfl_down_sync(0xffffffffu,v,o,16);
  s[g]=v;
 }"""
 return """
 float s[8];
 #pragma unroll
 for(int g=0;g<8;++g){
  float v=acc[g];
  for(int o=16;o>0;o>>=1)v+=__shfl_down_sync(0xffffffffu,v,o,32);
  s[g]=v;
 }"""

def make_kernel(name,width,load,prefetch):
 virt=256//width;rpb=256//width
 if prefetch:
  loop="""
   int q=tid;
   if(q<nvec){
    unsigned int cur=LOAD(w32+q);
    while(q<nvec){
     int nq=q+256;unsigned int nxt=(nq<nvec)?LOAD(w32+nq):0u;int k=q<<2;
     acc[vi]=fmaf(lut[cur&255u],sx[k],acc[vi]);
     acc[vi]=fmaf(lut[(cur>>8)&255u],sx[k+1],acc[vi]);
     acc[vi]=fmaf(lut[(cur>>16)&255u],sx[k+2],acc[vi]);
     acc[vi]=fmaf(lut[(cur>>24)&255u],sx[k+3],acc[vi]);
     cur=nxt;q=nq;
    }
   }"""
 else:
  loop="""
   for(int q=tid;q<nvec;q+=256){
    unsigned int cur=LOAD(w32+q);int k=q<<2;
    acc[vi]=fmaf(lut[cur&255u],sx[k],acc[vi]);
    acc[vi]=fmaf(lut[(cur>>8)&255u],sx[k+1],acc[vi]);
    acc[vi]=fmaf(lut[(cur>>16)&255u],sx[k+2],acc[vi]);
    acc[vi]=fmaf(lut[(cur>>24)&255u],sx[k+3],acc[vi]);
   }"""
 return f"""
#define WIDTH {width}
#define VIRT {virt}
#define RPB {rpb}
#define LOAD {load}
extern "C" __global__ void {name}(
 const unsigned char* W,const float* x,float* out,const float sc,
 const int rows,const int cols){{
 extern __shared__ float sm[];float* sx=sm;float* lut=sm+cols;
 for(int i=threadIdx.x;i<cols;i+=blockDim.x)sx[i]=x[i];
 for(int i=threadIdx.x;i<256;i+=blockDim.x)lut[i]=dec((unsigned char)i);
 __syncthreads();
 int lane=threadIdx.x&(WIDTH-1),sub=threadIdx.x/WIDTH,row=blockIdx.x*RPB+sub;
 bool valid=row<rows;const unsigned char* wb=W+(size_t)(valid?row:0)*cols;
 const unsigned int* w32=(const unsigned int*)wb;int nvec=cols>>2;
 float acc[VIRT];
 #pragma unroll
 for(int vi=0;vi<VIRT;++vi)acc[vi]=0.0f;
 #pragma unroll
 for(int vi=0;vi<VIRT;++vi){{
  int tid=lane+WIDTH*vi;
  if(valid){{{loop}}}
 }}
 {red(width)}
 if(lane==0&&valid){{
  float t0=s[0]+s[4],t1=s[1]+s[5],t2=s[2]+s[6],t3=s[3]+s[7];
  float u0=t0+t2,u1=t1+t3;out[row]=(u0+u1)*sc;
 }}
}}
#undef LOAD
#undef RPB
#undef VIRT
#undef WIDTH
"""

VARIANTS={
 "w8_default":(8,"ld_def",False),
 "w8_pf_cs":(8,"ld_cs",True),
 "w16_pf_default":(16,"ld_def",True),
 "w16_pf_cg":(16,"ld_cg",True),
 "w16_pf_cs":(16,"ld_cs",True),
 "w32_default":(32,"ld_def",False),
 "w32_pf_default":(32,"ld_def",True),
 "w32_pf_cg":(32,"ld_cg",True),
 "w32_pf_cs":(32,"ld_cs",True),
}
SOURCE=HEADER+"\n".join(make_kernel(n,*v) for n,v in VARIANTS.items())

class MambaERVF2:
 def __init__(self):
  import cupy as cp
  self.cp=cp;self.mod=cp.RawModule(code=SOURCE,options=("-std=c++14","--use_fast_math"))
  self.k={n:self.mod.get_function(n) for n in VARIANTS}
 def run(self,name,out,W,x,scale,rows,cols):
  width=VARIANTS[name][0];rpb=256//width
  self.k[name](((rows+rpb-1)//rpb,),(256,),
   (W,x,out,np.float32(scale),np.int32(rows),np.int32(cols)),
   shared_mem=(cols+256)*4)
