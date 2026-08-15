#!/usr/bin/env python3
"""PH0-R3 NVIDIA width-8 backend. CuPy import/initialization occurs only in run()."""
from __future__ import annotations
import hashlib, io
from het_next_l0_ph0r3_common import RECORD_BYTES,INPUT_BYTES,ROWS,COUNTER_BYTES

SRC=r'''
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
__device__ __forceinline__ float b2f(unsigned short x){return __uint_as_float(((unsigned)x)<<16);}
__device__ __forceinline__ unsigned short rbf(float x){unsigned b=__float_as_uint(x),l=(b>>16)&1U;b+=0x7fffU+l;return (unsigned short)(b>>16);}
__device__ __forceinline__ float rbff(float x){return b2f(rbf(x));}
extern "C" __global__ void ph0(const unsigned char* rec,const unsigned short* x,unsigned short* out,unsigned* count){
 cg::thread_block block=cg::this_thread_block();auto tile=cg::tiled_partition<8>(block);int lane=(int)tile.thread_rank(),row=(int)blockIdx.x*32+(int)threadIdx.x/8;if(row>=512)return;
 const unsigned char* code=rec+64;const unsigned short* scale=(const unsigned short*)(rec+64+655360);float p[32];
 #pragma unroll
 for(int v=0;v<32;v++){int pack=lane+8*v,col=pack*8;const unsigned char* s=code+(long long)row*1280LL+(long long)pack*5LL;unsigned long long w=(unsigned long long)s[0]|(unsigned long long)s[1]<<8|(unsigned long long)s[2]<<16|(unsigned long long)s[3]<<24|(unsigned long long)s[4]<<32;float a=0.0f,sc=b2f(scale[row*16+(col>>7)]);
  #pragma unroll
  for(int k=0;k<8;k++){int q=(int)((w>>(5*k))&31ULL)-15;a=fmaf(rbff((float)q*sc),b2f(x[col+k]),a);}p[v]=a;}
 #pragma unroll
 for(int d=16;d>=1;d>>=1){#pragma unroll for(int i=0;i<d;i++)p[i]=__fadd_rn(p[i],p[i+d]);}
 float z=p[0];
 #pragma unroll
 for(int d=4;d>=1;d>>=1){float o=tile.shfl_down(z,d);if(lane<d)z=__fadd_rn(z,o);}
 if(lane==0){out[row]=rbf(z);atomicAdd(&count[row],1U);}
}'''
OPTIONS=('--std=c++17','--fmad=true','--prec-div=true','--prec-sqrt=true','--ftz=false')
def sha(x):return hashlib.sha256(x).hexdigest()

def run(record:bytes,input_bytes:bytes):
 import numpy as np
 import cupy as cp
 from cupy.cuda import compiler
 device=cp.cuda.Device(0);props=cp.cuda.runtime.getDeviceProperties(0);name=props['name'].decode() if isinstance(props['name'],bytes) else props['name']
 if cp.cuda.runtime.getDeviceCount()!=1 or name!='NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU':raise RuntimeError('nvidia_identity')
 pci=cp.cuda.runtime.deviceGetPCIBusId(0)
 if isinstance(pci,bytes):pci=pci.decode()
 if pci!='0000:01:00.0':raise RuntimeError(f'nvidia_pci:{pci}')
 free,total=cp.cuda.runtime.memGetInfo();log=io.StringIO();ptx,_=compiler.compile_using_nvrtc(SRC,options=OPTIONS,name_expressions=('ph0',),log_stream=log,cache_in_memory=True)
 module=cp.RawModule(code=ptx,backend='ptx',name_expressions=('ph0',));kernel=module.get_function('ph0');stream=cp.cuda.Stream(non_blocking=True)
 pinned=[];dev=[];ledger=[]
 try:
  for name,size in (('record',RECORD_BYTES),('input',INPUT_BYTES),('output',ROWS*2),('counters',COUNTER_BYTES)):
   h=cp.cuda.alloc_pinned_memory(size);pinned.append((name,h,size));d=cp.cuda.alloc(size);dev.append((name,d,size));ledger.append({'kind':'allocation','name':name,'bytes':size,'host_pointer':int(h.ptr),'device_pointer':int(d.ptr)})
  memoryview(pinned[0][1])[:]=record;memoryview(pinned[1][1])[:]=input_bytes
  stream.memset_async(dev[2][1].ptr,0xff,ROWS*2);ledger.append({'op':'memset','target':'output','bytes':ROWS*2})
  stream.memset_async(dev[3][1].ptr,0,COUNTER_BYTES);ledger.append({'op':'memset','target':'counters','bytes':COUNTER_BYTES})
  cp.cuda.runtime.memcpyAsync(dev[0][1].ptr,pinned[0][1].ptr,RECORD_BYTES,cp.cuda.runtime.memcpyHostToDevice,stream.ptr);ledger.append({'op':'H2D','target':'record','bytes':RECORD_BYTES})
  cp.cuda.runtime.memcpyAsync(dev[1][1].ptr,pinned[1][1].ptr,INPUT_BYTES,cp.cuda.runtime.memcpyHostToDevice,stream.ptr);ledger.append({'op':'H2D','target':'input','bytes':INPUT_BYTES})
  kernel((16,),(256,),(dev[0][1],dev[1][1],dev[2][1],dev[3][1]),stream=stream);ledger.append({'op':'kernel','grid':[16],'block':[256]})
  cp.cuda.runtime.memcpyAsync(pinned[2][1].ptr,dev[2][1].ptr,ROWS*2,cp.cuda.runtime.memcpyDeviceToHost,stream.ptr);ledger.append({'op':'D2H','target':'output','bytes':ROWS*2})
  cp.cuda.runtime.memcpyAsync(pinned[3][1].ptr,dev[3][1].ptr,COUNTER_BYTES,cp.cuda.runtime.memcpyDeviceToHost,stream.ptr);ledger.append({'op':'D2H','target':'counters','bytes':COUNTER_BYTES})
  stream.synchronize();ledger.append({'op':'synchronize','code':0})
  output=bytes(memoryview(pinned[2][1]));counters=bytes(memoryview(pinned[3][1]))
  identity={'id':0,'name':name,'pci':pci,'driver_version':cp.cuda.runtime.driverGetVersion(),'runtime_version':cp.cuda.runtime.runtimeGetVersion(),'compute_capability':[props['major'],props['minor']],'free_start':free,'total':total}
  return {'identity':identity,'output_hex':output.hex(),'counters_hex':counters.hex(),'source':SRC,'source_sha256':sha(SRC.encode()),'ptx_hex':ptx.hex(),'ptx_sha256':sha(ptx),'compile_log':log.getvalue(),'options':list(OPTIONS),'ledger':ledger,'memset_calls':2,'h2d_calls':2,'kernel_calls':1,'d2h_calls':2,'sync_calls':1}
 finally:
  errors=[]
  for n,d,_ in reversed(dev):
   try:d.mem.free();ledger.append({'release':f'device_{n}','code':0})
   except Exception as e:errors.append(str(e))
  for n,h,_ in reversed(pinned):
   try:h.free();ledger.append({'release':f'pinned_{n}','code':0})
   except Exception as e:errors.append(str(e))
  try:stream.synchronize();ledger.append({'release':'stream_sync','code':0})
  except Exception as e:errors.append(str(e))
  ledger.append({'cleanup_complete':not errors,'errors':errors})
