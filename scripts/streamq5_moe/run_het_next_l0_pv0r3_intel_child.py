#!/usr/bin/env python3
"""PV0-R3 Intel child: all payload and stage buffers are host-USM."""
from __future__ import annotations
import argparse, ctypes as C, hashlib, json, os, sys, time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; PKG=ROOT/'reports/runs/streamq5_moe/het_next_l0_pv0r3_real_weight_process_validation/pv0r3_intel_package.npz'
OWNED=(50,199,237,474); STAGES=('gate','up','silu','activation','down','weighted')
KERNEL=r'''
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
__kernel void q5_ergv_host_usm(__global const uchar*a,__global const int*b,__global const float*c,__global uint*d,int e,int f,int g){if(get_global_id(0)==0&&d)d[0]=0;}
inline float bf(ushort x){return as_float(((uint)x)<<16);} inline ushort rb(float v){uint b=as_uint(v),l=(b>>16)&1;b+=0x7fff+l;return (ushort)(b>>16);}
inline float q5(const __global uchar*p,const __global ushort*s,const __global ushort*x,int row,int cols,int lane){int packs=cols>>3;float z[32];for(int v=0;v<32;v++){int t=lane+8*v;float a=0;if(t<packs){ulong w=0;for(int i=0;i<5;i++)w|=((ulong)p[(row*packs+t)*5+i])<<(8*i);int c=t<<3;float sc=bf(s[row*(cols>>7)+(c>>7)]);for(int i=0;i<8;i++)a=fma(bf(rb(((int)((w>>(5*i))&31)-15)*sc)),bf(x[c+i]),a);}z[v]=a;}for(int q=16;q;q>>=1)for(int i=0;i<q;i++)z[i]+=z[i+q];float v=z[0];for(int o=4;o;o>>=1){float y=intel_sub_group_shuffle_down(v,v,o);if(lane<o)v+=y;}return v;}
__kernel __attribute__((reqd_work_group_size(256,1,1))) __attribute__((intel_reqd_sub_group_size(8)))
void linear(__global const uchar*c,__global const ushort*s,__global const ushort*x,__global ushort*y,int rows,int cols){int row=get_group_id(0)*32+get_sub_group_id(),lane=get_sub_group_local_id();if(row<rows){float v=q5(c,s,x,row,cols,lane);if(lane==0)y[row]=rb(v);}}
__kernel void activate(__global const ushort*g,__global const ushort*u,__global ushort*si,__global ushort*a,int n){int i=get_global_id(0);if(i<n){float x=bf(g[i]);ushort s=rb(x/(1.0f+exp(-x)));si[i]=s;a[i]=rb(bf(s)*bf(u[i]));}}
__kernel void weight(__global const ushort*d,ushort w,__global ushort*o,int n){int i=get_global_id(0);if(i<n)o[i]=rb(bf(d[i])*bf(w));}
'''
def sha(b):return hashlib.sha256(b).hexdigest()
def emit(o):
 b=json.dumps(o,sort_keys=True,separators=(',',':')).encode();sys.stdout.buffer.write(len(b).to_bytes(8,'little')+b);sys.stdout.buffer.flush()
def receive():
 h=sys.stdin.buffer.read(8)
 if len(h)!=8:raise EOFError('frame header')
 n=int.from_bytes(h,'little')
 if n>1<<20:raise ValueError('frame size')
 b=sys.stdin.buffer.read(n)
 if len(b)!=n:raise EOFError('partial frame')
 return json.loads(b)
def run(nonce):
 from scripts.streamq5_moe import run_st2_mini_host_usm_q5 as base
 if not PKG.exists():raise FileNotFoundError(PKG)
 p=np.load(PKG,allow_pickle=False); ids=p['ids_i64']; weights=p['weights_u16']; x=p['x_u16'].reshape(16,2048); base.KERNEL_SOURCE=KERNEL;base.ALLOC_BYTES=8_060_928
 cl=base.OpenCL(); cap=cl.setup(); allocations=[]; calls={'clHostMemAllocINTEL':1,'clSetKernelArgMemPointerINTEL':0,'clCreateBuffer':0,'clEnqueueWriteBuffer':0,'clEnqueueReadBuffer':0,'clEnqueueCopyBuffer':0,'clEnqueueMigrateMemObjects':0}; raw={}; error=None
 def alloc(n):
  e=C.c_int();q=int(cl.clHostMemAllocINTEL(cl.context,None,n,4096,C.byref(e)));base.check(e.value,'clHostMemAllocINTEL');allocations.append((q,n));calls['clHostMemAllocINTEL']+=1;return q
 def setptr(k,i,q):base.check(cl.clSetKernelArgMemPointerINTEL(k,i,C.c_void_p(q)),'setptr');calls['clSetKernelArgMemPointerINTEL']+=1
 try:
  emit({'type':'ready','nonce':nonce,'role':'intel','seq':0,'pid':os.getpid(),'qpc_ns':time.perf_counter_ns(),'capability':cap})
  cmd=receive();
  if cmd!={'type':'start','nonce':nonce,'role':'intel','seq':1}:raise RuntimeError('start frame')
  program=cl.program;err=C.c_int();linear=cl.lib.clCreateKernel(program,b'linear',C.byref(err));base.check(err.value,'linear');act=cl.lib.clCreateKernel(program,b'activate',C.byref(err));base.check(err.value,'activate');weight=cl.lib.clCreateKernel(program,b'weight',C.byref(err));base.check(err.value,'weight')
  for e in OWNED:
   where=np.argwhere(ids==e); tok=where[:,0];pos=where[:,1]; xin=np.ascontiguousarray(x[tok]); hits=len(tok); xp=alloc(xin.nbytes);C.memmove(xp,xin.ctypes.data,xin.nbytes)
   vals={}
   for j,nm in enumerate(('gate','up')):
    c=p[f'e{e}_{nm}_codes'];s=p[f'e{e}_{nm}_scales'];cp=alloc(c.nbytes);sp=alloc(s.nbytes);yp=alloc(hits*512*2);C.memmove(cp,c.ctypes.data,c.nbytes);C.memmove(sp,s.ctypes.data,s.nbytes);setptr(linear,0,cp);setptr(linear,1,sp);setptr(linear,2,xp);setptr(linear,3,yp);cl.set_int_arg.__func__(cl,4,512);cl.set_int_arg.__func__(cl,5,2048);g=(C.c_size_t*1)(16*256);l=(C.c_size_t*1)(256);base.check(cl.lib.clEnqueueNDRangeKernel(cl.queue,linear,1,None,g,l,0,None,None),'linear');base.check(cl.lib.clFinish(cl.queue),'finish');v=np.empty((hits,512),'<u2');C.memmove(v.ctypes.data,yp,v.nbytes);vals[nm]=v;raw[f'e{e}_{nm}']=v
   gp=alloc(vals['gate'].nbytes);up=alloc(vals['up'].nbytes);sip=alloc(vals['up'].nbytes);ap=alloc(vals['up'].nbytes);C.memmove(gp,vals['gate'].ctypes.data,vals['gate'].nbytes);C.memmove(up,vals['up'].ctypes.data,vals['up'].nbytes);setptr(act,0,gp);setptr(act,1,up);setptr(act,2,sip);setptr(act,3,ap);cl.set_int_arg.__func__(cl,4,hits*512);g=(C.c_size_t*1)(((hits*512+255)//256)*256);l=(C.c_size_t*1)(256);base.check(cl.lib.clEnqueueNDRangeKernel(cl.queue,act,1,None,g,l,0,None,None),'act');base.check(cl.lib.clFinish(cl.queue),'finish');si=np.empty((hits,512),'<u2');av=np.empty_like(si);C.memmove(si.ctypes.data,sip,si.nbytes);C.memmove(av.ctypes.data,ap,av.nbytes);raw[f'e{e}_silu']=si;raw[f'e{e}_activation']=av
   c=p[f'e{e}_down_codes'];s=p[f'e{e}_down_scales'];cp=alloc(c.nbytes);sp=alloc(s.nbytes);dp=alloc(hits*2048*2);C.memmove(cp,c.ctypes.data,c.nbytes);C.memmove(sp,s.ctypes.data,s.nbytes);setptr(linear,0,cp);setptr(linear,1,sp);setptr(linear,2,ap);setptr(linear,3,dp);cl.set_int_arg.__func__(cl,4,2048);cl.set_int_arg.__func__(cl,5,512);g=(C.c_size_t*1)(64*256);base.check(cl.lib.clEnqueueNDRangeKernel(cl.queue,linear,1,None,g,l,0,None,None),'down');base.check(cl.lib.clFinish(cl.queue),'finish');dv=np.empty((hits,2048),'<u2');C.memmove(dv.ctypes.data,dp,dv.nbytes);raw[f'e{e}_down']=dv;idx=int(np.where(tok==15)[0][0]);wp=alloc(4096);setptr(weight,0,dp+idx*4096);cl.lib.clSetKernelArg(weight,1,2,C.byref(C.c_ushort(int(weights[15,np.where(ids[15]==e)[0][0]]))));setptr(weight,2,wp);cl.set_int_arg.__func__(cl,3,2048);g=(C.c_size_t*1)(2048);base.check(cl.lib.clEnqueueNDRangeKernel(cl.queue,weight,1,None,g,l,0,None,None),'weight');base.check(cl.lib.clFinish(cl.queue),'finish');wv=np.empty(2048,'<u2');C.memmove(wv.ctypes.data,wp,wv.nbytes);raw[f'e{e}_weighted_token15']=wv
  out=PKG.parent/'pv0r3_intel_raw.npz';
  with out.open('xb') as f:np.savez(f,**raw)
  emit({'type':'result','nonce':nonce,'role':'intel','seq':1,'pid':os.getpid(),'qpc_ns':time.perf_counter_ns(),'raw_path':str(out),'raw_sha256':sha(out.read_bytes()),'calls':calls,'allocations':[{'ptr':q,'bytes':n} for q,n in allocations],'error':None})
 finally:
  for q,n in reversed(allocations):
   try:cl.clMemFreeINTEL(cl.context,C.c_void_p(q))
   except Exception:error='cleanup'
  cl.close()
 return 0 if error is None else 3
def main():
 p=argparse.ArgumentParser();p.add_argument('--nonce',required=True);a=p.parse_args();return run(a.nonce)
if __name__=='__main__':raise SystemExit(main())
