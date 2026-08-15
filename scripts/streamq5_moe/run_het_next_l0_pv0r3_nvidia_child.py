#!/usr/bin/env python3
"""PV0-R3 NVIDIA child: staged real Q5, device-produced activation/down."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; PKG=ROOT/'reports/runs/streamq5_moe/het_next_l0_pv0r3_real_weight_process_validation/pv0r3_nvidia_package.npz'
OWNED=(8,12,168,239,245,374); NAMES=('gate','up','down')
SRC=r'''
extern "C" __device__ __forceinline__ float bf(unsigned short x){return __uint_as_float(((unsigned)x)<<16);} extern "C" __device__ __forceinline__ unsigned short rb(float v){unsigned b=__float_as_uint(v),l=(b>>16)&1;b+=0x7fff+l;return b>>16;}
extern "C" __device__ float q5(const unsigned char*p,const unsigned short*s,const unsigned short*x,int row,int cols,int lane){int packs=cols>>3;float z[32];
for(int v=0;v<32;v++){int t=lane+8*v;float a=0;if(t<packs){unsigned long long w=0;
for(int i=0;i<5;i++)w|=((unsigned long long)p[(row*packs+t)*5+i])<<(8*i);int c=t<<3;float sc=bf(s[row*(cols>>7)+(c>>7)]);
for(int i=0;i<8;i++)a=fmaf(bf(rb(((int)((w>>(5*i))&31)-15)*sc)),bf(x[c+i]),a);}z[v]=a;}for(int q=16;q;q>>=1)for(int i=0;i<q;i++)z[i]+=z[i+q];float v=z[0];for(int o=4;o;o>>=1)v+=__shfl_down_sync(0xff,v,o,8);return v;}
extern "C" __global__ void linear(const unsigned char*c,const unsigned short*s,const unsigned short*x,unsigned short*y,int rows,int cols,int hits){int lane=threadIdx.x&7,warp=(blockIdx.x*blockDim.x+threadIdx.x)>>3,hit=warp/rows,row=warp-hit*rows;if(hit<hits&&row<rows){float v=q5(c,s,x+hit*cols,row,cols,lane);if(lane==0)y[hit*rows+row]=rb(v);}}
extern "C" __global__ void activate(const unsigned short*g,const unsigned short*u,unsigned short*si,unsigned short*a,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float x=bf(g[i]);unsigned short s=rb(x/(1.0f+expf(-x)));si[i]=s;a[i]=rb(bf(s)*bf(u[i]));}}
extern "C" __global__ void weight(const unsigned short*d,unsigned short w,unsigned short*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=rb(bf(d[i])*bf(w));}
extern "C" __global__ void outer(const unsigned short*l,const unsigned short*r,unsigned short*s,unsigned short*g,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float x=bf(l[i]);unsigned short q=rb(1.0f/(1.0f+expf(-x)));s[i]=q;for(int j=threadIdx.x;j<2048;j+=blockDim.x)g[i*2048+j]=rb(bf(q)*bf(r[i*2048+j]));}}
'''
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(x):
 b=json.dumps(x,sort_keys=True,separators=(',',':')).encode();sys.stdout.buffer.write(len(b).to_bytes(8,'little')+b);sys.stdout.buffer.flush()
def receive():
 h=sys.stdin.buffer.read(8)
 if len(h)!=8:raise EOFError('frame header')
 n=int.from_bytes(h,'little')
 if n>1<<20:raise ValueError('frame size')
 b=sys.stdin.buffer.read(n)
 if len(b)!=n:raise EOFError('partial frame')
 return json.loads(b)
def run(nonce):
 import cupy as cp
 if not PKG.exists():raise FileNotFoundError(PKG)
 cp.cuda.set_allocator(None); p=np.load(PKG,allow_pickle=False); ids=p['ids_i64'];weights=p['weights_u16'];x=p['x_u16'].reshape(16,2048);sglin=p['shared_gate_u16'].reshape(16); stream=cp.cuda.Stream(non_blocking=True)
 mod=cp.RawModule(code=SRC,options=('--std=c++14','--fmad=true'),name_expressions=('linear','activate','weight','outer')); linear=mod.get_function('linear');activate=mod.get_function('activate');weight=mod.get_function('weight');outer=mod.get_function('outer')
 calls=[];arrays=[];raw={}; allocations=[]
 def dev(a,name,direction='H2D'):
  d=cp.asarray(a); allocations.append({'name':name,'ptr':int(d.data.ptr),'bytes':int(d.nbytes),'class':'cuda_device'});calls.append({'name':'memcpy','direction':direction,'bytes':int(d.nbytes),'dst':int(d.data.ptr)});arrays.append(d);return d
 emit({'type':'ready','nonce':nonce,'role':'nvidia','seq':0,'pid':os.getpid(),'qpc_ns':time.perf_counter_ns(),'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(errors='replace')})
 cmd=receive();
 if cmd!={'type':'start','nonce':nonce,'role':'nvidia','seq':1}:raise RuntimeError('start')
 for e in OWNED+(512,):
  if e==512:tok=np.arange(16,dtype=np.int64);pos=None
  else:where=np.argwhere(ids==e);tok=where[:,0];pos=where[:,1]
  xin=dev(np.ascontiguousarray(x[tok]),f'e{e}_input');hits=len(tok); vals={}
  for nm in ('gate','up'):
   c=dev(p[f'e{e}_{nm}_codes'],f'e{e}_{nm}_codes');s=dev(p[f'e{e}_{nm}_scales'].view('<u2'),f'e{e}_{nm}_scales');y=cp.empty((hits,512),dtype=cp.uint16);allocations.append({'name':f'e{e}_{nm}','ptr':int(y.data.ptr),'bytes':int(y.nbytes),'class':'cuda_device'});arrays.append(y);linear(((hits*512+31)//32,), (256,), (c,s,xin,y,np.int32(512),np.int32(2048),np.int32(hits)),stream=stream);vals[nm]=y
  si=cp.empty_like(vals['gate']);act=cp.empty_like(si);arrays.extend((si,act));allocations.extend(({'name':f'e{e}_silu','ptr':int(si.data.ptr),'bytes':int(si.nbytes),'class':'cuda_device'},{'name':f'e{e}_activation','ptr':int(act.data.ptr),'bytes':int(act.nbytes),'class':'cuda_device'}));activate(((hits*512+255)//256,), (256,), (vals['gate'],vals['up'],si,act,np.int32(hits*512)),stream=stream)
  c=dev(p[f'e{e}_down_codes'],f'e{e}_down_codes');s=dev(p[f'e{e}_down_scales'].view('<u2'),f'e{e}_down_scales');down=cp.empty((hits,2048),dtype=cp.uint16);arrays.append(down);allocations.append({'name':f'e{e}_down','ptr':int(down.data.ptr),'bytes':int(down.nbytes),'class':'cuda_device'});linear(((hits*2048+31)//32,), (256,), (c,s,act,down,np.int32(2048),np.int32(512),np.int32(hits)),stream=stream);stream.synchronize()
  for nm,v in [('gate',vals['gate']),('up',vals['up']),('silu',si),('activation',act),('down',down)]:raw[f'e{e}_{nm}']=cp.asnumpy(v);calls.append({'name':'memcpy','direction':'D2H','bytes':int(v.nbytes),'src':int(v.data.ptr)})
  if e!=512:
   idx=int(np.where(tok==15)[0][0]);w=int(weights[15,np.where(ids[15]==e)[0][0]]);o=cp.empty(2048,dtype=cp.uint16);arrays.append(o);allocations.append({'name':f'e{e}_weighted_token15','ptr':int(o.data.ptr),'bytes':int(o.nbytes),'class':'cuda_device'});weight((8,),(256,),(down[idx],np.uint16(w),o,np.int32(2048)),stream=stream);stream.synchronize();raw[f'e{e}_weighted_token15']=cp.asnumpy(o);calls.append({'name':'memcpy','direction':'D2H','bytes':4096,'src':int(o.data.ptr)})
  else:
   lin=dev(sglin,'shared_gate_linear');sig=cp.empty(16,dtype=cp.uint16);gated=cp.empty((16,2048),dtype=cp.uint16);arrays.extend((sig,gated));allocations.extend(({'name':'shared_sigmoid','ptr':int(sig.data.ptr),'bytes':32,'class':'cuda_device'},{'name':'shared_gated','ptr':int(gated.data.ptr),'bytes':65536,'class':'cuda_device'}));outer((16,),(256,),(lin,down,sig,gated,np.int32(16)),stream=stream);stream.synchronize();raw['shared_sigmoid']=cp.asnumpy(sig);raw['shared_gated']=cp.asnumpy(gated);calls.extend(({'name':'memcpy','direction':'D2H','bytes':32,'src':int(sig.data.ptr)},{'name':'memcpy','direction':'D2H','bytes':65536,'src':int(gated.data.ptr)}))
 out=PKG.parent/'pv0r3_nvidia_raw.npz';
 with out.open('xb') as f:np.savez(f,**raw)
 stream.synchronize();emit({'type':'result','nonce':nonce,'role':'nvidia','seq':1,'pid':os.getpid(),'qpc_ns':time.perf_counter_ns(),'raw_path':str(out),'raw_sha256':sha(out),'allocations':allocations,'copies':calls,'error':None});return 0
def main():p=argparse.ArgumentParser();p.add_argument('--nonce',required=True);a=p.parse_args();return run(a.nonce)
if __name__=='__main__':raise SystemExit(main())
