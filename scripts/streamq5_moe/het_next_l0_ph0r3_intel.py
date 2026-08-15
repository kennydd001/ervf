#!/usr/bin/env python3
"""PH0-R3 Intel host-USM backend. Import is device-free; calls occur only in run()."""
from __future__ import annotations
import ctypes as C, hashlib
import numpy as np
from het_next_l0_ph0r3_common import RECORD_BYTES,INPUT_BYTES,ROWS,COLS,COUNTER_BYTES

CL_SUCCESS=0;CL_DEVICE_TYPE_GPU=4;CL_DEVICE_NAME=0x102B;CL_DEVICE_VENDOR=0x102C;CL_DRIVER_VERSION=0x102D;CL_DEVICE_EXTENSIONS=0x1030;CL_DEVICE_PCI_BUS_INFO_KHR=0x410F
CL_CONTEXT_PLATFORM=0x1084;CL_PROGRAM_BUILD_LOG=0x1183;CL_PROGRAM_BINARY_SIZES=0x1165;CL_PROGRAM_BINARIES=0x1166
CL_MEM_ALLOC_TYPE_INTEL=0x419A;CL_MEM_ALLOC_BASE_PTR_INTEL=0x419B;CL_MEM_ALLOC_SIZE_INTEL=0x419C;CL_MEM_TYPE_HOST_INTEL=0x4197
SRC=r'''
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define CB 655360UL
inline float b2f(ushort x){return as_float(((uint)x)<<16);}
inline ushort rbf(float x){uint b=as_uint(x);uint l=(b>>16)&1U;b+=0x7fffU+l;return (ushort)(b>>16);}
inline float rbff(float x){return b2f(rbf(x));}
__kernel __attribute__((reqd_work_group_size(256,1,1))) __attribute__((intel_reqd_sub_group_size(8)))
void ph0(__global const uchar* rec,__global const ushort* x,__global ushort* out,__global uint* count){
 int sg=(int)get_sub_group_id(),lane=(int)get_sub_group_local_id(),row=(int)get_group_id(0)*32+sg;
 if(row>=512)return;__global const uchar* code=rec+64;__global const ushort* scale=(__global const ushort*)(rec+64+CB);float p[32];
 #pragma unroll
 for(int v=0;v<32;v++){int pack=lane+8*v,col=pack*8;__global const uchar* s=code+(ulong)row*1280UL+(ulong)pack*5UL;ulong w=(ulong)s[0]|(ulong)s[1]<<8|(ulong)s[2]<<16|(ulong)s[3]<<24|(ulong)s[4]<<32;float a=0.0f,sc=b2f(scale[row*16+(col>>7)]);
  #pragma unroll
  for(int k=0;k<8;k++){int q=(int)((w>>(5*k))&31UL)-15;a=fma(rbff((float)q*sc),b2f(x[col+k]),a);}p[v]=a;}
 #pragma unroll
 for(int d=16;d>=1;d>>=1){#pragma unroll for(int i=0;i<d;i++)p[i]=p[i]+p[i+d];}
 float z=p[0];
 #pragma unroll
 for(int d=4;d>=1;d>>=1){float o=intel_sub_group_shuffle_down(z,z,(uint)d);if(lane<d)z=z+o;}
 if(lane==0){out[row]=rbf(z);atomic_inc((volatile __global atomic_uint*)&count[row]);}
}'''

class PCI(C.Structure):_fields_=[('domain',C.c_uint),('bus',C.c_uint),('device',C.c_uint),('function',C.c_uint)]
def _check(code,op):
 if code!=CL_SUCCESS:raise RuntimeError(f'{op}:{code}')
def _sha(data):return hashlib.sha256(data).hexdigest()

class Intel:
 def __init__(self):
  self.l=C.WinDLL('OpenCL.dll');self.context=self.queue=self.program=self.kernel=None;self.ptrs=[];self.events=[];self.ledger=[];self._bind()
 def _bind(self):
  l=self.l;V=C.c_void_p;U=C.c_uint;S=C.c_size_t;I=C.c_int
  l.clGetPlatformIDs.argtypes=[U,C.POINTER(V),C.POINTER(U)];l.clGetPlatformIDs.restype=I
  l.clGetDeviceIDs.argtypes=[V,C.c_ulonglong,U,C.POINTER(V),C.POINTER(U)];l.clGetDeviceIDs.restype=I
  l.clGetDeviceInfo.argtypes=[V,U,S,V,C.POINTER(S)];l.clGetDeviceInfo.restype=I
  l.clGetExtensionFunctionAddressForPlatform.argtypes=[V,C.c_char_p];l.clGetExtensionFunctionAddressForPlatform.restype=V
  l.clCreateContext.argtypes=[C.POINTER(C.c_ssize_t),U,C.POINTER(V),V,V,C.POINTER(I)];l.clCreateContext.restype=V
  l.clCreateCommandQueue.argtypes=[V,V,C.c_ulonglong,C.POINTER(I)];l.clCreateCommandQueue.restype=V
  l.clCreateProgramWithSource.argtypes=[V,U,C.POINTER(C.c_char_p),C.POINTER(S),C.POINTER(I)];l.clCreateProgramWithSource.restype=V
  l.clBuildProgram.argtypes=[V,U,C.POINTER(V),C.c_char_p,V,V];l.clBuildProgram.restype=I
  l.clGetProgramBuildInfo.argtypes=[V,V,U,S,V,C.POINTER(S)];l.clGetProgramBuildInfo.restype=I
  l.clGetProgramInfo.argtypes=[V,U,S,V,C.POINTER(S)];l.clGetProgramInfo.restype=I
  l.clCreateKernel.argtypes=[V,C.c_char_p,C.POINTER(I)];l.clCreateKernel.restype=V
  l.clEnqueueNDRangeKernel.argtypes=[V,V,U,V,C.POINTER(S),C.POINTER(S),U,V,C.POINTER(V)];l.clEnqueueNDRangeKernel.restype=I
  l.clFinish.argtypes=[V];l.clFinish.restype=I
  for n in ('clReleaseEvent','clReleaseKernel','clReleaseProgram','clReleaseCommandQueue','clReleaseContext'):
   getattr(l,n).argtypes=[V];getattr(l,n).restype=I
 def info(self,dev,param):
  n=C.c_size_t();_check(self.l.clGetDeviceInfo(dev,param,0,None,C.byref(n)),'info_size');b=C.create_string_buffer(n.value);_check(self.l.clGetDeviceInfo(dev,param,n.value,b,None),'info');return b.value.decode(errors='replace')
 def select(self):
  n=C.c_uint();_check(self.l.clGetPlatformIDs(0,None,C.byref(n)),'platform_count');ps=(C.c_void_p*n.value)();_check(self.l.clGetPlatformIDs(n.value,ps,None),'platforms');found=[]
  for pv in ps:
   p=C.c_void_p(pv);nd=C.c_uint();code=self.l.clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,0,None,C.byref(nd))
   if code:continue
   ds=(C.c_void_p*nd.value)();_check(self.l.clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,nd.value,ds,None),'devices')
   for dv in ds:
    d=C.c_void_p(dv);name=self.info(d,CL_DEVICE_NAME);ext=self.info(d,CL_DEVICE_EXTENSIONS)
    if name=='Intel(R) Arc(TM) Pro 140T GPU (32GB)' and 'cl_intel_unified_shared_memory' in ext:found.append((p,d,ext))
  if len(found)!=1:raise RuntimeError(f'intel_cardinality:{len(found)}')
  p,d,ext=found[0];pci=PCI();_check(self.l.clGetDeviceInfo(d,CL_DEVICE_PCI_BUS_INFO_KHR,C.sizeof(pci),C.byref(pci),None),'pci')
  identity={'name':self.info(d,CL_DEVICE_NAME),'vendor':self.info(d,CL_DEVICE_VENDOR),'driver':self.info(d,CL_DRIVER_VERSION),'pci':f'{pci.domain:04x}:{pci.bus:02x}:{pci.device:02x}.{pci.function}','extensions':ext.split()}
  if identity['pci']!='0000:00:02.0':raise RuntimeError('intel_pci')
  return p,d,identity
 def extfn(self,p,name,proto):
  a=self.l.clGetExtensionFunctionAddressForPlatform(p,name.encode());
  if not a:raise RuntimeError(f'missing:{name}')
  return proto(a)
 def run(self,record:bytes,input_bytes:bytes):
  p=d=None;identity={};outputs=counters=b'';binary=log=b''
  try:
   p,d,identity=self.select();E=C.c_int();props=(C.c_ssize_t*3)(CL_CONTEXT_PLATFORM,int(p.value),0);devs=(C.c_void_p*1)(d.value)
   self.context=self.l.clCreateContext(props,1,devs,None,None,C.byref(E));_check(E.value,'context');self.queue=self.l.clCreateCommandQueue(self.context,d,0,C.byref(E));_check(E.value,'queue')
   src=SRC.encode();ss=(C.c_char_p*1)(src);nn=(C.c_size_t*1)(len(src));self.program=self.l.clCreateProgramWithSource(self.context,1,ss,nn,C.byref(E));_check(E.value,'program')
   code=self.l.clBuildProgram(self.program,1,devs,b'-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt',None,None);n=C.c_size_t();self.l.clGetProgramBuildInfo(self.program,d,CL_PROGRAM_BUILD_LOG,0,None,C.byref(n));bb=C.create_string_buffer(n.value);self.l.clGetProgramBuildInfo(self.program,d,CL_PROGRAM_BUILD_LOG,n.value,bb,None);log=bb.raw
   _check(code,'build');self.kernel=self.l.clCreateKernel(self.program,b'ph0',C.byref(E));_check(E.value,'kernel')
   host_alloc=self.extfn(p,'clHostMemAllocINTEL',C.WINFUNCTYPE(C.c_void_p,C.c_void_p,C.POINTER(C.c_ssize_t),C.c_size_t,C.c_uint,C.POINTER(C.c_int)))
   mem_free=self.extfn(p,'clMemFreeINTEL',C.WINFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p));setptr=self.extfn(p,'clSetKernelArgMemPointerINTEL',C.WINFUNCTYPE(C.c_int,C.c_void_p,C.c_uint,C.c_void_p));getinfo=self.extfn(p,'clGetMemAllocInfoINTEL',C.WINFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p,C.c_uint,C.c_size_t,C.c_void_p,C.POINTER(C.c_size_t)))
   for name,size in (('record',RECORD_BYTES),('input',INPUT_BYTES),('output',ROWS*2),('counters',COUNTER_BYTES)):
    q=int(host_alloc(self.context,None,size,4096,C.byref(E)));_check(E.value,f'alloc:{name}');self.ptrs.append((name,q,size,mem_free));C.memset(q,0xff if name=='output' else 0,size);self.ledger.append({'kind':'usm','name':name,'bytes':size,'pointer':q})
    typ=C.c_uint();base=C.c_void_p();sz=C.c_size_t();_check(getinfo(self.context,C.c_void_p(q),CL_MEM_ALLOC_TYPE_INTEL,C.sizeof(typ),C.byref(typ),None),'alloc_type');_check(getinfo(self.context,C.c_void_p(q),CL_MEM_ALLOC_BASE_PTR_INTEL,C.sizeof(base),C.byref(base),None),'alloc_base');_check(getinfo(self.context,C.c_void_p(q),CL_MEM_ALLOC_SIZE_INTEL,C.sizeof(sz),C.byref(sz),None),'alloc_size')
    if typ.value!=CL_MEM_TYPE_HOST_INTEL or base.value!=q or sz.value!=size:raise RuntimeError('usm_attestation')
   C.memmove(self.ptrs[0][1],record,len(record));C.memmove(self.ptrs[1][1],input_bytes,len(input_bytes))
   for i,(_,q,_,_) in enumerate(self.ptrs):_check(setptr(self.kernel,i,C.c_void_p(q)),f'arg{i}')
   g=(C.c_size_t*1)(16*256);loc=(C.c_size_t*1)(256);ev=C.c_void_p();_check(self.l.clEnqueueNDRangeKernel(self.queue,self.kernel,1,None,g,loc,0,None,C.byref(ev)),'enqueue');self.events.append(ev);_check(self.l.clFinish(self.queue),'finish')
   outputs=C.string_at(self.ptrs[2][1],ROWS*2);counters=C.string_at(self.ptrs[3][1],COUNTER_BYTES)
   sizes=(C.c_size_t*1)();_check(self.l.clGetProgramInfo(self.program,CL_PROGRAM_BINARY_SIZES,C.sizeof(sizes),sizes,None),'binary_size');buf=C.create_string_buffer(sizes[0]);arr=(C.c_void_p*1)(C.cast(buf,C.c_void_p));_check(self.l.clGetProgramInfo(self.program,CL_PROGRAM_BINARIES,C.sizeof(arr),arr,None),'binary');binary=buf.raw
   return {'identity':identity,'output_hex':outputs.hex(),'counters_hex':counters.hex(),'source':SRC,'source_sha256':_sha(SRC.encode()),'binary_hex':binary.hex(),'binary_sha256':_sha(binary),'build_log_hex':log.hex(),'ledger':self.ledger,'forbidden_copy_calls':0,'enqueue_calls':1}
  finally:self.close()
 def close(self):
  errors=[]
  for ev in self.events:
   try:_check(self.l.clReleaseEvent(ev),'release_event');self.ledger.append({'release':'event','code':0})
   except Exception as e:errors.append(str(e))
  for name,q,_,free in reversed(self.ptrs):
   try:code=free(self.context,C.c_void_p(q));_check(code,'free');self.ledger.append({'release':name,'code':code})
   except Exception as e:errors.append(str(e))
  for name,h,fn in (('kernel',self.kernel,'clReleaseKernel'),('program',self.program,'clReleaseProgram'),('queue',self.queue,'clReleaseCommandQueue'),('context',self.context,'clReleaseContext')):
   if h:
    try:code=getattr(self.l,fn)(h);_check(code,fn);self.ledger.append({'release':name,'code':code})
    except Exception as e:errors.append(str(e))
  self.ledger.append({'cleanup_complete':not errors,'errors':errors})

def run(record,input_bytes):return Intel().run(record,input_bytes)
