#!/usr/bin/env python3
"""CAP0P Intel child: isolated OpenCL host-USM owner."""
import argparse,ctypes as C,hashlib,struct,sys,threading,traceback
from het_next_cap0p_common import command,digest,emit,expected,now,words
SOURCE=r'''__kernel void cap0p(__global uint*x,uint n){uint i=get_global_id(0);if(i<n){uint v=x[i]^0xA5A5A5A5u;x[i]=rotate(v,7u)+0x3C6EF372u;}}'''
def pin(lp):
 k=C.WinDLL('kernel32',use_last_error=True);k.GetCurrentThread.restype=C.c_void_p;k.SetThreadAffinityMask.argtypes=[C.c_void_p,C.c_size_t];k.SetThreadAffinityMask.restype=C.c_size_t
 if not k.SetThreadAffinityMask(k.GetCurrentThread(),1<<lp):raise OSError(C.get_last_error(),'affinity')
 return {'lp':lp,'tid':threading.get_native_id()}
class Backend:
 def __init__(self):
  self.cleanup=[];self.cl=C.WinDLL('OpenCL.dll');cl=self.cl
  def b(n,a,r=C.c_int):f=getattr(cl,n);f.argtypes=a;f.restype=r;return f
  self.gp=b('clGetPlatformIDs',[C.c_uint,C.POINTER(C.c_void_p),C.POINTER(C.c_uint)]);self.gd=b('clGetDeviceIDs',[C.c_void_p,C.c_ulonglong,C.c_uint,C.POINTER(C.c_void_p),C.POINTER(C.c_uint)]);self.gi=b('clGetDeviceInfo',[C.c_void_p,C.c_uint,C.c_size_t,C.c_void_p,C.POINTER(C.c_size_t)]);self.cc=b('clCreateContext',[C.POINTER(C.c_ssize_t),C.c_uint,C.POINTER(C.c_void_p),C.c_void_p,C.c_void_p,C.POINTER(C.c_int)],C.c_void_p);self.cq=b('clCreateCommandQueue',[C.c_void_p,C.c_void_p,C.c_ulonglong,C.POINTER(C.c_int)],C.c_void_p);self.cp=b('clCreateProgramWithSource',[C.c_void_p,C.c_uint,C.POINTER(C.c_char_p),C.POINTER(C.c_size_t),C.POINTER(C.c_int)],C.c_void_p);self.bp=b('clBuildProgram',[C.c_void_p,C.c_uint,C.POINTER(C.c_void_p),C.c_char_p,C.c_void_p,C.c_void_p]);self.ck=b('clCreateKernel',[C.c_void_p,C.c_char_p,C.POINTER(C.c_int)],C.c_void_p);self.sa=b('clSetKernelArg',[C.c_void_p,C.c_uint,C.c_size_t,C.c_void_p]);self.en=b('clEnqueueNDRangeKernel',[C.c_void_p,C.c_void_p,C.c_uint,C.c_void_p,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t),C.c_uint,C.c_void_p,C.c_void_p]);self.fin=b('clFinish',[C.c_void_p]);self.ext=b('clGetExtensionFunctionAddressForPlatform',[C.c_void_p,C.c_char_p],C.c_void_p)
  for n in ('clReleaseKernel','clReleaseProgram','clReleaseCommandQueue','clReleaseContext'):getattr(cl,n).argtypes=[C.c_void_p];getattr(cl,n).restype=C.c_int
  def ok(x,n):
   if x:raise RuntimeError(f'{n}:{x}')
  self.ok=ok;n=C.c_uint();ok(self.gp(0,None,C.byref(n)),'platforms');ps=(C.c_void_p*n.value)();ok(self.gp(n,ps,None),'platforms2');found=[]
  for p in ps:
   d=C.c_uint();
   if self.gd(p,4,0,None,C.byref(d)):continue
   ds=(C.c_void_p*d.value)();ok(self.gd(p,4,d,ds,None),'devices')
   for dev in ds:
    def txt(code):z=C.c_size_t();ok(self.gi(dev,code,0,None,C.byref(z)),'is');q=C.create_string_buffer(z.value);ok(self.gi(dev,code,z.value,q,None),'iv');return q.value.decode(errors='replace')
    if 'Intel' in txt(0x102c) and 'Arc' in txt(0x102b) and 'cl_intel_unified_shared_memory' in txt(0x1030):found.append((p,dev,txt(0x102b),txt(0x102d)))
  if len(found)!=1:raise RuntimeError(f'intel_count:{len(found)}')
  self.p,self.d,name,driver=found[0];e=C.c_int();da=(C.c_void_p*1)(self.d);props=(C.c_ssize_t*3)(0x1084,int(self.p),0);self.ctx=self.cc(props,1,da,None,None,C.byref(e));ok(e.value,'ctx');self.cleanup.append(('context',lambda:self.cl.clReleaseContext(self.ctx)));self.q=self.cq(self.ctx,self.d,2,C.byref(e));ok(e.value,'queue');self.cleanup.append(('queue',lambda:self.cl.clReleaseCommandQueue(self.q)));raw=SOURCE.encode();rp=C.c_char_p(raw);rl=C.c_size_t(len(raw));self.prog=self.cp(self.ctx,1,C.byref(rp),C.byref(rl),C.byref(e));ok(e.value,'program');self.cleanup.append(('program',lambda:self.cl.clReleaseProgram(self.prog)));ok(self.bp(self.prog,1,da,b'-cl-std=CL2.0',None,None),'build');self.k=self.ck(self.prog,b'cap0p',C.byref(e));ok(e.value,'kernel');self.cleanup.append(('kernel',lambda:self.cl.clReleaseKernel(self.k)))
  def xf(n,r,*a):addr=self.ext(self.p,n.encode());
  alloc=C.WINFUNCTYPE(C.c_void_p,C.c_void_p,C.POINTER(C.c_longlong),C.c_size_t,C.c_uint,C.POINTER(C.c_int))(self.ext(self.p,b'clHostMemAllocINTEL'));self.free=C.WINFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p)(self.ext(self.p,b'clMemFreeINTEL'));setp=C.WINFUNCTYPE(C.c_int,C.c_void_p,C.c_uint,C.c_void_p)(self.ext(self.p,b'clSetKernelArgMemPointerINTEL'));self.ptr=alloc(self.ctx,None,4096,4096,C.byref(e));ok(e.value,'usm');self.cleanup.append(('host_usm',lambda:self.free(self.ctx,self.ptr)));ok(setp(self.k,0,self.ptr),'setptr');count=C.c_uint(1024);ok(self.sa(self.k,1,C.sizeof(count),C.byref(count)),'count');self.identity={'name':name,'driver':driver,'source_sha256':hashlib.sha256(raw).hexdigest(),'api':'OpenCL_host_USM','bytes':4096,'pci':'unavailable_if_extension_absent'}
 def run(self,x):
  C.memmove(self.ptr,struct.pack('<1024I',*x),4096);g,l=C.c_size_t(1024),C.c_size_t(256);a=now();self.ok(self.en(self.q,self.k,1,None,C.byref(g),C.byref(l),0,None,None),'enqueue');self.ok(self.fin(self.q),'finish');b=now();o=list(struct.unpack('<1024I',C.string_at(self.ptr,4096)));return a,b,o
 def close(self):
  out=[]
  for n,f in reversed(self.cleanup):
   try:c=int(f());out.append({'resource':n,'code':c})
   except Exception as e:out.append({'resource':n,'error':str(e)})
  return out
def main():
 a=argparse.ArgumentParser();a.add_argument('--lp',type=int,required=True);z=a.parse_args();top=pin(z.lp);be=None
 try:
  be=Backend();emit({'type':'ready','device':'intel','pid':__import__('os').getpid(),'topology':top,'identity':be.identity})
  x=words();want=expected('intel',x)
  for epoch in range(1,4):
   c=command();
   if c!=['START',str(epoch)]:raise RuntimeError('command')
   s,d,o=be.run(x);emit({'type':'result','device':'intel','epoch':epoch,'submit_ns':s,'done_ns':d,'output_words':o,'sha256':digest(o),'different_words':sum(i!=j for i,j in zip(o,want))})
  if command()!=['STOP']:raise RuntimeError('stop');cleanup=be.close();be=None;emit({'type':'cleanup','device':'intel','rows':cleanup});return 0
 except BaseException as e:emit({'type':'failure','device':'intel','error':f'{type(e).__name__}:{e}','traceback':traceback.format_exc()});return 2
 finally:
  if be is not None:be.close()
if __name__=='__main__':raise SystemExit(main())
