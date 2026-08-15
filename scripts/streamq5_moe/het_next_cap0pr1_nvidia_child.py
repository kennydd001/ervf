#!/usr/bin/env python3
"""CAP0P NVIDIA child: isolated CuPy/NVRTC owner."""
import argparse,ctypes as C,hashlib,os,threading,traceback
from het_next_cap0p_common import digest,expected,now,words\nfrom het_next_cap0pr1_child_common import args,receive,send
SOURCE=r'''extern "C" __global__ void cap0p(unsigned int*x,unsigned int n){unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){unsigned int v=x[i]+0x9E3779B9u;x[i]=((v>>11u)|(v<<21u))^0xC3C3C3C3u;}}'''
def pin(lp):
 k=C.WinDLL('kernel32',use_last_error=True);k.GetCurrentThread.restype=C.c_void_p;k.SetThreadAffinityMask.argtypes=[C.c_void_p,C.c_size_t];k.SetThreadAffinityMask.restype=C.c_size_t
 if not k.SetThreadAffinityMask(k.GetCurrentThread(),1<<lp):raise OSError(C.get_last_error(),'affinity')
 return {'lp':lp,'tid':threading.get_native_id()}
class Backend:
 def __init__(self):
  import cupy as cp
  self.cp=cp;self.stream=cp.cuda.Stream(non_blocking=True);self.pinned=cp.cuda.alloc_pinned_memory(4096);self.device=cp.cuda.alloc(4096);self.module=cp.RawModule(code=SOURCE,options=('--std=c++14','--fmad=false'),name_expressions=('cap0p',));self.kernel=self.module.get_function('cap0p');props=cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice());name=props['name'].decode() if isinstance(props['name'],bytes) else str(props['name']);pci=cp.cuda.runtime.deviceGetPCIBusId(cp.cuda.runtime.getDevice());pci=pci.decode() if isinstance(pci,bytes) else str(pci);self.identity={'name':name,'pci':pci,'driver':int(cp.cuda.runtime.driverGetVersion()),'runtime':int(cp.cuda.runtime.runtimeGetVersion()),'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'api':'CuPy_RawModule_NVRTC_explicit_nondefault_stream','bytes':4096}
 def run(self,x):
  cp=self.cp;raw=__import__('struct').pack('<1024I',*x);C.memmove(self.pinned.ptr,raw,4096);a=now()
  with self.stream:
   cp.cuda.runtime.memcpyAsync(self.device.ptr,self.pinned.ptr,4096,cp.cuda.runtime.memcpyHostToDevice,self.stream.ptr);self.kernel((4,),(256,),(self.device.ptr,1024),stream=self.stream);cp.cuda.runtime.memcpyAsync(self.pinned.ptr,self.device.ptr,4096,cp.cuda.runtime.memcpyDeviceToHost,self.stream.ptr)
  self.stream.synchronize();b=now();o=list(__import__('struct').unpack('<1024I',C.string_at(self.pinned.ptr,4096)));return a,b,o
 def close(self):
  rows=[]
  for name,fn in (('stream_sync',self.stream.synchronize),('device_free',self.device.mem.free),('pinned_free',self.pinned.mem.free)):
   try:fn();rows.append({'resource':name,'code':0})
   except Exception as e:rows.append({'resource':name,'error':str(e)})
  self.cp.get_default_memory_pool().free_all_blocks();self.cp.get_default_pinned_memory_pool().free_all_blocks();rows.append({'resource':'pools','used_bytes':int(self.cp.get_default_memory_pool().used_bytes())});return rows
def main():
 z=args();top=pin(z.lp);be=None
 try:
  be=Backend();send(z,0,'ready',{'device':'nvidia','pid':os.getpid(),'topology':top,'identity':be.identity});x=words();want=expected('nvidia',x)
  for epoch in range(1,4):
   receive({'cmd':'START','epoch':epoch,'nonce':z.nonce})
   s,d,o=be.run(x);send(z,epoch,'result',{'device':'nvidia','epoch':epoch,'submit_ns':s,'done_ns':d,'output_words':o,'sha256':digest(o),'different_words':sum(i!=j for i,j in zip(o,want))})
  if command()!=['STOP']:raise RuntimeError('stop')
  cleanup=be.close();be=None;emit({'type':'cleanup','device':'nvidia','rows':cleanup});return 0
 except BaseException as e:send(z,99,'failure',{'device':'nvidia','error':f'{type(e).__name__}:{e}','traceback':traceback.format_exc()});return 2
 finally:
  if be is not None:be.close()
if __name__=='__main__':raise SystemExit(main())


