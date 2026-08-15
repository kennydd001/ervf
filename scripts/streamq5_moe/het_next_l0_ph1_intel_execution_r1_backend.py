#!/usr/bin/env python3
"""R1 audited lifecycle wrapper over the frozen R0 Intel execution backend."""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import het_next_l0_ph1_intel_execution_r0_backend as r0

ROOT=Path(__file__).resolve().parents[2];BUFFER_TABLE=r0.BUFFER_TABLE;ARGUMENT_MAPS=r0.ARGUMENT_MAPS;LAUNCHES=r0.LAUNCHES
SOURCE_SHA=r0.SOURCE_SHA;BINARY_SHA=r0.BINARY_SHA;BINARY_BYTES=r0.BINARY_BYTES;ExecutionFailure=r0.ExecutionFailure
EXTENSION_ABI={
 'clHostMemAllocINTEL':(C.c_void_p,(C.c_void_p,C.POINTER(C.c_ssize_t),C.c_size_t,C.c_uint,C.POINTER(C.c_int))),
 'clMemFreeINTEL':(C.c_int,(C.c_void_p,C.c_void_p)),
 'clSetKernelArgMemPointerINTEL':(C.c_int,(C.c_void_p,C.c_uint,C.c_void_p)),
 'clGetMemAllocInfoINTEL':(C.c_int,(C.c_void_p,C.c_void_p,C.c_uint,C.c_size_t,C.c_void_p,C.POINTER(C.c_size_t))),}
FORBIDDEN=('clCreateBuffer','clEnqueueReadBuffer','clEnqueueWriteBuffer','clEnqueueCopyBuffer','clEnqueueMigrateMemObjects','clEnqueueMemAdviseINTEL')

class Backend(r0.Backend):
 def __init__(self):
  super().__init__();self.api_counts={name:0 for name in FORBIDDEN};self.ownership={};self.extension_counts={name:0 for name in EXTENSION_ABI}
 def extension(self,platform,name,prototype):
  if name not in EXTENSION_ABI:raise RuntimeError('unregistered_extension:'+name)
  result,args=EXTENSION_ABI[name]
  if prototype._restype_ is not result or tuple(prototype._argtypes_)!=args:raise RuntimeError('extension_abi:'+name)
  raw=super().extension(platform,name,prototype)
  def wrapped(*values):
   self.extension_counts[name]+=1;return raw(*values)
  return wrapped
 def close(self):
  attempted=[]
  def release(name,call):
   row={'op':'release','name':name,'attempt_index':len(attempted),'attempted':True,'code':None,'exception':None,'owned_before':True,'owned_after':True};attempted.append(row);self.ledger.append(row)
   try:
    row['code']=int(call())
    if row['code']==0:row['owned_after']=False
    else:self.cleanup_errors.append(f'{name}:code:{row["code"]}')
   except Exception as exc:row['exception']=f'{type(exc).__name__}:{exc}';self.cleanup_errors.append(f'{name}:{row["exception"]}')
  for name,pointer,_size,free in reversed(self.allocations):release('usm:'+name,lambda p=pointer,f=free:f(self.context,C.c_void_p(p)))
  for name,handle in reversed(self.kernels):release('kernel:'+name,lambda h=handle:self.lib.clReleaseKernel(h))
  for name,handle,fn in (('program',self.program,'clReleaseProgram'),('queue',self.queue,'clReleaseCommandQueue'),('context',self.context,'clReleaseContext')):
   if handle:release(name,lambda h=handle,f=fn:getattr(self.lib,f)(h))
  live=[row['name'] for row in attempted if row['owned_after']]
 self.ledger.append({'op':'cleanup','cleanup_complete':not live,'errors':list(self.cleanup_errors),'release_attempts':len(attempted),'live_owned_resources':len(live),'live_resource_names':live})
 def run(self,records,input_bytes,lut,authorization):
  try:evidence=super().run(records,input_bytes,lut,authorization)
  except ExecutionFailure as exc:
   exc.evidence['forbidden_calls']={name:sum(row.get('api')==name for row in self.ledger) for name in FORBIDDEN};exc.evidence['extension_counts']=dict(self.extension_counts);raise
  evidence['forbidden_calls']={name:sum(row.get('api')==name for row in self.ledger) for name in FORBIDDEN};evidence['extension_counts']=dict(self.extension_counts);return evidence
