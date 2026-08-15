#!/usr/bin/env python3
"""R6 lifecycle wrapper: immediate pending ownership and safe telemetry."""
from __future__ import annotations
import ctypes as C,time
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
  super().__init__();self.api_counts={name:0 for name in FORBIDDEN};self.ownership={};self.extension_counts={name:0 for name in EXTENSION_ABI};self.host_pointers=set();self.pending_kernels=[];self.pending_usm=[];self.mem_free_fn=None;self.ownership_ledger=[];self.telemetry_errors=[]
 def extension(self,platform,name,prototype):
  if name not in EXTENSION_ABI:raise RuntimeError('unregistered_extension:'+name)
  result,args=EXTENSION_ABI[name]
  if prototype._restype_ is not result or tuple(prototype._argtypes_)!=args:raise RuntimeError('extension_abi:'+name)
  raw=super().extension(platform,name,prototype)
  def wrapped(*values):
    self.extension_counts[name]+=1;row={'api':name,'attempted':True,'returned':None,'registered_pending':False,'exception':None};self.ownership_ledger.append(row)
    try:result=raw(*values)
    except Exception as exc:row['exception']=f'{type(exc).__name__}:{exc}';raise
    row['returned']=int(result or 0)
    if name=='clHostMemAllocINTEL':
     pointer=int(result or 0)
     if pointer and pointer not in self.host_pointers:self.pending_usm.append(pointer);row['registered_pending']=True
     if pointer==0 or pointer in self.host_pointers:raise RuntimeError('alloc_zero_or_alias')
     self.host_pointers.add(pointer)
    return result
  if name=='clMemFreeINTEL':self.mem_free_fn=wrapped
  return wrapped
 def bind(self):
  super().bind()
  for name in ('clCreateContext','clCreateCommandQueue','clCreateProgramWithBinary','clCreateKernel'):
   raw=getattr(self.lib,name)
   def wrapped(*args,_name=name,_raw=raw):
    row={'api':_name,'attempted':True,'returned':None,'registered_pending':False,'exception':None};self.ownership_ledger.append(row)
    try:result=_raw(*args)
    except Exception as exc:row['exception']=f'{type(exc).__name__}:{exc}';raise
    handle=int(result or 0);row['returned']=handle
    if handle:
     if _name=='clCreateContext':self.context=result
     elif _name=='clCreateCommandQueue':self.queue=result
     elif _name=='clCreateProgramWithBinary':self.program=result
     else:self.pending_kernels.append(handle)
     row['registered_pending']=True
    return result
   setattr(self.lib,name,wrapped)
  orig=self.lib.clEnqueueNDRangeKernel
  def enqueue(*args):
   self.sample('pre_launch:'+str(sum(x.get('op')=='enqueue' for x in self.ledger)));code=orig(*args);self.sample('post_launch:'+str(sum(x.get('op')=='enqueue' for x in self.ledger)));return code
  self.lib.clEnqueueNDRangeKernel=enqueue;finish=self.lib.clFinish
  def finish_wrapper(*args):self.sample('pre_finish');code=finish(*args);self.sample('post_finish');return code
  self.lib.clFinish=finish_wrapper
 def close(self):
  attempted=[]
  def release(name,call):
   row={'op':'release','name':name,'attempt_index':len(attempted),'attempted':True,'code':None,'exception':None,'owned_before':True,'owned_after':True};attempted.append(row);self.ledger.append(row)
   try:
    row['code']=int(call())
    if row['code']==0:row['owned_after']=False
    else:self.cleanup_errors.append(f'{name}:code:{row["code"]}')
   except Exception as exc:row['exception']=f'{type(exc).__name__}:{exc}';self.cleanup_errors.append(f'{name}:{row["exception"]}')
  promoted_usm={p for _n,p,_z,_f in self.allocations}
  for pointer in reversed([p for p in self.pending_usm if p not in promoted_usm]):release('pending_usm:'+str(pointer),lambda p=pointer:self.mem_free_fn(self.context,C.c_void_p(p)))
  for name,pointer,_size,free in reversed(self.allocations):release('usm:'+name,lambda p=pointer,f=free:f(self.context,C.c_void_p(p)))
  promoted_kernels={int(h) for _n,h in self.kernels}
  for handle in reversed([h for h in self.pending_kernels if h not in promoted_kernels]):release('pending_kernel:'+str(handle),lambda h=handle:self.lib.clReleaseKernel(C.c_void_p(h)))
  for name,handle in reversed(self.kernels):release('kernel:'+name,lambda h=handle:self.lib.clReleaseKernel(h))
  for name,handle,fn in (('program',self.program,'clReleaseProgram'),('queue',self.queue,'clReleaseCommandQueue'),('context',self.context,'clReleaseContext')):
   if handle:release(name,lambda h=handle,f=fn:getattr(self.lib,f)(h))
  live=[row['name'] for row in attempted if row['owned_after']]
  self.ledger.append({'op':'cleanup','cleanup_complete':not live,'errors':list(self.cleanup_errors),'release_attempts':len(attempted),'live_owned_resources':len(live),'live_resource_names':live})
  self.sample('post_cleanup')
 def sample(self,stage):
  try:
   import psutil;p=psutil.Process();m=p.memory_info();self.ledger.append({'op':'resource_sample','stage':stage,'qpc_ns':time.perf_counter_ns(),'available':psutil.virtual_memory().available,'rss':m.rss,'peak_wset':m.peak_wset,'telemetry_error':None});return True
  except Exception as exc:
   row={'op':'resource_sample','stage':stage,'qpc_ns':time.perf_counter_ns(),'available':None,'rss':None,'peak_wset':None,'telemetry_error':f'{type(exc).__name__}:{exc}'};self.ledger.append(row);self.telemetry_errors.append(row['telemetry_error']);return False
 def crosslink(self):
  create=[x for x in self.ownership_ledger if x['api'] in ('clCreateContext','clCreateCommandQueue','clCreateProgramWithBinary','clCreateKernel')]
  names=['context','queue','program']+[f'kernel:{n}' for n in ('gate_linear','up_linear','activation','down_linear')]
  for row,name in zip(create,names):row.update({'object_name':name,'object_pointer':row['returned'],'promoted':True,'release_name':name})
  hosts=[x for x in self.ownership_ledger if x['api']=='clHostMemAllocINTEL']
  for row,(name,pointer,_size,_free) in zip(hosts,self.allocations):row.update({'object_name':'usm:'+name,'object_pointer':pointer,'promoted':True,'release_name':'usm:'+name})
  frees=[x for x in self.ownership_ledger if x['api']=='clMemFreeINTEL']
  for row,(name,pointer,_size,_free) in zip(frees,reversed(self.allocations)):row.update({'object_name':'usm:'+name,'object_pointer':pointer,'promoted':True,'release_name':'usm:'+name})
  program=next((x for x in self.ledger if x.get('op')=='program_create_binary'),None)
  if program is not None and len(create)>=3:program['pointer']=create[2]['returned']
 def run(self,records,input_bytes,lut,authorization):
  self.sample('backend_entry')
  try:evidence=super().run(records,input_bytes,lut,authorization)
  except ExecutionFailure as exc:
   self.crosslink();exc.evidence['forbidden_calls']={name:sum(row.get('api')==name for row in self.ledger) for name in FORBIDDEN};exc.evidence['extension_counts']=dict(self.extension_counts);exc.evidence['ownership_ledger']=self.ownership_ledger;exc.evidence['telemetry_errors']=self.telemetry_errors;raise
  self.crosslink();evidence['forbidden_calls']={name:sum(row.get('api')==name for row in self.ledger) for name in FORBIDDEN};evidence['extension_counts']=dict(self.extension_counts);evidence['ownership_ledger']=self.ownership_ledger;evidence['telemetry_errors']=self.telemetry_errors;return evidence
