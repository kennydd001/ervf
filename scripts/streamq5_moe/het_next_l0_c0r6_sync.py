#!/usr/bin/env python3
"""Shared C0-R6 Win32 event/Interlocked protocol and mock implementation."""
from __future__ import annotations
import threading,time

ACTIVE_ORDER=('intel','nvidia');WAIT_MS=30000

class MockEvent:
 def __init__(self,manual,initial=False,name=''):self.manual=manual;self.state=initial;self.name=name
class MockPrimitives:
 def __init__(self):self.calls=[];self.qpc_value=0
 def event(self,manual,initial,name):self.calls.append(('CreateEventW',name,manual,initial));return MockEvent(manual,initial,name)
 def set(self,e):e.state=True;self.calls.append(('SetEvent',e.name));return True
 def reset(self,e):e.state=False;self.calls.append(('ResetEvent',e.name));return True
 def wait_one(self,e,ms=WAIT_MS):
  self.calls.append(('WaitForSingleObject',e.name,ms));ok=e.state
  if ok and not e.manual:e.state=False
  return ok
 def wait_all(self,events,ms=WAIT_MS):self.calls.append(('WaitForMultipleObjects',[e.name for e in events],ms));return all(e.state for e in events)
 def exchange(self,cell,value):old=cell[0];cell[0]=value;self.calls.append(('InterlockedExchange64',old,value));return old
 def read(self,cell):v=cell[0];self.calls.append(('InterlockedCompareExchange64',v));return v
 def barrier(self):self.calls.append(('MemoryBarrier',))
 def qpc(self):self.qpc_value+=100;self.calls.append(('QueryPerformanceCounter',self.qpc_value));return self.qpc_value
 def close(self,e):self.calls.append(('CloseHandle',e.name))

class WinPrimitives:
 def __init__(self):
  import ctypes as C
  from ctypes import wintypes
  self.C=C;self.w=wintypes;self.k=C.WinDLL('kernel32',use_last_error=True);self.calls=[]
  self.k.CreateEventW.argtypes=[C.c_void_p,wintypes.BOOL,wintypes.BOOL,wintypes.LPCWSTR];self.k.CreateEventW.restype=wintypes.HANDLE
  self.k.SetEvent.argtypes=[wintypes.HANDLE];self.k.SetEvent.restype=wintypes.BOOL;self.k.ResetEvent.argtypes=[wintypes.HANDLE];self.k.ResetEvent.restype=wintypes.BOOL
  self.k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];self.k.WaitForSingleObject.restype=wintypes.DWORD
  self.k.WaitForMultipleObjects.argtypes=[wintypes.DWORD,C.POINTER(wintypes.HANDLE),wintypes.BOOL,wintypes.DWORD];self.k.WaitForMultipleObjects.restype=wintypes.DWORD
  self.k.CloseHandle.argtypes=[wintypes.HANDLE];self.k.CloseHandle.restype=wintypes.BOOL;self.k.QueryPerformanceCounter.argtypes=[C.POINTER(C.c_longlong)];self.k.QueryPerformanceCounter.restype=wintypes.BOOL
  self.k.InterlockedExchange64.argtypes=[C.POINTER(C.c_longlong),C.c_longlong];self.k.InterlockedExchange64.restype=C.c_longlong;self.k.InterlockedCompareExchange64.argtypes=[C.POINTER(C.c_longlong),C.c_longlong,C.c_longlong];self.k.InterlockedCompareExchange64.restype=C.c_longlong
 def event(self,manual,initial,name):
  h=self.k.CreateEventW(None,bool(manual),bool(initial),name)
  if not h:raise OSError(self.C.get_last_error(),'CreateEventW')
  self.calls.append(('CreateEventW',name,bool(manual),bool(initial)));return h
 def set(self,e):
  if not self.k.SetEvent(e):raise OSError(self.C.get_last_error(),'SetEvent')
  self.calls.append(('SetEvent',int(e)));return True
 def reset(self,e):
  if not self.k.ResetEvent(e):raise OSError(self.C.get_last_error(),'ResetEvent')
  self.calls.append(('ResetEvent',int(e)));return True
 def wait_one(self,e,ms=WAIT_MS):
  r=int(self.k.WaitForSingleObject(e,ms));self.calls.append(('WaitForSingleObject',int(e),ms,r));return r==0
 def wait_all(self,events,ms=WAIT_MS):
  a=(self.w.HANDLE*len(events))(*events);r=int(self.k.WaitForMultipleObjects(len(events),a,True,ms));self.calls.append(('WaitForMultipleObjects',[int(x) for x in events],ms,r));return r==0
 def exchange(self,cell,value):
  old=int(self.k.InterlockedExchange64(self.C.byref(cell),value));self.calls.append(('InterlockedExchange64',old,value));return old
 def read(self,cell):
  v=int(self.k.InterlockedCompareExchange64(self.C.byref(cell),0,0));self.calls.append(('InterlockedCompareExchange64',v));return v
 def barrier(self):self.read(self._barrier_cell)
 _barrier_cell=__import__('ctypes').c_longlong(0)
 def qpc(self):
  v=self.C.c_longlong();
  if not self.k.QueryPerformanceCounter(self.C.byref(v)):raise OSError(self.C.get_last_error(),'QPC')
  self.calls.append(('QueryPerformanceCounter',int(v.value)));return int(v.value)
 def close(self,e):
  if e and not self.k.CloseHandle(e):raise OSError(self.C.get_last_error(),'CloseHandle')
  self.calls.append(('CloseHandle',int(e) if e else 0))

class Cell:
 def __init__(self,p,value=0):self.p=p;self.value=[value] if isinstance(p,MockPrimitives) else p.C.c_longlong(value)
 def read(self):return self.p.read(self.value)
 def set(self,v):return self.p.exchange(self.value,v)

class Channel:
 def __init__(self,p,name):
  self.name=name;self.command=p.event(False,False,f'c0r6_{name}_command');self.ready=p.event(True,False,f'c0r6_{name}_ready');self.done=p.event(True,False,f'c0r6_{name}_done');self.stop=p.event(True,False,f'c0r6_{name}_stop');self.last=Cell(p);self.ack=Cell(p);self.descriptor={};self.output=None
class Protocol:
 def __init__(self,p):
  self.p=p;self.channels={n:Channel(p,n) for n in ACTIVE_ORDER};self.start=p.event(True,False,'c0r6_start');self.epoch=0;self._lock=threading.RLock();self.log=[]
 def publish(self,active,arm):
  active=tuple(active)
  if not active or any(i not in ACTIVE_ORDER for i in active):raise RuntimeError('active')
  if any(self.channels[i].ack.read()!=self.channels[i].last.read() for i in active):raise RuntimeError('stale_ack')
  self.p.reset(self.start);self.epoch+=1
  with self._lock:
   for i in active:
    c=self.channels[i];self.p.reset(c.command);self.p.reset(c.ready);self.p.reset(c.done);c.descriptor={'epoch':self.epoch,'arm':arm,'active':list(active)};c.output=None;c.last.set(self.epoch);self.p.set(c.command)
  self.log.append({'op':'publish','epoch':self.epoch,'arm':arm,'active':list(active)});return self.epoch
 def worker_descriptor(self,name):
  c=self.channels[name]
  if not self.p.wait_one(c.command):raise TimeoutError('command')
  with self._lock:d=dict(c.descriptor)
  if d.get('epoch')<=c.ack.read():raise RuntimeError('epoch_not_increasing')
  self.p.set(c.ready);self.log.append({'op':'ready','worker':name,'epoch':d['epoch']});return d
 def wait_ready_release(self,active):
  handles=[self.channels[i].ready for i in active]
  if not self.p.wait_all(handles):raise TimeoutError('ready')
  t0=self.p.qpc();self.p.set(self.start);self.log.append({'op':'release','epoch':self.epoch,'active':list(active),'t0':t0});return t0
 def worker_started(self,name,epoch):
  if not self.p.wait_one(self.start):raise TimeoutError('start')
  if self.channels[name].last.read()!=epoch:raise RuntimeError('start_epoch')
 def worker_finish(self,name,epoch,output,telemetry):
  c=self.channels[name];c.output={'result':output,'telemetry':telemetry};self.p.barrier();c.ack.set(epoch);self.p.set(c.done);self.log.append({'op':'ack_done','worker':name,'epoch':epoch})
 def collect(self,active,epoch):
  handles=[self.channels[i].done for i in active]
  if not self.p.wait_all(handles):raise TimeoutError('done')
  for i in active:
   c=self.channels[i]
   if c.ack.read()!=epoch or c.last.read()!=epoch:raise RuntimeError('ack_mismatch')
  outputs={i:self.channels[i].output for i in active}
  if any(v is None for v in outputs.values()):raise RuntimeError('output_before_ack')
  t1=self.p.qpc();self.p.reset(self.start)
  for i in active:self.p.reset(self.channels[i].ready);self.p.reset(self.channels[i].done)
  self.log.append({'op':'collect_reset','epoch':epoch,'active':list(active),'t1':t1});return outputs,t1
 def close(self):
  for c in self.channels.values():
   for e in (c.command,c.ready,c.done,c.stop):self.p.close(e)
  self.p.close(self.start)

def simulate_protocol():
 p=MockPrimitives();q=Protocol(p);outs=[]
 for arm,active in (('A',('nvidia',)),('B',('intel','nvidia')),('S-I',('intel',)),('S-N',('nvidia',)),('A',('nvidia',))):
  e=q.publish(active,arm);desc={i:q.worker_descriptor(i) for i in active};t0=q.wait_ready_release(active)
  for i in active:q.worker_started(i,e);q.worker_finish(i,e,{'sentinel':i},{'finite':True})
  o,t1=q.collect(active,e);outs.append((arm,e,sorted(o),t0,t1))
 negative={}
 q.channels['intel'].ack.set(0)
 try:q.publish(('intel',),'NEG');negative['stale_ack']=False
 except RuntimeError:negative['stale_ack']=True
 try:q.worker_finish('intel',999,{},{});q.collect(('intel',),999);negative['wrong_epoch']=False
 except RuntimeError:negative['wrong_epoch']=True
 e=MockEvent(True,False,'timeout');negative['timeout']=not p.wait_one(e,1)
 q.close();return {'outputs':outs,'log':q.log,'calls':p.calls,'negative':negative,'pass':all(negative.values()) and len(outs)==5}
