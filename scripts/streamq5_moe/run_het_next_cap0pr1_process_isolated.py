#!/usr/bin/env python3
"""CAP0P-R1 native suspended/job/pipe coordinator; execution closed."""
import argparse,ctypes as C,hashlib,json,os,sys,time,traceback,uuid
from ctypes import wintypes
from pathlib import Path
from het_next_cap0pr1_protocol import Machine
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';OUT=ROOT/'reports/runs/streamq5_moe/het_next_cap0pr1_process_isolated';RES=OUT/'cap0pr1_result.json';COM=OUT/'cap0pr1_commit.json';FAIL=OUT/'cap0pr1_failure.json';LOCK=R/'het_next_cap0pr1_runner_lock.json';CHILD={'intel':S/'het_next_cap0pr1_intel_child.py','nvidia':S/'het_next_cap0pr1_nvidia_child.py'};PENDING='PENDING_INDEPENDENT_SOURCE_AUDIT';STILL_ACTIVE=259
class SA(C.Structure):_fields_=[('nLength',wintypes.DWORD),('lpSecurityDescriptor',C.c_void_p),('bInheritHandle',wintypes.BOOL)]
class SI(C.Structure):_fields_=[('cb',wintypes.DWORD),('lpReserved',wintypes.LPWSTR),('lpDesktop',wintypes.LPWSTR),('lpTitle',wintypes.LPWSTR),('dwX',wintypes.DWORD),('dwY',wintypes.DWORD),('dwXSize',wintypes.DWORD),('dwYSize',wintypes.DWORD),('dwXCountChars',wintypes.DWORD),('dwYCountChars',wintypes.DWORD),('dwFillAttribute',wintypes.DWORD),('dwFlags',wintypes.DWORD),('wShowWindow',wintypes.WORD),('cbReserved2',wintypes.WORD),('lpReserved2',C.POINTER(C.c_byte)),('hStdInput',wintypes.HANDLE),('hStdOutput',wintypes.HANDLE),('hStdError',wintypes.HANDLE)]
class PI(C.Structure):_fields_=[('hProcess',wintypes.HANDLE),('hThread',wintypes.HANDLE),('pid',wintypes.DWORD),('tid',wintypes.DWORD)]
class BASIC(C.Structure):_fields_=[('a',C.c_longlong),('b',C.c_longlong),('flags',wintypes.DWORD),('min',C.c_size_t),('max',C.c_size_t),('active_limit',wintypes.DWORD),('affinity',C.c_size_t),('priority',wintypes.DWORD),('schedule',wintypes.DWORD)]
class IO(C.Structure):_fields_=[(x,C.c_ulonglong) for x in ('a','b','c','d','e','f')]
class EXT(C.Structure):_fields_=[('basic',BASIC),('io',IO),('p',C.c_size_t),('j',C.c_size_t),('pp',C.c_size_t),('pj',C.c_size_t)]
class FILETIME(C.Structure):_fields_=[('low',wintypes.DWORD),('high',wintypes.DWORD)]
def api():
 k=C.WinDLL('kernel32',use_last_error=True);defs={'CreatePipe':([C.POINTER(wintypes.HANDLE),C.POINTER(wintypes.HANDLE),C.POINTER(SA),wintypes.DWORD],wintypes.BOOL),'SetHandleInformation':([wintypes.HANDLE,wintypes.DWORD,wintypes.DWORD],wintypes.BOOL),'CreateProcessW':([wintypes.LPCWSTR,wintypes.LPWSTR,C.c_void_p,C.c_void_p,wintypes.BOOL,wintypes.DWORD,C.c_void_p,wintypes.LPCWSTR,C.POINTER(SI),C.POINTER(PI)],wintypes.BOOL),'CreateJobObjectW':([C.c_void_p,wintypes.LPCWSTR],wintypes.HANDLE),'SetInformationJobObject':([wintypes.HANDLE,C.c_int,C.c_void_p,wintypes.DWORD],wintypes.BOOL),'AssignProcessToJobObject':([wintypes.HANDLE,wintypes.HANDLE],wintypes.BOOL),'ResumeThread':([wintypes.HANDLE],wintypes.DWORD),'PeekNamedPipe':([wintypes.HANDLE,C.c_void_p,wintypes.DWORD,C.POINTER(wintypes.DWORD),C.POINTER(wintypes.DWORD),C.POINTER(wintypes.DWORD)],wintypes.BOOL),'ReadFile':([wintypes.HANDLE,C.c_void_p,wintypes.DWORD,C.POINTER(wintypes.DWORD),C.c_void_p],wintypes.BOOL),'WriteFile':([wintypes.HANDLE,C.c_void_p,wintypes.DWORD,C.POINTER(wintypes.DWORD),C.c_void_p],wintypes.BOOL),'GetExitCodeProcess':([wintypes.HANDLE,C.POINTER(wintypes.DWORD)],wintypes.BOOL),'GetProcessTimes':([wintypes.HANDLE,C.POINTER(FILETIME),C.POINTER(FILETIME),C.POINTER(FILETIME),C.POINTER(FILETIME)],wintypes.BOOL),'WaitForMultipleObjects':([wintypes.DWORD,C.POINTER(wintypes.HANDLE),wintypes.BOOL,wintypes.DWORD],wintypes.DWORD),'TerminateJobObject':([wintypes.HANDLE,wintypes.UINT],wintypes.BOOL),'QueryInformationJobObject':([wintypes.HANDLE,C.c_int,C.c_void_p,wintypes.DWORD,C.POINTER(wintypes.DWORD)],wintypes.BOOL),'CloseHandle':([wintypes.HANDLE],wintypes.BOOL)}
 for n,(a,r) in defs.items():f=getattr(k,n);f.argtypes=a;f.restype=r
 return k
def atomic(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.inprogress')
 with t.open('xb') as f:f.write(json.dumps(o,sort_keys=True,separators=(',',':')).encode()+b'\n');f.flush();os.fsync(f.fileno())
 os.rename(t,p)
class Harness:
 def __init__(self,k):self.k=k;self.handles=[];self.children={};self.ledger=[];self.job=None
 def close(self,h,label):
  if h:self.ledger.append({'op':'CloseHandle','label':label,'handle':int(h),'ok':bool(self.k.CloseHandle(h))})
 def abort(self):
  if self.job:self.ledger.append({'op':'TerminateJobObject','ok':bool(self.k.TerminateJobObject(self.job,99))})
  self.wait(30000)
 def wait(self,ms):
  hs=[c['pi'].hProcess for c in self.children.values()]
  if not hs:return None
  a=(wintypes.HANDLE*len(hs))(*hs);return int(self.k.WaitForMultipleObjects(len(hs),a,True,ms))
 def write(self,c,obj):
  raw=(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n').encode();buf=C.create_string_buffer(raw);n=wintypes.DWORD()
  if not self.k.WriteFile(c['stdin_write'],buf,len(raw),C.byref(n),None) or n.value!=len(raw):raise OSError(C.get_last_error(),'WriteFile')
 def readframe(self,c,deadline_ns):
  while time.perf_counter_ns()<deadline_ns:
   avail=wintypes.DWORD()
   if not self.k.PeekNamedPipe(c['stdout_read'],None,0,None,C.byref(avail),None):raise OSError(C.get_last_error(),'PeekNamedPipe')
   if avail.value:
    b=C.create_string_buffer(min(avail.value,1<<20));n=wintypes.DWORD();
    if not self.k.ReadFile(c['stdout_read'],b,len(b),C.byref(n),None):raise OSError(C.get_last_error(),'ReadFile')
    c['buffer']+=b.raw[:n.value]
    if b'\n' in c['buffer']:
     line,c['buffer']=c['buffer'].split(b'\n',1);raw=line+b'\n';c['transcript'].append(raw.hex());return raw
   code=wintypes.DWORD();self.k.GetExitCodeProcess(c['pi'].hProcess,C.byref(code))
   if code.value!=STILL_ACTIVE:raise ChildProcessError(f'child_exit:{code.value}')
   time.sleep(.005)
  raise TimeoutError('pipe_deadline')
def run(token):
 l=json.loads(LOCK.read_text());
 if not(l['execution_open'] and token==l['audit_token'] and token!=PENDING):raise PermissionError('closed')
 if OUT.exists():raise FileExistsError('output_exists')
 k=api();h=Harness(k);nonce=uuid.uuid4().hex;machine=Machine(nonce);sa=SA(C.sizeof(SA),None,True);job=k.CreateJobObjectW(None,None);h.job=job;ex=EXT();ex.basic.flags=0x2000
 if not job or not k.SetInformationJobObject(job,9,C.byref(ex),C.sizeof(ex)):raise OSError(C.get_last_error(),'job')
 try:
  for role,path in CHILD.items():
   child_in,parent_write,parent_read,child_out=wintypes.HANDLE(),wintypes.HANDLE(),wintypes.HANDLE(),wintypes.HANDLE();
   if not k.CreatePipe(C.byref(child_in),C.byref(parent_write),C.byref(sa),0) or not k.CreatePipe(C.byref(parent_read),C.byref(child_out),C.byref(sa),0):raise OSError(C.get_last_error(),'pipe')
   k.SetHandleInformation(parent_write,1,0);k.SetHandleInformation(parent_read,1,0);si=SI();si.cb=C.sizeof(si);si.dwFlags=0x100;si.hStdInput=child_in;si.hStdOutput=child_out;si.hStdError=child_out;pi=PI();lp=2 if role=='intel' else 4;cmd=C.create_unicode_buffer(f'"{sys.executable}" "{path}" --lp {lp} --nonce {nonce} --role {role}')
   if not k.CreateProcessW(None,cmd,None,None,True,0x08000004,None,str(ROOT),C.byref(si),C.byref(pi)):raise OSError(C.get_last_error(),'CreateProcessW')
   if not k.AssignProcessToJobObject(job,pi.hProcess):raise OSError(C.get_last_error(),'assign')
   h.close(child_in,role+'_child_in_parent');h.close(child_out,role+'_child_out_parent');h.children[role]={'pi':pi,'stdin_write':parent_write,'stdout_read':parent_read,'buffer':b'','transcript':[],'create_filetime':None}
  for c in h.children.values():
   if k.ResumeThread(c['pi'].hThread)==0xffffffff:raise OSError(C.get_last_error(),'resume')
  ready={r:machine.accept(r,'ready',h.readframe(c,time.perf_counter_ns()+120_000_000_000)) for r,c in h.children.items()};reps=[]
  for epoch in range(1,4):
   for c in h.children.values():h.write(c,{'cmd':'START','epoch':epoch,'nonce':nonce})
   rows={r:machine.accept(r,'result',h.readframe(c,time.perf_counter_ns()+30_000_000_000)) for r,c in h.children.items()};reps.append({'epoch':epoch,**rows})
  for c in h.children.values():h.write(c,{'cmd':'STOP','nonce':nonce})
  cleanup={r:machine.accept(r,'cleanup',h.readframe(c,time.perf_counter_ns()+30_000_000_000)) for r,c in h.children.items()};h.wait(30000);process={}
  for r,c in h.children.items():
   code=wintypes.DWORD();k.GetExitCodeProcess(c['pi'].hProcess,C.byref(code));a,b,d,e=FILETIME(),FILETIME(),FILETIME(),FILETIME();k.GetProcessTimes(c['pi'].hProcess,C.byref(a),C.byref(b),C.byref(d),C.byref(e));process[r]={'pid':int(c['pi'].pid),'create_filetime':(a.high<<32)|a.low,'exit_filetime':(b.high<<32)|b.low,'exit_code':int(code.value),'exit_qpc_ns':time.perf_counter_ns(),'transcript_hex':c['transcript']}
  h.close(job,'job');h.job=None
  for r,c in h.children.items():h.close(c['stdin_write'],r+'_stdin');h.close(c['stdout_read'],r+'_stdout');h.close(c['pi'].hThread,r+'_thread');h.close(c['pi'].hProcess,r+'_process')
  result={'kind':'het_next_cap0pr1_process_isolated','nonce':nonce,'ready':ready,'repetitions':reps,'cleanup':cleanup,'process':process,'ledger':h.ledger,'job':{'assigned_before_resume':True,'kill_on_close':True,'closed':True}};OUT.mkdir();atomic(RES,result);atomic(COM,{'result':{'bytes':RES.stat().st_size,'sha256':hashlib.sha256(RES.read_bytes()).hexdigest()}});return result
 except BaseException:
  h.abort();raise
 finally:
  if h.job:h.close(h.job,'job_finally')
def main():
 p=argparse.ArgumentParser();p.add_argument('--ack',required=True);a=p.parse_args()
 try:run(a.ack);return 0
 except BaseException as e:
  if not OUT.exists():OUT.mkdir()
  if not FAIL.exists():atomic(FAIL,{'kind':'cap0pr1_failure','error':str(e),'traceback':traceback.format_exc()})
  raise
if __name__=='__main__':raise SystemExit(main())
