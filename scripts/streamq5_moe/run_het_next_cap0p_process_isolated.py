#!/usr/bin/env python3
"""CAP0P native Windows suspended/job coordinator (execution closed)."""
import argparse,ctypes as C,hashlib,json,msvcrt,os,sys,time,traceback,uuid
from ctypes import wintypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';OUT=ROOT/'reports/runs/streamq5_moe/het_next_cap0p_process_isolated';RESULT=OUT/'cap0p_result.json';COMMIT=OUT/'cap0p_commit.json';FAIL=OUT/'cap0p_failure.json';LOCK=R/'het_next_cap0p_runner_lock.json'
CHILDREN={'intel':S/'het_next_cap0p_intel_child.py','nvidia':S/'het_next_cap0p_nvidia_child.py'};ACK='PENDING_INDEPENDENT_SOURCE_AUDIT';CREATE_SUSPENDED=4;CREATE_NO_WINDOW=0x08000000;EXTENDED_STARTUPINFO_PRESENT=0x00080000;INFINITE=0xffffffff
class STARTUPINFOW(C.Structure):_fields_=[('cb',wintypes.DWORD),('lpReserved',wintypes.LPWSTR),('lpDesktop',wintypes.LPWSTR),('lpTitle',wintypes.LPWSTR),('dwX',wintypes.DWORD),('dwY',wintypes.DWORD),('dwXSize',wintypes.DWORD),('dwYSize',wintypes.DWORD),('dwXCountChars',wintypes.DWORD),('dwYCountChars',wintypes.DWORD),('dwFillAttribute',wintypes.DWORD),('dwFlags',wintypes.DWORD),('wShowWindow',wintypes.WORD),('cbReserved2',wintypes.WORD),('lpReserved2',C.POINTER(C.c_byte)),('hStdInput',wintypes.HANDLE),('hStdOutput',wintypes.HANDLE),('hStdError',wintypes.HANDLE)]
class PROCESS_INFORMATION(C.Structure):_fields_=[('hProcess',wintypes.HANDLE),('hThread',wintypes.HANDLE),('dwProcessId',wintypes.DWORD),('dwThreadId',wintypes.DWORD)]
class JOBOBJECT_BASIC_LIMIT_INFORMATION(C.Structure):_fields_=[('PerProcessUserTimeLimit',C.c_longlong),('PerJobUserTimeLimit',C.c_longlong),('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',C.c_size_t),('MaximumWorkingSetSize',C.c_size_t),('ActiveProcessLimit',wintypes.DWORD),('Affinity',C.c_size_t),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]
class IO_COUNTERS(C.Structure):_fields_=[(x,C.c_ulonglong) for x in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(C.Structure):_fields_=[('BasicLimitInformation',JOBOBJECT_BASIC_LIMIT_INFORMATION),('IoInfo',IO_COUNTERS),('ProcessMemoryLimit',C.c_size_t),('JobMemoryLimit',C.c_size_t),('PeakProcessMemoryUsed',C.c_size_t),('PeakJobMemoryUsed',C.c_size_t)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.inprogress')
 with t.open('xb') as f:f.write(json.dumps(o,sort_keys=True,separators=(',',':')).encode()+b'\n');f.flush();os.fsync(f.fileno())
 os.rename(t,p)
def api():
 k=C.WinDLL('kernel32',use_last_error=True)
 defs={'CreateJobObjectW':([C.c_void_p,wintypes.LPCWSTR],wintypes.HANDLE),'SetInformationJobObject':([wintypes.HANDLE,C.c_int,C.c_void_p,wintypes.DWORD],wintypes.BOOL),'AssignProcessToJobObject':([wintypes.HANDLE,wintypes.HANDLE],wintypes.BOOL),'CreatePipe':([C.POINTER(wintypes.HANDLE),C.POINTER(wintypes.HANDLE),C.c_void_p,wintypes.DWORD],wintypes.BOOL),'SetHandleInformation':([wintypes.HANDLE,wintypes.DWORD,wintypes.DWORD],wintypes.BOOL),'CreateProcessW':([wintypes.LPCWSTR,wintypes.LPWSTR,C.c_void_p,C.c_void_p,wintypes.BOOL,wintypes.DWORD,C.c_void_p,wintypes.LPCWSTR,C.POINTER(STARTUPINFOW),C.POINTER(PROCESS_INFORMATION)],wintypes.BOOL),'ResumeThread':([wintypes.HANDLE],wintypes.DWORD),'WaitForSingleObject':([wintypes.HANDLE,wintypes.DWORD],wintypes.DWORD),'GetExitCodeProcess':([wintypes.HANDLE,C.POINTER(wintypes.DWORD)],wintypes.BOOL),'CloseHandle':([wintypes.HANDLE],wintypes.BOOL)}
 for n,(a,r) in defs.items():f=getattr(k,n);f.argtypes=a;f.restype=r
 return k
def mainrun(token):
 lock=json.loads(LOCK.read_text());
 if not(lock['execution_open'] and token==lock['audit_token'] and token!=ACK):raise PermissionError('closed')
 if OUT.exists():raise FileExistsError('output_not_absent')
 k=api();job=k.CreateJobObjectW(None,None);info=JOBOBJECT_EXTENDED_LIMIT_INFORMATION();info.BasicLimitInformation.LimitFlags=0x2000
 if not job or not k.SetInformationJobObject(job,9,C.byref(info),C.sizeof(info)):raise OSError(C.get_last_error(),'job')
 children={};events=[]
 try:
  for name,path in CHILDREN.items():
   rin,win,rout,wout=wintypes.HANDLE(),wintypes.HANDLE(),wintypes.HANDLE(),wintypes.HANDLE();k.CreatePipe(C.byref(rin),C.byref(win),None,0);k.CreatePipe(C.byref(rout),C.byref(wout),None,0);k.SetHandleInformation(win,1,0);k.SetHandleInformation(rout,1,0);si=STARTUPINFOW();si.cb=C.sizeof(si);si.dwFlags=0x100;si.hStdInput=rin;si.hStdOutput=wout;si.hStdError=wout;pi=PROCESS_INFORMATION();lp=2 if name=='intel' else 4;cmd=C.create_unicode_buffer(f'"{sys.executable}" "{path}" --lp {lp}')
   if not k.CreateProcessW(None,cmd,None,None,True,CREATE_SUSPENDED|CREATE_NO_WINDOW,None,str(ROOT),C.byref(si),C.byref(pi)):raise OSError(C.get_last_error(),'CreateProcessW')
   if not k.AssignProcessToJobObject(job,pi.hProcess):raise OSError(C.get_last_error(),'AssignProcessToJobObject')
   children[name]={'pi':pi,'write':win,'read':rout,'pid':int(pi.dwProcessId),'create_ns':time.perf_counter_ns(),'rows':[]};k.CloseHandle(rin);k.CloseHandle(wout)
  for c in children.values():
   if k.ResumeThread(c['pi'].hThread)==0xffffffff:raise OSError(C.get_last_error(),'ResumeThread')
  for c in children.values():c['wf']=os.fdopen(msvcrt.open_osfhandle(int(c['write']),0),'w',buffering=1);c['rf']=os.fdopen(msvcrt.open_osfhandle(int(c['read']),os.O_RDONLY),'r',buffering=1)
  for name,c in children.items():row=json.loads(c['rf'].readline());c['rows'].append(row)
  reps=[]
  for epoch in range(1,4):
   send=time.perf_counter_ns()
   for c in children.values():c['wf'].write(f'START {epoch}\n')
   row={name:json.loads(c['rf'].readline()) for name,c in children.items()};reps.append({'epoch':epoch,'send_ns':send,**row,'strict_overlap':max(row['intel']['submit_ns'],row['nvidia']['submit_ns'])<min(row['intel']['done_ns'],row['nvidia']['done_ns'])})
  for c in children.values():c['wf'].write('STOP\n');c['wf'].close()
  cleanup={name:json.loads(c['rf'].readline()) for name,c in children.items()};exits={}
  for name,c in children.items():k.WaitForSingleObject(c['pi'].hProcess,30000);code=wintypes.DWORD();k.GetExitCodeProcess(c['pi'].hProcess,C.byref(code));exits[name]=int(code.value);c['rf'].close();k.CloseHandle(c['pi'].hThread);k.CloseHandle(c['pi'].hProcess)
  k.CloseHandle(job);job=None;result={'kind':'het_next_cap0p_process_isolated','status':'process_isolated_cohabitation_positive','children':{n:{'pid':c['pid'],'create_ns':c['create_ns'],'ready':c['rows'][0]} for n,c in children.items()},'repetitions':reps,'cleanup':cleanup,'exits':exits,'job':{'kill_on_close':True,'assigned_before_resume':True,'closed':True},'claim':'process-isolated correctness/cohabitation only'};OUT.mkdir(parents=True);atomic(RESULT,result);atomic(COMMIT,{'result':{'bytes':RESULT.stat().st_size,'sha256':sha(RESULT)}});return result
 finally:
  if job:k.CloseHandle(job)
def main():
 p=argparse.ArgumentParser();p.add_argument('--ack',required=True);a=p.parse_args()
 try:r=mainrun(a.ack);print(r['status']);return 0
 except BaseException as e:
  if not OUT.exists():OUT.mkdir(parents=True)
  if not FAIL.exists():atomic(FAIL,{'kind':'cap0p_failure','error':str(e),'traceback':traceback.format_exc()})
  raise
if __name__=='__main__':raise SystemExit(main())
