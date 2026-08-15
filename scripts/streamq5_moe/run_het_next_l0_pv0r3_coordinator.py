#!/usr/bin/env python3
"""PV0-R3 bounded process coordinator; correctness only, no timing claim."""
from __future__ import annotations
import argparse, ctypes as C, hashlib, json, os, shutil, subprocess, sys, time, traceback, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; S=ROOT/'scripts/streamq5_moe'; R=ROOT/'reports/streamq5_moe'; D=ROOT/'reports/runs/streamq5_moe/het_next_l0_pv0r3_real_weight_process_validation'
LOCK=R/'het_next_l0_pv0r3_runner_lock.json'; VL=R/'het_next_l0_pv0r3_verifier_lock.json'; BUILDER=S/'build_het_next_l0_pv0r3_source_oracle.py'; INTEL=S/'run_het_next_l0_pv0r3_intel_child.py'; NVIDIA=S/'run_het_next_l0_pv0r3_nvidia_child.py'; VER=S/'verify_het_next_l0_pv0r3_independent.py'
ORES=D/'pv0r3_cpu_builder.json'; IR=D/'pv0r3_intel_raw.npz'; NR=D/'pv0r3_nvidia_raw.npz'; RES=D/'pv0r3_result.json'; VRES=D/'pv0r3_verification.json'; COM=D/'pv0r3_commit.json'; FAIL=D/'pv0r3_failure.json'; ACK='PV0R3_REAL_WEIGHT_COMPONENT_AFTER_INDEPENDENT_SOURCE_AUDIT'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def atomic(p,data):
 p=Path(p);t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.inprogress')
 if p.exists():raise FileExistsError(p)
 t.write_bytes(data)
 with t.open('r+b') as f:os.fsync(f.fileno())
 os.rename(t,p)
def send(pipe,obj):
 b=canon(obj)
 if len(b)>1<<20:raise ValueError('frame')
 pipe.write(len(b).to_bytes(8,'little')+b);pipe.flush()
def recv(pipe,deadline,proc):
 # Bounded poll; child frames are small and stdout is binary.
 import msvcrt
 h=msvcrt.get_osfhandle(pipe.fileno());k=C.WinDLL('kernel32',use_last_error=True);k.PeekNamedPipe.argtypes=[C.c_void_p,C.c_void_p,C.c_uint,C.POINTER(C.c_uint),C.POINTER(C.c_uint),C.POINTER(C.c_uint)];k.PeekNamedPipe.restype=C.c_int
 buf=bytearray()
 while time.monotonic()<deadline:
  avail=C.c_uint()
  if not k.PeekNamedPipe(h,None,0,None,C.byref(avail),None):raise C.WinError(C.get_last_error())
  if avail.value:buf.extend(os.read(pipe.fileno(),avail.value))
  if len(buf)>=8:
   n=int.from_bytes(buf[:8],'little')
   if n>1<<20:raise ValueError('oversize')
   if len(buf)>=8+n:return json.loads(bytes(buf[8:8+n]))
  if proc.poll() is not None and not avail.value:raise RuntimeError('child exited before frame')
  time.sleep(.01)
 raise TimeoutError('frame timeout')
def valid_commit():
 try:
  c=json.loads(COM.read_text());return all((D/n).is_file() and sha(D/n)==h for n,h in c['files'].items())
 except Exception:return False
def run():
 if valid_commit():print('already_complete');return 0
 if RES.exists() or COM.exists() or FAIL.exists():raise FileExistsError('nonclean result')
 lock=json.loads(LOCK.read_text());
 if not(lock['execution_open'] and lock['audit_token']=='PV0R3_SOURCE_AUDIT_GO'):raise RuntimeError('closed')
 if not ORES.exists():raise RuntimeError('builder phase must complete separately')
 nonce=uuid.uuid4().hex;procs={};records={}
 try:
  for role,path in [('intel',INTEL),('nvidia',NVIDIA)]:
   p=subprocess.Popen([sys.executable,str(path),'--nonce',nonce],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));procs[role]=p;records[role]={'pid':p.pid,'start_qpc_ns':time.perf_counter_ns(),'argv':[sys.executable,str(path),'--nonce',nonce]}
  ready={role:recv(p.stdout,time.monotonic()+300,p) for role,p in procs.items()}
  for role,v in ready.items():
   if v.get('type')!='ready' or v.get('nonce')!=nonce or v.get('role')!=role or v.get('seq')!=0:raise RuntimeError('ready schema')
  release=time.perf_counter_ns()
  for role,p in procs.items():send(p.stdin,{'type':'start','nonce':nonce,'role':role,'seq':1})
  results={role:recv(p.stdout,time.monotonic()+1800,p) for role,p in procs.items()}
  for role,p in procs.items():
   p.wait(timeout=30);records[role].update(exit_code=p.returncode,end_qpc_ns=time.perf_counter_ns(),stderr=p.stderr.read().decode(errors='replace'))
   v=results[role]
   if p.returncode!=0 or v.get('type')!='result' or v.get('nonce')!=nonce or v.get('role')!=role or v.get('seq')!=1 or v.get('error') is not None:raise RuntimeError(role+' result')
  payload={'kind':'het_next_l0_pv0r3_candidate','status':'candidate_requires_independent_verifier','nonce':nonce,'release_qpc_ns':release,'processes':records,'child_results':results,'cpu_builder_sha256':sha(ORES),'intel_raw_sha256':sha(IR),'nvidia_raw_sha256':sha(NR),'runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'runner_lock_sha256':sha(LOCK),'verifier_lock_sha256':sha(VL)}
  atomic(RES,canon(payload)+b'\n')
  vr=subprocess.run([sys.executable,str(VER),'--candidate',str(RES)],cwd=ROOT,capture_output=True,text=True,timeout=1800,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  if vr.returncode!=0 or not VRES.exists() or json.loads(VRES.read_text()).get('pass') is not True:raise RuntimeError('independent verifier negative')
  commit={'kind':'het_next_l0_pv0r3_commit','files':{p.name:sha(p) for p in (ORES,IR,NR,RES,VRES)},'claim_boundary':'known p0/n16 real-weight process-isolated component validation only; no performance'};atomic(COM,canon(commit)+b'\n');print('real_weight_process_validation_positive');return 0
 except Exception as e:
  for p in procs.values():
   if p.poll() is None:p.kill()
   try:p.wait(timeout=10)
   except Exception:pass
  atomic(FAIL,canon({'kind':'het_next_l0_pv0r3_failure','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'processes':records,'survivors':[p.pid for p in procs.values() if p.poll() is None]})+b'\n');raise
def main():
 p=argparse.ArgumentParser();p.add_argument('--ack',required=True);a=p.parse_args()
 if a.ack!=ACK:raise SystemExit('ack')
 return run()
if __name__=='__main__':raise SystemExit(main())
