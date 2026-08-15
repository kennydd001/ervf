#!/usr/bin/env python3
"""One-shot PH1 Intel correctness execution; closed by immutable lock."""
from __future__ import annotations

import argparse, hashlib, importlib, json, os, sys, traceback, uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];SCRIPTS=ROOT/'scripts/streamq5_moe';sys.path.insert(0,str(SCRIPTS));REPORTS=ROOT/'reports/streamq5_moe'
import het_next_l0_ph1_intel_execution_r0_backend as backend
OUT=REPORTS/'het_next_l0_ph1_intel_execution_r0';FAILED=REPORTS/'het_next_l0_ph1_intel_execution_r0_failed_attempts';QUAR=REPORTS/'het_next_l0_ph1_intel_execution_r0_quarantine'
LOCK=REPORTS/'het_next_l0_ph1_intel_execution_r0_lock.json';PREREG=REPORTS/'HET_NEXT_L0_PH1_INTEL_EXECUTION_R0_PREREGISTRATION_2026-08-14.md';PREFLIGHT=SCRIPTS/'preflight_het_next_l0_ph1_intel_execution_r0.py';VERIFIER=SCRIPTS/'verify_het_next_l0_ph1_intel_execution_r0.py'
COMPILE=REPORTS/'het_next_l0_ph1_intel_compile_r2a';CPU=REPORTS/'het_next_l0_ph1_cpu_freeze_r2';CPU_VERIFY=REPORTS/'het_next_l0_ph1_cpu_freeze_r2_independent_verification.json';ACK='PH1_INTEL_EXECUTION_R0_AFTER_ARTIFACT_SOURCE_AND_PREFLIGHT_GO'
EXPECTED={'compile_commit_sha256':'c9f9ab3838d9d3d4ddd6e16a18f7989c16061901f08e046987081d9d975a152a','compile_result_sha256':'ac7c90e15c71cf2a481004f78954e9d78631078d3e08893d3f716120345df5cc','compile_binary_sha256':backend.BINARY_SHA,'compile_source_sha256':backend.SOURCE_SHA,'cpu_commit_sha256':'f3677e9610bea03649fec172b97c0c314f2f2e4c0d40bf9d864df0ec88a44f06','cpu_verification_sha256':'1c7f2772fb637485020be00f74b6f9295a18ec3d7d10af0587ea350e8756cbc8'}
STAGES={'gate':'e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867','up':'f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08','silu':'a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8','activation':'762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f','down':'142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc'}

def sha(b):return hashlib.sha256(b).hexdigest()
def fsha(p):return sha(Path(p).read_bytes())
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()+b'\n'
def write(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
def move(a,b):
 if b.exists():raise FileExistsError(b)
 if os.name=='nt':
  import ctypes as C;f=C.WinDLL('kernel32',use_last_error=True).MoveFileExW;f.argtypes=[C.c_wchar_p,C.c_wchar_p,C.c_uint32];f.restype=C.c_int
  if not f(str(a),str(b),8):raise C.WinError(C.get_last_error())
 else:os.rename(a,b)
def verify_bundle(d):
 r,m,c=(d/n for n in ('result.json','manifest.json','commit.json'))
 if not all(x.is_file() for x in (r,m,c)):raise RuntimeError('core')
 rr,mm,cc=(json.loads(x.read_text()) for x in (r,m,c))
 if rr.get('kind')!='ph1_intel_execution_r0' or mm.get('kind')!='ph1_intel_execution_r0_manifest' or cc!={'kind':'ph1_intel_execution_r0_commit','manifest_sha256':fsha(m),'result_sha256':fsha(r)}:raise RuntimeError('core_contract')
 if {x['name'] for x in mm['files']}|{'manifest.json','commit.json'}!={p.name for p in d.iterdir()}:raise RuntimeError('set')
 if not all((d/x['name']).stat().st_size==x['bytes'] and fsha(d/x['name'])==x['sha256'] for x in mm['files']):raise RuntimeError('hash')
 return rr
def archive(root,prefix,payload,path=None):
 root.mkdir(parents=True,exist_ok=True);d=root/(prefix+'_'+uuid.uuid4().hex)
 if path and path.exists():move(path,d)
 else:d.mkdir()
 write(d/'failure.json',canon(payload));return d
def recover():
 if OUT.exists():
  try:return {'already':True,'result':verify_bundle(OUT)}
  except Exception as e:archive(QUAR,'corrupt',{'error':str(e),'device_opened':False},OUT);raise RuntimeError('corrupt_quarantined')
 stale=list(REPORTS.glob(OUT.name+'.*.inprogress'))
 if stale:
  for p in stale:archive(QUAR,'stale',{'source':str(p),'device_opened':False},p)
  raise RuntimeError('stale_quarantined')
 return {'already':False}
def authorize():
 l=json.loads(LOCK.read_text());obs={'runner_sha256':fsha(Path(__file__)),'backend_sha256':fsha(Path(backend.__file__)),'verifier_sha256':fsha(VERIFIER),'preflight_sha256':fsha(PREFLIGHT),'prereg_sha256':fsha(PREREG),**EXPECTED}
 actual={**obs,'compile_commit_sha256':fsha(COMPILE/'commit.json'),'compile_result_sha256':fsha(COMPILE/'result.json'),'compile_binary_sha256':fsha(COMPILE/'intel_program.bin'),'compile_source_sha256':fsha(COMPILE/'intel_source.cl'),'cpu_commit_sha256':fsha(CPU/'commit.json'),'cpu_verification_sha256':fsha(CPU_VERIFY)}
 if not(l.get('kind')=='ph1_intel_execution_r0_lock' and l.get('execution_open') is True and l.get('audit_token')==ACK and all(l.get(k)==v and actual[k]==v for k,v in obs.items()) and json.loads(CPU_VERIFY.read_text()).get('pass') is True):raise RuntimeError('authorization')
 return {'lock_sha256':fsha(LOCK),'observed':actual}
def payload():
 cpu=importlib.import_module('generate_het_next_l0_ph1_cpu_freeze');inp=cpu.read_exact(cpu.D2,cpu.INPUT_OFFSET,cpu.INPUT_BYTES)
 records={}
 for spec in cpu.RECORDS:
  src=cpu.read_exact(cpu.SHARD,*((lambda a,b:(a,b-a))(*spec['absolute'])));record,_=cpu.build_record(spec,src);records[spec['projection']]=record
 return records,inp,(CPU/'bf16_silu_lut.bin').read_bytes()
def gates(ev):
 out={k:bytes.fromhex(v) for k,v in ev['outputs'].items()};rels=[r for r in ev['ledger'] if r.get('op')=='release'];alloc=[r for r in ev['ledger'] if r.get('op')=='host_usm_allocate'];args=[r for r in ev['ledger'] if r.get('op')=='set_pointer_arg'];launch=[r for r in ev['ledger'] if r.get('op')=='enqueue'];cleanup=ev['ledger'][-1]
 return {'stage_hashes':{k:sha(out[k]) for k in STAGES}==STAGES,'counter_ones':all(len(out[k])%4==0 and all(int.from_bytes(out[k][i:i+4],'little')==1 for i in range(0,len(out[k]),4)) for k in ('gate_counters','up_counters','activation_counters','down_counters')),'cardinality':len(alloc)==14 and len(args)==18 and len(launch)==4 and len(rels)==21,'allocation_contract':sum(r['bytes'] for r in alloc)==2185216 and all(r['alignment']==4096 and r['pointer']%4096==0 and r['base']==r['pointer'] and r['queried_size']==r['bytes'] for r in alloc),'launch_contract':[(r['kernel'],r['global'],r['local']) for r in launch]==list(backend.LAUNCHES),'cleanup':cleanup.get('cleanup_complete') is True and cleanup.get('release_attempts')==21 and cleanup.get('live_owned_resources')==0 and all(r['code']==0 for r in rels),'forbidden_zero':all(v==0 for v in ev['forbidden_calls'].values())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--ack',required=True);a=p.parse_args()
 if a.ack!=ACK:raise SystemExit('ack')
 rec=recover()
 if rec['already']:print(json.dumps({'status':'already_complete'}));return 0
 auth=authorize();attempt=REPORTS/(OUT.name+'.'+uuid.uuid4().hex+'.inprogress');attempt.mkdir();opened=False
 try:
  records,inp,lut=payload();opened=True;ev=backend.Backend().run(records,inp,lut,auth);gg=gates(ev);positive=all(gg.values());result={'kind':'ph1_intel_execution_r0','status':'intel_execution_positive' if positive else 'intel_execution_negative','positive':positive,'gates':gg,'evidence':ev,'claim':'one real expert/input Intel correctness component only'};write(attempt/'result.json',canon(result));files=[{'name':x.name,'bytes':x.stat().st_size,'sha256':fsha(x)} for x in attempt.iterdir()];write(attempt/'manifest.json',canon({'kind':'ph1_intel_execution_r0_manifest','files':files}));write(attempt/'commit.json',canon({'kind':'ph1_intel_execution_r0_commit','manifest_sha256':fsha(attempt/'manifest.json'),'result_sha256':fsha(attempt/'result.json')}));verify_bundle(attempt);move(attempt,OUT);verify_bundle(OUT);print(json.dumps({'status':result['status'],'positive':positive}));return 0 if positive else 3
 except Exception as e:archive(FAILED,'attempt',{'kind':'ph1_intel_execution_r0_failure','error':f'{type(e).__name__}:{e}','traceback':traceback.format_exc(),'device_opened':opened,'backend_evidence':getattr(e,'evidence',None)},attempt if attempt.exists() else (OUT if OUT.exists() else None));return 3
if __name__=='__main__':raise SystemExit(main())
