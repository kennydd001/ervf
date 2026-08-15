#!/usr/bin/env python3
"""Sequential PH0-R3 phase runner. Device modules import only inside their phase."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, platform, sys, time, traceback
from datetime import datetime,timezone
from pathlib import Path
import psutil
from het_next_l0_ph0r3_common import *

ACK='PH0R3_SINGLE_REAL_PROJECTION_SOURCE_AUDITED_AND_AUTHORIZED'
PHASES=('cpu','intel','nvidia','adjudicate')
FILES={p:RUN_DIR/f'{p}.json' for p in PHASES};FAIL=RUN_DIR/'failure.json';COMMIT=RUN_DIR/'commit.json'
LOCK=REPORTS/'het_next_l0_ph0r3_runner_lock.json'; VERIFIER=ROOT/'scripts/streamq5_moe/verify_het_next_l0_ph0r3_single_projection.py'
PREREG=REPORTS/'HET_NEXT_L0_PH0R3_SINGLE_REAL_PROJECTION_PREREGISTRATION_2026-08-13.md';DESIGN=REPORTS/'HET_NEXT_L0_PH0R3_SINGLE_REAL_PROJECTION_IMPLEMENTATION_DESIGN_2026-08-13.md'

def now():return datetime.now(timezone.utc).isoformat()
def resource(stage):
 p=psutil.Process();m=p.memory_info();return {'stage':stage,'rss':m.rss,'peak':getattr(m,'peak_wset',m.rss),'available':psutil.virtual_memory().available}
def locks():
 d=json.loads(LOCK.read_text());
 if not d['execution_open'] or d['audit_token']!='PH0R3_IMPLEMENTATION_AUDIT_GO':raise RuntimeError('execution_closed')
 for row in d['files'].values():
  if file_digest(ROOT/row['path'])!=row['sha256']:raise RuntimeError(f"hash_drift:{row['path']}")
 return d
def prior(phase):
 need={'cpu':[],'intel':['cpu'],'nvidia':['cpu','intel'],'adjudicate':['cpu','intel','nvidia']}[phase]
 for n in need:
  if not FILES[n].exists():raise RuntimeError(f'missing_prior:{n}')
 return {n:json.loads(FILES[n].read_text()) for n in need}
def payload():
 if SHARD.stat().st_size!=SHARD_BYTES or D2.stat().st_size!=D2_BYTES:raise RuntimeError('artifact_size')
 if file_digest(SHARD)!=SHARD_SHA or file_digest(D2)!=D2_SHA:raise RuntimeError('artifact_sha')
 source=read_exact(SHARD,SOURCE_OFFSET,SOURCE_BYTES);inp=read_exact(D2,INPUT_OFFSET,INPUT_BYTES)
 if digest(source)!=SOURCE_SHA or digest(inp)!=INPUT_SHA:raise RuntimeError('range_sha')
 return source,inp
def raw_compare(data,expected_hex):
 out=bytes.fromhex(data['output_hex']);cnt=np.frombuffer(bytes.fromhex(data['counters_hex']),'<u4');exp=bytes.fromhex(expected_hex)
 return {'output_bytes':len(out),'output_sha256':digest(out),'bit_differences':sum(a!=b for a,b in zip(out,exp))+(abs(len(out)-len(exp))), 'counter_count':len(cnt),'counters_all_one':bool(len(cnt)==ROWS and np.all(cnt==1)),'no_ffff':bool(np.all(np.frombuffer(out,'<u2')!=0xffff))}
def run_phase(phase):
 r0=resource('start')
 if r0['available']<2<<30:raise RuntimeError('start_ram')
 if FILES[phase].exists() or FAIL.exists() or COMMIT.exists():raise FileExistsError('target_exists')
 old=prior(phase);source,inp=payload();record,evidence=build_record(source);safe=safe_check(record,inp)
 base={'kind':f'het_next_l0_ph0r3_{phase}','phase':phase,'utc':now(),'locks_sha256':file_digest(LOCK),'prereg_sha256':file_digest(PREREG),'design_sha256':file_digest(DESIGN),'record_evidence':evidence,'safe_trace':safe['trace'],'resources':[r0],'runtime':{'python':sys.version,'platform':platform.platform(),'pid':os.getpid(),'cuda_loaded_before_backend':'cupy' in sys.modules}}
 if phase=='cpu':
  ctrl=controls(record,inp);oracle=cpu_oracle(record,inp);base.update({'controls':ctrl,'output_hex':oracle.tobytes().hex(),'output_sha256':digest(oracle.tobytes()),'status':'cpu_committed' if all(x['pass'] for x in ctrl) else 'negative_controls'})
 elif phase=='intel':
  from het_next_l0_ph0r3_intel import run
  base['backend']=run(record,inp);base['comparison']=raw_compare(base['backend'],old['cpu']['output_hex']);base['status']='intel_committed' if base['comparison']['bit_differences']==0 and base['comparison']['counters_all_one'] and base['backend']['ledger'][-1]['cleanup_complete'] else 'negative_intel'
 elif phase=='nvidia':
  if old['intel']['status']!='intel_committed' or not old['intel']['backend']['ledger'][-1]['cleanup_complete']:raise RuntimeError('intel_not_clean')
  from het_next_l0_ph0r3_nvidia import run
  base['backend']=run(record,inp);base['comparison']=raw_compare(base['backend'],old['cpu']['output_hex']);base['status']='nvidia_committed' if base['comparison']['bit_differences']==0 and base['comparison']['counters_all_one'] and base['backend']['ledger'][-1]['cleanup_complete'] else 'negative_nvidia'
 else:
  from subprocess import run as subprocess_run
  cmd=[sys.executable,str(VERIFIER),'--precommit'];proc=subprocess_run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=600)
  base.update({'verifier_precommit':{'command':cmd,'exit_code':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr},'status':'candidate_verified' if proc.returncode==0 else 'negative_verifier'})
 del record,source,inp;gc.collect();base['resources'].append(resource('cleanup'))
 if base['resources'][-1]['peak']>2<<30:base['status']='negative_resource'
 write_atomic_new(FILES[phase],canonical(base))
 if phase=='adjudicate' and base['status']=='candidate_verified':
  manifest={'kind':'het_next_l0_ph0r3_commit','status':'positive_single_real_projection_component','claim_boundary':'validation-only single real projection; no model/performance/concurrency claim','files':{p:{'bytes':FILES[p].stat().st_size,'sha256':file_digest(FILES[p])} for p in PHASES}}
  write_atomic_new(COMMIT,canonical(manifest))
 return base
def failure(phase,exc):
 RUN_DIR.mkdir(parents=True,exist_ok=True)
 row={'kind':'het_next_l0_ph0r3_failure','phase':phase,'utc':now(),'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'resources':[resource('failure')],'partial_files':[p.name for p in RUN_DIR.iterdir()]}
 if not COMMIT.exists() and not FAIL.exists():write_atomic_new(FAIL,canonical(row))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=PHASES,required=True);ap.add_argument('--acknowledge',required=True);a=ap.parse_args()
 if a.acknowledge!=ACK:raise SystemExit('bad acknowledgement')
 locks()
 try:r=run_phase(a.phase);print(json.dumps({'phase':a.phase,'status':r['status']}));return 0 if r['status'] in ('cpu_committed','intel_committed','nvidia_committed','candidate_verified') else 2
 except Exception as e:failure(a.phase,e);raise
if __name__=='__main__':raise SystemExit(main())
