#!/usr/bin/env python3
"""Truly static C0-R4 Phase-0 preflight. Standard library only; no payload/device calls."""
from __future__ import annotations
import ast, hashlib, json, math, os, struct, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/streamq5_moe';S=ROOT/'scripts/streamq5_moe'
RUN=S/'run_het_next_l0_c0r4_whole_expert_hybrid.py';VER=S/'verify_het_next_l0_c0r4_whole_expert_hybrid.py';KERNEL=S/'het_next_l0_c0r4_kernel_contract.py';LOCK=R/'het_next_l0_c0r4_runner_lock.json';VL=R/'het_next_l0_c0r4_verifier_lock.json'
PR=R/'HET_NEXT_L0_C0R3_WHOLE_EXPERT_HYBRID_PREREGISTRATION_2026-08-13.md';REV=R/'HET_NEXT_L0_C0R4_WORKER_EPOCH_REVISION_2026-08-13.md';DES=R/'HET_NEXT_L0_C0R3_CAPABILITY_PREFLIGHT_DESIGN_2026-08-13.md';ADD=R/'HET_NEXT_L0_C0R4_CAPABILITY_PREFLIGHT_ADDENDUM_2026-08-13.md';OUT=ROOT/'reports/runs/streamq5_moe/het_next_l0_c0r4_whole_expert_hybrid'
D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors';SHARD=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors'
EXPECTED={'d2_size':D2.stat().st_size if D2.exists() else -1,'shard_size':3999619288,'shard_sha_declared':'8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def subset_exec(source,names,env):
 tree=ast.parse(source);body=[]
 for n in tree.body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:body.append(n)
 exec(compile(ast.Module(body=body,type_ignores=[]),'<static-subset>','exec'),env);return env
def import_audit(path):
 tree=ast.parse(Path(path).read_text());top=[]
 for n in tree.body:
  if isinstance(n,ast.Import):top += [x.name.split('.')[0] for x in n.names]
  elif isinstance(n,ast.ImportFrom):top.append((n.module or '').split('.')[0])
 return not bool(set(top)&{'torch','cupy','pyopencl','safetensors','transformers','ctypes'}),top
def runner_subset():
 env={'hashlib':hashlib,'json':json,'math':math,'Path':Path,'struct':struct,'SEED':2026081302,'TEMPLATES':(('A','B','S','B','A','S','S','A','B','S','B','A'),('A','S','B','B','S','A','A','S','B','B','S','A'),('S','A','B','S','B','A','A','B','S','B','A','S')),'REVERSE':((3,2,1,0),(1,0,3,2),(3,2,1,0))}
 return subset_exec(RUN.read_text(),{'schedule','linear_q','gates','sm64','thrash_small','EpochMachine','simulate_sync','parse_header','SealedReader'},env)
def independent_schedule():
 ts=('ABSBASSABSBA','ASBBSAASBBSA','SABSBAABSBAS');out=[]
 for b in range(30):out.extend(ts[(2026081302+b)%3])
 return out
def simulation():
 e=runner_subset();sch=e['schedule']();sync=e['simulate_sync']();checks={}
 checks['schedule_len_counts']=len(sch)==360 and [sum(x['arm']==a for x in sch) for a in 'ASB']==[120,120,120] and ''.join(x['arm'] for x in sch)==''.join(independent_schedule())
 checks['schedule_pairs']=all(x['pair'][1]<=x['group']<=x['pair'][2] and x['pair'][1]!=x['pair'][2] for x in sch)
 checks['sync_actual']=sync['pass'] and all(sync['negative'])
 # Independent lifecycle prefix including inactive exclusion.
 M=e['EpochMachine'];m=M();m.arm(('nvidia',));m.release(('nvidia',));m.worker_done('nvidia');m.collect(('nvidia',));intel=(m.last['intel'],m.ack['intel']);m.arm(('intel','nvidia'));m.release(('intel','nvidia'));m.worker_done('intel');m.worker_done('nvidia');m.collect(('intel','nvidia'));checks['inactive_excluded']=intel==(0,0) and m.ack==m.last
 # Small actual thrash function and independent recurrence.
 n=4096;b=bytearray(e['sm64'](2026081302^i)&255 for i in range(n));c=bytearray(b);a=e['thrash_small'](b,'p0','warmup',0,0);L=n//64;start=int.from_bytes(hashlib.sha256(b'HET-NEXT-L0-C0-R2|p0|warmup|0').digest()[:8],'little')%L;v=0;counter=0
 for k in range(L):o=64*((start+k)%L);old=c[o];v^=old;c[o]=(old+(e['sm64'](2026081302^counter)&255))&255;counter+=1
 checks['thrash_actual']=a==(start,v,counter) and b==c
 # Execute actual sealed reader on TEMP bytes; no D2 payload.
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'tiny.bin';p.write_bytes(b'0123456789abcdef');h={'p0':{'absolute':[0,4]},'p1':{'absolute':[4,8]}};sr=e['SealedReader'](p,h,False);checks['p0_read']=sr.read('p0',0,'validation')==b'0123'
  try:sr.read('p1',1,'validation');checks['test_sealed']=False
  except PermissionError:checks['test_sealed']=True
  sr.opened=True;checks['test_open_after']=sr.read('p1',1,'test')==b'4567'
 # Fixed arrays exercise exact linear quantile gates.
 samples={'A':[10+i/100 for i in range(120)],'B':[8+i/100 for i in range(120)],'S':[9+i/100 for i in range(120)]};g=e['gates'](samples);checks['stats']=g['p50_ratio']<=.9 and g['p95_ratio']<=.95 and g['p50_b_lt_s'] and g['p95_b_lt_s']
 return checks
def source_contract():
 runner=RUN.read_text();ver=VER.read_text();kern=KERNEL.read_text();rt=ast.parse(runner);vt=ast.parse(ver);checks={};ok,imports=import_audit(__file__);checks['preflight_static_imports']=ok
 calls=[n.func.id for n in ast.walk(ast.parse(Path(__file__).read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
 checks['no_device_calls']=not bool(set(calls)&{'WinDLL','CDLL','RawModule','Device','clGetPlatformIDs','cudaSetDevice'})
 checks['closed_stubs']=all(x in runner for x in ("capability_open') is False","source_build_open') is False","execution_open') is False","capability backend remains closed","source builder remains closed","physical worker/backend binding remains closed"))
 checks['runner_top_imports']=import_audit(RUN)[0]
 checks['verifier_independent']=not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(x.name.endswith(('run_het_next_l0_c0r4_whole_expert_hybrid','het_next_l0_c0r4_kernel_contract')) for x in n.names) for n in vt.body)
 checks['kernel_pipeline']=all(x in kern for x in ('q5_linear','swiglu_bf16','field31','host_activation_input','device_swiglu_bf16'))
 checks['dataflow_controls']=all(x in runner for x in ('test_payload_sealed','sorted(int(v) for v in ids)','field31','wrong','ack_epoch','last_command_epoch'))
 return checks
def main():
 files=(RUN,VER,KERNEL,PR,REV,DES,ADD,LOCK,VL);l=json.loads(LOCK.read_text());v=json.loads(VL.read_text());checks={'files':all(x.exists() for x in files),'output_absent':not OUT.exists(),'shard_exact_size':SHARD.exists() and SHARD.stat().st_size==EXPECTED['shard_size'],'locks_closed':all(l.get(x) is False and v.get(x) is False for x in ('capability_open','source_build_open','execution_open'))}
 bindings={'runner_sha256':sha(RUN),'verifier_sha256':sha(VER),'kernel_sha256':sha(KERNEL),'preflight_sha256':sha(__file__),'prereg_sha256':sha(PR),'revision_sha256':sha(REV),'design_sha256':sha(DES),'addendum_sha256':sha(ADD),'verifier_lock_sha256':sha(VL)}
 rb={k:z for k,z in bindings.items() if k!='verifier_lock_sha256'};checks['runner_bindings']=all(l.get(k)==z for k,z in rb.items());vb=dict(rb);vb['runner_lock_sha256']=sha(LOCK);checks['verifier_bindings']=all(v.get(k)==z for k,z in vb.items());checks.update({f'sim_{k}':z for k,z in simulation().items()});checks.update({f'source_{k}':z for k,z in source_contract().items()});o={'kind':'het_next_l0_c0r4_static_phase0_preflight','pass':all(checks.values()),'checks':checks,'bindings':bindings,'physical_actions':False,'device_imports':False,'payload_reads':False};print(json.dumps(o,sort_keys=True));return 0 if o['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
