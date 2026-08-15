#!/usr/bin/env python3
"""C0-R6 true static Phase-0: never opens D2/shard/runtime/device libraries."""
from __future__ import annotations
import ast,hashlib,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/streamq5_moe';S=ROOT/'scripts/streamq5_moe';RUN=S/'run_het_next_l0_c0r6_capability.py';VER=S/'verify_het_next_l0_c0r6_capability.py';PREF=Path(__file__);KERNEL=S/'het_next_l0_c0r6_kernels.py';SYNC=S/'het_next_l0_c0r6_sync.py';LOCK=R/'het_next_l0_c0r6_runner_lock.json';VL=R/'het_next_l0_c0r6_verifier_lock.json';PM=R/'het_next_l0_c0r6_recorded_provenance_manifest.json';TM=R/'het_next_l0_c0r6_d2_sealed_tensor_manifest.json';OUT=ROOT/'reports/runs/streamq5_moe/het_next_l0_c0r6_capability';PR=R/'HET_NEXT_L0_C0R3_WHOLE_EXPERT_HYBRID_PREREGISTRATION_2026-08-13.md';REV=R/'HET_NEXT_L0_C0R4_WORKER_EPOCH_REVISION_2026-08-13.md';DES=R/'HET_NEXT_L0_C0R3_CAPABILITY_PREFLIGHT_DESIGN_2026-08-13.md';ADD=R/'HET_NEXT_L0_C0R4_CAPABILITY_PREFLIGHT_ADDENDUM_2026-08-13.md'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def imports(p):
 t=ast.parse(Path(p).read_text());return {x.name.split('.')[0] for n in t.body if isinstance(n,ast.Import) for x in n.names}|{(n.module or '').split('.')[0] for n in t.body if isinstance(n,ast.ImportFrom)}
def load_runner_contract():
 tree=ast.parse(RUN.read_text());env={'Path':Path,'hashlib':hashlib,'json':json,'MappingProxyType':__import__('types').MappingProxyType}
 names={'_VALIDATION_TOKEN','_TEST_TOKEN','SEED','ROUTES','TEMPLATES','REVERSE'};wanted={'schedule','SealedReader'};body=[]
 for n in tree.body:
  if isinstance(n,(ast.Assign,ast.AnnAssign)) and ({x.id for x in ast.walk(n) if isinstance(x,ast.Name)}&names):body.append(n)
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted:body.append(n)
 exec(compile(ast.Module(body=body,type_ignores=[]),'<runner-contract>','exec'),env);return env
def seal_sim(e):
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'tiny';p.write_bytes(b'abcdefghijklmnop');spec={'p0':{'row':0,'absolute':(0,4),'bytes':4,'sha256':hashlib.sha256(b'abcd').hexdigest()},'p1':{'row':1,'absolute':(4,8),'bytes':4,'sha256':hashlib.sha256(b'efgh').hexdigest()}};q=e['SealedReader'](p,spec);checks=[q.read('p0',e['_VALIDATION_TOKEN'])==b'abcd']
  for key,tok in (('p1',e['_VALIDATION_TOKEN']),('p1',object()),('spoof',e['_VALIDATION_TOKEN'])):
   try:q.read(key,tok);checks.append(False)
   except (PermissionError,KeyError):checks.append(True)
  try:e['SealedReader'](p,{**spec,'evil':{'row':0,'absolute':(3,7),'bytes':4,'sha256':'0'*64}});checks.append(False)
  except ValueError:checks.append(True)
  q.open_tests({'status':'validation_pass','verified':True});checks.append(q.read('p1',e['_TEST_TOKEN'])==b'efgh')
  return all(checks)
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VL.read_text());pm=json.loads(PM.read_text());tm=json.loads(TM.read_text());e=load_runner_contract();bindings={'runner_sha256':sha(RUN),'verifier_sha256':sha(VER),'preflight_sha256':sha(PREF),'kernel_sha256':sha(KERNEL),'sync_sha256':sha(SYNC),'provenance_manifest_sha256':sha(PM),'tensor_manifest_sha256':sha(TM),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PR),'revision_sha256':sha(REV),'design_sha256':sha(DES),'addendum_sha256':sha(ADD)};vb={k:z for k,z in bindings.items() if k!='verifier_lock_sha256'};checks={}
 checks['files']=all(x.exists() for x in (RUN,VER,PREF,KERNEL,SYNC,LOCK,VL,PM,TM,PR,REV,DES,ADD));checks['closed']=not any(l.get(k) or v.get(k) for k in ('capability_open','source_build_open','execution_open'));checks['absent']=not OUT.exists();checks['bindings']=all(l.get(k)==z for k,z in bindings.items()) and all(v.get(k)==z for k,z in vb.items());checks['static_imports']=not bool(imports(PREF)&{'ctypes','torch','cupy','numpy','safetensors','transformers','pyopencl'});checks['no_payload_paths']=all('d2_raw' not in str(x).lower() and 'model-00001' not in str(x).lower() for x in (PREF,));checks['manifest_recorded']=pm['payloads']['d2_raw']['fresh_verification_phase']=='source_build_closed' and pm['payloads']['official_shard1']['fresh_verification_phase']=='source_build_closed' and tm['payload_verification']=='closed until source_build';checks['small_evidence']=len(pm['small_evidence'])==12 and all((ROOT/x['path']).exists() and (ROOT/x['path']).stat().st_size==x['bytes'] and sha(ROOT/x['path'])==x['sha256'] for x in pm['small_evidence'].values());checks['tensor_manifest']=len(tm['keys'])==24 and len(tm['route_ids'])==4 and all(len(x)==10 for x in tm['route_ids']);ranges=sorted((x['absolute'][0],x['absolute'][1],k) for k,x in tm['keys'].items());checks['manifest_no_overlap']=all(ranges[i][1]<=ranges[i+1][0] for i in range(len(ranges)-1));checks['schedule_actual']=len(e['schedule']())==360 and [sum(x[3]==a for x in e['schedule']()) for a in 'ASB']==[120,120,120] and e['SEED']==2026081302;checks['sealed_actual']=seal_sim(e)
 # Execute actual shared simulator, never physical primitives.
 st=ast.parse(SYNC.read_text());env={'threading':__import__('threading'),'time':__import__('time')};body=[n for n in st.body if not isinstance(n,(ast.Import,ast.ImportFrom))];exec(compile(ast.Module(body=body,type_ignores=[]),'<sync-actual>','exec'),env);sim=env['simulate_protocol']();checks['sync_actual']=sim['pass'] and len(sim['outputs'])==5 and all(sim['negative'].values()) and sum(x[0]=='CreateEventW' for x in sim['calls'])==9;src=RUN.read_text();ks=KERNEL.read_text();checks['capability_source']=all(x in src for x in ('clHostMemAllocINTEL','clSetKernelArgMemPointerINTEL','PdhAddEnglishCounterW','WinPrimitives','threading.Barrier','RawModule','c0r6_ergv8_sentinel','not_implemented_and_closed'));checks['kernel']=all(x in ks for x in ('#define WIDTH 8','#define VIRTUAL 32','stride=128','intel_sub_group_shuffle_down','__shfl_down_sync','--fmad=false'))
 o={'kind':'c0r6_static_phase0_preflight','pass':all(checks.values()),'checks':checks,'bindings':bindings,'opened_payload_files':0,'device_calls':0};print(json.dumps(o,sort_keys=True));return 0 if o['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
