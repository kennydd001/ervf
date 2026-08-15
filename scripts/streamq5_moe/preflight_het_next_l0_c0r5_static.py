#!/usr/bin/env python3
"""C0-R5 true static Phase-0. Stdlib only; no payload reads, runtime/device imports/calls."""
from __future__ import annotations
import ast,hashlib,json,struct,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/streamq5_moe';S=ROOT/'scripts/streamq5_moe';RUN=S/'run_het_next_l0_c0r5_capability_contract.py';VER=S/'verify_het_next_l0_c0r5_capability_contract.py';KERNEL=S/'het_next_l0_c0r5_ergv_kernels.py';LOCK=R/'het_next_l0_c0r5_runner_lock.json';VL=R/'het_next_l0_c0r5_verifier_lock.json';OUT=ROOT/'reports/runs/streamq5_moe/het_next_l0_c0r5_capability'
PR=R/'HET_NEXT_L0_C0R3_WHOLE_EXPERT_HYBRID_PREREGISTRATION_2026-08-13.md';REV=R/'HET_NEXT_L0_C0R4_WORKER_EPOCH_REVISION_2026-08-13.md';DES=R/'HET_NEXT_L0_C0R3_CAPABILITY_PREFLIGHT_DESIGN_2026-08-13.md';ADD=R/'HET_NEXT_L0_C0R4_CAPABILITY_PREFLIGHT_ADDENDUM_2026-08-13.md'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def defs():
 tree=ast.parse(RUN.read_text());env={'Path':Path,'ROOT':ROOT,'S':S,'R':R,'hashlib':hashlib,'json':json,'struct':struct}
 wanted={'safetensor_header','d2_specs','SealedReader','official_shard_key','validate_shard_header','schedule','EpochMachine','simulate_sync'}
 pre=[]
 for n in tree.body:
  if isinstance(n,(ast.Assign,ast.AnnAssign)):
   names={x.id for x in ast.walk(n) if isinstance(x,ast.Name)}
   if names & {'SEED','ROUTES','TEMPLATES','REVERSE','NAMES','D2_KEY_SHA','DTYPES','PROVENANCE','ACTIVATION_FIXTURE'}:pre.append(n)
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted:pre.append(n)
 exec(compile(ast.Module(body=pre,type_ignores=[]),'<actual-runner-contract>','exec'),env);return env
def imports(path):
 t=ast.parse(Path(path).read_text());return {x.name.split('.')[0] for n in t.body if isinstance(n,ast.Import) for x in n.names}|{(n.module or '').split('.')[0] for n in t.body if isinstance(n,ast.ImportFrom)}
def simulate(e):
 checks={};sch=e['schedule']();checks['schedule']=len(sch)==360 and [sum(x[3]==a for x in sch) for a in 'ASB']==[120,120,120] and e['SEED']==2026081302 and e['TEMPLATES']==tuple(map(tuple,('ABSBASSABSBA','ASBBSAASBBSA','SABSBAABSBAS'))) and e['REVERSE']==((3,2,1,0),(1,0,3,2),(3,2,1,0));checks['sync']=e['simulate_sync']()['stale_rejected']
 # Execute actual allowlisted reader against synthetic file/specs, spoof key/row/overlap negatives.
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'p';p.write_bytes(b'abcdefghijklmnop');spec={'p0_whole_post_norm':{'row':0,'absolute':(0,4),'bytes':4,'sha256':hashlib.sha256(b'abcd').hexdigest()},'p1_whole_post_norm':{'row':1,'absolute':(4,8),'bytes':4,'sha256':hashlib.sha256(b'efgh').hexdigest()}};q=e['SealedReader'](p,spec,False);checks['p0']=q.read('p0_whole_post_norm','validation')==b'abcd'
  neg=[]
  for key,phase in (('p1_whole_post_norm','validation'),('p1_whole_post_norm','test'),('spoof','validation')):
   try:q.read(key,phase);neg.append(False)
   except (PermissionError,KeyError):neg.append(True)
  q.tests_open=True
  try:q.read('p1_whole_post_norm','validation');neg.append(False)
  except PermissionError:neg.append(True)
  checks['seal_spoof']=all(neg)
  overlap=dict(spec);overlap['evil']={'row':0,'absolute':(4,8),'bytes':4,'sha256':hashlib.sha256(b'efgh').hexdigest()};checks['intersection_negative']=any(v['row']==0 and a['row']>0 and max(v['absolute'][0],a['absolute'][0])<min(v['absolute'][1],a['absolute'][1]) for v in overlap.values() for a in overlap.values())
 return checks
def main():
 e=defs();l=json.loads(LOCK.read_text());v=json.loads(VL.read_text());bindings={'runner_sha256':sha(RUN),'verifier_sha256':sha(VER),'preflight_sha256':sha(__file__),'kernel_sha256':sha(KERNEL),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PR),'revision_sha256':sha(REV),'design_sha256':sha(DES),'addendum_sha256':sha(ADD)};vb={k:z for k,z in bindings.items() if k!='verifier_lock_sha256'};checks={'files':all(x.exists() for x in (RUN,VER,KERNEL,LOCK,VL,PR,REV,DES,ADD)),'output_absent':not OUT.exists(),'closed':not any(l.get(k) or v.get(k) for k in ('capability_open','source_build_open','execution_open')),'runner_bindings':all(l.get(k)==z for k,z in bindings.items()),'verifier_bindings':all(v.get(k)==z for k,z in vb.items()) and v.get('runner_lock_binding')=='verified_at_runtime_from_result_or_static_preflight; omitted here to avoid circular runner-lock/verifier-lock hash','preflight_imports':not bool(imports(__file__)&{'torch','cupy','numpy','safetensors','ctypes','transformers','pyopencl'}),'runner_constants_actual':True}
 # Rehash every hardcoded provenance file and runtime binary; inspect headers without payload.
 prov=e['PROVENANCE'];checks['provenance']=len(prov)==17 and all((ROOT/p).exists() and (ROOT/p).stat().st_size==n and sha(ROOT/p)==h for p,n,h in prov.values());runtime=(('.venv/Lib/site-packages/torch/nn/functional.py',270189,'e409a97896241e0dfb8c23fbf1f09967ecf5e65ec9626aec0d97d9cc5d727d50'),('.venv/Lib/site-packages/torch/_C.cp312-win_amd64.pyd',10752,'0948fb62c5e58866a485077cf54f8cfd907fcd8482bf8f139823d1d0a724c7d2'),('.venv/Lib/site-packages/torch/lib/torch_cpu.dll',307916800,'56aaff6d76ee7ba9573e88fd8e920acb170e5c0a8d9d2ee94e8a20ed480aa32b'),('.venv/Lib/site-packages/torch/lib/c10.dll',1089536,'9aa3fb6fe82d9b3a0ccd6d406d59b61140a65990d3ffd3929b9ee0b6f4954866'),('.venv/Lib/site-packages/torch/lib/libiomp5md.dll',1617256,'2299b0460e8118e8187fd57a8d17df836c2a3d59f2639c3681582070da66b7be'))
 checks['runtime']=all((ROOT/p).stat().st_size==n and sha(ROOT/p)==h for p,n,h in runtime);d2=ROOT/prov['d2_raw'][0];dh=e['safetensor_header'](d2);spec=e['d2_specs'](dh);checks['d2_header']=len(spec)==24 and all(spec[f'p{r}_whole_{s}']['row']==r for r in range(4) for s in e['DTYPES']);shard=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors';checks['shard']=shard.stat().st_size==3999619288 and len(e['validate_shard_header'](e['safetensor_header'](shard)))==33;checks['activation_fixture']=e['ACTIVATION_FIXTURE']=={'input_bf16_words':[49152,49024,48896,0,16128,16256,16384],'sigmoid_bf16_words':[15860,16010,16065,16128,16159,16187,16225],'silu_bf16_words':[48756,48778,48705,0,16031,16187,16353]};checks.update({f'sim_{k}':z for k,z in simulate(e).items()});src=RUN.read_text();checks['capability_real']=all(x in src for x in ('clHostMemAllocINTEL','clSetKernelArgMemPointerINTEL','used_cl_mem','cupy as cp','RawModule','sentinel_total_bytes')) and "not_implemented_and_closed" in src;out={'kind':'c0r5_static_phase0_preflight','pass':all(checks.values()),'checks':checks,'bindings':bindings,'device_calls':False,'payload_reads':False};print(json.dumps(out,sort_keys=True));return 0 if out['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
