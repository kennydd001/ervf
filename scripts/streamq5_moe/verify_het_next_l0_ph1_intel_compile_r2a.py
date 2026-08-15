#!/usr/bin/env python3
"""Independent read-only verifier for the PH1 Intel compile-only R2A bundle."""
from __future__ import annotations
import ast, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/streamq5_moe'; S=ROOT/'scripts/streamq5_moe'
PKG=R/'het_next_l0_ph1_intel_compile_r2a'; OUT=R/'het_next_l0_ph1_intel_compile_r2a_independent_verification.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def const(path,name):
 tree=ast.parse(Path(path).read_text(encoding='utf-8'))
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets): return ast.literal_eval(n.value)
 raise RuntimeError(name)
def source():
 x=const(S/'het_next_l0_ph1_intel_backend.py','SRC')
 old='#pragma OPENCL EXTENSION cl_intel_required_sub_group_size : enable\n#pragma OPENCL EXTENSION cl_khr_int64 : enable\n'
 x=x.replace(old,'#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n')
 return x.replace('#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n','',1).replace('ulong half =','ulong halfway =',1).replace('remainder > half || (remainder == half','remainder > halfway || (remainder == halfway',1)
def main():
 result=load(PKG/'result.json'); manifest=load(PKG/'manifest.json'); commit=load(PKG/'commit.json'); c=result['compile']; ledger=c['ledger']
 actual={p.name:(p.stat().st_size,sha(p)) for p in PKG.iterdir() if p.is_file()}
 rows={r['name']:(r['bytes'],r['sha256']) for r in manifest['files']}
 expected={'commit.json','manifest.json'}|set(rows)
 observed=result['bindings']['observed']
 paths={
  'authorization_preflight_sha256':S/'preflight_het_next_l0_ph1_intel_compile_r2a.py','authorization_sha256':R/'HET_NEXT_L0_PH1_INTEL_COMPILE_R2A_AUTHORIZATION_2026-08-14.md','backend_sha256':S/'het_next_l0_ph1_intel_compile_r2a_backend.py','cpu_commit_sha256':R/'het_next_l0_ph1_cpu_freeze_r2/commit.json','cpu_verification_sha256':R/'het_next_l0_ph1_cpu_freeze_r2_independent_verification.json','prior_audit_sha256':R/'HET_NEXT_L0_PH1_INTEL_COMPILE_R0_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md','r1b_failure_sha256':R/'het_next_l0_ph1_intel_compile_r1b_failed_attempts/attempt_failure_06df3c72c9c44379a04d39b43d301b53/failure.json','r2_backend_sha256':S/'het_next_l0_ph1_intel_compile_r2_backend.py','r2_closed_lock_sha256':R/'het_next_l0_ph1_intel_compile_r2_lock.json','r2_design_sha256':R/'HET_NEXT_L0_PH1_INTEL_COMPILE_R2_SOURCE_REVISION_2026-08-14.md','r2_prereg_sha256':R/'HET_NEXT_L0_PH1_INTEL_COMPILE_R2_PREREGISTRATION_2026-08-14.md','r2_runner_sha256':S/'run_het_next_l0_ph1_intel_compile_r2.py','r2_source_module_sha256':S/'het_next_l0_ph1_intel_compile_r2_source.py','r2p1_lock_sha256':R/'het_next_l0_ph1_intel_compile_r2p1_lock.json','r2p1_pass_sha256':R/'het_next_l0_ph1_intel_compile_r2p1_static_preflight.json','r2p1_preflight_sha256':S/'preflight_het_next_l0_ph1_intel_compile_r2p1.py','r2p1_revision_sha256':R/'HET_NEXT_L0_PH1_INTEL_COMPILE_R2P1_PREFLIGHT_REVISION_2026-08-14.md','runner_sha256':S/'run_het_next_l0_ph1_intel_compile_r2a.py'}
 src=source().encode(); binary=(PKG/'intel_program.bin').read_bytes(); log=(PKG/'intel_build.log').read_bytes()
 checks={
  'exact_files':set(actual)==expected and all(actual[k]==v for k,v in rows.items()),
  'manifest_commit':manifest['kind']=='het_next_l0_ph1_intel_compile_r2a_manifest' and commit=={'kind':'het_next_l0_ph1_intel_compile_r2a_commit','manifest_sha256':sha(PKG/'manifest.json'),'result_sha256':sha(PKG/'result.json')},
  'source_exact':len(src)==7852 and hashlib.sha256(src).hexdigest()==sha(PKG/'intel_source.cl')=='f1b3ccdae6d202ed210810e3cd419f726ea89ffa8fba0c84df5c2bfca3a84d21' and (PKG/'intel_source.cl').read_bytes()==src,
  'binary_exact_nonempty':len(binary)==c['declared_binary_bytes']==c['read_binary_bytes']==result['artifacts']['binary_bytes']==186352 and sha(PKG/'intel_program.bin')==c['binary_sha256']==result['artifacts']['binary_sha256']=='8b57db279fbb1d7d8df17ebab5cfb54203ef8da8cc31df2d136650820548f629',
  'build_log_exact':len(log)==1 and sha(PKG/'intel_build.log')==c['build_log_sha256']==result['artifacts']['build_log_sha256'],
  'identity':c['identity']['name']=='Intel(R) Arc(TM) Pro 140T GPU (32GB)' and c['identity']['vendor']=='Intel(R) Corporation' and c['identity']['driver']=='32.0.101.8517' and c['identity']['pci']=='0000:00:02.0' and 'cl_intel_unified_shared_memory' in c['identity']['extensions'],
  'ledger':len(ledger)==8 and [x['op'] for x in ledger]==['identity','context_create','program_create','program_build','program_binary_read','release','release','cleanup'] and ledger[3]['code']==0 and ledger[4]['declared_bytes']==ledger[4]['read_bytes']==186352 and ledger[5:7]==[{'code':0,'name':'program','op':'release'},{'code':0,'name':'context','op':'release'}] and ledger[-1]['cleanup_complete'] is True and ledger[-1]['live_owned_resources']==0,
  'zero_forbidden':c['payload_read'] is False and all(c[k]==0 for k in ('queues_created','kernels_created','events_created','memory_objects_created','allocations','kernels_launched')),
  'positive_schema':result['kind']=='het_next_l0_ph1_intel_compile_r2a' and result['status']=='compile_positive' and result['positive'] is True and c['binary_nonempty'] is True and c['queried_program_devices']==1 and c['cleanup_errors']==[],
  'provenance':all(observed[k]==sha(v) for k,v in paths.items()) and observed['source_sha256']==hashlib.sha256(src).hexdigest() and result['bindings']['lock_sha256']==sha(R/'het_next_l0_ph1_intel_compile_r2a_lock.json') and c['authorization']['lock_sha256']==result['bindings']['lock_sha256'] and c['authorization']['audit_token']=='PH1_INTEL_COMPILE_R2A_AFTER_R2P1_PASS_AND_INDEPENDENT_FINAL_AUDIT_GO',
 }
 out={'kind':'het_next_l0_ph1_intel_compile_r2a_independent_verification','pass':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'binary':{'bytes':len(binary),'sha256':hashlib.sha256(binary).hexdigest()},'source_sha256':hashlib.sha256(src).hexdigest(),'claim':'compile eligibility only; no payload, kernel creation/launch, numerical correctness, timing or performance evidence'}
 OUT.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n',encoding='utf-8'); print(json.dumps(out,indent=2)); return 0 if out['pass'] else 3
if __name__=='__main__': raise SystemExit(main())
