#!/usr/bin/env python3
"""Independent C0-R5 static/capability verifier. No runner or kernel imports."""
from __future__ import annotations
import argparse,hashlib,json,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/streamq5_moe';S=ROOT/'scripts/streamq5_moe';OUT=ROOT/'reports/runs/streamq5_moe/het_next_l0_c0r5_capability';CAP=OUT/'c0r5_capability.json';FAIL=OUT/'c0r5_capability_failure.json';COM=OUT/'c0r5_capability_commit.json'
LOCK=R/'het_next_l0_c0r5_verifier_lock.json';RL=R/'het_next_l0_c0r5_runner_lock.json';RUN=S/'run_het_next_l0_c0r5_capability_contract.py';PREF=S/'preflight_het_next_l0_c0r5_static.py';KERNEL=S/'het_next_l0_c0r5_ergv_kernels.py';PR=R/'HET_NEXT_L0_C0R3_WHOLE_EXPERT_HYBRID_PREREGISTRATION_2026-08-13.md';REV=R/'HET_NEXT_L0_C0R4_WORKER_EPOCH_REVISION_2026-08-13.md';DES=R/'HET_NEXT_L0_C0R3_CAPABILITY_PREFLIGHT_DESIGN_2026-08-13.md';ADD=R/'HET_NEXT_L0_C0R4_CAPABILITY_PREFLIGHT_ADDENDUM_2026-08-13.md'
REQ_INTEL=('cl_intel_unified_shared_memory','cl_intel_subgroups','cl_intel_required_subgroup_size');LP=(0,2,4,6)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def static():
 l=json.loads(LOCK.read_text());x={'verifier_sha256':sha(__file__),'runner_sha256':sha(RUN),'preflight_sha256':sha(PREF),'kernel_sha256':sha(KERNEL),'prereg_sha256':sha(PR),'revision_sha256':sha(REV),'design_sha256':sha(DES),'addendum_sha256':sha(ADD)};c={f'bind_{k}':l.get(k)==v for k,v in x.items()};c['runner_lock_noncyclic']=l.get('runner_lock_binding')=='verified_at_runtime_from_result_or_static_preflight; omitted here to avoid circular runner-lock/verifier-lock hash';c['closed']=not any(l.get(k) for k in ('capability_open','source_build_open','execution_open'));c['absent']=not OUT.exists();return {'kind':'c0r5_static_verification','pass':all(c.values()),'checks':c}
def expected_sync(rows):
 act=(('nvidia',),('intel','nvidia'),('intel',),('nvidia',),('nvidia',));return len(rows)==5 and tuple(tuple(x['active']) for x in rows)==act and [x['epoch'] for x in rows]==[1,2,3,4,5]
def capability():
 if FAIL.exists():
  f=json.loads(FAIL.read_text());return {'kind':'c0r5_capability_verification','pass':False,'valid_failure':f.get('kind')=='het_next_l0_c0r5_capability_failure' and f.get('runner_sha256')==sha(RUN) and f.get('no_weight_payload') is True}
 if not all(x.exists() for x in (CAP,COM)):return {'kind':'c0r5_capability_verification','pass':False,'error':'missing'}
 r=json.loads(CAP.read_text());m=json.loads(COM.read_text());c={};c['commit']=m.get('files')=={CAP.name:{'bytes':CAP.stat().st_size,'sha256':sha(CAP)}};c['kind_status']=r.get('kind')=='het_next_l0_c0r5_capability' and r.get('status')=='capability_positive';c['source']=r.get('runner_sha256')==sha(RUN) and r.get('runner_lock_sha256')==sha(RL) and r.get('verifier_sha256')==sha(__file__) and r.get('verifier_lock_sha256')==sha(LOCK) and r.get('kernel_sha256')==sha(KERNEL);c['no_payload']=r.get('device_payload_reads')==0 and r.get('weight_payload_reads')==0 and r.get('sentinel_total_bytes')==8192
 i=r.get('intel',{});n=r.get('nvidia',{});c['intel_identity']=all(isinstance(i.get(k),str) and i[k] for k in ('platform','platform_vendor','device','vendor')) and 'Intel' in i['vendor'];c['intel_extensions']=all(x in i.get('extensions',()) for x in REQ_INTEL) and tuple(i.get('required_extensions',()))==REQ_INTEL;c['intel_usm']=i.get('buffer_bytes')==4096 and i.get('alignment')==64 and i.get('copyless_host_usm') is True and i.get('used_cl_mem') is False and i.get('used_enqueue_write') is False and i.get('used_migrate') is False and i.get('after_sha256')==i.get('expected_after_sha256')!=i.get('before_sha256') and all(v==0 for v in i.get('cleanup_counts',{}).values())
 c['nvidia_identity']=isinstance(n.get('name'),str) and bool(n['name']) and isinstance(n.get('pci_bus_id'),str) and len(n.get('compute_capability',()))==2 and n.get('total_global_mem',0)>0;c['nvidia_sentinel']=n.get('buffer_bytes')==4096 and n.get('after_sha256')==n.get('expected_after_sha256')!=n.get('before_sha256') and n.get('cleanup_pool_used_bytes')==0
 sy=r.get('sync',{});c['sync']=expected_sync(sy.get('rows',())) and sy.get('final_last')==sy.get('final_ack')=={'intel':3,'nvidia':5} and sy.get('stale_rejected') is True;c['topology_contract']=r.get('topology_required_lps')==list(LP);c['time']=isinstance(r.get('started_ns'),int) and r['ended_ns']>=r['started_ns'];c['cardinality']=len(i)==17 and len(n)==10 and len(sy)==4
 return {'kind':'c0r5_capability_verification','pass':all(c.values()),'checks':c}
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('static','capability'),required=True);a=p.parse_args();o=static() if a.phase=='static' else capability();print(json.dumps(o,sort_keys=True));return 0 if o['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
