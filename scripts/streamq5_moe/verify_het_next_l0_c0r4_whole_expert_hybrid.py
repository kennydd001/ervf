#!/usr/bin/env python3
"""Independent C0-R4 verifier; never imports runner, codec, kernel or helpers."""
from __future__ import annotations
import argparse, hashlib, json, math, struct, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/streamq5_moe';S=ROOT/'scripts/streamq5_moe'
LOCK=R/'het_next_l0_c0r4_verifier_lock.json'; RUNLOCK=R/'het_next_l0_c0r4_runner_lock.json'
RUNNER=S/'run_het_next_l0_c0r4_whole_expert_hybrid.py';PREF=S/'preflight_het_next_l0_c0r4_static.py';KERNEL=S/'het_next_l0_c0r4_kernel_contract.py'
PR=R/'HET_NEXT_L0_C0R3_WHOLE_EXPERT_HYBRID_PREREGISTRATION_2026-08-13.md';REV=R/'HET_NEXT_L0_C0R4_WORKER_EPOCH_REVISION_2026-08-13.md';DES=R/'HET_NEXT_L0_C0R3_CAPABILITY_PREFLIGHT_DESIGN_2026-08-13.md';ADD=R/'HET_NEXT_L0_C0R4_CAPABILITY_PREFLIGHT_ADDENDUM_2026-08-13.md'
OUT=ROOT/'reports/runs/streamq5_moe/het_next_l0_c0r4_whole_expert_hybrid'
ROUTES=((50,199,237,474,245,374,239,8,168,12),(42,162,267,299,467,307,326,145,297,182),(474,232,382,80,31,450,103,372,286,206),(26,159,28,176,253,84,431,294,386,356))
T=(tuple('ABSBASSABSBA'),tuple('ASBBSAASBBSA'),tuple('SABSBAABSBAS'));REV=((3,2,1,0),(1,0,3,2),(3,2,1,0));SEED=2026081302

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def expected_schedule():
 rows=[]
 for b in range(30):
  ti=(SEED+b)%3
  for k,a in enumerate(T[ti]):
   g=k//3;rows.append({'observation':len(rows),'block':b,'template':ti,'group':g,'arm':a,'pair':(b,min(g,REV[ti][g]),max(g,REV[ti][g]))})
 return rows
def qlin(v,q):
 v=sorted(float(x) for x in v);h=(len(v)-1)*q;a=math.floor(h);b=math.ceil(h);return v[a]+(h-a)*(v[b]-v[a])
def check_lifecycle(rows):
 last={'intel':0,'nvidia':0};ack={'intel':0,'nvidia':0};ok=True
 for r in rows:
  active=tuple(r['active']);epoch=r['epoch'];ok &= epoch>max(last.values())
  for i in active:ok &= r['prior_ack'][i]==last[i] and r['prior_last'][i]==last[i];last[i]=epoch
  ok &= r['command_order']==[i for i in ('intel','nvidia') if i in active]
  ok &= r['ready_before_t0'] and r['t0']<=r['start_qpc']<=r['submit_qpc_min']<=r['done_qpc_max']<=r['merge_qpc']<=r['t1']
  for i in active:ack[i]=r['observed_ack'][i];ok &= ack[i]==last[i]
  for i in set(last)-set(active):ok &= r['prior_last'][i]==last[i] and r['observed_ack'][i]==ack[i]
  ok &= r['start_reset_after_ack']
 return bool(ok)
def verify_static():
 l=json.loads(LOCK.read_text());expect={'verifier_sha256':sha(__file__),'runner_sha256':sha(RUNNER),'preflight_sha256':sha(PREF),'kernel_sha256':sha(KERNEL),'runner_lock_sha256':sha(RUNLOCK),'prereg_sha256':sha(PR),'revision_sha256':sha(REV),'design_sha256':sha(DES),'addendum_sha256':sha(ADD)}
 checks={f'bind_{k}':l.get(k)==v for k,v in expect.items()};checks['closed']=all(l.get(x) is False for x in ('capability_open','source_build_open','execution_open'));checks['output_absent']=not OUT.exists();checks['schedule']=len(expected_schedule())==360 and all(sum(r['arm']==a for r in expected_schedule())==120 for a in 'ASB')
 return {'kind':'het_next_l0_c0r4_static_verification','pass':all(checks.values()),'checks':checks,'bindings':expect}
def raw_words(t):return t.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()
def finite_tensor(t):return bool(__import__('torch').isfinite(t.float()).all())
def verify_evidence():
 import torch
 from safetensors import safe_open
 result_path=OUT/'c0r4_result.json';raw_path=OUT/'c0r4_raw.safetensors';commit_path=OUT/'c0r4_commit.json';failure=OUT/'c0r4_failure.json'
 if failure.exists():
  f=json.loads(failure.read_text());return {'kind':'het_next_l0_c0r4_evidence_verification','pass':False,'valid_failure':f.get('kind')=='het_next_l0_c0r4_failure' and f.get('runner_sha256')==sha(RUNNER),'failure':f}
 if not all(x.exists() for x in (result_path,raw_path,commit_path)):return {'kind':'het_next_l0_c0r4_evidence_verification','pass':False,'error':'missing_bundle'}
 r=json.loads(result_path.read_text());c=json.loads(commit_path.read_text());checks={}
 checks['commit']=c.get('files')=={raw_path.name:{'bytes':raw_path.stat().st_size,'sha256':sha(raw_path)},result_path.name:{'bytes':result_path.stat().st_size,'sha256':sha(result_path)}}
 checks['provenance']=r.get('runner_sha256')==sha(RUNNER) and r.get('verifier_sha256')==sha(__file__) and r.get('runner_lock_sha256')==sha(RUNLOCK) and r.get('verifier_lock_sha256')==sha(LOCK)
 checks['routes']=tuple(tuple(x) for x in r.get('routes',()))==ROUTES
 checks['schedule']=r.get('schedule')==expected_schedule()
 checks['seal']=all(not x.get('completed') or x.get('row')==0 for x in r['validation_access_ledger']) and r['tests_opened_after_validation'] is True
 checks['test_ledger']=all(x.get('row') in (1,2,3) and x.get('completed') for x in r['test_access_ledger'])
 checks['lifecycle']=check_lifecycle(r['worker_lifecycle'])
 checks['resources']=r['resources']['start_available']>=16*2**30 and r['resources']['min_available']>=2*2**30 and r['resources']['peak_wset']<=12*2**30 and r['resources']['intel_allocated']<2**30 and r['resources']['nvidia_allocated']<256*2**20 and r['resources']['retained_bytes']<10*2**20
 checks['pdh']=r['pdh']['valid'] and all(80<=x['interval_ms']<=120 and x['lateness_ms']<=20 for x in r['pdh']['samples'][1:]) and not r['pdh']['paging_triggered']
 checks['clocks']=r['clock_thermal']['valid'] and not r['clock_thermal']['triggered']
 with safe_open(raw_path,framework='pt',device='cpu') as sf:
  actual={k:{'dtype':str(sf.get_tensor(k).dtype),'shape':list(sf.get_tensor(k).shape),'bytes':sf.get_tensor(k).numel()*sf.get_tensor(k).element_size(),'sha256':hashlib.sha256(raw_words(sf.get_tensor(k))).hexdigest()} for k in sf.keys()}
  checks['raw_manifest']=actual==r['raw_manifest'] and all(finite_tensor(sf.get_tensor(k)) for k in sf.keys())
  # Device and oracle arrays are compared independently for every declared binding.
  exact=True
  for b in r['bitwise_bindings']:
   a=sf.get_tensor(b['oracle_key']);z=sf.get_tensor(b['device_key']);exact &= torch.equal(a,z) and hashlib.sha256(raw_words(a)).hexdigest()==b['oracle_sha256'] and hashlib.sha256(raw_words(z)).hexdigest()==b['device_sha256']
  checks['bitwise']=bool(exact)
 for row in range(4):
  s=r['samples'][str(row)];checks[f'count_p{row}']=all(len(s[a])==120 and all(math.isfinite(float(x)) and x>0 for x in s[a]) for a in 'ASB')
  q={a:{p:qlin(s[a],p) for p in (.5,.95)} for a in 'ASB'};calc={'p50_ratio':q['B'][.5]/q['A'][.5],'p95_ratio':q['B'][.95]/q['A'][.95],'p50_b_lt_s':q['B'][.5]<q['S'][.5],'p95_b_lt_s':q['B'][.95]<q['S'][.95]};checks[f'stats_p{row}']=calc==r['statistics'][str(row)] and calc['p50_ratio']<=.90 and calc['p95_ratio']<=.95 and calc['p50_b_lt_s'] and calc['p95_b_lt_s']
 checks['controls']=all(x['rejected_before_enqueue'] and x['enqueue_count_before']==x['enqueue_count_after'] and x['unsafe_changed'] for x in r['controls'])
 checks['cleanup']=all(v['before']==v['after'] and v['released_once'] for v in r['cleanup'].values())
 checks['status']=r.get('status')=='heterogeneous_component_positive' and all(checks.values())
 return {'kind':'het_next_l0_c0r4_evidence_verification','pass':all(checks.values()),'checks':checks}
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('static','evidence'),required=True);a=p.parse_args();o=verify_static() if a.phase=='static' else verify_evidence();print(json.dumps(o,sort_keys=True));return 0 if o['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
