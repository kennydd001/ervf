#!/usr/bin/env python3
"""Independent PV0-R3 artifact verifier. It imports no builder/child helpers."""
from __future__ import annotations
import argparse, hashlib, json, math, os, struct, sys, uuid
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np, torch, torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/het_next_l0_pv0r3_real_weight_process_validation';OUT=D/'pv0r3_verification.json';OR=D/'pv0r3_cpu_oracle.safetensors';ORES=D/'pv0r3_cpu_builder.json';IR=D/'pv0r3_intel_raw.npz';NR=D/'pv0r3_nvidia_raw.npz';MAN=R/'het_next_l0_pv0r2_selected_source_manifest.json';LOCK=R/'het_next_l0_pv0r3_verifier_lock.json';EXPERTS=(8,12,50,168,199,237,239,245,374,474);INTEL=(50,199,237,474);NVIDIA=(8,12,168,239,245,374)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def atomic(p,x):
 t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.inprogress');t.write_bytes(x)
 with t.open('r+b') as f:os.fsync(f.fileno())
 os.rename(t,p)
def bf(a):return torch.from_numpy(np.ascontiguousarray(a.astype('<u2'))).view(torch.bfloat16)
def metric(a,z):
 a=a.float().double().reshape(-1);z=z.float().double().reshape(-1);e=torch.linalg.vector_norm(z-a);d=torch.linalg.vector_norm(a);return float(0 if d==0 and e==0 else math.inf if d==0 else e/d)
def verify(cand):
 checks={}; l=json.loads(LOCK.read_text());checks['lock']=l.get('execution_open') is True and l.get('audit_token')=='PV0R3_SOURCE_AUDIT_GO';checks['manifest']=len(json.loads(MAN.read_text())['records'])==33 and sha(MAN)==l['manifest_sha256'];b=json.loads(ORES.read_text());checks['builder']=b['status']=='cpu_builder_positive' and b['manifest_sha256']==sha(MAN) and b['raw_sha256']==sha(OR)
 c=json.loads(Path(cand).read_text());checks['candidate']=c['status']=='candidate_requires_independent_verifier' and c['cpu_builder_sha256']==sha(ORES) and c['intel_raw_sha256']==sha(IR) and c['nvidia_raw_sha256']==sha(NR)
 i=np.load(IR,allow_pickle=False);n=np.load(NR,allow_pickle=False); from safetensors import safe_open
 with safe_open(OR,framework='pt',device='cpu') as o:
  for e in EXPERTS:
   q=i if e in INTEL else n
   for st in ('gate','up'):
    checks[f'e{e}_{st}_exact']=torch.equal(bf(q[f'e{e}_{st}']),o.get_tensor(f'cpu_q5_e{e}_{st}'))
   for st in ('silu','activation','down'):
    checks[f'e{e}_{st}_metric']=math.isfinite(metric(o.get_tensor(f'cpu_q5_e{e}_{st}'),bf(q[f'e{e}_{st}']))) and metric(o.get_tensor(f'cpu_q5_e{e}_{st}'),bf(q[f'e{e}_{st}']))<=.001
  for st in ('gate','up'):
   checks[f'shared_{st}_exact']=torch.equal(bf(n[f'e512_{st}']),o.get_tensor(f'cpu_q5_shared_{st}'))
  for st in ('silu','activation','down'):
   checks[f'shared_{st}_metric']=metric(o.get_tensor('cpu_q5_shared_'+st),bf(n[f'e512_{st}']))<=.001
  checks['shared_sigmoid']=metric(o.get_tensor('shared_sigmoid'),bf(n['shared_sigmoid']))<=.001
  out=torch.zeros(2048,dtype=torch.bfloat16)
  for e in EXPERTS:out.add_(bf((i if e in INTEL else n)[f'e{e}_weighted_token15']))
  checks['merge']=metric(o.get_tensor('cpu_q5_routed_token15'),out)<=.001
  checks['shared_raw']=metric(o.get_tensor('cpu_q5_shared'),bf(n['e512_down']))<=.001
  checks['shared_gated']=metric(o.get_tensor('cpu_q5_shared_gated'),bf(n['shared_gated']))<=.001
  checks['source_routed_exact']=torch.equal(o.get_tensor('source_routed_token15'),o.get_tensor('source_routed_token15'))
  checks['quality']=all(metric(o.get_tensor('source_'+k),o.get_tensor('cpu_q5_'+k))<=.08 for k in ('routed_token15','shared','shared_gated'))
 for role in ('intel','nvidia'):
  pr=c['processes'][role];checks[role+'_exit']=pr['exit_code']==0 and pr['end_qpc_ns']>c['release_qpc_ns']>pr['start_qpc_ns'] and not pr['stderr']
 checks['intel_no_copy']=all(c['child_results']['intel']['calls'].get(k)==0 for k in ('clCreateBuffer','clEnqueueWriteBuffer','clEnqueueReadBuffer','clEnqueueCopyBuffer','clEnqueueMigrateMemObjects'))
 checks['nvidia_copies']=len(c['child_results']['nvidia']['copies'])>0 and all(x['bytes']>0 and x['direction'] in ('H2D','D2H') for x in c['child_results']['nvidia']['copies'])
 passed=all(checks.values());result={'kind':'het_next_l0_pv0r3_independent_verification','pass':passed,'checks':checks,'check_count':len(checks),'candidate_sha256':sha(cand),'manifest_sha256':sha(MAN),'claim_boundary':'known p0/n16 component correctness only; no performance'};atomic(OUT,canon(result)+b'\n');return 0 if passed else 3
def main():p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);a=p.parse_args();return verify(a.candidate)
if __name__=='__main__':raise SystemExit(main())
