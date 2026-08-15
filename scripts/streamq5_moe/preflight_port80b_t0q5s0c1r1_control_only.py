#!/usr/bin/env python3
"""S0-C1-R1 self-bound CPU/no-forward preflight; TEMP lifecycle only."""
import ast,hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';SELF=Path(__file__);RUN=S/'run_port80b_t0q5s0c1r1_control_only.py';VER=S/'verify_port80b_t0q5s0c1r1_control_only.py';PR=R/'PORT80B_T0Q5S0C1R1_CONTROL_ONLY_PREREGISTRATION_2026-08-13.md';LOCK=R/'port80b_t0q5s0c1r1_runner_lock.json';VL=R/'port80b_t0q5s0c1r1_verifier_lock.json';PL=R/'port80b_t0q5s0c1r1_preflight_lock.json';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0c1r1_control_only'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def durable(p):
 with Path(p).open('r+b' if os.name=='nt' else 'rb') as h:os.fsync(h.fileno())
def ast_clean():
 t=ast.parse(SELF.read_text());imports=set();calls=set()
 for n in ast.walk(t):
  if isinstance(n,ast.Import):imports.update(a.name.split('.')[0] for a in n.names)
  elif isinstance(n,ast.ImportFrom) and n.module:imports.add(n.module.split('.')[0])
  elif isinstance(n,ast.Call):calls.add(n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else '')
 return not(imports&{'torch','safetensors','transformers','numpy'}) and not(calls&{'safe_open','from_pretrained','load_file','snapshot_download'})
def simulation():
 with tempfile.TemporaryDirectory(prefix='s0c1r1_') as z:
  d=Path(z);rt=d/'raw.tmp';jt=d/'result.tmp';ct=d/'commit.tmp';raw=d/'raw';res=d/'result';com=d/'commit';rt.write_bytes(b'raw');jt.write_bytes(b'result');durable(rt);durable(jt);ct.write_text(json.dumps({'files':{'raw':{'bytes':3,'sha256':sha(rt)},'result':{'bytes':6,'sha256':sha(jt)}}}));durable(ct);os.rename(rt,raw);os.rename(jt,res);os.rename(ct,com);c=json.loads(com.read_text());ok=c['files']['raw']=={'bytes':3,'sha256':sha(raw)} and c['files']['result']=={'bytes':6,'sha256':sha(res)};ft=d/'failure.tmp';ff=d/'failure';ft.write_text('{"kind":"failure"}');durable(ft);os.rename(ft,ff);return ok and ff.read_text()=='{"kind":"failure"}'
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VL.read_text());p=json.loads(PL.read_text());actual={'preflight_sha256':sha(SELF),'runner_sha256':sha(RUN),'verifier_sha256':sha(VER),'prereg_sha256':sha(PR),'runner_lock_sha256':sha(LOCK),'verifier_lock_sha256':sha(VL)};checks={'self_bound':all(p[k]==x for k,x in actual.items()),'source_locks':l['runner_sha256']==actual['runner_sha256'] and l['verifier_sha256']==actual['verifier_sha256'] and l['prereg_sha256']==actual['prereg_sha256'] and l['verifier_lock_sha256']==actual['verifier_lock_sha256'] and v['verifier_sha256']==actual['verifier_sha256'],'closed_until_audit':l['execution_open'] is False and l['control_only_authorization'] is True and l['physical_run_requires_independent_source_go'] is True and l['implementation_audit_token']=='S0C1R1_IMPLEMENTATION_SOURCE_AUDIT_PENDING' and v['implementation_audit_token']=='S0C1R1_IMPLEMENTATION_SOURCE_AUDIT_PENDING','output_absent':not D.exists(),'ast_no_physical':ast_clean(),'temp_atomic_simulation':simulation()};o={'kind':'t0q5s0c1r1_preflight','pass':all(checks.values()),'checks':checks,'physical_actions':{'temp_only':True,'shard':False,'model':False,'gpu':False}};print(json.dumps(o,indent=2));return 0 if o['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
