#!/usr/bin/env python3
"""Self-bound S0-R4 preflight with AST physical-call audit and TEMP lifecycle simulation."""
import ast,hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';SELF=Path(__file__);RUN=S/'run_port80b_t0q5s0r4_selected_route_validation.py';VER=S/'verify_port80b_t0q5s0r4_selected_route_validation.py';LOCK=R/'port80b_t0q5s0r4_runner_lock.json';VL=R/'port80b_t0q5s0r4_verifier_lock.json';SL=R/'port80b_t0q5s0r4_preflight_lock.json';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0r4_selected_route_validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def physical_ast_clean():
 tree=ast.parse(SELF.read_text(encoding='utf-8'));imports=set();calls=set()
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):imports.update(a.name.split('.')[0] for a in node.names)
  elif isinstance(node,ast.ImportFrom) and node.module:imports.add(node.module.split('.')[0])
  elif isinstance(node,ast.Call):
   f=node.func
   if isinstance(f,ast.Name):calls.add(f.id)
   elif isinstance(f,ast.Attribute):calls.add(f.attr)
 return not(imports & {'torch','safetensors','transformers','huggingface_hub'}) and not(calls & {'safe_open','from_pretrained','snapshot_download','load_file'})
def simulation():
 with tempfile.TemporaryDirectory(prefix='s0r4_') as t:
  d=Path(t);bad=d/'failed_attempts';bad.mkdir();stale=d/'raw.inprogress';stale.write_bytes(b'x');os.rename(stale,bad/'x_raw.inprogress');raw=d/'raw';res=d/'result';com=d/'commit';rt=d/'raw.inprogress';jt=d/'result.inprogress';ct=d/'commit.inprogress';rt.write_bytes(b'raw');jt.write_bytes(b'result');ct.write_text(json.dumps({'files':{'raw':{'bytes':3,'sha256':sha(rt)},'result':{'bytes':6,'sha256':sha(jt)}}}));os.rename(rt,raw);os.rename(jt,res);os.rename(ct,com);m=json.loads(com.read_text());valid=m['files']['raw']=={'bytes':raw.stat().st_size,'sha256':sha(raw)} and m['files']['result']=={'bytes':res.stat().st_size,'sha256':sha(res)};return valid and sorted(x.name for x in d.iterdir())==['commit','failed_attempts','raw','result']
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VL.read_text());sl=json.loads(SL.read_text());a={'preflight_sha256':sha(SELF),'runner_sha256':sha(RUN),'verifier_sha256':sha(VER),'runner_lock_sha256':sha(LOCK),'verifier_lock_sha256':sha(VL)};checks={'self_bound':all(sl[k]==x for k,x in a.items()),'open_token':l['execution_open'] is True and l['s0_validation_authorization'] is True and v['execution_open'] is True and l['implementation_audit_token']==v['implementation_audit_token']=='S0R4_IMPLEMENTATION_AUDIT_GO' and l['dependency_use']=='runtime fields/hash only; dependency outputs_opened false is not S0 authorization','sources':l['runner_sha256']==a['runner_sha256'] and l['verifier_sha256']==a['verifier_sha256'] and l['verifier_lock_sha256']==a['verifier_lock_sha256'],'temp_recovery_commit_simulation':simulation(),'output_absent':not D.exists(),'no_physical_calls_ast':physical_ast_clean()};o={'kind':'s0r4_preflight','pass':all(checks.values()),'checks':checks,'physical_actions':{'temp_only':True,'shard':False,'model':False,'gpu':False}};print(json.dumps(o,indent=2));return 0 if o['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
