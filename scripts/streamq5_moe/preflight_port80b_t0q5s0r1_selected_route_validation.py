#!/usr/bin/env python3
"""Self-bound semantic/static S0-R1 preflight plus TEMP bundle recovery simulation."""
import ast,hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';RUN=S/'run_port80b_t0q5s0r1_selected_route_validation.py';VER=S/'verify_port80b_t0q5s0r1_selected_route_validation.py';SELF=Path(__file__);LOCK=R/'port80b_t0q5s0r1_runner_lock.json';VL=R/'port80b_t0q5s0r1_verifier_lock.json';SL=R/'port80b_t0q5s0r1_preflight_lock.json';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0r1_selected_route_validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def simulate():
 with tempfile.TemporaryDirectory(prefix='s0r1_') as q:
  q=Path(q);a=q/'raw.tmp.inprogress';b=q/'result.tmp.inprogress';a.write_bytes(b'a');b.write_bytes(b'b');
  with a.open('rb') as h:os.fsync(h.fileno())
  with b.open('rb') as h:os.fsync(h.fileno())
  ar=q/'raw';br=q/'result';os.rename(a,ar);os.rename(b,br);marker=q/'commit';marker.write_text(json.dumps({'files':{ar.name:sha(ar),br.name:sha(br)}}));return ar.exists() and br.exists() and marker.exists() and not list(q.glob('*.inprogress'))
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VL.read_text());sl=json.loads(SL.read_text());rs=RUN.read_text();vs=VER.read_text();actual={'preflight_sha256':sha(SELF),'runner_sha256':sha(RUN),'verifier_sha256':sha(VER),'runner_lock_sha256':sha(LOCK),'verifier_lock_sha256':sha(VL)};checks={'self_bound':all(sl[k]==x for k,x in actual.items()),'source_locks':l['runner_sha256']==actual['runner_sha256'] and l['verifier_sha256']==actual['verifier_sha256'] and v['verifier_sha256']==actual['verifier_sha256'],'ast':ast.parse(rs) is not None and ast.parse(vs) is not None,'standalone':'importlib' not in vs and 'run_port80b' not in vs,'semantic_contract':all(x in rs for x in ('len(union)!=252','len(evidence)','field31_absent','zero_group_count','max_bf16_ulp','shared_down_code_mutation_graph_wide','sample(resources,\'cleanup\')')),'narrow_no_forward_bank':all(x not in rs for x in ('from_pretrained','DynamicCache','.sq5m','candidate_layer','complete_mlp')),'transaction_contract':all(x in rs for x in ('.inprogress','os.fsync','os.rename','recover()','atomic_failure')),'temp_simulation':simulate(),'output_absent':not D.exists()};o={'kind':'t0q5s0r1_preflight','pass':all(checks.values()),'checks':checks,'physical_actions':{'temp_only':True,'shard':False,'model':False,'forward':False,'gpu':False}};print(json.dumps(o,indent=2));return 0 if o['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
