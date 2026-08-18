"""Fresh process legacy-QFAST baseline or full phase-6 candidate arm."""
from __future__ import annotations
import argparse,gc,json,subprocess,traceback
from common import REPO,percentiles,utc_now,write_json_atomic
from diag_component_marginals_graph import _run,_prefill,_reset_exact_state
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from graph_e1f22 import _load_prompt_set
from s100_phase6_runtime import build_phase6_runtime,record
CANDS=REPO/'pro_research'/'results'/'S100_PHASE6_CANDIDATES.json';HELD=REPO/'pro_research'/'results'/'S100_PHASE6_HELDOUT.json';BSEL=REPO/'pro_research'/'results'/'S100_PHASE6_BACKEND_SELECT.json'
def smi():
 p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate','--format=csv,noheader,nounits'],capture_output=True,text=True)
 if p.returncode:return {'error':(p.stderr or p.stdout).strip()}
 v=[x.strip() for x in p.stdout.splitlines()[0].split(',')];return {'memory_used_mib':int(v[0]),'utilization_percent':int(v[1]),'sm_mhz':float(v[2]),'mem_mhz':float(v[3]),'power_w':float(v[4]),'temperature_c':float(v[5]),'pstate':v[6]}
def preheat(rt,pids,n=128):
 _reset_exact_state(rt);_prefill(rt,pids)
 for _ in range(n):rt.step_graph(None)
 rt._graph_stream.synchronize()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--role',choices=('base_a','cand_a','cand_b','base_b'),required=True);a=ap.parse_args();out=REPO/'pro_research'/'results'/f'S100_PHASE6_TIMING_{a.candidate.upper()}_{a.role.upper()}.json';p={'kind':'s100_phase6_candidate_arm','status':'started','candidate':a.candidate,'role':a.role,'started_utc':utc_now()}
 try:
  c=json.loads(CANDS.read_text());h=json.loads(HELD.read_text());spec=c['selected'][a.candidate]
  if h['results'][a.candidate]['status']!='v18_fidelity_candidate':raise RuntimeError('candidate is not heldout green')
  selected_backend=(json.loads(BSEL.read_text()).get('selected_backend','legacy') if BSEL.exists() else 'legacy');isbase=a.role.startswith('base');backend='legacy' if isbase else selected_backend;m={} if isbase else {int(k):int(v) for k,v in spec.get('layer_k',{}).items()};alpha=0.0 if isbase else float(spec.get('alpha',0))
  p['gpu_idle_preflight']=_require_gpu_idle_wddm();import cupy as cp
  prompts,_e,n,capacity=_load_prompt_set('full');n=max(int(n),256);b=build_phase6_runtime(int(capacity),m,alpha,backend);rt=b.rt;preheat(rt,prompts[0]['prompt_ids'],128);before=smi();raw=[];ids={}
  for q in prompts:
   x,ms=_run(rt,q['prompt_ids'],n);ids[q['prompt']]=[int(z) for z in x];raw.extend(float(z) for z in ms)
  rt._graph_stream.synchronize();after=smi();finite=bool(cp.isfinite(rt.logits).all().item());p.update({'status':'measured','selected_backend':selected_backend,'runtime':record(b),'timing':percentiles(raw),'raw_timing_ms':raw,'ids':ids,'finite':finite,'vram_mib':max(int(before.get('memory_used_mib',0)),int(after.get('memory_used_mib',0))),'smi_before':before,'smi_after':after,'completed_utc':utc_now()});b.restore_combined();b.restore_selective();del rt,b;cp.get_default_memory_pool().free_all_blocks();gc.collect()
 except Exception as e:p.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},'completed_utc':utc_now()})
 write_json_atomic(out,p,archive=True);print(json.dumps({'status':p.get('status'),'candidate':a.candidate,'role':a.role,'backend':p.get('selected_backend'),'timing':p.get('timing'),'error':(p.get('error') or {}).get('message'),'output':str(out)},indent=2));return 2 if p.get('status')=='technical_failure' else 0
if __name__=='__main__':raise SystemExit(main())
