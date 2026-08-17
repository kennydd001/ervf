"""One fresh-process exact backend timing arm."""
from __future__ import annotations
import argparse,gc,json,subprocess,traceback
from typing import Any
from common import REPO,percentiles,utc_now,write_json_atomic
from diag_component_marginals_graph import _run,_prefill,_reset_exact_state
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from graph_e1f22 import _load_prompt_set
from s100_phase6_runtime import build_phase6_runtime,record

BACKENDS=('ballot_fused','direct','direct_opt')
ROLES=('base_a','cand_a','cand_b','base_b')

def smi():
 p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate','--format=csv,noheader,nounits'],capture_output=True,text=True)
 if p.returncode:return {'error':(p.stderr or p.stdout).strip()}
 v=[x.strip() for x in p.stdout.splitlines()[0].split(',')];return {'memory_used_mib':int(v[0]),'utilization_percent':int(v[1]),'sm_mhz':float(v[2]),'mem_mhz':float(v[3]),'power_w':float(v[4]),'temperature_c':float(v[5]),'pstate':v[6]}

def preheat(rt,pids,n):
 _reset_exact_state(rt);_prefill(rt,pids)
 for _ in range(n):rt.step_graph(None)
 rt._graph_stream.synchronize()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=BACKENDS,required=True);ap.add_argument('--role',choices=ROLES,required=True);ap.add_argument('--mode',choices=('smoke','full'),required=True);a=ap.parse_args()
 out=REPO/'pro_research'/'results'/f'S100_PHASE6_BACKEND_{a.backend.upper()}_{a.mode.upper()}_{a.role.upper()}.json';p={'kind':'s100_phase6_backend_arm','status':'started','backend':a.backend,'role':a.role,'mode':a.mode,'started_utc':utc_now()}
 try:
  p['gpu_idle_preflight']=_require_gpu_idle_wddm();import cupy as cp
  prompts,_e,n,capacity=_load_prompt_set(a.mode);n=min(int(n),32) if a.mode=='smoke' else max(int(n),256);pre=48 if a.mode=='smoke' else 128
  backend='legacy' if a.role.startswith('base') else a.backend
  b=build_phase6_runtime(int(capacity),backend=backend);rt=b.rt;preheat(rt,prompts[0]['prompt_ids'],pre)
  before=smi();raw=[];ids={}
  for q in prompts:
   x,ms=_run(rt,q['prompt_ids'],n);ids[q['prompt']]=[int(z) for z in x];raw.extend(float(z) for z in ms)
  rt._graph_stream.synchronize();after=smi();finite=bool(cp.isfinite(rt.logits).all().item())
  p.update({'status':'measured','runtime':record(b),'timing':percentiles(raw),'raw_timing_ms':raw,'ids':ids,'finite':finite,'vram_mib':max(int(before.get('memory_used_mib',0)),int(after.get('memory_used_mib',0))),'smi_before':before,'smi_after':after,'completed_utc':utc_now()})
  b.restore_combined();b.restore_selective();del rt,b;cp.get_default_memory_pool().free_all_blocks();gc.collect()
 except Exception as e:p.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},'completed_utc':utc_now()})
 write_json_atomic(out,p,archive=True);print(json.dumps({'status':p.get('status'),'backend':a.backend,'role':a.role,'mode':a.mode,'timing':p.get('timing'),'vram_mib':p.get('vram_mib'),'error':(p.get('error') or {}).get('message'),'output':str(out)},indent=2));return 2 if p.get('status')=='technical_failure' else 0
if __name__=='__main__':raise SystemExit(main())
