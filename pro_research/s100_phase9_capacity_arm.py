from __future__ import annotations
import argparse,gc,json,subprocess,traceback
from common import REPO,percentiles,utc_now,write_json_atomic
from diag_component_marginals_graph import _run,_prefill,_reset_exact_state
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from graph_e1f22 import _load_prompt_set
from s100_phase9_capacity_runtime import Phase9VRAMInfeasible,build,record
PROF=REPO/'pro_research'/'results'/'s100_phase9'/'S100_PHASE9_CAPACITY_PROFILES.json'
def smi():
 p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate','--format=csv,noheader,nounits'],capture_output=True,text=True)
 if p.returncode:return {'error':(p.stderr or p.stdout).strip()}
 v=[x.strip() for x in p.stdout.splitlines()[0].split(',')];return {'memory_used_mib':int(v[0]),'utilization_percent':int(v[1]),'sm_mhz':float(v[2]),'mem_mhz':float(v[3]),'power_w':float(v[4]),'temperature_c':float(v[5]),'pstate':v[6]}
def preheat(rt,pids,n=96):
 _reset_exact_state(rt);_prefill(rt,pids)
 for _ in range(n):rt.step_graph(None)
 rt._graph_stream.synchronize()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',required=True);ap.add_argument('--role',choices=('base_a','cand_a','cand_b','base_b'),required=True);a=ap.parse_args();out=REPO/'pro_research'/'results'/'s100_phase9'/f'CAP_{a.profile.upper()}_{a.role.upper()}.json';p={'status':'started','profile':a.profile,'role':a.role,'started_utc':utc_now()}
 try:
  rows=json.loads(PROF.read_text())['profiles'];name='current' if a.role.startswith('base') else a.profile;mp={int(k):int(v) for k,v in rows[name].items()};p['gpu_idle_preflight']=_require_gpu_idle_wddm();import cupy as cp
  prompts,_e,n,_capacity=_load_prompt_set('full');n=max(int(n),256);b=build(mp);rt=b.rt;preheat(rt,prompts[0]['prompt_ids']);before=smi();raw=[];ids={}
  for q in prompts:x,ms=_run(rt,q['prompt_ids'],n);ids[q['prompt']]=[int(z) for z in x];raw.extend(float(z) for z in ms)
  rt._graph_stream.synchronize();after=smi();p.update({'status':'measured','runtime':record(b),'timing':percentiles(raw),'raw_timing_ms':raw,'ids':ids,'finite':bool(cp.isfinite(rt.logits).all().item()),'vram_mib':max(int(before.get('memory_used_mib',0)),int(after.get('memory_used_mib',0))),'smi_before':before,'smi_after':after,'completed_utc':utc_now()});b.restore_combined();b.restore_sel();del rt,b;cp.get_default_memory_pool().free_all_blocks();gc.collect()
 except Phase9VRAMInfeasible as e:p.update({'status':'infeasible_vram','error':{'type':type(e).__name__,'message':str(e),'stage':e.stage,'planned_plane_bytes':e.planned,'free_bytes_at_plan':e.free,'traceback':traceback.format_exc()},'completed_utc':utc_now()})
 except Exception as e:
  # A raw CuPy OOM outside build() (preheat/run buffers) is the same honest
  # verdict: this candidate does not fit on the target GPU.
  s='infeasible_vram' if type(e).__name__=='OutOfMemoryError' else 'technical_failure'
  p.update({'status':s,'error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},'completed_utc':utc_now()})
 write_json_atomic(out,p,archive=True);print(json.dumps({'status':p.get('status'),'profile':a.profile,'role':a.role,'timing':p.get('timing'),'vram_mib':p.get('vram_mib'),'error':(p.get('error') or {}).get('message')},indent=2));return 2 if p.get('status')=='technical_failure' else 0
if __name__=='__main__':raise SystemExit(main())
