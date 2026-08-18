"""Run fixed phase-6 candidate grid on validation only."""
from __future__ import annotations
import gc,json,traceback
from common import REPO,utc_now,write_json_atomic
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from s100_phase5_quality import evaluate
from s100_phase6_common import CANDIDATES,public_spec
from s100_phase6_runtime import build_phase6_runtime,recapture
OUT=REPO/'pro_research'/'results'/'S100_PHASE6_VALIDATION.json';CANDS=REPO/'pro_research'/'results'/'S100_PHASE6_CANDIDATES.json'
def main():
 p={'kind':'s100_phase6_validation','status':'started','started_utc':utc_now(),'fixed_grid':{k:public_spec(v) for k,v in CANDIDATES.items()},'results':{}}
 try:
  p['gpu_idle_preflight']=_require_gpu_idle_wddm();import cupy as cp
  b=build_phase6_runtime(backend='legacy')
  for idx,(name,spec) in enumerate(CANDIDATES.items(),1):
   b.config['layer_k'].clear();b.config['layer_k'].update({int(k):int(v) for k,v in spec['layer_k'].items()});b.config['alpha']=float(spec['alpha']);recapture(b)
   ev=evaluate(b,'validation',False);p['results'][name]={'spec':public_spec(spec),'status':'validation_candidate' if ev['official_pass'] else 'validation_failed',**ev};print(f'validation {idx:02d}/{len(CANDIDATES)} {name}: official={ev["official_pass"]} top1={ev["summary"]["top1_agreement"]:.5f} KL={ev["summary"]["mean_coarse_kl"]:.5f}',flush=True)
  selected={k:v['spec'] for k,v in p['results'].items() if v['status']=='validation_candidate'}
  c={'kind':'s100_phase6_candidates','created_utc':utc_now(),'selection_policy':'all fixed-grid candidates passing original official validation gates; heldout untouched','selected':selected,'validation_results':p['results']}
  write_json_atomic(CANDS,c,archive=True);p.update({'status':'complete','selected':selected,'candidates_path':str(CANDS.relative_to(REPO)),'completed_utc':utc_now()});b.restore_combined();b.restore_selective();del b;cp.get_default_memory_pool().free_all_blocks();gc.collect()
 except Exception as e:p.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},'completed_utc':utc_now()})
 write_json_atomic(OUT,p,archive=True);print(json.dumps({'status':p.get('status'),'selected':p.get('selected'),'error':(p.get('error') or {}).get('message'),'output':str(OUT)},indent=2));return 2 if p.get('status')=='technical_failure' else 0
if __name__=='__main__':raise SystemExit(main())
