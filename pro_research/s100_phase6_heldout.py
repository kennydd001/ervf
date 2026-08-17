"""Untouched heldout evaluation for phase-6 validation-green candidates."""
from __future__ import annotations
import gc,json,traceback
from common import REPO,utc_now,write_json_atomic
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from s100_phase5_quality import evaluate
from s100_phase6_runtime import build_phase6_runtime
CANDS=REPO/'pro_research'/'results'/'S100_PHASE6_CANDIDATES.json';OUT=REPO/'pro_research'/'results'/'S100_PHASE6_HELDOUT.json'
def main():
 p={'kind':'s100_phase6_heldout','status':'started','started_utc':utc_now(),'results':{}}
 try:
  c=json.loads(CANDS.read_text());selected=c.get('selected',{})
  if not selected:
   p.update({'status':'complete','note':'no validation-green phase6 candidate','completed_utc':utc_now()});write_json_atomic(OUT,p,archive=True);print(json.dumps(p,indent=2));return 0
  p['gpu_idle_preflight']=_require_gpu_idle_wddm();import cupy as cp
  for idx,(name,spec) in enumerate(selected.items(),1):
   try:
    m={int(k):int(v) for k,v in spec.get('layer_k',{}).items()};a=float(spec.get('alpha',0));b=build_phase6_runtime(layer_k=m,alpha=a,backend='legacy');ev=evaluate(b,'heldout',True);p['results'][name]={'spec':spec,'status':'v18_fidelity_candidate' if ev['official_pass'] else 'v18_fidelity_failed',**ev};print(f'heldout {idx:02d}/{len(selected)} {name}: {p["results"][name]["status"]} top1={ev["summary"]["top1_agreement"]:.5f} KL={ev["summary"]["mean_coarse_kl"]:.5f}',flush=True);b.restore_combined();b.restore_selective();del b;cp.get_default_memory_pool().free_all_blocks();gc.collect()
   except Exception as e:p['results'][name]={'spec':spec,'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}}
  p.update({'status':'complete','completed_utc':utc_now()})
 except Exception as e:p.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},'completed_utc':utc_now()})
 write_json_atomic(OUT,p,archive=True);print(json.dumps({'status':p.get('status'),'results':{k:{'status':v.get('status'),'summary':v.get('summary')} for k,v in p.get('results',{}).items()},'error':(p.get('error') or {}).get('message'),'output':str(OUT)},indent=2));return 2 if p.get('status')=='technical_failure' else 0
if __name__=='__main__':raise SystemExit(main())
