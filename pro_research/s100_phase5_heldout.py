"""Heldout fidelity evaluation of candidates frozen before heldout."""
from __future__ import annotations
import gc,json,traceback
from common import REPO,utc_now,write_json_atomic
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from s100_phase5_runtime import build_phase5_runtime
from s100_phase5_quality import evaluate
CANDS=REPO/'pro_research'/'results'/'S100_PHASE5_CANDIDATES.json'; OUT=REPO/'pro_research'/'results'/'S100_PHASE5_HELDOUT.json'

def main():
    payload={'kind':'s100_phase5_heldout','status':'started','started_utc':utc_now(),'results':{}}
    try:
        c=json.loads(CANDS.read_text(encoding='utf-8')); selected=c.get('selected',{})
        if not selected:
            payload.update({'status':'complete','note':'validation selected no nonzero phase5 candidate','completed_utc':utc_now()})
            write_json_atomic(OUT,payload,archive=True)
            print(json.dumps(payload,indent=2)); return 0
        payload['gpu_idle_preflight']=_require_gpu_idle_wddm(); import cupy as cp
        for name,spec in selected.items():
            try:
                m={int(k):int(v) for k,v in spec.get('layer_k',{}).items()}; alpha=float(spec.get('alpha',0.0)); b=build_phase5_runtime(layer_k=m,alpha=alpha)
                ev=evaluate(b,'heldout',True); payload['results'][name]={'status':'v18_fidelity_candidate' if ev['official_pass'] else 'v18_fidelity_failed','spec':spec,**ev}
                print(f'heldout {name}: {payload["results"][name]["status"]} top1={ev["summary"]["top1_agreement"]:.5f} KL={ev["summary"]["mean_coarse_kl"]:.5f}',flush=True)
                b.restore_combined(); b.restore_selective(); del b; cp.get_default_memory_pool().free_all_blocks(); gc.collect()
            except Exception as e:
                payload['results'][name]={'status':'technical_failure','spec':spec,'error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}}
        payload['status']='complete'; payload['completed_utc']=utc_now()
    except Exception as e:
        payload.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},'completed_utc':utc_now()})
    write_json_atomic(OUT,payload,archive=True); print(json.dumps({'status':payload.get('status'),'results':{k:{'status':v.get('status'),'summary':v.get('summary')} for k,v in payload.get('results',{}).items()},'error':(payload.get('error') or {}).get('message'),'output':str(OUT)},indent=2))
    return 2 if payload.get('status')=='technical_failure' else 0
if __name__=='__main__': raise SystemExit(main())
