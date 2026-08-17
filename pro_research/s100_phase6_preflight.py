"""Exact backend parity and destructive-control preflight."""
from __future__ import annotations
import gc, json, traceback
from common import REPO, first_divergence, percentiles, utc_now, write_json_atomic
from diag_component_marginals_graph import _run
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from graph_e1f22 import _load_prompt_set
from s100_phase6_runtime import build_phase6_runtime, recapture, record

OUT = REPO/'pro_research'/'results'/'S100_PHASE6_PREFLIGHT.json'
BACKENDS = ('legacy','ballot_fused','direct','direct_opt')


def run_backend(name, bad=False):
    import cupy as cp
    prompts, _e, _n, capacity = _load_prompt_set('smoke')
    b = build_phase6_runtime(int(capacity), backend=name)
    if bad:
        b.rt._bad_pick = 1
        recapture(b)
    ids, raw = {}, []
    for p in prompts:
        x, ms = _run(b.rt, p['prompt_ids'], 32)
        ids[p['prompt']] = [int(z) for z in x]
        raw.extend(float(z) for z in ms)
    b.rt._graph_stream.synchronize()
    finite = bool(cp.isfinite(b.rt.logits).all().item())
    rec = {'backend':name,'bad_pick':bad,'runtime':record(b),'ids':ids,
           'timing':percentiles(raw),'finite':finite}
    b.restore_combined(); b.restore_selective(); del b
    cp.get_default_memory_pool().free_all_blocks(); gc.collect()
    return rec


def main():
    p={'kind':'s100_phase6_preflight','status':'started','started_utc':utc_now()}
    try:
        p['gpu_idle_preflight']=_require_gpu_idle_wddm()
        runs={name:run_backend(name) for name in BACKENDS}
        base=runs['legacy']['ids']
        parity={name:{k:first_divergence(base[k],r['ids'][k]) for k in base}
                for name,r in runs.items()}
        control=run_backend('direct_opt',True)
        control_div={k:first_divergence(base[k],control['ids'][k]) for k in base}
        gates={
            'P1_all_exact_backends_token_parity':all(
                all(v is None for v in parity[n].values()) for n in BACKENDS),
            'P2_all_finite':all(r['finite'] for r in runs.values()),
            'P3_runtime_K6_alpha0':all(
                int(r['runtime']['top_k'])==6 and
                float(r['runtime']['config']['alpha'])==0.0 for r in runs.values()),
            'P4_bad_pick_control_diverges':any(v is not None for v in control_div.values()),
        }
        p.update({'status':'preflight_pass' if all(gates.values()) else 'preflight_failed',
                  'runs':runs,'parity':parity,'control':{'divergence':control_div,
                  'timing':control['timing']},'gates':gates,'completed_utc':utc_now()})
    except Exception as e:
        p.update({'status':'technical_failure','error':{'type':type(e).__name__,
                  'message':str(e),'traceback':traceback.format_exc()},
                  'completed_utc':utc_now()})
    write_json_atomic(OUT,p,archive=True)
    print(json.dumps({'status':p.get('status'),'gates':p.get('gates'),
          'timing':{k:v['timing'] for k,v in p.get('runs',{}).items()},
          'error':(p.get('error') or {}).get('message'),'output':str(OUT)},indent=2))
    return 2 if p.get('status') in {'technical_failure','preflight_failed'} else 0
if __name__=='__main__': raise SystemExit(main())
