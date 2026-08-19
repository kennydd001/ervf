from __future__ import annotations

import gc,json,traceback,types
import numpy as np
from common import REPO,require_model_dir,utc_now,write_json_atomic
from s100_phase20b_verifier import GroupedMoEH4,nrmse,H

OUT=REPO/'pro_research'/'results'/'s100_phase20b'/'S100_PHASE20B_MOE_PREFLIGHT.json'

PROMPT='Explain how a compiler optimizes a program while preserving its meaning.'


def main():
    payload={'kind':'s100_phase20b_moe_preflight','status':'started','started_utc':utc_now()}
    try:
        import cupy as cp
        from transformers import AutoTokenizer
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

        s20=json.loads((REPO/'pro_research'/'results'/'s100_phase20s'/'S100_PHASE20S_SUMMARY.json').read_text(encoding='utf-8'))
        if not s20.get('PHASE20B_FULL_VERIFIER_OPEN'):
            raise RuntimeError('Phase20S did not open Phase20B')
        if s20.get('phase20b_kv_policy')!='fp8_kv=False':
            raise RuntimeError(f"unexpected KV policy {s20.get('phase20b_kv_policy')}")

        rt=LightningRuntime(require_model_dir(),contexts_max=512,embed_on_host=True,fp8_kv=False,verbose=False)
        rt.load_routed_bank();rt.enable_cache(48);rt.deterministic_accum=True
        layers=[int(x) for x in rt.moe_layers]
        chosen=sorted({layers[0],layers[len(layers)//2],layers[-1]})
        cap={i:[] for i in chosen};orig=rt._moe

        def wrap(self,i,out):
            i=int(i);take=i in cap and len(cap[i])<H
            if take: rec={'normed':cp.asnumpy(self.normed).astype(np.float32,copy=True)}
            idx,w=orig(i,out)
            if take:
                rec.update({'out':cp.asnumpy(out).astype(np.float32,copy=True),
                            'ids':np.asarray(idx,dtype=np.int32).tolist(),
                            'weights':np.asarray(w,dtype=np.float64).tolist()})
                cap[i].append(rec)
            return idx,w
        rt._moe=types.MethodType(wrap,rt)
        tok=AutoTokenizer.from_pretrained(str(require_model_dir()),local_files_only=True,trust_remote_code=True,use_fast=True)
        ids=tok.encode(PROMPT,add_special_tokens=False)
        rt.reset();nxt=None
        for t in ids:nxt=int(rt.step(int(t)))
        cur=int(nxt)
        for _ in range(H):cur=int(rt.step(cur))
        rt._moe=orig
        if any(len(cap[i])!=H for i in chosen):
            raise RuntimeError({i:len(cap[i]) for i in chosen})

        grouped=GroupedMoEH4(rt);results=[]
        for layer in chosen:
            normed=cp.asarray(np.stack([r['normed'] for r in cap[layer]]))
            out=cp.empty((H,int(rt.hidden)),cp.float32)
            gids,gw,stats=grouped(layer,normed,out,collect_stats=True)
            cp.cuda.get_current_stream().synchronize()
            refout=np.stack([r['out'] for r in cap[layer]])
            rid=np.asarray([r['ids'] for r in cap[layer]],np.int32)
            rw=np.asarray([r['weights'] for r in cap[layer]],np.float64)
            got=cp.asnumpy(out)
            rec={
                'layer':layer,
                'route_ids_equal':bool(np.array_equal(gids,rid)),
                'route_weight_max_abs':float(np.max(np.abs(gw.astype(np.float64)-rw))),
                'output_nrmse':nrmse(got,refout),
                'finite':bool(np.isfinite(got).all()),
                'census':stats,
            }
            rec['pass']=bool(rec['route_ids_equal'] and rec['route_weight_max_abs']<=2e-6 and rec['output_nrmse']<=5e-4 and rec['finite'])
            results.append(rec)
            print(f"20B MoE preflight layer={layer}: nrmse={rec['output_nrmse']:.3e} routes={rec['route_ids_equal']} pass={rec['pass']}",flush=True)

        payload.update({'status':'measured','layers':chosen,'results':results,
                        'GROUPED_MOE_H4_GREEN':all(x['pass'] for x in results),
                        'completed_utc':utc_now()})
        rt.bank={};rt.cache={};del rt;gc.collect();cp.get_default_memory_pool().free_all_blocks();cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({'status':'technical_failure','error':{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},'completed_utc':utc_now()})
    OUT.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({'status':payload.get('status'),'GROUPED_MOE_H4_GREEN':payload.get('GROUPED_MOE_H4_GREEN'),'results':payload.get('results'),'error':(payload.get('error') or {}).get('message'),'output':str(OUT)},indent=2))
    return 0 if payload.get('status')=='measured' else 2
if __name__=='__main__':raise SystemExit(main())
