from __future__ import annotations
import argparse,ctypes,json,os,sys,traceback
from pathlib import Path
import numpy as np

def add_repo(repo):
    sys.path.insert(0,str(repo/'pro_research'));sys.path.insert(0,str(repo/'src'));os.chdir(repo)

def closure_dict(fn):
    out={}
    for n,c in zip(fn.__code__.co_freevars,fn.__closure__ or ()):
        try:out[n]=c.cell_contents
        except ValueError:pass
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--out',required=True);ap.add_argument('--skip-tokens',type=int,default=0);a=ap.parse_args()
    repo=Path(a.repo).resolve();out=Path(a.out).resolve();meta={'kind':'s100_p8_overnight_real_snapshot','status':'started','skip_tokens':a.skip_tokens}
    try:
        add_repo(repo);import cupy as cp
        from s100_phase5_runtime import build_phase5_runtime
        from graph_e1f22 import _load_prompt_set
        from diag_component_marginals_graph import _reset_exact_state,_prefill
        from moe_dev_batched import DOWN_PANEL_BYTES
        prompts,_e,_n,capacity=_load_prompt_set('full');b=build_phase5_runtime(int(capacity),layer_k={},alpha=0.0);rt=b.rt
        _reset_exact_state(rt);_prefill(rt,prompts[0]['prompt_ids'])
        for _ in range(int(a.skip_tokens)+1):rt.step_graph(None);rt._graph_stream.synchronize()
        free=closure_dict(rt._moe_dev.__func__);state=free.get('state')
        if not isinstance(state,dict):
            for v in free.values():
                if isinstance(v,dict) and v and isinstance(next(iter(v.values())),dict) and 'act' in next(iter(v.values())):state=v;break
        if not isinstance(state,dict):raise RuntimeError(f'phase5 state not found: {list(free)}')
        layers=[int(x) for x in rt.moe_layers];arrays={};layer_meta=[];topk=int(rt.top_k);inter=int(rt.moe_inter);rows=int(rt.hidden);npanel=inter//16
        for layer in layers:
            bs=state[layer];dev=rt._dev_cache[layer];bank=rt.bank[layer]
            ids=cp.asnumpy(dev['ids'][:topk]).astype(np.int32);route=cp.asnumpy(dev['w'][:topk]).astype(np.float32);act=cp.asnumpy(bs['act'][:topk*inter]).reshape(topk,inter).astype(np.float32);masks=cp.asnumpy(bs['masks'][:topk*npanel]).reshape(topk,npanel).astype(np.uint32);pcount=cp.asnumpy(bs['pcount'][:topk]).astype(np.int32);nzc=cp.asnumpy(bs['nzc'][:topk]).astype(np.int32)
            base=int(bank['down_base_ptr']);records=np.empty((topk,DOWN_PANEL_BYTES),np.uint8);gg=bank['globals'];gg=cp.asnumpy(gg).reshape(-1) if hasattr(gg,'get') else np.asarray(gg).reshape(-1);g=np.empty(topk,np.float32)
            for s,eid in enumerate(ids):
                records[s]=np.frombuffer(ctypes.string_at(base+int(eid)*DOWN_PANEL_BYTES,DOWN_PANEL_BYTES),dtype=np.uint8,count=DOWN_PANEL_BYTES);g[s]=np.float32(gg[int(eid)*2])
            pref=f'L{layer}';arrays[pref+'_ids']=ids;arrays[pref+'_route_w']=route;arrays[pref+'_act']=act;arrays[pref+'_masks']=masks;arrays[pref+'_pcount']=pcount;arrays[pref+'_nzc']=nzc;arrays[pref+'_records']=records;arrays[pref+'_globals']=g
            layer_meta.append({'layer':layer,'ids':ids.tolist(),'pcount':pcount.tolist(),'nzc':nzc.tolist(),'nonzero_fraction':[float(x/inter) for x in nzc]})
        arrays['e2m1']=cp.asnumpy(rt.fused.e2m1).astype(np.float32);arrays['e4m3']=cp.asnumpy(rt.fused.e4m3).astype(np.float32);arrays['meta_json']=np.asarray(json.dumps({'hidden':rows,'moe_inter':inter,'top_k':topk,'npanel':npanel,'down_panel_bytes':int(DOWN_PANEL_BYTES),'layers':layer_meta,'skip_tokens':int(a.skip_tokens)}))
        out.parent.mkdir(parents=True,exist_ok=True);np.savez(out,**arrays);meta.update({'status':'measured','path':str(out),'bytes':out.stat().st_size,'hidden':rows,'moe_inter':inter,'top_k':topk,'layer_count':len(layers),'layers':layer_meta});b.restore_combined();b.restore_selective()
    except Exception as e:meta.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}})
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':meta.get('status'),'skip_tokens':a.skip_tokens,'bytes':meta.get('bytes'),'layer_count':meta.get('layer_count'),'error':(meta.get('error') or {}).get('message'),'output':str(out)},indent=2));return 0 if meta['status']=='measured' else 2
if __name__=='__main__':raise SystemExit(main())
