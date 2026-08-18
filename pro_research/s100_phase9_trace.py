from __future__ import annotations
import argparse, json, os, sys, traceback
from pathlib import Path
import numpy as np

def add_repo(repo):
    sys.path.insert(0,str(repo/'pro_research'));sys.path.insert(0,str(repo/'src'));os.chdir(repo)

def load_prompts(repo):
    from transformers import AutoTokenizer
    from common import require_model_dir
    obj=json.loads((repo/'pro_research'/'S100_PHASE3_PROMPTS.json').read_text(encoding='utf-8'))
    rows=obj.get('prompts',[]);tok=AutoTokenizer.from_pretrained(str(require_model_dir()),local_files_only=True,trust_remote_code=True,use_fast=True);out=[]
    for i,r in enumerate(rows):
        text=r.get('prompt','');ids=tok.encode(text,add_special_tokens=False)
        if ids: out.append({'index':i,'id':r.get('id',str(i)),'prompt_ids':[int(x) for x in ids]})
    if not out: raise RuntimeError('frozen 40-prompt pool unavailable')
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--outdir',required=True);ap.add_argument('--tokens',type=int,default=8192);a=ap.parse_args()
    repo=Path(a.repo).resolve();od=Path(a.outdir).resolve();od.mkdir(parents=True,exist_ok=True)
    js=od/'S100_PHASE9_TRACE.json';npz=od/'S100_PHASE9_TRACE.npz';p={'kind':'s100_phase9_trace','status':'started'}
    try:
        add_repo(repo);import cupy as cp
        from s100_phase5_runtime import build_phase5_runtime
        from diag_component_marginals_graph import _reset_exact_state,_prefill
        from graph_e1f22 import _load_prompt_set
        prompts=load_prompts(repo);_p,_e,_n,capacity=_load_prompt_set('full')
        b=build_phase5_runtime(int(capacity),layer_k={},alpha=0.0);rt=b.rt;layers=[int(x) for x in rt.moe_layers];L=len(layers);K=int(rt.top_k)
        ids_rows=[];need_rows=[];counted=[];sessions=[];prompt_indices=[];token_in_session=[]
        measured_need=[];target=int(a.tokens);done=0;sid=0
        while done<target:
            pr=prompts[sid%len(prompts)];_reset_exact_state(rt);_prefill(rt,pr['prompt_ids'])
            total_steps=32+min(224,target-done)
            for j in range(total_steps):
                rt.step_graph(None);rt._graph_stream.synchronize()
                ids=cp.asnumpy(cp.stack([rt._dev_cache[i]['ids'][:K] for i in layers])).astype(np.int16,copy=False)
                need=cp.asnumpy(cp.stack([rt._dev_cache[i]['need'][:K] for i in layers])).astype(np.int8,copy=False)
                is_count=j>=32
                ids_rows.append(ids.copy());need_rows.append(need.copy());counted.append(is_count);sessions.append(sid);prompt_indices.append(pr['index']);token_in_session.append(j)
                if is_count:
                    measured_need.append(need.copy());done+=1
            sid+=1
            if sid%8==0:print(f'trace {done}/{target} measured tokens, {sid} sessions',flush=True)
        ids_arr=np.stack(ids_rows);need_arr=np.stack(need_rows);cnt=np.asarray(counted,bool);ses=np.asarray(sessions,np.int16);pi=np.asarray(prompt_indices,np.int16);tis=np.asarray(token_in_session,np.int16)
        mn=np.stack(measured_need);actual=float(mn.sum()/mn.size)
        np.savez_compressed(npz,ids=ids_arr,need=need_arr,counted=cnt,session=ses,prompt_index=pi,token_in_session=tis,layers=np.asarray(layers,np.int16))
        p.update({'status':'measured','measured_tokens':int(cnt.sum()),'all_steps':int(len(cnt)),'sessions':int(sid),'prompt_pool':len(prompts),'shape':{'hidden':int(rt.hidden),'moe_inter':int(rt.moe_inter),'n_experts':int(rt.n_experts),'top_k':K,'moe_layers':layers},'measured_route_slot_miss_fraction':actual,'npz':str(npz),'npz_bytes':npz.stat().st_size})
        b.restore_combined();b.restore_selective()
    except Exception as e:p.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}})
    js.write_text(json.dumps(p,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(p,indent=2));return 0 if p.get('status')=='measured' else 2
if __name__=='__main__':raise SystemExit(main())
