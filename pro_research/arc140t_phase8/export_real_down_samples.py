from __future__ import annotations
import argparse, ctypes, json, os, sys, traceback
from pathlib import Path
import numpy as np

def add_repo(repo):
    sys.path.insert(0,str(repo/'pro_research'))
    sys.path.insert(0,str(repo/'src'))
    os.chdir(repo)

def closure_dict(fn):
    vals={}
    cells=fn.__closure__ or ()
    for name,cell in zip(fn.__code__.co_freevars,cells):
        try: vals[name]=cell.cell_contents
        except ValueError: pass
    return vals

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    repo=Path(args.repo).resolve();out=Path(args.out).resolve()
    meta={"kind":"s100_phase8_real_down_sample","status":"started"}
    try:
        add_repo(repo)
        import cupy as cp
        from s100_phase5_runtime import build_phase5_runtime
        from graph_e1f22 import _load_prompt_set
        from diag_component_marginals_graph import _reset_exact_state,_prefill
        from moe_dev_batched import DOWN_PANEL_BYTES

        prompts,_e,_n,capacity=_load_prompt_set('full')
        b=build_phase5_runtime(int(capacity),layer_k={},alpha=0.0)
        rt=b.rt
        _reset_exact_state(rt);_prefill(rt,prompts[0]['prompt_ids'])
        rt.step_graph(None);rt._graph_stream.synchronize()

        free=closure_dict(rt._moe_dev.__func__)
        state=None
        for v in free.values():
            if isinstance(v,dict) and v and all(isinstance(k,int) for k in v.keys()):
                sample=next(iter(v.values()))
                if isinstance(sample,dict) and 'act' in sample and 'masks' in sample:
                    state=v;break
        if state is None:
            raise RuntimeError(f"phase5 MoE closure state not found; freevars={list(free)}")

        layers=[int(x) for x in rt.moe_layers]
        chosen=sorted(set([layers[0],layers[len(layers)//2],layers[-1]]))
        arrays={}
        layer_meta=[]
        topk=int(rt.top_k); inter=int(rt.moe_inter); rows=int(rt.hidden)
        for layer in chosen:
            bs=state[layer];dev=rt._dev_cache[layer];bank=rt.bank[layer]
            ids=cp.asnumpy(dev['ids'][:topk]).astype(np.int32)
            route_w=cp.asnumpy(dev['w'][:topk]).astype(np.float32)
            act=cp.asnumpy(bs['act'][:topk*inter]).reshape(topk,inter).astype(np.float32)
            masks=cp.asnumpy(bs['masks'][:topk*(inter//16)]).reshape(topk,inter//16).astype(np.uint32)
            pcount=cp.asnumpy(bs['pcount'][:topk]).astype(np.int32)
            nzc=cp.asnumpy(bs['nzc'][:topk]).astype(np.int32)
            base=int(bank['down_base_ptr'])
            records=np.empty((topk,DOWN_PANEL_BYTES),dtype=np.uint8)
            globals_obj=bank['globals']
            if hasattr(globals_obj,'get'):
                globals_all=cp.asnumpy(globals_obj).astype(np.float32,copy=False).reshape(-1)
            else:
                globals_all=np.asarray(globals_obj,dtype=np.float32).reshape(-1)
            g=np.empty(topk,dtype=np.float32)
            for s,eid in enumerate(ids):
                records[s]=np.frombuffer(ctypes.string_at(base+int(eid)*DOWN_PANEL_BYTES,DOWN_PANEL_BYTES),
                                         dtype=np.uint8,count=DOWN_PANEL_BYTES)
                g[s]=globals_all[int(eid)*2+0]
            prefix=f"L{layer}"
            arrays[prefix+"_ids"]=ids
            arrays[prefix+"_route_w"]=route_w
            arrays[prefix+"_act"]=act
            arrays[prefix+"_masks"]=masks
            arrays[prefix+"_pcount"]=pcount
            arrays[prefix+"_nzc"]=nzc
            arrays[prefix+"_records"]=records
            arrays[prefix+"_globals"]=g
            layer_meta.append({
                "layer":layer,"ids":ids.tolist(),"pcount":pcount.tolist(),
                "nzc":nzc.tolist(),
                "nonzero_fraction":[float(x/inter) for x in nzc],
            })
        arrays["e2m1"]=cp.asnumpy(rt.fused.e2m1).astype(np.float32)
        arrays["e4m3"]=cp.asnumpy(rt.fused.e4m3).astype(np.float32)
        arrays["meta_json"]=np.asarray(json.dumps({
            "hidden":rows,"moe_inter":inter,"top_k":topk,
            "npanel":inter//16,"down_panel_bytes":int(DOWN_PANEL_BYTES),
            "layers":layer_meta
        }))
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,**arrays)
        meta.update({"status":"measured","path":str(out),
                     "bytes":out.stat().st_size,
                     "hidden":rows,"moe_inter":inter,"top_k":topk,
                     "down_panel_bytes":int(DOWN_PANEL_BYTES),
                     "layers":layer_meta})
        b.restore_combined();b.restore_selective()
    except Exception as e:
        meta.update({"status":"technical_failure",
                     "error":{"type":type(e).__name__,"message":str(e),
                              "traceback":traceback.format_exc()}})
    meta_path=out.with_suffix('.json')
    meta_path.write_text(json.dumps(meta,indent=2,allow_nan=False)+"\n",encoding='utf-8')
    print(json.dumps(meta,indent=2,allow_nan=False))
    return 0 if meta["status"]=="measured" else 2
if __name__=="__main__": raise SystemExit(main())
