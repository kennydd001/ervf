from __future__ import annotations
import argparse, collections, json, math, os, sys, traceback
from pathlib import Path
import numpy as np

def add_repo(repo: Path):
    sys.path.insert(0,str(repo/'pro_research'))
    sys.path.insert(0,str(repo/'src'))
    os.chdir(repo)

def lru_curve(trace_by_layer, capacities):
    out={}
    for cap in capacities:
        hits=misses=0
        for layer, seq in trace_by_layer.items():
            cache=collections.OrderedDict()
            for ids in seq:
                for e in ids:
                    e=int(e)
                    if e in cache:
                        hits+=1; cache.move_to_end(e)
                    else:
                        misses+=1; cache[e]=None
                        if len(cache)>cap: cache.popitem(last=False)
        out[str(cap)]={"hits":hits,"misses":misses,
                       "hit_fraction":hits/(hits+misses) if hits+misses else None}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--tokens',type=int,default=768)
    args=ap.parse_args()
    repo=Path(args.repo).resolve(); out=Path(args.out).resolve()
    payload={"kind":"s100_phase8_route_miss_census","status":"started"}
    try:
        add_repo(repo)
        import cupy as cp
        from s100_phase5_runtime import build_phase5_runtime
        from graph_e1f22 import _load_prompt_set
        from diag_component_marginals_graph import _reset_exact_state,_prefill

        prompts,_e,_n,capacity=_load_prompt_set('full')
        b=build_phase5_runtime(int(capacity),layer_k={},alpha=0.0)
        rt=b.rt
        layers=[int(x) for x in rt.moe_layers]
        topk=int(rt.top_k)
        target=max(1,int(args.tokens))
        trace={i:[] for i in layers}
        actual_need={i:[] for i in layers}
        overlap={i:[] for i in layers}
        prev={i:None for i in layers}
        done=0
        for p in prompts:
            if done>=target: break
            _reset_exact_state(rt); _prefill(rt,p['prompt_ids'])
            while done<target:
                rt.step_graph(None); rt._graph_stream.synchronize()
                # Stack on device so the diagnostic pays only two D2H copies/token.
                ids_gpu=cp.stack([rt._dev_cache[i]['ids'][:topk] for i in layers])
                need_gpu=cp.stack([rt._dev_cache[i]['need'][:topk] for i in layers])
                ids=cp.asnumpy(ids_gpu).astype(np.int32,copy=False)
                need=cp.asnumpy(need_gpu).astype(np.int32,copy=False)
                for li,layer in enumerate(layers):
                    row=ids[li].tolist(); trace[layer].append(row)
                    actual_need[layer].append(int(need[li].sum()))
                    if prev[layer] is not None:
                        overlap[layer].append(len(set(prev[layer]) & set(row)))
                    prev[layer]=row
                done+=1
                if done>=target: break

        layer_rows={}
        all_need=[]
        for layer in layers:
            needs=np.asarray(actual_need[layer],dtype=np.int32)
            all_need.extend(needs.tolist())
            flat=[e for row in trace[layer] for e in row]
            freq=collections.Counter(flat)
            total=len(flat)
            coverage=[]
            running=0
            for rank,(_e,c) in enumerate(freq.most_common(),1):
                running+=c
                if rank in (8,16,32,48,64,72,96):
                    coverage.append({"hotset":rank,"route_coverage":running/total})
            layer_rows[str(layer)]={
                "tokens":len(needs),
                "actual_up_cache_miss_mean":float(needs.mean()),
                "actual_up_cache_miss_p95":float(np.percentile(needs,95)),
                "miss_distribution":{str(k):int((needs==k).sum()) for k in range(topk+1)},
                "previous_token_route_overlap_mean":float(np.mean(overlap[layer])) if overlap[layer] else None,
                "unique_experts":len(freq),
                "hotset_coverage":coverage,
            }
        an=np.asarray(all_need,dtype=np.int32)
        payload.update({
            "status":"measured",
            "shape_contract":{
                "hidden":int(rt.hidden),"moe_inter":int(rt.moe_inter),
                "n_experts":int(rt.n_experts),"top_k":topk,
                "moe_layers":layers,"moe_layer_count":len(layers),
                "capacity":int(capacity),
            },
            "tokens":done,
            "actual_up_cache":{
                "misses_per_layer_token_mean":float(an.mean()),
                "misses_per_layer_token_p95":float(np.percentile(an,95)),
                "route_slot_miss_fraction":float(an.sum()/(len(an)*topk)),
                "distribution":{str(k):int((an==k).sum()) for k in range(topk+1)},
            },
            "offline_lru":lru_curve(trace,(16,32,48,64,72,96)),
            "per_layer":layer_rows,
        })
        b.restore_combined();b.restore_selective()
    except Exception as e:
        payload.update({"status":"technical_failure",
                        "error":{"type":type(e).__name__,"message":str(e),
                                 "traceback":traceback.format_exc()}})
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding='utf-8')
    print(json.dumps(payload,indent=2,allow_nan=False))
    return 0 if payload["status"]=="measured" else 2
if __name__=="__main__": raise SystemExit(main())
