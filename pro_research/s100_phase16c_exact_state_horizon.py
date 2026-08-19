from __future__ import annotations
import json,traceback
import numpy as np
from common import write_json_atomic,utc_now
from s100_phase16_common import (RESULTS,make_runtime,release_runtime,prompt_rows,
    compact_snapshot,restore_snapshot,collect_cases)

OUT=RESULTS/"S100_PHASE16C_EXACT_STATE_HORIZON.json"
HORIZONS=(1,2,4,8);BLOCKS=4

def run_scope(rt,dispatch,rows,label,names):
    import cupy as cp
    records=[]
    for H in HORIZONS:
        for p in rows:
            dispatch.set_selected([]);rt.reset()
            for t in p["prompt_ids"]:rt.step(int(t))
            inp=int(cp.argmax(rt.logits).item())
            for bi in range(BLOCKS):
                pre=compact_snapshot(rt)
                dispatch.set_selected(names)
                cand=[];tok=inp
                for _ in range(H):
                    tok=int(rt.step(int(tok)));cand.append(tok)
                restore_snapshot(rt,pre);del pre
                dispatch.set_selected([])
                exact=[];tok=inp
                for _ in range(H):
                    tok=int(rt.step(int(tok)));exact.append(tok)
                accepted=H
                for j,(a,b) in enumerate(zip(exact,cand)):
                    if a!=b:accepted=j;break
                records.append({"scope":label,"horizon":H,"prompt_id":p["id"],
                    "domain":p["domain"],"block":bi,"accepted_exact_prefix":accepted,
                    "first_prediction_match":exact[0]==cand[0],"full_block_match":accepted==H,
                    "position_agreement":float(np.mean(np.asarray(exact)==np.asarray(cand))),
                    "exact":exact,"candidate":cand})
                inp=int(exact[-1])
    summary={}
    for H in HORIZONS:
        x=[r for r in records if r["horizon"]==H]
        a=np.asarray([r["accepted_exact_prefix"] for r in x],np.float64)
        summary[str(H)]={"blocks":len(x),
          "first_prediction_agreement":float(np.mean([r["first_prediction_match"] for r in x])),
          "mean_accepted_exact_prefix":float(a.mean()),
          "p50_accepted_exact_prefix":float(np.percentile(a,50)),
          "full_block_match_rate":float(np.mean([r["full_block_match"] for r in x])),
          "mean_position_agreement":float(np.mean([r["position_agreement"] for r in x]))}
    h4=summary["4"]
    go=bool(h4["first_prediction_agreement"]>=.95 and
            h4["mean_accepted_exact_prefix"]>=1.5 and h4["full_block_match_rate"]>=.25)
    return summary,go,records

def main():
    payload={"kind":"s100_phase16c_exact_state_horizon","status":"started",
      "started_utc":utc_now(),"claim_boundary":"candidate blocks start from exact state"}
    try:
        a=json.loads((RESULTS/"S100_PHASE16A_LOCAL_SENSITIVITY.json").read_text(encoding="utf-8"))
        b=json.loads((RESULTS/"S100_PHASE16B_SUBSET_VALIDATION.json").read_text(encoding="utf-8"))
        rt,dispatch=make_runtime();cases=collect_cases(rt)
        attention=[c.name for c in cases if c.family=="attention"]
        safe=[x["name"] for x in a.get("safe_matrices_ranked",[])]
        selected=(b.get("selected_strict_subset") or {}).get("names") or safe
        scopes={"attention_all":attention}
        if selected:scopes["selected_safe_subset"]=selected
        rows=prompt_rows("validation");results={}
        for label,names in scopes.items():
            s,go,recs=run_scope(rt,dispatch,rows,label,names)
            results[label]={"names":names,"summary":s,"H4_BLOCK_RESEARCH_GO":go,
                            "records":recs}
            print(f"16C {label}: H4={s['4']} go={go}",flush=True)
        payload.update({"status":"measured","results":results,
          "ANY_H4_BLOCK_RESEARCH_GO":any(v["H4_BLOCK_RESEARCH_GO"] for v in results.values()),
          "completed_utc":utc_now()})
        release_runtime(rt)
    except Exception as e:
        payload.update({"status":"technical_failure","error":{"type":type(e).__name__,
          "message":str(e),"traceback":traceback.format_exc()},"completed_utc":utc_now()})
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({"status":payload.get("status"),
      "ANY_H4_BLOCK_RESEARCH_GO":payload.get("ANY_H4_BLOCK_RESEARCH_GO"),
      "results":{k:{"summary":v["summary"],"go":v["H4_BLOCK_RESEARCH_GO"]}
                 for k,v in payload.get("results",{}).items()},
      "error":(payload.get("error") or {}).get("message"),"output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
