from __future__ import annotations
import json,traceback
from common import write_json_atomic,utc_now
from s100_phase16_common import (RESULTS,make_runtime,release_runtime,prompt_rows,
    collect_cases,compact_snapshot,restore_snapshot,compare_logits,summarize,phase14_savings)

OUT=RESULTS/"S100_PHASE16A_LOCAL_SENSITIVITY.json"
TOKENS=8

def manifest(cases):
    def names(fn):return [c.name for c in cases if fn(c)]
    d={
      "attention_all":names(lambda c:c.family=="attention"),
      "attention_k":names(lambda c:c.family=="attention" and c.side=="k"),
      "attention_v":names(lambda c:c.family=="attention" and c.side=="v"),
      "attention_o":names(lambda c:c.family=="attention" and c.side=="o"),
      "attention_q":names(lambda c:c.family=="attention" and c.side=="q"),
      "mamba_in":names(lambda c:c.family=="mamba" and c.side=="in"),
      "mamba_out":names(lambda c:c.family=="mamba" and c.side=="out"),
    }
    d={k:v for k,v in d.items() if v}
    for c in cases:d[f"matrix:{c.name}"]=[c.name]
    return d

def run_scope(rt,dispatch,rows,names):
    import cupy as cp
    cand=cp.empty_like(rt.logits); records=[]
    for p in rows:
        dispatch.set_selected([]);rt.reset()
        for t in p["prompt_ids"]:rt.step(int(t))
        inp=int(cp.argmax(rt.logits).item())
        for step in range(TOKENS):
            pre=compact_snapshot(rt)
            dispatch.set_selected(names);rt.step(inp);cand[...]=rt.logits
            restore_snapshot(rt,pre);del pre
            dispatch.set_selected([]);rt.step(inp)
            r=compare_logits(cp,cand,rt.logits)
            r.update({"prompt_id":p["id"],"domain":p["domain"],"step":step})
            records.append(r);inp=int(r["target"])
    return summarize(records)

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    payload={"kind":"s100_phase16a_local_sensitivity","status":"started",
             "tokens_per_prompt":TOKENS,"started_utc":utc_now(),
             "claim_boundary":"one native transition from exact state"}
    try:
        rt,dispatch=make_runtime()
        cases=collect_cases(rt);scopes=manifest(cases)
        rows=prompt_rows("calibration");sav=phase14_savings();results={}
        for i,(scope,names) in enumerate(scopes.items()):
            s=run_scope(rt,dispatch,rows,names)
            safe=bool(s and s["top1"]>=.995 and s["top5"]==1.0 and s["K16"]==1.0
                      and s["mean_ce"]<=.01 and s["mean_kl"]<=.005 and s["finite"])
            rec={"selected_names":names,"summary":s,"locally_safe":safe}
            if scope.startswith("matrix:"):rec["phase14_component"]=sav.get(names[0])
            results[scope]=rec
            print(f"16A {i+1}/{len(scopes)} {scope}: top1={s['top1']:.4f} KL={s['mean_kl']:.5f} safe={safe}",flush=True)
        safe=[]
        for c in cases:
            r=results[f"matrix:{c.name}"]
            if r["locally_safe"]:
                comp=sav.get(c.name,{})
                safe.append({"name":c.name,"family":c.family,"side":c.side,"layer":c.layer,
                             "summary":r["summary"],"phase14_component":comp,
                             "B1_saving_ms":float(comp.get("B1_saving_ms",0.0))})
        safe.sort(key=lambda x:(-x["B1_saving_ms"],x["summary"]["mean_kl"]))
        payload.update({"status":"measured","case_count":len(cases),"results":results,
                        "safe_matrix_count":len(safe),"safe_matrices_ranked":safe,
                        "completed_utc":utc_now()})
        release_runtime(rt)
    except Exception as e:
        payload.update({"status":"technical_failure","error":{"type":type(e).__name__,
            "message":str(e),"traceback":traceback.format_exc()},"completed_utc":utc_now()})
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({"status":payload.get("status"),"safe_matrix_count":payload.get("safe_matrix_count"),
      "safe_matrix_names":[x["name"] for x in payload.get("safe_matrices_ranked",[])],
      "group_scopes":{k:v["summary"] for k,v in payload.get("results",{}).items()
                      if not k.startswith("matrix:")},
      "error":(payload.get("error") or {}).get("message"),"output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
