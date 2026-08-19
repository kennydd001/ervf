from __future__ import annotations
import json,traceback
from common import write_json_atomic,utc_now
from s100_phase16_common import (RESULTS,make_runtime,release_runtime,prompt_rows,
    compact_snapshot,restore_snapshot,compare_logits,summarize)

OUT=RESULTS/"S100_PHASE16B_SUBSET_VALIDATION.json";TOKENS=24

def main():
    payload={"kind":"s100_phase16b_subset_validation","status":"started",
      "tokens_per_prompt":TOKENS,"started_utc":utc_now(),
      "claim_boundary":"exact token prefixes; candidate state accumulates"}
    try:
        a=json.loads((RESULTS/"S100_PHASE16A_LOCAL_SENSITIVITY.json").read_text(encoding="utf-8"))
        safe=[x["name"] for x in a.get("safe_matrices_ranked",[])]
        if not safe:
            payload.update({"status":"measured","safe_matrix_count":0,"subsets":[],
                            "selected_strict_subset":None,"completed_utc":utc_now()})
        else:
            import cupy as cp
            sizes=sorted(set(min(n,len(safe)) for n in (1,2,4,8,16,len(safe))))
            rt,dispatch=make_runtime();rows=prompt_rows("validation")
            # Exact greedy target chains.
            exact={}
            dispatch.set_selected([])
            for p in rows:
                rt.reset()
                for t in p["prompt_ids"]:rt.step(int(t))
                chain=[]
                for j in range(TOKENS):
                    chain.append(int(cp.argmax(rt.logits).item()))
                    if j+1<TOKENS:rt.step(chain[-1])
                exact[p["id"]]=chain

            cand_logits=cp.empty_like(rt.logits);results=[]
            for size in sizes:
                names=safe[:size];records=[]
                for p in rows:
                    dispatch.set_selected(names);rt.reset()
                    for t in p["prompt_ids"]:rt.step(int(t))
                    for step,target in enumerate(exact[p["id"]]):
                        cand_logits[...]=rt.logits
                        cand_state=compact_snapshot(rt)

                        # Exact logits for the same exact prefix, using the same
                        # pinned-bank runtime. Replay is slow but selection-only.
                        dispatch.set_selected([]);rt.reset()
                        for t in p["prompt_ids"]:rt.step(int(t))
                        for j in range(step):rt.step(int(exact[p["id"]][j]))
                        exact_logits=rt.logits.copy()

                        restore_snapshot(rt,cand_state);del cand_state
                        dispatch.set_selected(names)
                        r=compare_logits(cp,cand_logits,exact_logits)
                        r.update({"prompt_id":p["id"],"step":step,"subset_size":size})
                        records.append(r)
                        if step+1<TOKENS:rt.step(int(target))
                s=summarize(records)
                strict=bool(s["top1"]>=.970 and s["top5"]>=.999 and s["mean_ce"]<=.025
                            and s["mean_kl"]<=.015 and s["p95_kl"]<=.060 and s["finite"])
                results.append({"size":size,"names":names,"summary":s,"strict_pass":strict})
                print(f"16B N={size}: top1={s['top1']:.4f} KL={s['mean_kl']:.5f} strict={strict}",flush=True)
            green=[x for x in results if x["strict_pass"]]
            selected=max(green,key=lambda x:x["size"]) if green else None
            payload.update({"status":"measured","safe_matrix_count":len(safe),
                            "subsets":results,"selected_strict_subset":selected,
                            "completed_utc":utc_now()})
            release_runtime(rt)
    except Exception as e:
        payload.update({"status":"technical_failure","error":{"type":type(e).__name__,
            "message":str(e),"traceback":traceback.format_exc()},"completed_utc":utc_now()})
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({"status":payload.get("status"),
      "selected_strict_subset":payload.get("selected_strict_subset"),
      "subsets":payload.get("subsets"),"error":(payload.get("error") or {}).get("message"),
      "output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
