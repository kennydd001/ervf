from __future__ import annotations

import json
import time
import traceback
from collections import defaultdict

import numpy as np

from common import write_json_atomic,utc_now
from s100_phase21_common import (
    RESULTS,identity_gate,load_trace,make_rt,verifier_for,
    prefill_to,expected_for_block,release,
)

OUT=RESULTS/"S100_PHASE21_PROFILE.json"
CONTEXT=1024

def timed_sync(cp,fn):
    cp.cuda.get_current_stream().synchronize()
    t0=time.perf_counter_ns()
    result=fn()
    cp.cuda.get_current_stream().synchronize()
    return result,(time.perf_counter_ns()-t0)/1e6

def main():
    payload={"kind":"s100_phase21_profile","status":"started",
             "context":CONTEXT,"started_utc":utc_now(),
             "claim_boundary":"synchronized diagnostic profile; not throughput"}
    try:
        import cupy as cp
        identity_gate();tr=load_trace();tokens=tr["tokens"]
        rt,keep=make_rt(CONTEXT,"current_grouped")
        v=verifier_for(rt,"current_grouped")
        prefill_to(rt,tokens,CONTEXT)

        # One ordinary block first, with only end synchronization, for a fresh
        # same-process reference.
        drafts,expected=expected_for_block(tokens,int(rt.pos))
        cp.cuda.get_current_stream().synchronize()
        t0=time.perf_counter_ns()
        got,_=v.block(drafts.tolist(),False)
        cp.cuda.get_current_stream().synchronize()
        ordinary_ms=(time.perf_counter_ns()-t0)/1e6
        if not np.array_equal(got,expected):
            raise RuntimeError("ordinary profile-reference block diverged")

        layer_rows=[]
        family=defaultdict(float)
        pos=int(rt.pos)
        drafts,expected=expected_for_block(tokens,pos)

        _,ms=timed_sync(cp,lambda:v._embed(drafts.tolist()))
        family["embed"]+=ms

        for i,ch in enumerate(rt.pattern):
            d=rt.layer[i]
            _,nms=timed_sync(cp,lambda d=d:v._norm_rows(d["norm"],v.h,v.normed))
            family["layer_norm"]+=nms

            if ch=="M":
                _,fms=timed_sync(cp,lambda i=i:v._mamba(i,v.normed,v.acc))
                name="mamba"
            elif ch=="*":
                _,fms=timed_sync(
                    cp,lambda i=i:v._attention(i,v.normed,v.acc,pos)
                )
                name="attention"
            else:
                (_,_,_),fms=timed_sync(
                    cp,lambda i=i:v.moeb(i,v.normed,v.acc,False)
                )
                name="moe"
            family[name]+=fms

            _,ams=timed_sync(
                cp,lambda:rt.k.add_(v.h[0],v.acc[0],v.hidden)
            )
            # Above measures one add. Execute the remaining 3 and report the
            # real four-row add time separately to avoid silently multiplying.
            add_ms=ams
            for t in range(1,4):
                _,x=timed_sync(cp,lambda t=t:rt.k.add_(v.h[t],v.acc[t],v.hidden))
                add_ms+=x
            family["residual_add"]+=add_ms
            layer_rows.append({"layer":i,"kind":name,
                               "norm_ms":nms,"body_ms":fms,"add_ms":add_ms})

        _,fnms=timed_sync(cp,lambda:v._norm_rows(rt.norm_f,v.h,v.final_normed))
        family["final_norm"]+=fnms
        _,hms=timed_sync(
            cp,lambda:v.bk.nvfp4(
                rt.lm_head_codes,rt.lm_head_scales,rt.fused.e2m1,rt.fused.e4m3,
                v.final_normed,v.logits,rt.lm_head_g,v.vocab,v.hidden,4,False
            )
        )
        family["lm_head"]+=hms
        got,ams=timed_sync(cp,lambda:cp.asnumpy(cp.argmax(v.logits,axis=1)).astype(np.int32))
        family["argmax_readback"]+=ams
        if not np.array_equal(got,expected):
            raise RuntimeError("synchronized profiled block diverged")
        rt.pos=pos+4

        # Census on the next canonical block; not timed.
        drafts2,expected2=expected_for_block(tokens,int(rt.pos))
        got2,census=v.block(drafts2.tolist(),True)
        cp.cuda.get_current_stream().synchronize()
        if not np.array_equal(got2,expected2):
            raise RuntimeError("census block diverged")
        unique=[int(x["unique_experts"]) for x in census]
        repeat=[float(x["repeat_rate"]) for x in census]
        hist=defaultdict(int)
        for row in census:
            for m in row["rows_per_expert"]:
                hist[int(m)]+=1

        payload.update({
            "status":"measured",
            "ordinary_block_ms":ordinary_ms,
            "profile_sync_sum_ms":float(sum(family.values())),
            "family_ms":dict(family),
            "family_fraction_of_sync_sum":{
                k:float(vv/sum(family.values())) for k,vv in family.items()
            },
            "layers":layer_rows,
            "route_census":{
                "moe_layers":len(census),
                "median_unique_experts":float(np.median(unique)),
                "median_repeat_rate":float(np.median(repeat)),
                "rows_per_expert_histogram":{str(k):v for k,v in sorted(hist.items())},
                "m1_fraction":float(hist.get(1,0)/max(sum(hist.values()),1)),
            },
            "all_token_exact":True,
            "completed_utc":utc_now(),
        })
        release(rt)
    except Exception as exc:
        payload.update({"status":"technical_failure",
            "error":{"type":type(exc).__name__,"message":str(exc),
                     "traceback":traceback.format_exc()},
            "completed_utc":utc_now()})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),
      "ordinary_block_ms":payload.get("ordinary_block_ms"),
      "family_ms":payload.get("family_ms"),
      "route_census":payload.get("route_census"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
