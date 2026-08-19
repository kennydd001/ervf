from __future__ import annotations
import json,traceback
import numpy as np
from common import REPO,write_json_atomic,utc_now
from s100_phase23_group_kernels import Phase23Kernels,ROUTES,GROUPS,MAXM

OUT=REPO/"pro_research"/"results"/"s100_phase23"/"S100_PHASE23_PREFLIGHT.json"

def main():
    payload={"kind":"s100_phase23_preflight","status":"started","started_utc":utc_now()}
    try:
        import cupy as cp
        k=Phase23Kernels()  # forces NVRTC module creation on first function lookup
        # Four tokens x six unique-per-token routes, repeated across tokens.
        ids=np.asarray([
          1,2,3,4,5,6,
          1,7,8,9,10,11,
          1,2,12,13,14,15,
          1,2,3,16,17,18,
        ],np.int32)
        di=cp.asarray(ids)
        rg=cp.empty(ROUTES,cp.int32);gi=cp.empty(GROUPS,cp.int32)
        gc=cp.empty(GROUPS,cp.int32);gr=cp.empty(GROUPS*MAXM,cp.int32)
        ng=cp.zeros(1,cp.int32)
        k.group(di,rg,gi,gc,gr,ng)
        cp.cuda.get_current_stream().synchronize()
        ngi=int(cp.asnumpy(ng)[0]);counts=cp.asnumpy(gc)[:ngi].tolist()
        gids=cp.asnumpy(gi)[:ngi].tolist()

        # Tiny exact-LRU state, same algorithmic contract.
        ne=32;cap=8
        dev={
          "slot_of":cp.full(ne,-1,cp.int32),
          "expert_of":cp.full(cap,-1,cp.int32),
          "last_used":cp.full(cap,-1,cp.int32),
          "state2":cp.zeros(2,cp.int32),
          "stats2":cp.zeros(2,cp.int32),
        }
        slots=cp.empty(ROUTES,cp.int32);need=cp.empty(ROUTES,cp.int32)
        k.cache_assign(dev,di,slots,need,cap)
        cp.cuda.get_current_stream().synchronize()
        need_h=cp.asnumpy(need)
        group_ok=bool(
          ngi==18 and gids[:3]==[1,2,3]
          and counts[0]==4 and counts[1]==3 and counts[2]==2
          and all(1<=int(x)<=4 for x in counts)
        )
        lru_ok=bool(
          need_h[0]==1 and need_h[6]==0 and need_h[12]==0 and need_h[18]==0
        )
        payload.update({"status":"measured","ngroups":ngi,"group_ids":gids,
          "group_counts":counts,"need":need_h.tolist(),
          "GROUP_PREFLIGHT_GREEN":group_ok,
          "LRU_PREFLIGHT_GREEN":lru_ok,
          "PREFLIGHT_GREEN":bool(group_ok and lru_ok),
          "completed_utc":utc_now()})
    except Exception as exc:
        payload.update({"status":"technical_failure",
          "error":{"type":type(exc).__name__,"message":str(exc),
                   "traceback":traceback.format_exc()},
          "completed_utc":utc_now()})
    OUT.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,payload,archive=True)
    print(json.dumps(payload,indent=2))
    return 0 if payload.get("status")=="measured" and payload.get("PREFLIGHT_GREEN") else 2
if __name__=="__main__":raise SystemExit(main())
