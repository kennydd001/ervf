from __future__ import annotations

import json
import traceback
import numpy as np

from common import REPO,utc_now,write_json_atomic
from s100_phase17_kernels import Phase17Kernels
from s100_phase19_residual_projection import FP8ProjectionBlock
from s100_phase25_common import RESULTS,phase24_gate,OFFICIAL_PARENT_H8_MS,ADOPTION_ABS_MS
from s100_phase25_h8_kernels import H8GroupedKernels,ROUTES,GROUPS,MAXM
from s100_phase25_sres_h8 import H8ScaleResidentGather

OUT=RESULTS/"S100_PHASE25_PREFLIGHT.json"

def main():
    payload={"kind":"s100_phase25_preflight","status":"started","started_utc":utc_now(),
      "claim_boundary":"compile/ABI/parent gate only; no H8 throughput claim"}
    try:
        import cupy as cp
        cfg,p24,th,st=phase24_gate()
        k=H8GroupedKernels();sg=H8ScaleResidentGather();p17=Phase17Kernels();fp8=FP8ProjectionBlock(cp)

        # Compile and execute group_routes48 on a pattern with an M8 expert.
        ids=np.arange(ROUTES,dtype=np.int32)+100
        ids[::6]=7  # route slot 0 of every token -> same expert exactly 8 times
        ids_g=cp.asarray(ids);route_group=cp.empty(ROUTES,cp.int32);group_ids=cp.empty(GROUPS,cp.int32)
        group_count=cp.empty(GROUPS,cp.int32);group_refs=cp.empty(GROUPS*MAXM,cp.int32);ng=cp.zeros(1,cp.int32)
        k.group(ids_g,route_group,group_ids,group_count,group_refs,ng)
        cp.cuda.get_current_stream().synchronize()
        n=int(cp.asnumpy(ng)[0]);counts=cp.asnumpy(group_count)[:n].astype(np.int32)
        grouping_green=bool(n>0 and counts.max()==8 and counts.min()>=1 and counts.max()<=8)

        # Force T8 FP8 residual2 kernel compilation on a tiny finite problem.
        rng=np.random.default_rng(2501);rows,cols=32,64
        W=cp.asarray(rng.integers(0,240,size=(rows,cols),dtype=np.uint8))
        x=cp.asarray(rng.standard_normal((8,cols),dtype=np.float32));out=cp.empty((8,rows),cp.float32)
        fp8.apply_residual2(W,x,out,1.0);cp.cuda.get_current_stream().synchronize()
        fp8_green=bool(cp.isfinite(out).all().item())

        selected_layers=sorted(int(x) for x in cfg.sres_layers)
        payload.update({
          "status":"measured","device":str(cp.cuda.runtime.getDeviceProperties(0).get("name")),
          "phase24_selected_layers":selected_layers,"phase24_resident_layer_count":len(selected_layers),
          "phase24_thermal_adopted":bool(th.get("BEST_OF_ALL_ADOPTED")),
          "phase24_state_green":bool(st.get("BEST_OF_ALL_STATE_GREEN")),
          "official_parent_h8_ms":OFFICIAL_PARENT_H8_MS,"adoption_abs_ms":ADOPTION_ABS_MS,
          "routes":ROUTES,"groups_capacity":GROUPS,"max_m":MAXM,
          "grouping_test":{"ngroups":n,"counts":counts.tolist(),"green":grouping_green},
          "fp8_t8_compile_green":fp8_green,"h8_sres_module_compiled":True,"phase17_t8_module_compiled":True,
          "PREFLIGHT_GREEN":bool(grouping_green and fp8_green and len(selected_layers)==23),
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({"status":"technical_failure","error":{"type":type(exc).__name__,
          "message":str(exc),"traceback":traceback.format_exc()},"completed_utc":utc_now()})
    OUT.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,payload,archive=True)
    print(json.dumps(payload,indent=2))
    return 0 if payload.get("PREFLIGHT_GREEN") else 2

if __name__=="__main__":raise SystemExit(main())
