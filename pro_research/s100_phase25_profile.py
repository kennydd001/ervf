from __future__ import annotations

import json
import traceback
import numpy as np

from common import utc_now,write_json_atomic
from s100_phase21_common import load_trace,prefill_to,release
from s100_phase25_common import RESULTS,make_h8,expected_for_h8
OUT=RESULTS/"S100_PHASE25_PROFILE.json"

def main():
    payload={"kind":"s100_phase25_profile","status":"started","context":1024,"started_utc":utc_now(),
      "claim_boundary":"synchronized diagnostic; not throughput"};rt=None
    try:
        import cupy as cp
        sel=json.loads((RESULTS/"S100_PHASE25_SELECTION.json").read_text(encoding="utf-8"));s=sel.get("selected")
        if not s:raise RuntimeError("no selected H8 variant")
        variant=str(s["variant"]);rt,g,keep=make_h8(1024,variant,diagnostic=True)
        tr=load_trace();tokens=tr["tokens"];prefill_to(rt,tokens,1024);g.prepare_after_prefill();blocks=[]
        for bi in range(2):
            pos=int(rt.pos);draft,expected=expected_for_h8(tokens,pos);g.tok_dev[...] = cp.asarray(draft);g.set_pos_from_host()
            import time;t0=time.perf_counter_ns();g.body();cp.cuda.get_current_stream().synchronize();rt.copy_stream.synchronize();wall=(time.perf_counter_ns()-t0)/1e6
            got=cp.asnumpy(g.ids_dev).astype(np.int32)
            if not np.array_equal(got,expected):raise RuntimeError(f"profile mismatch pos={pos}")
            rt.pos += 8;blocks.append({"block":bi,"pos":pos,"wall_ms":wall})
        rows=g.gmoe.profile_rows
        stage_names=("router","cache_group","shared","routed_up","mask_union","down_gather","down_compute_reduce")
        totals={k:float(sum(float(r.get("stage_ms",{}).get(k,0.0)) for r in rows)/2.0) for k in stage_names}
        ideal=float(sum(r["ideal_weight_streams"] for r in rows)/2.0);split=float(sum(r["split4_weight_streams"] for r in rows)/2.0)
        chosen=float(sum(r["selected_weight_streams"] for r in rows)/2.0)
        hist={}
        for r in rows:
            for k,v in r["m_histogram"].items():hist[k]=hist.get(k,0)+int(v)
        payload.update({"status":"measured","variant":variant,"blocks":blocks,
          "moe_layer_calls":len(rows),"stage_totals_ms_per_h8":totals,
          "weight_streams_per_h8":{"ideal_unique":ideal,"split4":split,"selected":chosen,
            "selected_vs_split4_reduction_fraction":0.0 if split==0 else 1.0-chosen/split},
          "global_m_histogram":hist,"max_m":max((r["max_m"] for r in rows),default=None),
          "records":rows,"completed_utc":utc_now()})
    except Exception as exc:
        payload.update({"status":"technical_failure","error":{"type":type(exc).__name__,"message":str(exc),
          "traceback":traceback.format_exc()},"completed_utc":utc_now()})
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass
    OUT.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({"status":payload.get("status"),"variant":payload.get("variant"),
      "stage_totals_ms_per_h8":payload.get("stage_totals_ms_per_h8"),"weight_streams_per_h8":payload.get("weight_streams_per_h8"),
      "max_m":payload.get("max_m"),"error":(payload.get("error") or {}).get("message"),"output":str(OUT)},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
