from __future__ import annotations

import json
import traceback
import numpy as np

from common import utc_now,write_json_atomic
from s100_phase21_common import load_trace,prefill_to,expected_for_block,release
from s100_phase27_common import RESULTS,Variant,make_candidate,phase27_gate

OUT=RESULTS/"S100_PHASE27_PIPELINE_PROFILE.json"
CTX=1024


def main():
    payload={
      "kind":"s100_phase27_pipeline_profile","status":"started",
      "context":CTX,"started_utc":utc_now(),
      "claim_boundary":"captured event spans for selected candidate; not adoption timing",
    }
    rt=None
    try:
        phase27_gate()
        sel=json.loads(
          (RESULTS/"S100_PHASE27_SELECTION.json").read_text(encoding="utf-8")
        )
        v=sel.get("selected_variant") or {}
        variant=Variant(
          int(v["gather_y"]),int(v["batches"]),bool(v["shared_overlap"])
        )
        rt,g,keep=make_candidate(CTX,variant,diagnostic=True)
        cap=g.setup_graph()
        tr=load_trace();tokens=tr["tokens"]
        prefill_to(rt,tokens,CTX)
        g.prepare_after_prefill()

        records=[]
        for _ in range(2):
            pos=int(rt.pos)
            draft,expected=expected_for_block(tokens,pos)
            got=g.launch(draft.tolist())
            if not np.array_equal(got,expected):
                raise RuntimeError(
                  f"profile ids mismatch pos={pos} got={got.tolist()} "
                  f"expected={expected.tolist()}"
                )
            snap=g.gmoe.profile_snapshot()
            snap["pos"]=pos
            records.append(snap)

        pipe=[x["pipeline_span_ms_per_h4"] for x in records]
        gather=[x["gather_stream_span_ms_per_h4"] for x in records]
        payload.update({
          "status":"measured",
          "variant":variant.as_dict(),
          "capture_info":cap,
          "records":records,
          "median_pipeline_span_ms_per_h4":float(np.median(pipe)),
          "median_gather_stream_span_ms_per_h4":float(np.median(gather)),
          "phase24_reference_ms_per_h4":{
            "down_gather":17.96296000480652,
            "down_compute_reduce":3.3009280152618885,
            "serial_sum":21.263888020068408,
          },
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({
          "status":"technical_failure",
          "error":{
            "type":type(exc).__name__,
            "message":str(exc),
            "traceback":traceback.format_exc(),
          },
          "completed_utc":utc_now(),
        })
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass

    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),
      "variant":payload.get("variant"),
      "median_pipeline_span_ms_per_h4":payload.get("median_pipeline_span_ms_per_h4"),
      "median_gather_stream_span_ms_per_h4":payload.get("median_gather_stream_span_ms_per_h4"),
      "reference":payload.get("phase24_reference_ms_per_h4"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(OUT),
    },indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
