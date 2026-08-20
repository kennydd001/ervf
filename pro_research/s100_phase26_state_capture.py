from __future__ import annotations

import argparse
import json
import traceback
import numpy as np

from common import utc_now, write_json_atomic
from s100_phase21_common import load_trace, prefill_to, expected_for_block, release
from s100_phase24_common import selected_config, make_synth
from s100_phase25_common import expected_for_h8
from s100_phase26_common import (
    RESULTS, phase26_gate, make_h4_overlap, make_h8_overlap, capture_arrays,
)

CTX=1024

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("h4_parent","h4_overlap","h8_parent","h8_overlap"),
        required=True,
    )
    args=ap.parse_args()

    out_json=RESULTS/f"S100_PHASE26_STATE_{args.mode.upper()}.json"
    out_npz=RESULTS/f"S100_PHASE26_STATE_{args.mode.upper()}.npz"
    payload={
      "kind":"s100_phase26_state_capture","status":"started",
      "mode":args.mode,"context":CTX,"started_utc":utc_now(),
    }
    rt=None
    try:
        phase26_gate()
        tr=load_trace();tokens=tr["tokens"]
        horizon=4 if args.mode.startswith("h4") else 8
        candidate=args.mode.endswith("overlap")

        if args.mode=="h4_parent":
            cfg=selected_config()
            rt,g,keep=make_synth(CTX,cfg)
        elif args.mode=="h4_overlap":
            rt,g,keep=make_h4_overlap(CTX)
        elif args.mode=="h8_parent":
            cfg=selected_config()
            rt,g,keep=make_synth(CTX,cfg)
        else:
            rt,g,keep=make_h8_overlap(CTX)

        cap=g.setup_graph()

        def run_once():
            rt.reset()
            prefill_to(rt,tokens,CTX)
            g.prepare_after_prefill()
            if horizon==4:
                draft,expected=expected_for_block(tokens,CTX)
                ids=g.launch(draft.tolist())
                logits_tail=g.v.logits
            elif args.mode=="h8_parent":
                draft,expected=expected_for_h8(tokens,CTX)
                a=g.launch(draft[:4].tolist())
                b=g.launch(draft[4:].tolist())
                ids=np.concatenate([a,b])
                logits_tail=g.v.logits
            else:
                draft,expected=expected_for_h8(tokens,CTX)
                ids=g.launch(draft.tolist())
                logits_tail=g.v.logits[4:]
            if not np.array_equal(ids,expected):
                raise RuntimeError(
                    f"{args.mode} ids mismatch got={ids.tolist()} "
                    f"expected={expected.tolist()}"
                )
            return np.asarray(ids,np.int32), expected, logits_tail

        ids,expected,logits_tail=run_once()
        ids_repeat=None
        if candidate:
            ids_repeat,expected2,_=run_once()
            if not np.array_equal(ids_repeat,expected2):
                raise RuntimeError("candidate deterministic replay diverged")
            # Recreate the state corresponding to the primary capture.
            ids,expected,logits_tail=run_once()

        arrays=capture_arrays(
            rt,logits_tail,CTX+horizon,ids,ids_repeat=ids_repeat
        )
        out_npz.parent.mkdir(parents=True,exist_ok=True)
        np.savez(out_npz,**arrays)

        payload.update({
          "status":"measured","horizon":horizon,
          "candidate":candidate,"capture_info":cap,
          "expected":expected.tolist(),"ids":ids.tolist(),
          "ids_repeat":(
              None if ids_repeat is None else ids_repeat.tolist()
          ),
          "array_count":len(arrays),"npz":str(out_npz),
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({
          "status":"technical_failure",
          "error":{
            "type":type(exc).__name__,"message":str(exc),
            "traceback":traceback.format_exc(),
          },
          "completed_utc":utc_now(),
        })
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass

    out_json.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out_json,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),"mode":args.mode,
      "horizon":payload.get("horizon"),"ids":payload.get("ids"),
      "ids_repeat":payload.get("ids_repeat"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(out_json),
    },indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
