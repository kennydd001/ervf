from __future__ import annotations

import argparse
import json
import traceback
import numpy as np

from common import utc_now,write_json_atomic
from s100_phase21_common import load_trace,prefill_to,expected_for_block,release
from s100_phase24_common import make_synth
from s100_phase27_common import (
    RESULTS,Variant,phase27_gate,make_candidate,capture_arrays,compare_npz,
)

CTX=1024


def selected_variant():
    d=json.loads(
      (RESULTS/"S100_PHASE27_SELECTION.json").read_text(encoding="utf-8")
    )
    v=d.get("selected_variant") or {}
    return Variant(
      int(v["gather_y"]),int(v["batches"]),bool(v["shared_overlap"])
    )


def capture(mode):
    out_json=RESULTS/f"S100_PHASE27_STATE_{mode.upper()}.json"
    out_npz=RESULTS/f"S100_PHASE27_STATE_{mode.upper()}.npz"
    payload={
      "kind":"s100_phase27_state_capture","status":"started",
      "mode":mode,"context":CTX,"started_utc":utc_now(),
    }
    rt=None
    try:
        cfg,_,_=phase27_gate()
        variant=selected_variant()
        tr=load_trace();tokens=tr["tokens"]

        if mode=="parent":
            rt,g,keep=make_synth(CTX,cfg)
        else:
            rt,g,keep=make_candidate(CTX,variant)

        cap=g.setup_graph()

        def run_once():
            rt.reset()
            prefill_to(rt,tokens,CTX)
            g.prepare_after_prefill()
            draft,expected=expected_for_block(tokens,CTX)
            ids=g.launch(draft.tolist())
            if not np.array_equal(ids,expected):
                raise RuntimeError(
                  f"{mode} ids mismatch got={ids.tolist()} "
                  f"expected={expected.tolist()}"
                )
            return np.asarray(ids,np.int32),expected,g.v.logits

        ids,expected,logits=run_once()
        ids_repeat=None
        if mode=="candidate":
            ids_repeat,expected2,_=run_once()
            if not np.array_equal(ids_repeat,expected2):
                raise RuntimeError("candidate deterministic replay mismatch")
            ids,expected,logits=run_once()

        arrays=capture_arrays(
          rt,logits,CTX+4,ids,ids_repeat=ids_repeat
        )
        out_npz.parent.mkdir(parents=True,exist_ok=True)
        np.savez(out_npz,**arrays)
        payload.update({
          "status":"measured",
          "variant":variant.as_dict() if mode=="candidate" else None,
          "capture_info":cap,
          "ids":ids.tolist(),
          "ids_repeat":None if ids_repeat is None else ids_repeat.tolist(),
          "expected":expected.tolist(),
          "array_count":len(arrays),
          "npz":str(out_npz),
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

    write_json_atomic(out_json,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),"mode":mode,
      "variant":payload.get("variant"),
      "ids":payload.get("ids"),"ids_repeat":payload.get("ids_repeat"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(out_json),
    },indent=2))
    return payload


def adjudicate():
    pm=json.loads(
      (RESULTS/"S100_PHASE27_STATE_PARENT.json").read_text(encoding="utf-8")
    )
    cm=json.loads(
      (RESULTS/"S100_PHASE27_STATE_CANDIDATE.json").read_text(encoding="utf-8")
    )
    if pm.get("status")!="measured" or cm.get("status")!="measured":
        raise RuntimeError("state captures incomplete")

    state,gates=compare_npz(
      RESULTS/"S100_PHASE27_STATE_PARENT.npz",
      RESULTS/"S100_PHASE27_STATE_CANDIDATE.npz",
    )
    out={
      "kind":"s100_phase27_state_check","status":"measured",
      "created_utc":utc_now(),
      "state":state,"gates":gates,
      "parent_ids":pm.get("ids"),
      "candidate_ids":cm.get("ids"),
      "candidate_repeat_ids":cm.get("ids_repeat"),
      "selected_variant":cm.get("variant"),
      "SELECTED_STATE_GREEN":bool(all(gates.values())),
    }
    path=RESULTS/"S100_PHASE27_STATE_CHECK.json"
    write_json_atomic(path,out,archive=True)
    print(json.dumps(out,indent=2))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=("parent","candidate","compare"),required=True)
    args=ap.parse_args()
    if args.mode=="compare":
        out=adjudicate()
        return 0 if out.get("SELECTED_STATE_GREEN") else 2
    out=capture(args.mode)
    return 0 if out.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
