from __future__ import annotations

import argparse
import json
import traceback

import numpy as np

from common import utc_now,write_json_atomic
from s100_phase21_common import load_trace,prefill_to,expected_for_block,release
from s100_phase22_common import make_v6,selected_head_mode
from s100_phase23_common import GraphH4VerifierGrouped
from s100_phase24_common import RESULTS,selected_config,make_synth

CTX=1024

def capture_arrays(rt,v,pos,ids,ids_repeat=None):
    import cupy as cp
    arrays={
      "ids":np.asarray(ids,np.int32),
      "ids_repeat":np.asarray(
        ids_repeat if ids_repeat is not None else ids,np.int32
      ),
      "logits":cp.asnumpy(v.logits).astype(np.float32,copy=True),
    }
    for k,x in rt.ssm.items():
        arrays[f"ssm_{int(k)}"]=cp.asnumpy(x).astype(np.float32,copy=True)
    for k,x in rt.conv.items():
        arrays[f"conv_{int(k)}"]=cp.asnumpy(x).astype(np.float32,copy=True)
    nk,mc,hd=int(rt.n_kv),int(rt.max_ctx),int(rt.head_dim)
    for li in rt.attn_layers:
        i=int(li)
        arrays[f"k_{i}"]=cp.asnumpy(
          rt.kc[i].reshape(nk,mc,hd)[:,:pos,:]
        ).astype(np.float32,copy=True)
        arrays[f"v_{i}"]=cp.asnumpy(
          rt.vc[i].reshape(nk,mc,hd)[:,:pos,:]
        ).astype(np.float32,copy=True)
    return arrays

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=("baseline","selected"),required=True)
    args=ap.parse_args()
    out_json=RESULTS/f"S100_PHASE24_STATE_{args.mode.upper()}.json"
    out_npz=RESULTS/f"S100_PHASE24_STATE_{args.mode.upper()}.npz"
    payload={"kind":"s100_phase24_state_capture","status":"started",
      "mode":args.mode,"context":CTX,"started_utc":utc_now()}
    rt=None
    try:
        tr=load_trace();tokens=tr["tokens"]
        if args.mode=="baseline":
            rt,keep=make_v6(CTX)
            g=GraphH4VerifierGrouped(rt,selected_head_mode())
            config={"baseline":True}
            g.setup_graph()
            prefill_to(rt,tokens,CTX)
            g.set_pos_from_host()
        else:
            cfg=selected_config()
            if cfg is None:
                raise RuntimeError("selected arm is baseline")
            rt,g,keep=make_synth(CTX,cfg)
            config=cfg.as_dict()
            g.setup_graph()
            prefill_to(rt,tokens,CTX)
            g.prepare_after_prefill()

        draft,expected=expected_for_block(tokens,CTX)
        ids=g.launch(draft.tolist())
        if not np.array_equal(ids,expected):
            raise RuntimeError(f"{args.mode} ids diverged")

        ids_repeat=None
        if args.mode=="selected":
            rt.reset()
            prefill_to(rt,tokens,CTX)
            g.prepare_after_prefill()
            ids_repeat=g.launch(draft.tolist())
            if not np.array_equal(ids_repeat,expected):
                raise RuntimeError("selected deterministic repeat diverged")
            # Recreate first-state capture after deterministic replay.
            rt.reset()
            prefill_to(rt,tokens,CTX)
            g.prepare_after_prefill()
            ids=g.launch(draft.tolist())

        arrays=capture_arrays(rt,g.v,CTX+4,ids,ids_repeat)
        out_npz.parent.mkdir(parents=True,exist_ok=True)
        np.savez(out_npz,**arrays)
        payload.update({
          "status":"measured","config":config,
          "expected":expected.tolist(),"ids":np.asarray(ids).tolist(),
          "ids_repeat":(
            None if ids_repeat is None else np.asarray(ids_repeat).tolist()
          ),
          "npz":str(out_npz),
          "array_count":len(arrays),
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({"status":"technical_failure",
          "error":{"type":type(exc).__name__,"message":str(exc),
                   "traceback":traceback.format_exc()},
          "completed_utc":utc_now()})
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass

    out_json.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out_json,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),"mode":args.mode,
      "config":payload.get("config"),"ids":payload.get("ids"),
      "ids_repeat":payload.get("ids_repeat"),
      "array_count":payload.get("array_count"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(out_json)},indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
