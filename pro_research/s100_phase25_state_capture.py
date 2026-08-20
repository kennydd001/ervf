from __future__ import annotations

import argparse
import json
import traceback
import numpy as np

from common import utc_now,write_json_atomic
from s100_phase21_common import load_trace,prefill_to,release
from s100_phase24_common import selected_config,make_synth
from s100_phase25_common import RESULTS,VARIANTS,make_h8,expected_for_h8,phase24_gate

CTX=1024

def capture_arrays(rt,logits8,pos,ids,ids_repeat=None):
    import cupy as cp
    arrays={"ids":np.asarray(ids,np.int32),"ids_repeat":np.asarray(ids if ids_repeat is None else ids_repeat,np.int32),
            "logits":np.asarray(logits8,np.float32).copy()}
    for k,x in rt.ssm.items():arrays[f"ssm_{int(k)}"]=cp.asnumpy(x).astype(np.float32,copy=True)
    for k,x in rt.conv.items():arrays[f"conv_{int(k)}"]=cp.asnumpy(x).astype(np.float32,copy=True)
    nk,mc,hd=int(rt.n_kv),int(rt.max_ctx),int(rt.head_dim)
    for li in rt.attn_layers:
        i=int(li);arrays[f"k_{i}"]=cp.asnumpy(rt.kc[i].reshape(nk,mc,hd)[:,:pos,:]).astype(np.float32,copy=True)
        arrays[f"v_{i}"]=cp.asnumpy(rt.vc[i].reshape(nk,mc,hd)[:,:pos,:]).astype(np.float32,copy=True)
    return arrays

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=("parent","candidate"),required=True)
    ap.add_argument("--variant",choices=tuple(VARIANTS),default="split4_route");args=ap.parse_args()
    label="PARENT" if args.mode=="parent" else args.variant.upper()
    out_json=RESULTS/f"S100_PHASE25_STATE_{label}.json";out_npz=RESULTS/f"S100_PHASE25_STATE_{label}.npz"
    payload={"kind":"s100_phase25_state_capture","status":"started","mode":args.mode,
      "variant":args.variant,"context":CTX,"started_utc":utc_now()};rt=None
    try:
        import cupy as cp
        tr=load_trace();tokens=tr["tokens"];draft,expected=expected_for_h8(tokens,CTX)
        if args.mode=="parent":
            cfg,_,_,_=phase24_gate();rt,g,keep=make_synth(CTX,cfg);g.setup_graph();prefill_to(rt,tokens,CTX);g.prepare_after_prefill()
            a=g.launch(draft[:4].tolist());la=cp.asnumpy(g.v.logits).astype(np.float32,copy=True)
            b=g.launch(draft[4:].tolist());lb=cp.asnumpy(g.v.logits).astype(np.float32,copy=True)
            ids=np.concatenate([a,b]);logits8=np.concatenate([la,lb],axis=0);ids_repeat=None
            config={"phase24_parent":True,"phase24_config":cfg.as_dict()}
        else:
            rt,g,keep=make_h8(CTX,args.variant);g.setup_graph();prefill_to(rt,tokens,CTX);g.prepare_after_prefill()
            ids=g.launch(draft.tolist());logits8=cp.asnumpy(g.v.logits).astype(np.float32,copy=True)
            if not np.array_equal(ids,expected):raise RuntimeError(f"{args.variant} first ids diverged")
            rt.reset();prefill_to(rt,tokens,CTX);g.prepare_after_prefill();ids_repeat=g.launch(draft.tolist())
            if not np.array_equal(ids_repeat,expected):raise RuntimeError(f"{args.variant} repeat ids diverged")
            rt.reset();prefill_to(rt,tokens,CTX);g.prepare_after_prefill();ids=g.launch(draft.tolist())
            logits8=cp.asnumpy(g.v.logits).astype(np.float32,copy=True)
            config={"variant":args.variant,"up_mode":g.up_mode,"down_mode":g.down_mode,"phase24_config":g.config.as_dict()}
        if not np.array_equal(ids,expected):raise RuntimeError(f"{args.mode}/{args.variant} ids diverged")
        arrays=capture_arrays(rt,logits8,CTX+8,ids,ids_repeat)
        out_npz.parent.mkdir(parents=True,exist_ok=True);np.savez(out_npz,**arrays)
        payload.update({"status":"measured","config":config,"expected":expected.tolist(),"ids":np.asarray(ids).tolist(),
          "ids_repeat":None if ids_repeat is None else np.asarray(ids_repeat).tolist(),"npz":str(out_npz),
          "array_count":len(arrays),"completed_utc":utc_now()})
    except Exception as exc:
        payload.update({"status":"technical_failure","error":{"type":type(exc).__name__,"message":str(exc),
          "traceback":traceback.format_exc()},"completed_utc":utc_now()})
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass
    out_json.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(out_json,payload,archive=True)
    print(json.dumps({"status":payload.get("status"),"mode":args.mode,"variant":args.variant,
      "ids":payload.get("ids"),"ids_repeat":payload.get("ids_repeat"),"array_count":payload.get("array_count"),
      "error":(payload.get("error") or {}).get("message"),"output":str(out_json)},indent=2))
    return 0 if payload.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
