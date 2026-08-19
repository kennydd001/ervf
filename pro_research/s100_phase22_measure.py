from __future__ import annotations
import argparse,json,traceback
from common import write_json_atomic,utc_now
from s100_phase21_common import measure_blocks
from s100_phase22_common import (
    RESULTS,identity_gate,load_trace,make_v6,selected_head_mode,
    eager_verifier,GraphH4Verifier,timed_graph_blocks,release,
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=("eager","graph"),required=True)
    ap.add_argument("--context",type=int,required=True)
    ap.add_argument("--tag",required=True)
    ap.add_argument("--blocks",type=int,default=8)
    ap.add_argument("--warmup",type=int,default=1)
    args=ap.parse_args()

    out=RESULTS/f"S100_PHASE22_{args.mode.upper()}_CTX{args.context}_{args.tag.upper()}.json"
    payload={"kind":"s100_phase22_measure","status":"started",
      "mode":args.mode,"context":args.context,"tag":args.tag,
      "blocks":args.blocks,"warmup":args.warmup,"started_utc":utc_now(),
      "claim_boundary":"single-stream perfect-draft target-only H4 timing"}
    rt=None
    try:
        identity_gate()
        state=json.loads(
          (RESULTS/"S100_PHASE22_GRAPH_STATE_CHECK.json").read_text(encoding="utf-8")
        )
        if args.mode=="graph" and not state.get("GRAPH_CORRECTNESS_GREEN"):
            raise RuntimeError("graph correctness gate is not green")
        tr=load_trace();tokens=tr["tokens"];mode=selected_head_mode()
        need=args.context+4*(args.blocks+args.warmup)+1
        if need>len(tokens):
            raise RuntimeError(f"canonical trace too short: need {need}")

        rt,keep=make_v6(args.context)
        cap=None
        if args.mode=="eager":
            v=eager_verifier(rt,mode)
            records,summary=measure_blocks(
              rt,v,tokens,args.context,args.blocks,args.warmup
            )
        else:
            g=GraphH4Verifier(rt,mode)
            cap=g.setup_graph()
            records,summary=timed_graph_blocks(
              rt,g,tokens,args.context,args.blocks,args.warmup
            )

        payload.update({"status":"measured","head_mode":mode,
          "capture_info":cap,"records":records,"summary":summary,
          "correctness_green":bool(summary.get("all_token_exact")),
          "cache_stats":dict(getattr(rt,"cache_stats",{})),
          "completed_utc":utc_now()})
    except Exception as exc:
        payload.update({"status":"technical_failure",
          "error":{"type":type(exc).__name__,"message":str(exc),
                   "traceback":traceback.format_exc()},
          "completed_utc":utc_now()})
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass

    out.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),"mode":args.mode,
      "context":args.context,"tag":args.tag,
      "head_mode":payload.get("head_mode"),
      "summary":payload.get("summary"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(out)},indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":raise SystemExit(main())
