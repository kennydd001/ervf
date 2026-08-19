from __future__ import annotations
import argparse,json,traceback
from common import REPO,write_json_atomic,utc_now
from s100_phase21_common import identity_gate,load_trace,release
from s100_phase22_common import make_v6,GraphH4Verifier,selected_head_mode,timed_graph_blocks
from s100_phase23_common import GraphH4VerifierGrouped,RESULTS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=("parent","grouped"),required=True)
    ap.add_argument("--context",type=int,required=True)
    ap.add_argument("--tag",required=True)
    ap.add_argument("--blocks",type=int,default=16)
    ap.add_argument("--warmup",type=int,default=4)
    args=ap.parse_args()

    out=RESULTS/f"S100_PHASE23_{args.mode.upper()}_CTX{args.context}_{args.tag.upper()}.json"
    payload={"kind":"s100_phase23_measure","status":"started",
      "mode":args.mode,"context":args.context,"tag":args.tag,
      "blocks":args.blocks,"warmup":args.warmup,"started_utc":utc_now(),
      "claim_boundary":"perfect-draft target-only H4 graph timing"}
    rt=None
    try:
        identity_gate();tr=load_trace();tokens=tr["tokens"]
        if args.mode=="grouped":
            st=json.loads((RESULTS/"S100_PHASE23_STATE_CHECK.json").read_text(encoding="utf-8"))
            if not st.get("GPU_GROUPED_CORRECTNESS_GREEN"):
                raise RuntimeError("Phase23 grouped correctness is not green")
        else:
            p22=json.loads(
              (REPO/"pro_research"/"results"/"s100_phase22"/"S100_PHASE22_SUMMARY.json")
              .read_text(encoding="utf-8")
            )
            if not p22.get("GRAPH_CORRECTNESS_GREEN"):
                raise RuntimeError("Phase22 parent graph correctness is not green")

        need=args.context+4*(args.blocks+args.warmup)+1
        if need>len(tokens):
            raise RuntimeError(f"canonical trace too short: need={need}, have={len(tokens)}")

        rt,keep=make_v6(args.context)
        mode=selected_head_mode()
        g=(GraphH4VerifierGrouped(rt,mode)
           if args.mode=="grouped" else GraphH4Verifier(rt,mode))
        cap=g.setup_graph()
        records,summary=timed_graph_blocks(
            rt,g,tokens,args.context,args.blocks,args.warmup
        )
        payload.update({"status":"measured","head_mode":mode,
          "capture_info":cap,"records":records,"summary":summary,
          "correctness_green":bool(summary.get("all_token_exact")),
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
      "status":payload.get("status"),"mode":args.mode,"context":args.context,
      "tag":args.tag,"summary":payload.get("summary"),
      "error":(payload.get("error") or {}).get("message"),"output":str(out)},indent=2))
    return 0 if payload.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
