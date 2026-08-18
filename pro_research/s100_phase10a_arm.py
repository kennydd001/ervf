from __future__ import annotations
import argparse,json,subprocess,traceback
from common import REPO,percentiles,write_json_atomic
from diag_component_marginals_graph import _run,_prefill,_reset_exact_state
from graph_e1f22 import _load_prompt_set
from s100_phase10a_runtime import build
R=REPO/"pro_research"/"results"/"s100_phase10a";P=R/"S100_PHASE10A_PANEL_PROFILE.json"
def smi():
 p=subprocess.run(["nvidia-smi","--query-gpu=memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate","--format=csv,noheader,nounits"],capture_output=True,text=True);return p.stdout.strip()
def main():
 a=argparse.ArgumentParser();a.add_argument("--budget",type=int,required=True);a.add_argument("--mode",choices=("smoke","full"),required=True);a.add_argument("--role",choices=("base_a","cand_a","cand_b","base_b","bad"),required=True);q=a.parse_args()
 R.mkdir(parents=True,exist_ok=True);o=R/f"P10A_{q.budget}_{q.mode}_{q.role.upper()}.json";p={"status":"started","budget":q.budget,"mode":q.mode,"role":q.role}
 try:
  import cupy as cp
  pr=json.loads(P.read_text());s=pr["selections"][str(q.budget)];items=s["items"] if q.role.startswith("cand") or q.role=="bad" else None
  b=build(pr["alpha"],items=items,bad=q.role=="bad");rt=b.rt;ps,_e,n,_c=_load_prompt_set(q.mode);n=32 if q.mode=="smoke" else max(256,int(n))
  _reset_exact_state(rt);_prefill(rt,ps[0]["prompt_ids"]);[rt.step_graph(None) for _ in range(64)];rt._graph_stream.synchronize()
  bef=smi();raw=[];ids={}
  for z in ps:x,ms=_run(rt,z["prompt_ids"],n);ids[z["prompt"]]=[int(v) for v in x];raw.extend(float(v) for v in ms)
  rt._graph_stream.synchronize();aft=smi();p.update({"status":"measured","selection_sha256":s["selection_sha256"],"timing":percentiles(raw),"ids":ids,"finite":bool(cp.isfinite(rt.logits).all().item()),"vram_mib":int(cp.cuda.runtime.memGetInfo()[1]-cp.cuda.runtime.memGetInfo()[0])//(1024**2),"smi_before":bef,"smi_after":aft})
 except Exception as e:p.update({"status":"technical_failure","error":{"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}})
 write_json_atomic(o,p,archive=True);print(json.dumps({"status":p.get("status"),"timing":p.get("timing"),"error":(p.get("error") or {}).get("message")},indent=2));return 0 if p.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
