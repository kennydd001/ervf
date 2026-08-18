from __future__ import annotations
import argparse,json,traceback
from common import REPO,percentiles,write_json_atomic
from diag_component_marginals_graph import _run,_prefill,_reset_exact_state
from graph_e1f22 import _load_prompt_set
from s100_phase10b_runtime import build
R=REPO/"pro_research"/"results"/"s100_phase10b"
def main():
 a=argparse.ArgumentParser();a.add_argument("--variant",required=True);a.add_argument("--mode",choices=("smoke","full"),required=True);a.add_argument("--role",choices=("base_a","cand_a","cand_b","base_b"),required=True);q=a.parse_args()
 R.mkdir(parents=True,exist_ok=True);o=R/f"P10B_{q.variant}_{q.mode}_{q.role.upper()}.json";p={"status":"started","variant":q.variant,"mode":q.mode,"role":q.role}
 try:
  import cupy as cp
  b=build(q.variant if q.role.startswith("cand") else None);rt=b.rt;ps,_e,n,_c=_load_prompt_set(q.mode);n=32 if q.mode=="smoke" else max(256,int(n))
  _reset_exact_state(rt);_prefill(rt,ps[0]["prompt_ids"]);[rt.step_graph(None) for _ in range(96)];rt._graph_stream.synchronize()
  raw=[];ids={}
  for z in ps:x,ms=_run(rt,z["prompt_ids"],n);ids[z["prompt"]]=[int(v) for v in x];raw.extend(float(v) for v in ms)
  rt._graph_stream.synchronize();p.update({"status":"measured","timing":percentiles(raw),"ids":ids,"finite":bool(cp.isfinite(rt.logits).all().item())})
 except Exception as e:p.update({"status":"technical_failure","error":{"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}})
 write_json_atomic(o,p,archive=True);print(json.dumps({"status":p.get("status"),"timing":p.get("timing"),"error":(p.get("error") or {}).get("message")},indent=2));return 0 if p.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
