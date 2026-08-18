from __future__ import annotations
import argparse,json
from common import REPO,write_json_atomic
R=REPO/"pro_research"/"results"/"s100_phase10a"
def ld(b,m,r):
 d=json.loads((R/f"P10A_{b}_{m}_{r}.json").read_text())
 if d.get("status")!="measured":raise RuntimeError(d.get("status"))
 return d
def main():
 a=argparse.ArgumentParser();a.add_argument("--budget",type=int,required=True);a.add_argument("--mode",choices=("smoke","full"),required=True);q=a.parse_args()
 A,C1,C2,B=[ld(q.budget,q.mode,x) for x in ("BASE_A","CAND_A","CAND_B","BASE_B")]
 pa,pb=A["timing"]["p50"],B["timing"]["p50"];pc=(C1["timing"]["p50"]+C2["timing"]["p50"])/2;base=(pa+pb)/2;gain=base-pc
 g={"base_repeat":A["ids"]==B["ids"],"cand_repeat":C1["ids"]==C2["ids"],"cand_equals_base":A["ids"]==C1["ids"],"finite":C1["finite"] and C2["finite"],"base_drift_le_1ms":abs(pa-pb)<=1,"cand_drift_le_1ms":abs(C1["timing"]["p50"]-C2["timing"]["p50"])<=1}
 if q.mode=="smoke":g["bad_diverges"]=A["ids"]!=ld(q.budget,q.mode,"BAD")["ids"]
 else:g["samples_ge_765"]=min(x["timing"]["count"] for x in (A,C1,C2,B))>=765;g["gain_ge_0_15ms"]=gain>=.15
 out={"kind":"s100_phase10a_compare","status":"pass" if all(g.values()) else "fail","budget":q.budget,"mode":q.mode,"base_mid_ms":base,"cand_mid_ms":pc,"gain_ms":gain,"cand_tok_s":1000/pc,"gates":g}
 write_json_atomic(R/f"P10A_COMPARE_{q.budget}_{q.mode}.json",out,archive=True);print(json.dumps(out,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
