import json
from common import REPO,write_json_atomic
R=REPO/"pro_research"/"results"/"s100_phase10b";s=json.loads((R/"S100_PHASE10B_STREAM.json").read_text());rows=[]
for v in s["selected_for_integration"]:
 p=R/f"P10B_COMPARE_{v}_full.json"
 if p.exists():rows.append(json.loads(p.read_text()))
good=[x for x in rows if x.get("status")=="pass"];best=min(good,key=lambda x:x["cand_mid_ms"]) if good else None
out={"kind":"s100_phase10b_summary","stream_selected":s["selected_for_integration"],"integrated":rows,"selected":best,"MAMBA_ERVF2_PROMOTE":bool(best),"s100_single_achieved":bool(best and best["cand_mid_ms"]<=10)}
write_json_atomic(R/"S100_PHASE10B_SUMMARY.json",out,archive=True)
text=f"S100 PHASE 10B MAMBA BANDWIDTH\nMAMBA_ERVF2_PROMOTE: {bool(best)}\nSelected: {best}\nS100 SINGLE ACHIEVED: {out['s100_single_achieved']}\n"
(R/"S100_PHASE10B_SUMMARY.txt").write_text(text,encoding="utf-8");print(text)
