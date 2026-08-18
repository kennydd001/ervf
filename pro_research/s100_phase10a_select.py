import json
from common import REPO,write_json_atomic
R=REPO/"pro_research"/"results"/"s100_phase10a"
rows=[]
for b in (8,16,24,32,40,48):
 p=R/f"P10A_COMPARE_{b}_full.json"
 if p.exists():rows.append(json.loads(p.read_text()))
good=[x for x in rows if x.get("status")=="pass"];best=min(good,key=lambda x:x["cand_mid_ms"]) if good else None
out={"kind":"s100_phase10a_summary","selected":best,"PANEL_CACHE_PROMOTE":bool(best),"s100_single_achieved":bool(best and best["cand_mid_ms"]<=10),"results":rows}
write_json_atomic(R/"S100_PHASE10A_SUMMARY.json",out,archive=True)
text=f"S100 PHASE 10A PANEL CACHE\nPANEL_CACHE_PROMOTE: {bool(best)}\nSelected: {best}\nS100 SINGLE ACHIEVED: {out['s100_single_achieved']}\n"
(R/"S100_PHASE10A_SUMMARY.txt").write_text(text,encoding="utf-8");print(text)
