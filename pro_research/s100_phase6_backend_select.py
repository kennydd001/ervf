from __future__ import annotations
import json
from common import REPO,utc_now,write_json_atomic
OUT=REPO/'pro_research'/'results'/'S100_PHASE6_BACKEND_SELECT.json'
def main():
 rows=[]
 for b in ('ballot_fused','direct','direct_opt'):
  p=REPO/'pro_research'/'results'/f'S100_PHASE6_BACKEND_COMPARE_{b.upper()}_FULL.json'
  if p.exists():
   d=json.loads(p.read_text());rows.append({'backend':b,'status':d.get('status'),'summary':d.get('summary'),'gates':d.get('gates')})
 good=[r for r in rows if r['status']=='exact_backend_candidate']
 selected=min(good,key=lambda r:float(r['summary']['candidate_midpoint_ms']))['backend'] if good else 'legacy'
 payload={'kind':'s100_phase6_backend_select','created_utc':utc_now(),'selected_backend':selected,'minimum_gain_ms':.15,'results':rows}
 write_json_atomic(OUT,payload,archive=True);print(json.dumps(payload,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
