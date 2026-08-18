from __future__ import annotations
import json
from pathlib import Path
from common import REPO,utc_now,write_json_atomic
OD=REPO/'pro_research'/'results'/'s100_phase9'
def load(p):
 try:return json.loads(p.read_text())
 except:return None
def main():
 oracle=load(OD/'S100_PHASE9_CACHE_ORACLE.json') or {};econ=load(OD/'S100_PHASE9_MISS_ECONOMICS.json') or {};caps=[]
 for p in sorted(OD.glob('CAP_COMPARE_*.json')):
  d=load(p)
  if d:caps.append({'profile':d.get('profile'),'status':d.get('status'),'summary':d.get('summary'),'gates':d.get('gates')})
 promoted=[x for x in caps if x['status']=='capacity_promote'];best=min(promoted,key=lambda x:x['summary']['candidate_ms']) if promoted else None
 pf=oracle.get('prefetch',[]);pfbest=min(pf,key=lambda x:x.get('demand_miss_fraction',1)) if pf else None;prefetch_research=bool(pfbest and pfbest.get('prefetch_precision',0)>=.40 and pfbest.get('demand_miss_fraction',1)<=.7*oracle.get('test_current',{}).get('miss_fraction',0))
 payload={'kind':'s100_phase9_summary','created_utc':utc_now(),'frozen_quality_green_parent':{'profile':'QFAST+thr_0003','heldout_green':True,'phase6_candidate_ms':18.6276,'phase6_tok_s':53.68378105606734},'cache_oracle':oracle,'capacity_timings':caps,'best_capacity_promote':best,'miss_economics':econ,'PREFETCH_RESEARCH':prefetch_research,'DIRECTHOST_PROMOTE':bool(econ.get('DIRECTHOST_PROMOTE')),'ARC_MISS_PROMOTE':bool(econ.get('ARC_MISS_PROMOTE')),'s100_single_achieved':False}
 write_json_atomic(OD/'S100_PHASE9_SUMMARY.json',payload,archive=True);lines=['S100 PHASE 9 SUMMARY',f'Best exact capacity promote: {best}',f'DIRECTHOST_PROMOTE: {payload["DIRECTHOST_PROMOTE"]}',f'ARC_MISS_PROMOTE: {payload["ARC_MISS_PROMOTE"]}',f'PREFETCH_RESEARCH: {payload["PREFETCH_RESEARCH"]}',f'Belady current-map miss fraction: {(oracle.get("belady_current_map_test") or {}).get("miss_fraction")}',f'Current test miss fraction: {(oracle.get("test_current") or {}).get("miss_fraction")}',f'S100 SINGLE ACHIEVED: False'];(OD/'S100_PHASE9_SUMMARY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines));return 0
if __name__=='__main__':raise SystemExit(main())
