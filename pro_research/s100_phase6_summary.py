from __future__ import annotations
import json
from common import REPO,utc_now,write_json_atomic
OUT=REPO/'pro_research'/'results'/'S100_PHASE6_SUMMARY.json';TXT=OUT.with_suffix('.txt')
def load(p):return json.loads(p.read_text()) if p.exists() else None
def main():
 bsel=load(REPO/'pro_research'/'results'/'S100_PHASE6_BACKEND_SELECT.json') or {};val=load(REPO/'pro_research'/'results'/'S100_PHASE6_VALIDATION.json') or {};held=load(REPO/'pro_research'/'results'/'S100_PHASE6_HELDOUT.json') or {};rows=[]
 for name,r in held.get('results',{}).items():
  t=load(REPO/'pro_research'/'results'/f'S100_PHASE6_TIMING_COMPARE_{name.upper()}.json');rows.append({'candidate':name,'fidelity':r.get('status'),'summary':r.get('summary'),'timing_status':t.get('status') if t else None,'timing':t.get('summary') if t else None})
 good=[x for x in rows if x['fidelity']=='v18_fidelity_candidate' and x['timing_status']=='fresh_timing_candidate'];fast=min(good,key=lambda x:float(x['timing']['candidate_midpoint_ms'])) if good else None;s100=bool(fast and float(fast['timing']['candidate_midpoint_ms'])<=10.0)
 payload={'kind':'s100_phase6_summary','created_utc':utc_now(),'selected_exact_backend':bsel.get('selected_backend','legacy'),'validation_selected':list((val.get('selected') or {}).keys()),'candidates':rows,'fastest_fidelity_green':fast,'s100_single_achieved':s100}
 write_json_atomic(OUT,payload,archive=True);lines=['S100 PHASE 6 SUMMARY',f'Exact backend: {payload["selected_exact_backend"]}',f'Validation selected: {", ".join(payload["validation_selected"])}','', 'candidate | fidelity | ms | tok/s | top1 | dCE | KL']
 for r in rows:
  s=r.get('summary') or {};t=r.get('timing') or {};lines.append(f'{r["candidate"]} | {r["fidelity"]} | {t.get("candidate_midpoint_ms")} | {t.get("candidate_tok_s")} | {s.get("top1_agreement")} | {s.get("mean_ce_delta")} | {s.get("mean_coarse_kl")}')
 lines+=['',f'FASTEST FIDELITY-GREEN: {fast["candidate"] if fast else None}',f'S100 SINGLE ACHIEVED: {s100}'];TXT.write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines));return 0
if __name__=='__main__':raise SystemExit(main())
