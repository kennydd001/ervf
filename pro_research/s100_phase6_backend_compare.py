from __future__ import annotations
import argparse,json,traceback
from common import REPO,first_divergence,utc_now,write_json_atomic
ROLES=('base_a','cand_a','cand_b','base_b'); BACKENDS=('ballot_fused','direct','direct_opt')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=BACKENDS,required=True);ap.add_argument('--mode',choices=('smoke','full'),required=True);a=ap.parse_args();out=REPO/'pro_research'/'results'/f'S100_PHASE6_BACKEND_COMPARE_{a.backend.upper()}_{a.mode.upper()}.json';p={'kind':'s100_phase6_backend_compare','status':'started','backend':a.backend,'mode':a.mode,'created_utc':utc_now()}
 try:
  d={}
  for r in ROLES:
   q=REPO/'pro_research'/'results'/f'S100_PHASE6_BACKEND_{a.backend.upper()}_{a.mode.upper()}_{r.upper()}.json';x=json.loads(q.read_text());
   if x.get('status')!='measured':raise RuntimeError(f'{q}: {x.get("status")}')
   d[r]=x
  v={r:float(d[r]['timing']['p50']) for r in ROLES};base=.5*(v['base_a']+v['base_b']);cand=.5*(v['cand_a']+v['cand_b']);bd=abs(v['base_a']-v['base_b']);cd=abs(v['cand_a']-v['cand_b']);saving=base-cand
  def div(x,y):return {k:first_divergence(x[k],y[k]) for k in x}
  bdiv=div(d['base_a']['ids'],d['base_b']['ids']);cdiv=div(d['cand_a']['ids'],d['cand_b']['ids']);xdiv=div(d['base_a']['ids'],d['cand_a']['ids'])
  gates={'G1_base_parity':all(x is None for x in bdiv.values()),'G2_candidate_repeat':all(x is None for x in cdiv.values()),'G3_exact_backend_parity':all(x is None for x in xdiv.values()),'G4_finite':bool(d['cand_a']['finite'] and d['cand_b']['finite']),'M1_base_drift':bd<=1.0,'M2_candidate_drift':cd<=1.0,'M3_samples':True if a.mode=='smoke' else all(int(d[r]['timing']['count'])>=765 for r in ROLES),'M4_vram':all(int(d[r]['vram_mib'])<=7987 for r in ROLES)}
  stable=all(gates.values());status='exact_backend_candidate' if stable and saving>=.15 else 'exact_backend_below_gate' if stable else 'measurement_failed'
  p.update({'status':status,'summary':{'legacy_midpoint_ms':base,'candidate_midpoint_ms':cand,'saving_ms':saving,'candidate_tok_s':1000/cand,'remaining_ms_to_s100':cand-10,'base_drift_ms':bd,'candidate_drift_ms':cd},'divergence':{'base':bdiv,'candidate_repeat':cdiv,'candidate_vs_legacy':xdiv},'gates':gates,'completed_utc':utc_now()})
 except Exception as e:p.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},'completed_utc':utc_now()})
 write_json_atomic(out,p,archive=True);print(json.dumps(p,indent=2));return 2 if p.get('status')=='technical_failure' else 0
if __name__=='__main__':raise SystemExit(main())
