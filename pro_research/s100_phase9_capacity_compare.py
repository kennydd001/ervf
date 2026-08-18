from __future__ import annotations
import argparse,json,traceback
from common import REPO,first_divergence,utc_now,write_json_atomic
ROLES=('base_a','cand_a','cand_b','base_b')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',required=True);a=ap.parse_args();od=REPO/'pro_research'/'results'/'s100_phase9';out=od/f'CAP_COMPARE_{a.profile.upper()}.json';p={'status':'started','profile':a.profile,'created_utc':utc_now()}
 try:
  d={}
  for r in ROLES:
   q=od/f'CAP_{a.profile.upper()}_{r.upper()}.json';x=json.loads(q.read_text());
   if x.get('status')!='measured':raise RuntimeError(f'{q}: {x.get("status")}')
   d[r]=x
  v={r:float(d[r]['timing']['p50']) for r in ROLES};base=.5*(v['base_a']+v['base_b']);cand=.5*(v['cand_a']+v['cand_b']);bd=abs(v['base_a']-v['base_b']);cd=abs(v['cand_a']-v['cand_b'])
  def div(x,y):return {k:first_divergence(x[k],y[k]) for k in x}
  gates={'base_repeat':all(x is None for x in div(d['base_a']['ids'],d['base_b']['ids']).values()),'cand_repeat':all(x is None for x in div(d['cand_a']['ids'],d['cand_b']['ids']).values()),'exact_token_parity':all(x is None for x in div(d['base_a']['ids'],d['cand_a']['ids']).values()),'finite':bool(d['cand_a']['finite'] and d['cand_b']['finite']),'base_drift_le1':bd<=1,'cand_drift_le1':cd<=1,'samples':all(int(d[r]['timing']['count'])>=765 for r in ROLES),'vram':all(int(d[r]['vram_mib'])<=7987 for r in ROLES)}
  p.update({'status':'capacity_promote' if all(gates.values()) and base-cand>=.15 else 'capacity_below_gate' if all(gates.values()) else 'measurement_failed','summary':{'base_ms':base,'candidate_ms':cand,'saving_ms':base-cand,'candidate_tok_s':1000/cand,'vram_mib':max(d['cand_a']['vram_mib'],d['cand_b']['vram_mib'])},'gates':gates,'completed_utc':utc_now()})
 except Exception as e:p.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}})
 write_json_atomic(out,p,archive=True);print(json.dumps(p,indent=2));return 2 if p.get('status')=='technical_failure' else 0
if __name__=='__main__':raise SystemExit(main())
