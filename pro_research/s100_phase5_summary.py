from __future__ import annotations
import json
from common import REPO,utc_now,write_json_atomic,write_text_atomic

def main():
 cal=json.loads((REPO/'pro_research'/'results'/'S100_PHASE5_CALIBRATION.json').read_text());held=json.loads((REPO/'pro_research'/'results'/'S100_PHASE5_HELDOUT.json').read_text());rows=[]
 for name,r in held.get('results',{}).items():
  t=REPO/'pro_research'/'results'/f'S100_PHASE5_TIMING_COMPARE_{name.upper()}.json';td=json.loads(t.read_text()) if t.exists() else None;rows.append({'candidate':name,'fidelity_status':r.get('status'),'top1':(r.get('summary') or {}).get('top1_agreement'),'top5':(r.get('summary') or {}).get('target_in_top5'),'dCE':(r.get('summary') or {}).get('mean_ce_delta'),'KL':(r.get('summary') or {}).get('mean_coarse_kl'),'candidate_ms':((td or {}).get('summary') or {}).get('candidate_midpoint_ms'),'tok_s':((td or {}).get('summary') or {}).get('candidate_tok_s'),'saving_vs_qfast_ms':((td or {}).get('summary') or {}).get('saving_vs_qfast_ms'),'timing_status':(td or {}).get('status')})
 green=[x for x in rows if x['fidelity_status']=='v18_fidelity_candidate' and x['timing_status']=='fresh_timing_candidate'];best=min(green,key=lambda x:x['candidate_ms']) if green else None
 payload={'kind':'s100_phase5_summary','created_utc':utc_now(),'phase4_qfast_reference':{'ms':18.75165,'tok_s':53.32864041297699,'full_fidelity_green':True},'selected_budget':cal.get('selected_budget'),'selected_alpha':cal.get('selected_alpha'),'candidates':rows,'fastest_fidelity_green':best,'s100_single_achieved':bool(best and best['candidate_ms']<=10.0)}
 out=REPO/'pro_research'/'results'/'S100_PHASE5_SUMMARY.json';write_json_atomic(out,payload,archive=True)
 lines=['S100 PHASE 5 SUMMARY',f"Selected K-drop budget: {payload['selected_budget']}",f"Selected threshold alpha: {payload['selected_alpha']}",'','candidate | fidelity | ms | tok/s | top1 | dCE | KL']
 for x in rows:lines.append(f"{x['candidate']} | {x['fidelity_status']} | {x['candidate_ms']} | {x['tok_s']} | {x['top1']} | {x['dCE']} | {x['KL']}")
 lines+=['',f"FASTEST FIDELITY-GREEN: {best}",f"S100 SINGLE ACHIEVED: {payload['s100_single_achieved']}"]
 txt=REPO/'pro_research'/'results'/'S100_PHASE5_SUMMARY.txt';write_text_atomic(txt,'\n'.join(lines)+'\n',archive=True);print('\n'.join(lines));return 0
if __name__=='__main__':raise SystemExit(main())
