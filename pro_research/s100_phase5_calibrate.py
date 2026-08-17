"""Calibration scan, static K-map construction and validation selection."""
from __future__ import annotations
import gc,json,traceback
from pathlib import Path
from common import REPO,utc_now,write_json_atomic
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from s100_phase5_runtime import build_phase5_runtime,recapture,record
from s100_phase5_quality import evaluate

OUT=REPO/'pro_research'/'results'/'S100_PHASE5_CALIBRATION.json'
CANDS=REPO/'pro_research'/'results'/'S100_PHASE5_CANDIDATES.json'
BUDGETS=(4,8,12,16,20,24); ALPHAS=(0.0001,0.0003,0.001,0.003)

def cost(cur,base):
    return max(0.0,cur['mean_coarse_kl']-base['mean_coarse_kl']) + .25*max(0.0,cur['mean_ce_delta']-base['mean_ce_delta']) + .20*max(0.0,base['top1_agreement']-cur['top1_agreement'])

def main():
    payload={'kind':'s100_phase5_calibration','status':'started','started_utc':utc_now(),'budgets':BUDGETS,'alphas':ALPHAS}
    try:
        payload['gpu_idle_preflight']=_require_gpu_idle_wddm(); import cupy as cp
        b=build_phase5_runtime(); rt=b.rt; moe=[int(i) for i in rt.moe_layers]
        base=evaluate(b,'calibration',False); bsum=base['summary']; scans=[]
        for pos,layer in enumerate(moe,1):
            row={'layer':layer}
            for kval in (5,4):
                b.config['layer_k'].clear(); b.config['layer_k'][layer]=kval; b.config['alpha']=0.0; recapture(b)
                ev=evaluate(b,'calibration',False); row[f'k{kval}']=ev; row[f'k{kval}_cost']=cost(ev['summary'],bsum)
            scans.append(row); print(f'layer scan {pos:02d}/{len(moe)}: layer {layer}',flush=True)
        b.config['layer_k'].clear(); b.config['alpha']=0.0; recapture(b)

        # Precedence-constrained greedy expert drops.
        state={i:6 for i in moe}; actions=[]; portfolios={}
        by={x['layer']:x for x in scans}
        for step in range(1,max(BUDGETS)+1):
            eligible=[]
            for i in moe:
                if state[i]==6: eligible.append((float(by[i]['k5_cost']),i,5))
                elif state[i]==5:
                    inc=max(0.0,float(by[i]['k4_cost'])-float(by[i]['k5_cost']))
                    eligible.append((inc,i,4))
            c,i,newk=min(eligible,key=lambda x:(x[0],x[1],x[2])); state[i]=newk; actions.append({'drop_index':step,'layer':i,'new_k':newk,'incremental_cost':c})
            if step in BUDGETS: portfolios[str(step)]={str(i):int(k) for i,k in state.items() if k!=6}

        validation_portfolios={}
        for budget in BUDGETS:
            m={int(k):int(v) for k,v in portfolios[str(budget)].items()}; b.config['layer_k'].clear(); b.config['layer_k'].update(m); b.config['alpha']=0.0; recapture(b)
            ev=evaluate(b,'validation',False); validation_portfolios[str(budget)]=ev; print(f'validation K budget {budget}: strict={ev["strict_pass"]}',flush=True)
        validation_thresholds={}
        b.config['layer_k'].clear()
        for alpha in ALPHAS:
            b.config['alpha']=alpha; recapture(b); ev=evaluate(b,'validation',False); validation_thresholds[str(alpha)]=ev; print(f'validation alpha {alpha}: strict={ev["strict_pass"]}',flush=True)

        passed_b=[x for x in BUDGETS if validation_portfolios[str(x)]['strict_pass']]
        passed_a=[x for x in ALPHAS if validation_thresholds[str(x)]['strict_pass']]
        sel_b=max(passed_b) if passed_b else None; sel_a=max(passed_a) if passed_a else None
        selected={}
        if sel_b is not None: selected['selective_k']={'layer_k':portfolios[str(sel_b)],'alpha':0.0,'source':f'budget_{sel_b}'}
        if sel_a is not None: selected['threshold']={'layer_k':{},'alpha':sel_a,'source':f'alpha_{sel_a}'}
        if sel_b is not None and sel_a is not None: selected['combined']={'layer_k':portfolios[str(sel_b)],'alpha':sel_a,'source':f'budget_{sel_b}+alpha_{sel_a}'}
        candidates={'kind':'s100_phase5_candidates','created_utc':utc_now(),'selection_uses_only':'calibration+validation; heldout untouched','selected_budget':sel_b,'selected_alpha':sel_a,'selected':selected,'all_portfolios':portfolios,'validation_portfolios':validation_portfolios,'validation_thresholds':validation_thresholds}
        write_json_atomic(CANDS,candidates,archive=True)
        payload.update({'status':'calibrated','runtime':record(b),'moe_layers':moe,'qfast_calibration':base,'single_layer_scans':scans,'greedy_actions':actions,'portfolios':portfolios,'validation_portfolios':validation_portfolios,'validation_thresholds':validation_thresholds,'selected_budget':sel_b,'selected_alpha':sel_a,'selected_candidates':selected,'candidates_path':str(CANDS.relative_to(REPO)),'completed_utc':utc_now()})
        b.restore_combined(); b.restore_selective(); del rt,b; cp.get_default_memory_pool().free_all_blocks(); gc.collect()
    except Exception as e:
        payload.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},'completed_utc':utc_now()})
    write_json_atomic(OUT,payload,archive=True); print(json.dumps({'status':payload.get('status'),'selected_budget':payload.get('selected_budget'),'selected_alpha':payload.get('selected_alpha'),'selected_candidates':payload.get('selected_candidates'),'error':(payload.get('error') or {}).get('message'),'output':str(OUT)},indent=2))
    return 2 if payload.get('status')=='technical_failure' else 0
if __name__=='__main__': raise SystemExit(main())
