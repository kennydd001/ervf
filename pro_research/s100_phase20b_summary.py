from __future__ import annotations
import json
import numpy as np
from common import REPO,utc_now,write_json_atomic

R=REPO/'pro_research'/'results'/'s100_phase20b'
OUT=R/'S100_PHASE20B_SUMMARY.json'
CONTEXTS=(128,1024,4096)

def load(p):
    try:return json.loads(p.read_text(encoding='utf-8'))
    except Exception:return {}

def main():
    pf=load(R/'S100_PHASE20B_MOE_PREFLIGHT.json')
    sc=load(R/'S100_PHASE20B_STATE_CHECK.json')
    ctxout={};all_green=bool(pf.get('GROUPED_MOE_H4_GREEN') and sc.get('FULL_H4_STATE_PARITY_GREEN'))
    for ctx in CONTEXTS:
        arms={}
        for mode,tags in (('baseline',('B1','B2')),('candidate',('C1','C2'))):
            vals=[];ids=[];raw=[]
            for tag in tags:
                d=load(R/f'S100_PHASE20B_{mode.upper()}_CTX{ctx}_{tag}.json')
                raw.append({'tag':tag,'status':d.get('status'),'summary':d.get('summary')})
                if d.get('status')=='measured':
                    vals.extend(float(x['block_ms']) for x in d.get('rows',[]))
                    ids.append(bool((d.get('summary') or {}).get('all_ids_match')))
                else:ids.append(False)
            if vals:
                a=np.asarray(vals,np.float64)
                arms[mode]={'samples':len(a),'median_block_ms':float(np.median(a)),
                            'p10_block_ms':float(np.percentile(a,10)),'p90_block_ms':float(np.percentile(a,90)),
                            'all_ids_match':all(ids),'raw_arms':raw}
            else:arms[mode]={'samples':0,'median_block_ms':None,'all_ids_match':False,'raw_arms':raw}
        b=arms['baseline']['median_block_ms'];c=arms['candidate']['median_block_ms']
        green=bool(c is not None and b is not None and arms['candidate']['all_ids_match'] and arms['baseline']['all_ids_match'])
        all_green=all_green and green
        ctxout[str(ctx)]={'baseline':arms['baseline'],'candidate':arms['candidate'],
                          'speedup':(b/c if b and c else None),
                          'candidate_ms_per_useful_token':(c/4.0 if c else None),
                          'candidate_target_only_tok_s':(4000.0/c if c else None),
                          'correctness_green':green,
                          'under_40ms':bool(green and c<=40.0) if c else False,
                          'under_32ms':bool(green and c<=32.0) if c else False}

    census=load(R/'S100_PHASE20B_CENSUS_CTX1024_CENSUS.json')
    rc=census.get('route_census') or []
    route_summary=None
    if rc:
        uniques=[x['unique_experts'] for x in rc];repeats=[x['repeat_rate'] for x in rc]
        route_summary={'layers':len(rc),'median_unique_experts':float(np.median(uniques)),
                       'min_unique_experts':int(np.min(uniques)),'max_unique_experts':int(np.max(uniques)),
                       'median_repeat_rate':float(np.median(repeats)),
                       'total_cache_hits':int(sum(x['cache_hits'] for x in rc)),
                       'total_cache_misses':int(sum(x['cache_misses'] for x in rc)),
                       'total_up_bytes_loaded':int(sum(x['up_bytes_loaded'] for x in rc)),
                       'total_down_sparse_bytes_loaded':int(sum(x['down_sparse_bytes_loaded'] for x in rc))}

    under40=bool(all_green and all(ctxout[str(c)]['under_40ms'] for c in CONTEXTS))
    under32=bool(all_green and all(ctxout[str(c)]['under_32ms'] for c in CONTEXTS))
    if under32:route='OPEN_PHASE20C_DSPARK_MTP_DFLASH_SHOOTOUT'
    elif under40:route='TARGET_S100_CEILING_OPEN_OPTIMIZE_BEFORE_DRAFTER'
    elif all_green:route='FULL_VERIFIER_CORRECT_BUT_TOO_SLOW_PROFILE_MOE_ATTENTION_HEAD'
    else:route='REPAIR_FULL_VERIFIER_CORRECTNESS_OR_INCOMPLETE_EVIDENCE'
    out={'kind':'s100_phase20b_summary','created_utc':utc_now(),'GROUPED_MOE_H4_GREEN':pf.get('GROUPED_MOE_H4_GREEN'),
         'FULL_H4_STATE_PARITY_GREEN':sc.get('FULL_H4_STATE_PARITY_GREEN'),
         'contexts':ctxout,'route_census_summary':route_summary,'FULL_VERIFIER_CORRECTNESS_GREEN':all_green,
         'TARGET_H4_40MS_OPEN':under40,'DRAFTER_SHOOTOUT_OPEN':under32,
         'NEXT_ROUTE':route,'S100_SINGLE_ACHIEVED':False,
         'claim_boundary':'perfect-draft target-only verifier; drafter excluded'}
    R.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,out,archive=True)
    text=('S100 PHASE 20B — FULL H4 PERFECT-DRAFT VERIFIER\n'
          f"Grouped MoE H4 green: {out['GROUPED_MOE_H4_GREEN']}\n"
          f"Full H4 state parity green: {out['FULL_H4_STATE_PARITY_GREEN']}\n"
          f"Full verifier correctness green: {all_green}\n")
    for c in CONTEXTS:
        x=ctxout[str(c)];text+=(f"ctx {c}: baseline={x['baseline']['median_block_ms']} ms  candidate={x['candidate']['median_block_ms']} ms  "
            f"ms/useful={x['candidate_ms_per_useful_token']}  target_tok_s={x['candidate_target_only_tok_s']}\n")
    text+=(f"TARGET_H4_40MS_OPEN: {under40}\nDRAFTER_SHOOTOUT_OPEN: {under32}\nNEXT_ROUTE: {route}\nS100 SINGLE ACHIEVED: False\n")
    (R/'S100_PHASE20B_SUMMARY.txt').write_text(text,encoding='utf-8')
    report=REPO/'reports'/'S100_PHASE20B_RUN_REPORT.md'
    lines=['# S100 Phase 20B — Full H=4 Perfect-Draft Verifier','',
           'Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`','',
           f"Grouped MoE H4 green: **{out['GROUPED_MOE_H4_GREEN']}**",
           f"Full H4 state parity green: **{out['FULL_H4_STATE_PARITY_GREEN']}**",
           f"Full verifier correctness green: **{all_green}**",'']
    lines += ['| context | baseline H4 ms | candidate H4 ms | speedup | ms/useful token | target-only tok/s |',
              '|---:|---:|---:|---:|---:|---:|']
    for c in CONTEXTS:
        x=ctxout[str(c)]
        lines.append(f"| {c} | {x['baseline']['median_block_ms']} | {x['candidate']['median_block_ms']} | {x['speedup']} | {x['candidate_ms_per_useful_token']} | {x['candidate_target_only_tok_s']} |")
    lines += ['',f"`TARGET_H4_40MS_OPEN = {under40}`",f"`DRAFTER_SHOOTOUT_OPEN = {under32}`",
              f"`NEXT_ROUTE = {route}`",'','`S100_SINGLE_ACHIEVED = False`','',
              'Phase20B is perfect-draft target-only timing. Drafter generation, acceptance loss and rejection recovery are not included.']
    if route_summary:
        lines += ['', '## H=4 route-union census','', '```json', json.dumps(route_summary,indent=2), '```']
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(text);return 0
if __name__=='__main__':raise SystemExit(main())
