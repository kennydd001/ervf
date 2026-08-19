from __future__ import annotations

import argparse,gc,json,time,traceback
import numpy as np
from common import REPO,require_model_dir,utc_now,write_json_atomic
from s100_phase20b_verifier import FullH4Verifier,H

R=REPO/'pro_research'/'results'/'s100_phase20b'
TRACE=R/'S100_PHASE20B_CANONICAL_TRACE.json'


def make_rt():
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    rt=LightningRuntime(require_model_dir(),contexts_max=4352,embed_on_host=True,fp8_kv=False,verbose=False)
    rt.load_routed_bank();rt.enable_cache(48);rt.deterministic_accum=True
    return rt


def prefill(rt,tokens,n):
    rt.reset();nxt=None
    for j,t in enumerate(tokens[:n]):
        nxt=int(rt.step(int(t)))
        if (j+1)%512==0:print(f'prefill {j+1}/{n}',flush=True)
    return int(nxt)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('baseline','candidate','census'),required=True)
    ap.add_argument('--context',type=int,required=True)
    ap.add_argument('--tag',default='X')
    ap.add_argument('--blocks',type=int,default=12)
    args=ap.parse_args()
    out=R/f'S100_PHASE20B_{args.mode.upper()}_CTX{args.context}_{args.tag}.json'
    payload={'kind':'s100_phase20b_measure','status':'started','mode':args.mode,'context':args.context,
             'tag':args.tag,'blocks':args.blocks,'H':H,'started_utc':utc_now(),
             'claim_boundary':'single-stream perfect-draft target verifier'}
    try:
        import cupy as cp
        trace=json.loads(TRACE.read_text(encoding='utf-8'))
        if trace.get('status')!='measured':raise RuntimeError('canonical trace not measured')
        tokens=[int(x) for x in trace['tokens']]
        need=args.context+args.blocks*H+1
        if len(tokens)<need:raise RuntimeError(f'trace too short {len(tokens)} < {need}')
        rt=make_rt();seed=prefill(rt,tokens,args.context)
        expected_seed=tokens[args.context]
        seed_ok=(seed==expected_seed)
        if not seed_ok:raise RuntimeError(f'prefill seed mismatch {seed} != {expected_seed}')

        verifier=None
        if args.mode in ('candidate','census'):verifier=FullH4Verifier(rt)
        rows=[];all_ids=[];all_census=[]
        pos=args.context
        for b in range(args.blocks):
            draft=np.asarray(tokens[pos:pos+H],np.int32)
            expected=np.asarray(tokens[pos+1:pos+H+1],np.int32)
            # Pre-existing prefix logit predicts draft[0]; checked before timing.
            if args.mode=='baseline':
                cp.cuda.get_current_stream().synchronize();t0=time.perf_counter()
                got=[]
                for token in draft:got.append(int(rt.step(int(token))))
                cp.cuda.get_current_stream().synchronize();ms=(time.perf_counter()-t0)*1000.0
                got=np.asarray(got,np.int32);census=None
            else:
                cp.cuda.get_current_stream().synchronize();t0=time.perf_counter()
                got,census=verifier.block(draft.tolist(),collect_census=(args.mode=='census'))
                cp.cuda.get_current_stream().synchronize();ms=(time.perf_counter()-t0)*1000.0
            ok=bool(np.array_equal(got,expected))
            rows.append({'block':b,'start_pos':pos,'block_ms':ms,'draft':draft.tolist(),
                         'predicted':got.tolist(),'expected':expected.tolist(),'ids_match':ok})
            all_ids.extend(got.tolist())
            if census:all_census.extend(census)
            print(f'20B {args.mode} ctx={args.context} {args.tag} block={b+1}/{args.blocks} ms={ms:.3f} ids={ok}',flush=True)
            pos+=H
            if args.mode=='census':break

        vals=np.asarray([r['block_ms'] for r in rows],np.float64)
        correctness=bool(seed_ok and all(r['ids_match'] for r in rows))
        summary={'samples':len(vals),'median_block_ms':float(np.median(vals)),
                 'p10_block_ms':float(np.percentile(vals,10)),'p90_block_ms':float(np.percentile(vals,90)),
                 'mean_block_ms':float(vals.mean()),'ms_per_useful_token':float(np.median(vals)/H),
                 'target_only_tok_s':float(1000.0*H/np.median(vals)),
                 'all_ids_match':correctness,'all_finite':bool(np.isfinite(vals).all())}
        payload.update({'status':'measured','prefill_seed_ok':seed_ok,'rows':rows,'summary':summary,
                        'committed_ids':all_ids,'route_census':all_census if all_census else None,
                        'completed_utc':utc_now()})
        rt.bank={};rt.cache={};del verifier,rt;gc.collect();cp.get_default_memory_pool().free_all_blocks();cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({'status':'technical_failure','error':{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},'completed_utc':utc_now()})
    R.mkdir(parents=True,exist_ok=True);write_json_atomic(out,payload,archive=True)
    print(json.dumps({'status':payload.get('status'),'mode':args.mode,'context':args.context,'tag':args.tag,
                      'summary':payload.get('summary'),'error':(payload.get('error') or {}).get('message'),'output':str(out)},indent=2))
    return 0 if payload.get('status')=='measured' else 2
if __name__=='__main__':raise SystemExit(main())
