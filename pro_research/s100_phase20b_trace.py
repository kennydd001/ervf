from __future__ import annotations
import gc,json,traceback
from common import REPO,require_model_dir,utc_now,write_json_atomic

OUT=REPO/'pro_research'/'results'/'s100_phase20b'/'S100_PHASE20B_CANONICAL_TRACE.json'
TOTAL=4096+12*4+1
PROMPT='A rigorous explanation of efficient artificial intelligence inference begins with'

def main():
    payload={'kind':'s100_phase20b_canonical_trace','status':'started','target_length':TOTAL,'started_utc':utc_now()}
    try:
        import cupy as cp
        from transformers import AutoTokenizer
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
        model=require_model_dir();tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=True,use_fast=True)
        tokens=[int(x) for x in tok.encode(PROMPT,add_special_tokens=False)]
        rt=LightningRuntime(model,contexts_max=4352,embed_on_host=True,fp8_kv=False,verbose=False)
        rt.load_routed_bank();rt.enable_cache(48);rt.deterministic_accum=True
        rt.reset();nxt=None
        for t in tokens:nxt=int(rt.step(t))
        while len(tokens)<TOTAL:
            tokens.append(int(nxt))
            if len(tokens)<TOTAL:nxt=int(rt.step(int(nxt)))
            if len(tokens)%256==0:print(f'20B trace {len(tokens)}/{TOTAL}',flush=True)
        payload.update({'status':'measured','prompt':PROMPT,'prompt_length':len(tok.encode(PROMPT,add_special_tokens=False)),
                        'tokens':tokens,'token_count':len(tokens),'model_dir':str(model),'completed_utc':utc_now()})
        rt.bank={};rt.cache={};del rt;gc.collect();cp.get_default_memory_pool().free_all_blocks();cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({'status':'technical_failure','error':{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},'completed_utc':utc_now()})
    OUT.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({'status':payload.get('status'),'token_count':payload.get('token_count'),'error':(payload.get('error') or {}).get('message'),'output':str(OUT)},indent=2))
    return 0 if payload.get('status')=='measured' else 2
if __name__=='__main__':raise SystemExit(main())
