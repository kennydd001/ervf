#!/usr/bin/env python3
"""Standalone tokenizer-only canonical prompt/disjointness verifier."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];GEN=ROOT/'scripts/streamq5_moe/generate_port80b_t0q5r1_prompts.py';PL=ROOT/'reports/streamq5_moe/port80b_t0q5r5_prompt_lock.json';OLD=ROOT/'reports/streamq5_moe/port80b_t0r12_prompt_lock.json'
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def main():
 if not PL.exists():print(json.dumps({'kind':'t0q5r5_prompt_verification','pass':False,'blocked':'prompt_lock_absent'}));return 3
 q=importlib.util.spec_from_file_location('gen',GEN);g=importlib.util.module_from_spec(q);sys.modules['gen']=g;q.loader.exec_module(g);new=json.loads(PL.read_text());old=json.loads(OLD.read_text());checks={'canonical_replay':canon(new)==canon(g.generate()),'four_exact16':len(new['prompts'])==4 and all(len(x['token_ids'])==16 for x in new['prompts'])};pairs=[]
 for i,x in enumerate(new['prompts']):
  for y in old['prompts']+new['prompts'][i+1:]:pairs.append(x['utf8_text']!=y['utf8_text'] and x['token_ids']!=y['token_ids'] and x['token_ids_le_u32_sha256']!=y['token_ids_le_u32_sha256'])
 checks['all_pair_disjoint']=len(pairs)==22 and all(pairs);out={'kind':'t0q5r5_prompt_verification','pass':all(checks.values()),'checks':checks,'prompt_lock_sha256':hashlib.sha256(PL.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(GEN.read_bytes()).hexdigest()};print(json.dumps(out,indent=2));return 0 if out['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
