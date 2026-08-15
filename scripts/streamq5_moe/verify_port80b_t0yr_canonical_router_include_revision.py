from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];REPORTS=ROOT/'reports/streamq5_moe';RUN=ROOT/'reports/runs/streamq5_moe/port80b_t0yr_canonical_router';LOCK=REPORTS/'port80b_t0yr_canonical_router_lock.json';RESULT=RUN/'t0yr_canonical_router_result.json';RAW=RUN/'t0yr_canonical_router_raw.safetensors'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def preflight():
 l=json.loads(LOCK.read_text());c={k:sha(ROOT/l[k])==l['source_sha256'][k] for k in ('runner','verifier','prereg')};c['outputs_closed']=not RESULT.exists() and not RAW.exists();o={'kind':'t0yr_preflight','pass':all(c.values()),'checks':c};print(json.dumps(o,indent=2));return 0 if o['pass'] else 1
def verify():
 j=json.loads(RESULT.read_text());
 with safe_open(str(RAW),framework='np') as f:t={k:f.get_tensor(k) for k in f.keys()}
 logits=np.array_equal(t['cpu_logits'].view(np.uint32),t['gpu1_logits'].view(np.uint32));ids=np.array_equal(t['cpu_ids'],t['gpu1_ids']);repeat=np.array_equal(t['gpu1_logits'].view(np.uint32),t['gpu2_logits'].view(np.uint32)) and np.array_equal(t['gpu1_ids'],t['gpu2_ids']);finite=bool(np.isfinite(t['cpu_logits']).all() and np.isfinite(t['gpu1_logits']).all());valid=all(len(set(map(int,row)))==10 and min(row)>=0 and max(row)<512 for row in t['gpu1_ids']);passed=logits and ids and repeat and finite and valid and j['gates']['resource'];c={'raw_sha':sha(RAW)==j['raw_sha256'],'logits':logits==j['gates']['cpu_cuda_logits_bitexact'],'ids':ids==j['gates']['cpu_cuda_ids_exact'],'repeat':repeat==j['gates']['cuda_repeat'],'finite':finite==j['gates']['finite'],'valid':valid==j['gates']['ids_valid'],'verdict':passed==j['overall_pass']};o={'kind':'t0yr_independent_verification','verification_pass':all(c.values()),'scientific_pass':passed,'checks':c,'claim_boundary':'Stored tensor replay; no CUDA rerun.'};(REPORTS/'port80b_t0yr_canonical_router_independent_verification.json').write_text(json.dumps(o,indent=2));print(json.dumps(o,indent=2));return 0 if all(c.values()) else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('preflight','verify'),required=True);a=p.parse_args();raise SystemExit(preflight() if a.phase=='preflight' else verify())
