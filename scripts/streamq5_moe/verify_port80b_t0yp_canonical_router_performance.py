from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];REPORTS=ROOT/'reports/streamq5_moe';RUN=ROOT/'reports/runs/streamq5_moe/port80b_t0yp_canonical_router_performance';LOCK=REPORTS/'port80b_t0yp_canonical_router_performance_lock.json';RESULT=RUN/'t0yp_canonical_router_performance.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def preflight():
 l=json.loads(LOCK.read_text());c={k:sha(ROOT/l[k])==l['source_sha256'][k] for k in ('runner','verifier','prereg')};c['outputs_closed']=not RESULT.exists();o={'kind':'t0yp_preflight','pass':all(c.values()),'checks':c};print(json.dumps(o,indent=2));return 0 if o['pass'] else 1
def p(xs,q):return float(np.percentile(np.asarray(xs),q,method='linear'))
def verify():
 j=json.loads(RESULT.read_text());checks={'correctness':j['correctness_replay_pass'] is True};
 for name in ('validation','test'):
  d=j.get(name)
  if d is None:continue
  raw=d['raw'];r=[x['reference_ms'] for x in raw];c=[x['candidate_ms'] for x in raw];checks[name+'_samples']=len(raw)==(80 if name=='validation' else 240) and all(x['order']==(['reference','candidate'] if i%2==0 else ['candidate','reference']) for i,x in enumerate(raw));checks[name+'_stats']=abs(d['ratio_p50']-p(c,50)/p(r,50))<1e-12 and abs(d['ratio_p95']-p(c,95)/p(r,95))<1e-12
 opened=j['validation']['ratio_p50']<=4 and j['validation']['ratio_p95']<=5;passed=bool(opened and j.get('test') and j['test']['ratio_p50']<=2 and j['test']['ratio_p95']<=2.5);checks['gates']=opened==j['test_opened'] and passed==j['overall_pass'];o={'kind':'t0yp_independent_verification','verification_pass':all(checks.values()),'scientific_pass':passed,'checks':checks,'claim_boundary':j['claim_boundary']};(REPORTS/'port80b_t0yp_canonical_router_performance_independent_verification.json').write_text(json.dumps(o,indent=2));print(json.dumps(o,indent=2));return 0 if all(checks.values()) else 1
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=('preflight','verify'),required=True);x=a.parse_args();raise SystemExit(preflight() if x.phase=='preflight' else verify())
