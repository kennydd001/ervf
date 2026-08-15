#!/usr/bin/env python3
"""Static/source-only PH0-R3 preflight. Never import backends or read payload."""
import ast,hashlib,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REP=ROOT/'reports/streamq5_moe';LOCK=REP/'het_next_l0_ph0r3_runner_lock.json';VLOCK=REP/'het_next_l0_ph0r3_verifier_lock.json';OUT=ROOT/'reports/runs/streamq5_moe/het_next_l0_ph0r3_single_projection'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 l=json.loads(LOCK.read_text());vl=json.loads(VLOCK.read_text());checks=[]
 checks += [not l['execution_open'],l['audit_token']=='PENDING_INDEPENDENT_SOURCE_AUDIT',not OUT.exists(),l['verifier_lock_sha256']==sha(VLOCK)]
 for row in l['files'].values():checks.append(sha(ROOT/row['path'])==row['sha256'])
 for row in vl['files'].values():checks.append(sha(ROOT/row['path'])==row['sha256'])
 texts={k:(ROOT/v['path']).read_text() for k,v in l['files'].items() if v['path'].endswith('.py')};trees={k:ast.parse(v) for k,v in texts.items()}
 forbidden=('transformers','from_pretrained','routing','shared_expert','throughput','persistent_bank');checks.append(all(not any(x in t.lower() for x in forbidden) for t in texts.values()))
 checks += ['cooperative_groups::tiled_partition<8>' in texts['nvidia'],'intel_reqd_sub_group_size(8)' in texts['intel'],'clEnqueueWriteBuffer' not in texts['intel'],'clEnqueueReadBuffer' not in texts['intel'],'unpack_fields' in texts['common'],'expected_digest' not in texts['common']]
 checks += [675840==64+655360+16384+4032,683008==675840+4096+1024+2048,1366016==2*683008,16*32==512]
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'x';p.write_bytes(b'x');checks.append(p.read_bytes()==b'x')
 print(json.dumps({'kind':'ph0r3_static_preflight','passed':sum(checks),'total':len(checks),'all_pass':all(checks),'checks':checks}));return 0 if all(checks) else 2
if __name__=='__main__':raise SystemExit(main())
