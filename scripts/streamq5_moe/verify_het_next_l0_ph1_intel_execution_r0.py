#!/usr/bin/env python3
"""Independent CPU verifier for frozen PH1 Intel execution evidence."""
from __future__ import annotations
import hashlib,json,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPORTS=ROOT/'reports/streamq5_moe';OUT=REPORTS/'het_next_l0_ph1_intel_execution_r0';VERIFY=REPORTS/'het_next_l0_ph1_intel_execution_r0_independent_verification.json'
STAGES={'gate':'e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867','up':'f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08','silu':'a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8','activation':'762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f','down':'142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc'}
LAUNCH=[('gate_linear',4096,256),('up_linear',4096,256),('activation',512,256),('down_linear',16384,256)]
def sha(b):return hashlib.sha256(b).hexdigest()
def fsha(p):return sha(Path(p).read_bytes())
def main():
 r,m,c=(OUT/n for n in ('result.json','manifest.json','commit.json'));rr,mm,cc=(json.loads(x.read_text()) for x in (r,m,c));ev=rr['evidence'];led=ev['ledger'];outputs={k:bytes.fromhex(v) for k,v in ev['outputs'].items()};alloc=[x for x in led if x.get('op')=='host_usm_allocate'];args=[x for x in led if x.get('op')=='set_pointer_arg'];launch=[x for x in led if x.get('op')=='enqueue'];release=[x for x in led if x.get('op')=='release'];cleanup=led[-1]
 checks={'bundle':cc=={'kind':'ph1_intel_execution_r0_commit','manifest_sha256':fsha(m),'result_sha256':fsha(r)} and all((OUT/x['name']).stat().st_size==x['bytes'] and fsha(OUT/x['name'])==x['sha256'] for x in mm['files']),'schema':rr['kind']=='ph1_intel_execution_r0' and rr['status']=='intel_execution_positive' and rr['positive'] is True,'stages':{k:sha(outputs[k]) for k in STAGES}==STAGES,'finite':all(all(((w>>7)&255)!=255 for w in struct.unpack('<'+'H'*(len(outputs[k])//2),outputs[k])) for k in STAGES),'counters':all(all(v==1 for v in struct.unpack('<'+'I'*(len(outputs[k])//4),outputs[k])) for k in ('gate_counters','up_counters','activation_counters','down_counters')),'allocations':len(alloc)==14 and sum(x['bytes'] for x in alloc)==2185216 and len({x['pointer'] for x in alloc})==14 and all(x['pointer']%4096==0 and x['base']==x['pointer'] and x['queried_size']==x['bytes'] for x in alloc),'args':len(args)==18 and len({(x['kernel'],x['index']) for x in args})==18,'launches':[(x['kernel'],x['global'],x['local']) for x in launch]==LAUNCH and all(x['event_requested'] is False for x in launch),'release':len(release)==21 and all(x['code']==0 for x in release) and cleanup['release_attempts']==21 and cleanup['cleanup_complete'] is True and cleanup['live_owned_resources']==0,'forbidden':all(v==0 for v in ev['forbidden_calls'].values()),'compile_binding':ev['binary_sha256']=='8b57db279fbb1d7d8df17ebab5cfb54203ef8da8cc31df2d136650820548f629' and ev['source_sha256']=='f1b3ccdae6d202ed210810e3cd419f726ea89ffa8fba0c84df5c2bfca3a84d21'}
 out={'kind':'ph1_intel_execution_r0_independent_verification','checks':checks,'pass':all(checks.values()),'passed':sum(checks.values()),'total':len(checks)}
 if VERIFY.exists():raise FileExistsError(VERIFY)
 VERIFY.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps(out,indent=2));return 0 if out['pass'] else 3
if __name__=='__main__':raise SystemExit(main())
