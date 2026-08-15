#!/usr/bin/env python3
"""Static/no-exec S0-R2 preflight candidate; source-audit first."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';RUN=S/'run_port80b_t0q5s0r2_selected_route_validation.py';VER=S/'verify_port80b_t0q5s0r2_selected_route_validation.py';SELF=Path(__file__);LOCK=R/'port80b_t0q5s0r2_runner_lock.json';VL=R/'port80b_t0q5s0r2_verifier_lock.json';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0r2_selected_route_validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VL.read_text());rs=RUN.read_text();vs=VER.read_text();checks={'hashes':l['runner_sha256']==sha(RUN) and l['verifier_sha256']==sha(VER) and l['verifier_lock_sha256']==sha(VL) and v['verifier_sha256']==sha(VER),'ast':ast.parse(rs) is not None and ast.parse(vs) is not None,'output_absent':not D.exists(),'r2_repairs':all(x in rs for x in ('post_serialization','mutcodes=bytearray(codes)','runner_lock_content','d2_result_sha256','shard_expected_sha256','gc.collect()','dispositions')),'verifier_replay':all(x in vs for x in ('mutcodes=bytearray(codes)','runner_lock_content','post_serialization','d2_audit_sha256','SHARD_SHA')),'no_model_bank':all(x not in rs for x in ('from_pretrained','DynamicCache','.sq5m'))};o={'kind':'s0r2_static_preflight','pass':all(checks.values()),'checks':checks,'physical_actions':False};print(json.dumps(o,indent=2));return 0 if o['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
