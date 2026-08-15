#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'scripts/streamq5_moe/verify_port80b_t0r12d2_full_stage_diagnostic.py'
spec=importlib.util.spec_from_file_location('d2verify',SRC)
v=importlib.util.module_from_spec(spec);sys.modules['d2verify']=v;spec.loader.exec_module(v)
R=ROOT/'reports/streamq5_moe'
RUN=ROOT/'scripts/streamq5_moe/run_port80b_t0r12d2r2_contiguous_serialization.py'
LOCK=R/'port80b_t0r12d2r2_runner_lock.json'
VL=R/'port80b_t0r12d2r2_verifier_lock.json'
PR=R/'PORT80B_T0R12D2R2_CONTIGUOUS_SERIALIZATION_REPAIR_2026-08-13.md'
D2RUN=ROOT/'scripts/streamq5_moe/run_port80b_t0r12d2_full_stage_diagnostic.py'
SERFAIL=R/'port80b_t0r12d2_postcompute_serialization_failure.json'
OUT=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r2_contiguous_serialization'

def configure():
 v.D=OUT;v.RUN=RUN;v.LOCK=LOCK;v.VL=VL;v.PR=PR;v.__file__=str(Path(__file__))

def provenance():
 lock=json.loads(LOCK.read_text());vl=json.loads(VL.read_text())
 actual={'runner_sha256':v.sha(RUN),'verifier_sha256':v.sha(Path(__file__)),'verifier_lock_sha256':v.sha(VL),'prereg_sha256':v.sha(PR),'d2_source_sha256':v.sha(D2RUN),'failure_sha256':v.sha(SERFAIL)}
 return {'bindings':all(lock.get(k)==z for k,z in actual.items()),'verifier_bound':vl.get('verifier_sha256')==actual['verifier_sha256'],'bindings_actual':actual}

def preflight():
 configure();p=provenance();checks={'bindings':p['bindings'],'verifier_bound':p['verifier_bound'],'outputs_absent':not OUT.exists()}
 return {'kind':'d2r2_verifier_preflight','pass':all(checks.values()),'checks':checks}

def verify():
 configure();base=v.verify();r=json.loads((OUT/'t0r12d2_result.json').read_text());p=provenance()
 extra={'base_verifier_valid':base['valid_diagnostic'],'bindings':p['bindings'],'verifier_bound':p['verifier_bound'],'revision':r.get('revision')=='t0r12d2r2_contiguous_serialization','repair_only':r.get('serialization_repair_only') is True,'d2_source':r.get('d2_source_sha256')==v.sha(D2RUN),'serialization_failure':r.get('serialization_failure_sha256')==v.sha(SERFAIL)}
 return {'kind':'d2r2_independent_verification','pass':False,'valid_diagnostic':all(extra.values()),'checks':extra,'base_checks':base['checks']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('preflight','verify'),required=True);a=p.parse_args();r=preflight() if a.phase=='preflight' else verify();print(json.dumps(r,indent=2));return 0 if r.get('pass') else (3 if r.get('valid_diagnostic') else 2)
if __name__=='__main__':raise SystemExit(main())
