#!/usr/bin/env python3
import ast,hashlib,json,py_compile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/"reports/streamq5_moe";X=ROOT/"scripts/streamq5_moe/run_port80b_t0r12_official_cpu_reference_only.py";V=ROOT/"scripts/streamq5_moe/verify_port80b_t0r12_official_cpu_reference_only.py";L=R/"port80b_t0r12_runner_lock.json";VL=R/"port80b_t0r12_verifier_lock.json";PY=ROOT/".venv-next-ref/Scripts/python.exe"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 l=json.loads(L.read_text());vl=json.loads(VL.read_text());c={'hashes':sha(X)==l['runner_sha256'] and sha(V)==l['verifier_sha256'] and sha(VL)==l['verifier_lock_sha256'],'run_dir_absent':not (ROOT/'reports/runs/streamq5_moe/port80b_t0r12_official_cpu_reference_only').exists()}
 for p in (X,V):py_compile.compile(str(p),doraise=True)
 forbidden_functions={'stream_records','matrix_record','quantize_matrix','quantize_source','codec_sentinel','verify_committed_bank','transaction_simulation'}
 forbidden_constants={'RECORD_ARTIFACT','RECORD_MANIFEST','BANK_BYTES','EXPERT_BYTES','MATRIX_BYTES','CODE_BYTES','SCALE_BYTES','PADDING_BYTES','PAD_BYTES','HEADER_FORMAT','HEADER_BYTES','GROUP'}
 forbidden_calls={'stream_records','matrix_record','quantize_matrix','quantize_source','codec_sentinel','verify_committed_bank','transaction_simulation'}
 ast_checks=[]
 for source_path in (X,V):
  tree=ast.parse(source_path.read_text());names={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))};assigned={n.targets[0].id for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)};calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)};ast_checks.append(not (forbidden_functions&names) and not (forbidden_constants&assigned) and not (forbidden_calls&calls))
 c['no_bank_or_codec_ast_both_sources']=all(ast_checks)
 source=X.read_text();c['kinds']=vl['result_kind'] in source and vl['failure_kind'] in source and vl['result_kind'] in V.read_text() and vl['failure_kind'] in V.read_text()
 for phase in ('lockcheck','ledger-unit'):
  p=subprocess.run([str(PY),str(X),'--phase',phase],capture_output=True,text=True,timeout=90);d=json.loads(p.stdout) if p.returncode==0 else {};c[phase]=p.returncode==0 and d.get('pass') and not any(d['physical_actions'].values())
 r={'kind':'port80b_t0r12_cpu_preflight','pass':all(c.values()),'checks':c,'claim_boundary':'Hash/AST/exact lockcheck/ledger-unit only; no model/forward/bank/GPU.'};print(json.dumps(r,indent=2));return 0 if r['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
