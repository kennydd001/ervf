#!/usr/bin/env python3
"""R7A authorization-only no-device preflight."""
from __future__ import annotations
import ast,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe'
RUN=S/'run_het_next_l0_ph1_intel_execution_r7a.py';VER=S/'verify_het_next_l0_ph1_intel_execution_r7a.py';R7RUN=S/'run_het_next_l0_ph1_intel_execution_r7.py';R7VER=S/'verify_het_next_l0_ph1_intel_execution_r7.py';BACK=S/'het_next_l0_ph1_intel_execution_r6_backend.py';COMMON=S/'het_next_l0_ph1_intel_execution_r6_common.py';LOCK=R/'het_next_l0_ph1_intel_execution_r7a_lock.json';PR=R/'HET_NEXT_L0_PH1_INTEL_EXECUTION_R7A_PREREGISTRATION_2026-08-14.md';OUT=R/'het_next_l0_ph1_intel_execution_r7a';RESULT=R/'het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json'
R7PPRE=S/'preflight_het_next_l0_ph1_intel_execution_r7p.py';R7PLOCK=R/'het_next_l0_ph1_intel_execution_r7p_lock.json';R7PPR=R/'HET_NEXT_L0_PH1_INTEL_EXECUTION_R7P_PREREGISTRATION_2026-08-14.md';R7PAUD=R/'HET_NEXT_L0_PH1_INTEL_EXECUTION_R7P_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md';R7PRESULT=R/'het_next_l0_ph1_intel_execution_r7p_static_preflight.json';ACK='PH1_INTEL_EXECUTION_R7A_AFTER_R7P_PASS18_AND_FINAL_AUDIT_GO'
COMPILE=R/'het_next_l0_ph1_intel_compile_r2a';CPU=R/'het_next_l0_ph1_cpu_freeze_r2'
PROVENANCE={'runner_sha256':RUN,'backend_sha256':BACK,'common_sha256':COMMON,'verifier_sha256':VER,'preflight_sha256':Path(__file__),'prereg_sha256':PR,'audit_sha256':R/'HET_NEXT_L0_PH1_INTEL_EXECUTION_R5_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md','r0_backend_sha256':S/'het_next_l0_ph1_intel_execution_r0_backend.py','r0_runner_sha256':S/'run_het_next_l0_ph1_intel_execution_r0.py','compile_commit_sha256':COMPILE/'commit.json','compile_result_sha256':COMPILE/'result.json','compile_manifest_sha256':COMPILE/'manifest.json','compile_build_log_sha256':COMPILE/'intel_build.log','compile_binary_sha256':COMPILE/'intel_program.bin','compile_source_sha256':COMPILE/'intel_source.cl','compile_verification_sha256':R/'het_next_l0_ph1_intel_compile_r2a_independent_verification.json','compile_verification_report_sha256':R/'HET_NEXT_L0_PH1_INTEL_COMPILE_R2A_INDEPENDENT_VERIFICATION_REPORT_2026-08-14.md','cpu_commit_sha256':CPU/'commit.json','cpu_manifest_sha256':CPU/'manifest.json','cpu_verification_sha256':R/'het_next_l0_ph1_cpu_freeze_r2_independent_verification.json','cpu_verification_report_sha256':R/'HET_NEXT_L0_PH1_CPU_FREEZE_R2_INDEPENDENT_VERIFICATION_REPORT_2026-08-14.md','generator_sha256':S/'generate_het_next_l0_ph1_cpu_freeze.py','r6p1_preflight_sha256':S/'preflight_het_next_l0_ph1_intel_execution_r6p1.py','r6p1_lock_sha256':R/'het_next_l0_ph1_intel_execution_r6p1_lock.json','r6p1_audit_sha256':R/'HET_NEXT_L0_PH1_INTEL_EXECUTION_R6P1_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md','r6p1_result_sha256':R/'het_next_l0_ph1_intel_execution_r6p1_static_preflight.json','r6p1_diagnosis_sha256':R/'HET_NEXT_L0_PH1_INTEL_EXECUTION_R6P1_ORACLE_OUTPUTS_NEGATIVE_DIAGNOSIS_2026-08-14.md','r7p_preflight_sha256':R7PPRE,'r7p_lock_sha256':R7PLOCK,'r7p_prereg_sha256':R7PPR,'r7p_audit_sha256':R7PAUD,'r7p_result_sha256':R7PRESULT}
MUTATIONS=('getinfo_status','setptr_status','ownership_missing','ownership_duplicate','ownership_return','ownership_pending','ownership_pointer','identity','control_missing','output','pointer_alias','alignment','usm_type','usm_base','arg_pointer','launch_geometry','launch_event','read_order','release_order','release_owned','release_code','cleanup','provenance','resource_summary','resource_order','resource_peak','forbidden_api','stage_hash')
def fs(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Normalize(ast.NodeTransformer):
 def visit_Constant(self,node):
  if isinstance(node.value,str):return ast.copy_location(ast.Constant(node.value.replace('r7a','r7').replace('R7A','R7')),node)
  return node
def funcs(path):
 tree=ast.parse(Path(path).read_text());return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))},tree
def same_functions(a,b,names,normalize=False):
 af,_=funcs(a);bf,_=funcs(b)
 if not all(n in af and n in bf for n in names):return False
 def dump(x):
  x=ast.fix_missing_locations(Normalize().visit(x)) if normalize else x
  return ast.dump(x,include_attributes=False)
 return all(dump(af[n])==dump(bf[n]) for n in names)
def no_device_ast(paths):
 trees=[ast.parse(Path(p).read_text()) for p in paths];imports={n.module for t in trees for n in ast.walk(t) if isinstance(n,ast.ImportFrom)}|{a.name for t in trees for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names};loaders={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for t in trees for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name)) and (n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id) in {'WinDLL','CDLL','LoadLibrary','LoadLibraryEx'}}
 return not ({'pyopencl','cupy','torch','safetensors','transformers'}&imports) and not loaders
def pass18(j):
 checks=j.get('checks',{});vf=j.get('verifier_fixture_evidence',{});base=vf.get('baseline_checks',{});lin=j.get('linear_sentinel',{}).get('shapes',{});bad=j.get('write_after_loop_negative',{});expected_lin={'gate_up':'3c7b2fca9822c349c42b87f8115eba4e9f7a794e4f7b33eb8ec70226ca870ca0','down':'64391e466f846b7383e376d155c621bb98c80ca57eb4845c441d506796ec4bb1'}
 linear_ok=set(lin)==set(expected_lin) and all(lin[k].get('all_rows_equal') is True and lin[k].get('repeat_equal') is True and lin[k].get('first_sha256')==lin[k].get('second_sha256')==lin[k].get('expected_sha256')==v for k,v in expected_lin.items())
 bad_ok=bad.get('pass') is True and bad.get('poison_word')==0x7e00 and bad.get('r7_assignment_inside_row_loop') is True and bad.get('mutant_assignment_inside_row_loop') is False and all(x.get('poison_prefix') is True and x.get('last_correct') is True and x.get('repeat_equal') is True and x.get('differs_from_target') is True for x in bad.get('shapes',{}).values())
 return j.get('kind')=='ph1_intel_execution_r7p_static_preflight' and j.get('pass') is True and j.get('passed')==j.get('total')==18 and j.get('no_payload_compiler_device') is True and len(checks)==18 and all(v is True for v in checks.values()) and len(base)==20 and all(v is True for v in base.values()) and vf.get('baseline_false_names')==[] and tuple(vf.get('mutation_names',()))==MUTATIONS==tuple(vf.get('rejected_mutations',())) and linear_ok and bad_ok
def main():
 lock=json.loads(LOCK.read_text());p7=json.loads(R7PRESULT.read_text());runner_text=RUN.read_text();observed={k:fs(v) for k,v in PROVENANCE.items()};chain={'auth_preflight_sha256':fs(Path(__file__)),'r7p_result_sha256':fs(R7PRESULT),'r7p_audit_sha256':fs(R7PAUD),'r7p_preflight_sha256':fs(R7PPRE),'r7p_lock_sha256':fs(R7PLOCK),'r7p_prereg_sha256':fs(R7PPR)}
 runner_same=same_functions(RUN,R7RUN,('safe_available','verify_bundle','configure','package_exact','ledger_gates','execute','execute_authorized','main'),True);verifier_same=same_functions(VER,R7VER,('b2f','f2b','codec','check_record','rebuild_controls','parts','rse','pack','fma','add','rb','mul','linear','linear_all_row_sentinel','verify_dict'),False)
 checks={'hash_bindings':all(lock.get(k)==v for k,v in observed.items()) and all(lock.get(k)==v for k,v in chain.items()),'open_exact':lock.get('kind')=='ph1_intel_execution_r7a_lock' and lock.get('execution_open') is True and lock.get('audit_token')==ACK and ACK in runner_text,'r7p_pass18':fs(R7PRESULT)=='e10c513fdbecb27e08319c462ba1d1020b1c94c4ff5d9199047ae513197dd959' and pass18(p7),'authorization_only_runner':runner_same and 'import het_next_l0_ph1_intel_execution_r6_backend as backend' in runner_text and 'import het_next_l0_ph1_intel_execution_r6_common as common' in runner_text,'fixed_verifier_unchanged':verifier_same,'no_device_static':no_device_ast((Path(__file__),RUN,VER)),'output_absent':not OUT.exists() and not RESULT.exists()}
 out={'kind':'ph1_intel_execution_r7a_authorization_preflight','checks':checks,'pass':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'no_payload_compiler_device':True,'ack':ACK,'r7p_result_sha256':fs(R7PRESULT)}
 if RESULT.exists():raise FileExistsError(RESULT)
 RESULT.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps(out,indent=2));return 0 if out['pass'] else 3
if __name__=='__main__':raise SystemExit(main())
