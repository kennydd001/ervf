#!/usr/bin/env python3
"""R12-D2 full-stage localization diagnostic; scientific pass impossible."""
from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/"scripts/streamq5_moe/run_port80b_t0r12_official_cpu_reference_only.py";spec=importlib.util.spec_from_file_location("r12base",BASE);b=importlib.util.module_from_spec(spec);sys.modules['r12base']=b;spec.loader.exec_module(b)
torch=b.torch;F=b.F;DynamicCache=b.DynamicCache;save_file=b.save_file;psutil=b.psutil;gc=b.gc
R=ROOT/"reports/streamq5_moe";D=ROOT/"reports/runs/streamq5_moe/port80b_t0r12d2_full_stage_diagnostic";LOCK=R/"port80b_t0r12d2_runner_lock.json";VER=ROOT/"scripts/streamq5_moe/verify_port80b_t0r12d2_full_stage_diagnostic.py";VL=R/"port80b_t0r12d2_verifier_lock.json";PR=R/"PORT80B_T0R12D2_FULL_STAGE_DIAGNOSTIC_PREREGISTRATION_2026-08-13.md";FAIL=ROOT/"reports/runs/streamq5_moe/port80b_t0r12_official_cpu_reference_only/t0r12_capture_1_failure.json";ACK="T0R12D2_FULL_STAGE_DIAGNOSTIC_ONLY"
STAGES=("input_norm","gdn","post_norm","official_router_logits","official_router_weights","official_router_ids","experts","shared","shared_gate")
def lockcheck():
 l=json.loads(LOCK.read_text());actual={"runner_sha256":b.sha256(Path(__file__)),"verifier_sha256":b.sha256(VER),"verifier_lock_sha256":b.sha256(VL),"prereg_sha256":b.sha256(PR),"base_sha256":b.sha256(BASE),"failure_sha256":b.sha256(FAIL)};return {"pass":all(l.get(k)==v for k,v in actual.items()),"bindings":actual}
def route_evidence(layer,cap):
 native=cap['official_router_logits'];weights=cap['official_router_weights'];ids=cap['official_router_ids'];sl,sw,si=layer.mlp.gate(cap['router_input']);probs=torch.softmax(native.float(),-1);v11,i11=torch.topk(probs,11,-1);pre=v11[:,:10];pre=pre/pre.sum(-1,keepdim=True);boundary=probs==v11[:,9:10]
 return {'diagnostic_router_logits':sl.detach().cpu(),'diagnostic_router_weights':sw.detach().cpu(),'diagnostic_router_ids':si.detach().cpu(),'router_logits_fp32':native.float(),'router_probs_fp32':probs,'router_top10_ids_recomputed':i11[:,:10],'router_weights_precast_fp32':pre,'router_weights_recomputed_bf16':pre.to(torch.bfloat16),'router_top11_ids':i11,'router_top11_native_bf16_logits':torch.gather(native,1,i11),'router_top10_top11_margin_fp32':v11[:,9]-v11[:,10],'router_boundary_tie_mask':boundary,'router_selected_boundary_mask':torch.gather(boundary,1,ids)}
def retain(raw,prefix,output,cap,cache,p,n,layer):
 for stage in STAGES:raw[f"{prefix}_{stage}"]=cap[stage].detach().cpu().contiguous()
 raw[f"{prefix}_layer_output"]=output.detach().cpu().contiguous();meta,t=b.cache_state(cache,p,n);raw[f"{prefix}_cache_conv"]=t[f"p{p}_s{n}_cache_conv"];raw[f"{prefix}_cache_recurrent"]=t[f"p{p}_s{n}_cache_recurrent"]
 route=route_evidence(layer,cap)
 for k,v in route.items():raw[f"{prefix}_{k}"]=v
def last(t,stage,n):
 return t[n-1:n] if stage in ('official_router_logits','official_router_weights','official_router_ids','experts','shared','shared_gate') else t[:,n-1:n]
def metric(a,z):
 aa=a.detach().cpu().contiguous();zz=z.detach().cpu().contiguous();neq=int((aa!=zz).sum());out={'dtype':str(aa.dtype),'shape':list(aa.shape),'different_elements':neq,'exact_equal':bool(torch.equal(aa,zz)),'reference_sha256':b.tensor_sha(aa),'observed_sha256':b.tensor_sha(zz)}
 if aa.dtype==torch.bfloat16:out.update(max_bf16_ulp=b.max_bf16_ulp(aa,zz),max_abs=float((aa.float()-zz.float()).abs().max()),rel_l2=float(torch.linalg.vector_norm(aa.float()-zz.float())/torch.linalg.vector_norm(aa.float()).clamp_min(1e-30)))
 return out
def all_metrics(raw):
 rows=[]
 for p in range(4):
  for n in range(1,17):
   sm={s:metric(last(raw[f'p{p}_whole_{s}'],s,n),last(raw[f'p{p}_n{n}_{s}'],s,n)) for s in STAGES+('layer_output',)};rows.append({'prompt':p,'length':n,'stages':sm})
 repeats={}
 for a,z,n in (('p1_whole','p1_whole_repeat',16),('p1_n3','p1_n3_repeat',3)):
  repeats[z]={'stages':{s:metric(raw[f'{a}_{s}'],raw[f'{z}_{s}']) for s in STAGES+('layer_output',)},'cache_conv':metric(raw[f'{a}_cache_conv'],raw[f'{z}_cache_conv']),'cache_recurrent':metric(raw[f'{a}_cache_recurrent'],raw[f'{z}_cache_recurrent'])}
 first={}
 for p in range(4):
  first[str(p)]={s:next((r['length'] for r in rows if r['prompt']==p and not r['stages'][s]['exact_equal']),None) for s in STAGES+('layer_output',)}
 cache16={str(p):{'conv':metric(raw[f'p{p}_whole_cache_conv'],raw[f'p{p}_n16_cache_conv']),'recurrent':metric(raw[f'p{p}_whole_cache_recurrent'],raw[f'p{p}_n16_cache_recurrent'])} for p in range(4)}
 return rows,repeats,{'same_length_nondeterminism_observed':any(not m['exact_equal'] for q in repeats.values() for m in list(q['stages'].values())+[q['cache_conv'],q['cache_recurrent']]),'first_divergent_length_by_prompt_stage':first,'whole_prefix16_cache':cache16,'whole_prefix16_cache_divergence':any(not m['exact_equal'] for q in cache16.values() for m in q.values())}
def run():
 if D.exists():raise FileExistsError("D2 output exists")
 if not lockcheck()["pass"]:raise RuntimeError("D2 lock")
 if psutil.virtual_memory().available<b.MIN_START_RAM or torch.cuda.is_initialized():raise RuntimeError("resources")
 proc=psutil.Process();want=json.loads(b.DEPENDENCY_LOCK.read_text())["runtime"]["process_affinity"];proc.cpu_affinity(want);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.mkldnn.enabled=True;torch.set_flush_denormal(False)
 peak={};b.rss_guard('start',peak);inputs=b.locked_inputs();config=b.Qwen3NextConfig.from_pretrained(b.SNAPSHOT,local_files_only=True,trust_remote_code=False);layer,embed,idsource=b.load_official(config,peak);prompts=json.loads(b.PROMPT_LOCK.read_text())['prompts'];ids=torch.tensor([x['token_ids'] for x in prompts]);hidden=F.embedding(ids,embed).to(torch.bfloat16);del embed;gc.collect();raw={'token_ids':ids,'embedding':hidden.cpu()}
 with torch.inference_mode():
  for p in range(4):
   c=DynamicCache(config=config);out,cap=b.capture_official(layer,hidden[p:p+1],c);retain(raw,f"p{p}_whole",out,cap,c,p,16,layer)
   for n in range(1,17):
    c=DynamicCache(config=config);o,ca=b.capture_official(layer,hidden[p:p+1,:n],c);retain(raw,f"p{p}_n{n}",o,ca,c,p,n,layer)
  for label,length in (("p1_whole_repeat",16),("p1_n3_repeat",3)):
   c=DynamicCache(config=config);o,ca=b.capture_official(layer,hidden[1:2,:length],c);retain(raw,label,o,ca,c,1,length,layer)
 rows,repeats,classes=all_metrics(raw);manifest=b.tensor_manifest(raw);D.mkdir(parents=True);rp=D/'t0r12d2_raw.safetensors';jp=D/'t0r12d2_result.json';save_file(raw,rp);b.rss_guard('serialized',peak);result={"kind":"port80b_t0r12d2_full_stage_diagnostic","status":"diagnostic_only_not_pass","runner_sha256":b.sha256(Path(__file__)),"verifier_sha256":b.sha256(VER),"verifier_lock_sha256":b.sha256(VL),"prereg_sha256":b.sha256(PR),"base_sha256":b.sha256(BASE),"failure_sha256":b.sha256(FAIL),"raw_manifest":manifest,"raw_sha256":b.sha256(rp),"inputs":inputs,"source_tensor_sha256":idsource,"stage_metrics":rows,"repeat_metrics":repeats,"interpretation_classes":classes,"resources":peak,"cuda_initialized":torch.cuda.is_initialized(),"claim_boundary":"Interpretation classes only; no threshold/pass/manual/Q5/bank/P4/GPU."};jp.write_text(json.dumps(result,indent=2)+'\n');return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('lockcheck','diagnostic'),required=True);p.add_argument('--acknowledge-diagnostic');a=p.parse_args()
 if a.phase=='lockcheck':print(json.dumps({'kind':'r12d2_lockcheck',**lockcheck(),'physical_actions':{'model':False,'forward':False,'gpu':False}}));return 0
 if a.acknowledge_diagnostic!=ACK:raise SystemExit('exact acknowledgement required')
 print(json.dumps({'status':run()['status']}));return 3
if __name__=='__main__':raise SystemExit(main())
