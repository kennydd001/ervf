#!/usr/bin/env python3
"""R6-D: one-prompt immutable router diagnostic. Never a scientific pass."""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE_PATH=ROOT/"scripts/streamq5_moe/run_port80b_t0r4r5_official_layer0_reference.py"
spec=importlib.util.spec_from_file_location("r5base",BASE_PATH);base=importlib.util.module_from_spec(spec);sys.modules["r5base"]=base;spec.loader.exec_module(base)
torch=base.torch;F=base.F;DynamicCache=base.DynamicCache;save_file=base.save_file;psutil=base.psutil;gc=base.gc
REPORTS=ROOT/"reports/streamq5_moe";RUN_DIR=ROOT/"reports/runs/streamq5_moe/port80b_t0r6d_router_diagnostic"
LOCK=REPORTS/"port80b_t0r6d_runner_lock.json";VERIFIER=ROOT/"scripts/streamq5_moe/verify_port80b_t0r6d_router_diagnostic.py";VLOCK=REPORTS/"port80b_t0r6d_verifier_lock.json"
ACK="T0R6D_ROUTER_DIAGNOSTIC_ONLY_AFTER_PREFLIGHT"

def diagnostic():
 lock=json.loads(LOCK.read_text());
 if base.sha256(Path(__file__))!=lock["runner_sha256"] or base.sha256(VERIFIER)!=lock["verifier_sha256"] or base.sha256(VLOCK)!=lock["verifier_lock_sha256"]:raise RuntimeError("R6-D lock mismatch")
 if psutil.virtual_memory().available<base.MIN_START_RAM or torch.cuda.is_initialized():raise RuntimeError("CPU/resource gate")
 proc=psutil.Process();wanted=json.loads(base.DEPENDENCY_LOCK.read_text())["runtime"]["process_affinity"];proc.cpu_affinity(wanted)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision("highest");torch.backends.mkldnn.enabled=True;torch.set_flush_denormal(False)
 peak={};base.rss_guard("r6d_start",peak);inputs=base.locked_inputs();config=base.Qwen3NextConfig.from_pretrained(base.SNAPSHOT,local_files_only=True,trust_remote_code=False);layer,embedding,identities=base.load_official(config,peak)
 prompt=json.loads(base.PROMPT_LOCK.read_text())["prompts"][0];token_ids=torch.tensor([prompt["token_ids"]],dtype=torch.long);hidden=F.embedding(token_ids,embedding).to(torch.bfloat16);del embedding;gc.collect();base.rss_guard("r6d_embedding",peak)
 captured={}
 def out_hook(_m,_a,o):captured.update({k:v.detach().cpu().contiguous().clone() for k,v in zip(("logits","weights","ids"),o)})
 def in_hook(_m,a):captured["input"]=a[0].detach().cpu().contiguous().clone()
 h1=layer.mlp.gate.register_forward_hook(out_hook);h2=layer.mlp.gate.register_forward_pre_hook(in_hook)
 with torch.inference_mode():
  if not torch.is_inference_mode_enabled() or torch.is_autocast_enabled("cpu"):raise RuntimeError("runtime mode")
  cache=DynamicCache(config=config);empty=torch.empty(0,dtype=torch.bfloat16);output=layer(hidden,position_embeddings=(empty,empty),attention_mask=None,past_key_values=cache).detach().cpu().contiguous().clone()
 h1.remove();h2.remove()
 with torch.inference_mode():second=tuple(x.detach().cpu().contiguous().clone() for x in layer.mlp.gate(captured["input"]))
 raw={"token_ids":token_ids,"official_layer_output":output,"official_gate_input":captured["input"],"official_logits":captured["logits"],"official_weights":captured["weights"],"official_ids":captured["ids"],"second_logits":second[0],"second_weights":second[1],"second_ids":second[2]}
 RUN_DIR.mkdir(parents=True,exist_ok=True);raw_path=RUN_DIR/"t0r6d_router_raw.safetensors";result_path=RUN_DIR/"t0r6d_router_diagnostic.json"
 if raw_path.exists() or result_path.exists():raise FileExistsError("refusing to overwrite R6-D evidence")
 save_file(raw,raw_path)
 logits=captured["logits"];probs=torch.softmax(logits.float(),dim=-1);v11,i11=torch.topk(probs,11,dim=-1);pre=v11[:,:10];pre=pre/pre.sum(-1,keepdim=True);bf=pre.to(torch.bfloat16);margins=v11[:,9]-v11[:,10];l11=torch.gather(logits,1,i11);rows=[];failed=set()
 for r in range(16):
  boundary=v11[r,9];ties=torch.nonzero(probs[r]==boundary).flatten();selected=captured["ids"][r]
  checks={"ids_equal":torch.equal(selected,i11[r,:10]),"weights_equal":torch.equal(captured["weights"][r],bf[r]),"finite":bool(torch.isfinite(probs[r]).all() and torch.isfinite(captured["weights"][r].float()).all()),"positive":bool((pre[r]>0).all() and (captured["weights"][r]>0).all()),"monotonic":bool((pre[r,:-1]>=pre[r,1:]).all() and (captured["weights"][r,:-1]>=captured["weights"][r,1:]).all()),"unique":torch.unique(selected).numel()==10,"bounds":bool((selected>=0).all() and (selected<512).all())};failed.update(k for k,v in checks.items() if not v);bits=l11[r].contiguous().view(torch.uint16)
  rows.append({"row":r,"checks":checks,"fp32_sum_error":float((pre[r].sum()-1).abs()),"bf16_sum_error":float((bf[r].float().sum()-1).abs()),"probability_margin":float(margins[r]),"rank10":{"expert":int(i11[r,9]),"native_bf16_logit":float(l11[r,9]),"u16_bits":int(bits[9])},"rank11":{"expert":int(i11[r,10]),"native_bf16_logit":float(l11[r,10]),"u16_bits":int(bits[10])},"boundary_tie_expert_ids":[int(x) for x in ties],"selected_boundary_subset":[int(x) for x in selected if bool((ties==x).any())]})
 strict=bool((margins==0).any());verdict="strict_margin_negative" if strict else ("failed_conjunct:"+",".join(sorted(failed)) if failed else "no_failure_reproduced")
 result={"kind":"port80b_t0r6d_router_diagnostic","status":"diagnostic_negative_not_pass","verdict":verdict,"rows":rows,"official_second_call":{"logits_equal":torch.equal(logits,second[0]),"weights_equal":torch.equal(captured["weights"],second[1]),"ids_equal":torch.equal(captured["ids"],second[2])},"raw_artifact":str(raw_path.relative_to(ROOT)).replace("\\","/"),"raw_sha256":base.sha256(raw_path),"source_tensor_sha256":identities,"inputs":inputs,"resources":peak,"cuda_initialized":torch.cuda.is_initialized(),"bank_built":False,"claim_boundary":"Diagnostic-only evidence; cannot pass R4-REF, Q5, or T0-P."}
 with result_path.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2);f.write("\n")
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument("--phase",choices=("smoke","diagnostic"),required=True);p.add_argument("--acknowledge-diagnostic");a=p.parse_args()
 if a.phase=="smoke":print(json.dumps({"kind":"port80b_t0r6d_smoke","pass":BASE_PATH.is_file() and not RUN_DIR.exists(),"no_forward":True,"no_bank":True}));return 0
 if a.acknowledge_diagnostic!=ACK:raise SystemExit("exact diagnostic acknowledgement required")
 try:r=diagnostic();print(json.dumps({"status":r["status"],"verdict":r["verdict"]},indent=2));return 3
 except BaseException as e:
  RUN_DIR.mkdir(parents=True,exist_ok=True);p=RUN_DIR/"t0r6d_failure.json"
  if not p.exists():
   with p.open("x",encoding="utf-8") as f:json.dump({"kind":"port80b_t0r6d_failure","status":"diagnostic_failure_not_pass","type":type(e).__name__,"error":str(e),"runner_sha256":base.sha256(Path(__file__)),"cuda_initialized":torch.cuda.is_initialized(),"bank_built":False},f,indent=2);f.write("\n")
  raise
if __name__=="__main__":raise SystemExit(main())
