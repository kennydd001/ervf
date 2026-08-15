#!/usr/bin/env python3
"""R12-D diagnostic-only whole/prefix evidence; never a pass."""
from __future__ import annotations
import argparse,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/"scripts/streamq5_moe/run_port80b_t0r12_official_cpu_reference_only.py";spec=importlib.util.spec_from_file_location("r12base",BASE);b=importlib.util.module_from_spec(spec);sys.modules["r12base"]=b;spec.loader.exec_module(b)
torch=b.torch;F=b.F;DynamicCache=b.DynamicCache;save_file=b.save_file;np=b.np;psutil=b.psutil;gc=b.gc
R=ROOT/"reports/streamq5_moe";D=ROOT/"reports/runs/streamq5_moe/port80b_t0r12d_prefix_diagnostic";LOCK=R/"port80b_t0r12d_runner_lock.json";VER=ROOT/"scripts/streamq5_moe/verify_port80b_t0r12d_prefix_diagnostic.py";VL=R/"port80b_t0r12d_verifier_lock.json";ACK="T0R12D_PREFIX_DIAGNOSTIC_ONLY"
def lockcheck():
 l=json.loads(LOCK.read_text());return {"pass":b.sha256(Path(__file__))==l["runner_sha256"] and b.sha256(VER)==l["verifier_sha256"] and b.sha256(VL)==l["verifier_lock_sha256"] and b.sha256(BASE)==l["r12_runner_sha256"]}
def metrics(a,c):
 af=a.float();cf=c.float();delta=af-cf;den=float(torch.linalg.vector_norm(af));aa=a.contiguous().view(torch.uint16);cc=c.contiguous().view(torch.uint16)
 return {"diff_words":int((aa!=cc).sum()),"max_bf16_ulp":b.max_bf16_ulp(a,c),"max_abs":float(delta.abs().max()),"relative_l2":float(torch.linalg.vector_norm(delta))/max(den,1e-30),"finite":bool(torch.isfinite(af).all() and torch.isfinite(cf).all()),"a_sha256":b.tensor_sha(a),"b_sha256":b.tensor_sha(c)}
def run():
 if D.exists():raise FileExistsError("R12-D output exists")
 if not lockcheck()["pass"]:raise RuntimeError("R12-D lock")
 if psutil.virtual_memory().available<b.MIN_START_RAM or torch.cuda.is_initialized():raise RuntimeError("resource gate")
 proc=psutil.Process();want=json.loads(b.DEPENDENCY_LOCK.read_text())["runtime"]["process_affinity"];proc.cpu_affinity(want);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision("highest");torch.backends.mkldnn.enabled=True;torch.set_flush_denormal(False)
 peak={};b.rss_guard("start",peak);inputs=b.locked_inputs();config=b.Qwen3NextConfig.from_pretrained(b.SNAPSHOT,local_files_only=True,trust_remote_code=False);layer,embedding,identities=b.load_official(config,peak);prompts=json.loads(b.PROMPT_LOCK.read_text())["prompts"];ids=torch.tensor([x["token_ids"] for x in prompts]);hidden=F.embedding(ids,embedding).to(torch.bfloat16);del embedding;gc.collect()
 raw={"token_ids":ids,"embedding":hidden.cpu()};rows=[];cache_rows=[]
 with torch.inference_mode():
  for p in range(4):
   wc=DynamicCache(config=config);whole,cap=b.capture_official(layer,hidden[p:p+1],wc);route=b.router_artifacts(layer,cap);wm,wt=b.cache_state(wc,p,16);raw[f"p{p}_whole_output"]=whole;raw[f"p{p}_whole_cache_conv"]=wt[f"p{p}_s16_cache_conv"];raw[f"p{p}_whole_cache_recurrent"]=wt[f"p{p}_s16_cache_recurrent"]
   for k,v in {**cap,**route}.items():raw[f"p{p}_whole_{k}"]=v
   for n in range(1,17):
    pc=DynamicCache(config=config);out,_=b.capture_official(layer,hidden[p:p+1,:n],pc);final=out[:,-1:].contiguous();raw[f"p{p}_prefix{n}_final"]=final;pm,pt=b.cache_state(pc,p,n)
    if n==16:raw[f"p{p}_prefix16_cache_conv"]=pt[f"p{p}_s16_cache_conv"];raw[f"p{p}_prefix16_cache_recurrent"]=pt[f"p{p}_s16_cache_recurrent"]
    rows.append({"prompt":p,"length":n,**metrics(whole[:,n-1:n],final)})
   cache_rows.append({"prompt":p,"conv":metrics(raw[f"p{p}_whole_cache_conv"],raw[f"p{p}_prefix16_cache_conv"]),"recurrent":metrics(raw[f"p{p}_whole_cache_recurrent"],raw[f"p{p}_prefix16_cache_recurrent"])})
  p=1
  wc=DynamicCache(config=config);wr,_=b.capture_official(layer,hidden[p:p+1],wc);pc=DynamicCache(config=config);pr,_=b.capture_official(layer,hidden[p:p+1,:3],pc);raw["p1_whole_repeat_output"]=wr;raw["p1_prefix3_repeat_final"]=pr[:,-1:].contiguous()
 raw_manifest=b.tensor_manifest(raw);D.mkdir(parents=True);rp=D/"t0r12d_raw.safetensors";jp=D/"t0r12d_result.json";save_file(raw,rp);b.rss_guard("after_raw",peak)
 result={"kind":"port80b_t0r12d_prefix_diagnostic","status":"diagnostic_only_not_pass","whole_prefix_metrics":rows,"cache_metrics":cache_rows,"same_length_repeat":{"whole_prompt1":metrics(raw["p1_whole_output"],raw["p1_whole_repeat_output"]),"prefix3_prompt1":metrics(raw["p1_prefix3_final"],raw["p1_prefix3_repeat_final"])},"raw_manifest":raw_manifest,"raw_sha256":b.sha256(rp),"inputs":inputs,"source_tensor_sha256":identities,"resources":peak,"cuda_initialized":torch.cuda.is_initialized(),"claim_boundary":"Diagnostic only; no pass/manual/Q5/bank/P4/GPU."};jp.write_text(json.dumps(result,indent=2)+"\n");return result
def main():
 p=argparse.ArgumentParser();p.add_argument("--phase",choices=("lockcheck","diagnostic"),required=True);p.add_argument("--acknowledge-diagnostic");a=p.parse_args()
 if a.phase=="lockcheck":print(json.dumps({"kind":"r12d_lockcheck",**lockcheck(),"physical_actions":{"model":False,"forward":False,"gpu":False}}));return 0
 if a.acknowledge_diagnostic!=ACK:raise SystemExit("exact acknowledgement required")
 r=run();print(json.dumps({"status":r["status"]}));return 3
if __name__=="__main__":raise SystemExit(main())
