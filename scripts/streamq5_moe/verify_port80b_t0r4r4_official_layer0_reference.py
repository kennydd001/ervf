#!/usr/bin/env python3
"""Independent reconstruction verifier for PORT80B T0-R4-REF-R4."""
from __future__ import annotations
import argparse, hashlib, json, struct, zlib
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import transformers, safetensors, tokenizers

ROOT=Path(__file__).resolve().parents[2]; REPORTS=ROOT/"reports/streamq5_moe"
RUN_DIR=ROOT/"reports/runs/streamq5_moe/port80b_t0r4r4_official_layer0"
RUNNER=ROOT/"scripts/streamq5_moe/run_port80b_t0r4r4_official_layer0_reference.py"
LOCK=REPORTS/"port80b_t0r4r4_runner_lock.json"; VLOCK=REPORTS/"port80b_t0r4r4_verifier_lock.json"
BANK=RUN_DIR/"layer0_513_real_q5_records.bin"; MATRIX_BYTES=675_840; BANK_BYTES=1_040_117_760
HEADER_FORMAT="<4sHHHBBIIH2xIII28s"; HEADER_BYTES=struct.calcsize(HEADER_FORMAT); CODE_BYTES=655_360; SCALE_BYTES=16_384; PAD_BYTES=4_032
SNAP=Path.home()/".cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f"
SHARD=SNAP/"model-00001-of-00040.safetensors"
DEP=REPORTS/"port80b_t0r4_dependency_execution_lock.json"

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8*2**20),b""):h.update(b)
 return h.hexdigest()
def tb(t): return t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
def tsha(t): return hashlib.sha256(tb(t)).hexdigest()
def ordered(t):
 x=t.detach().contiguous().view(torch.uint16).cpu().numpy(); return np.where((x&0x8000)!=0,0x8000-(x&0x7fff),0x8000+x).astype(np.int32)
def ulp(a,b): return int(np.max(np.abs(ordered(a)-ordered(b))))

def quantize_source(value):
 rows,cols=value.shape; work=value.float().reshape(rows,cols//128,128); maximum=work.abs().amax(-1,keepdim=True)
 scale=torch.where(maximum>0,maximum/15,torch.ones_like(maximum)); signed=torch.round(work/scale).clamp(-15,15).to(torch.int8).reshape(rows,cols)
 fields=(signed.numpy().astype(np.int16)+15).astype(np.uint64).reshape(rows,cols//8,8); shifts=(np.arange(8,dtype=np.uint64)*5).reshape(1,1,8);words=np.bitwise_or.reduce(fields<<shifts,axis=-1); bs=(np.arange(5,dtype=np.uint64)*8).reshape(1,1,5)
 codes=((words[...,None]>>bs)&255).astype(np.uint8).tobytes(); scales=scale.squeeze(-1).to(torch.bfloat16).contiguous().view(torch.uint16).numpy().astype("<u2",copy=False).tobytes();return codes,scales

def expected_schema():
 s={"token_ids":("torch.int64",[4,16]),"embedding":("torch.bfloat16",[4,16,2048])}
 cap={"input_norm":[1,16,2048],"gdn":[1,16,2048],"post_norm":[1,16,2048],"router":[16,512],"experts":[16,2048],"shared":[16,2048],"shared_gate":[16,1]}
 routes={"router_logits_native_bf16":("torch.bfloat16",[16,512]),"router_logits_fp32":("torch.float32",[16,512]),"router_probs_fp32":("torch.float32",[16,512]),"router_ids":("torch.int64",[16,10]),"router_top10_top11_margin_fp32":("torch.float32",[16]),"router_weights_precast_fp32":("torch.float32",[16,10]),"router_weights_native_bf16":("torch.bfloat16",[16,10])}
 manual={"manual_gate":[16,10,512],"manual_up":[16,10,512],"manual_down":[16,10,2048],"manual_routed":[16,2048],"manual_shared_raw":[16,2048],"manual_shared_gate":[16,1],"manual_shared_gated":[16,2048]}
 for p in range(4):
  s[f"p{p}_whole_post_attention_residual"]=("torch.bfloat16",[1,16,2048])
  for k,v in cap.items():s[f"p{p}_whole_{k}"]=("torch.bfloat16",v)
  for k,(d,v) in routes.items():s[f"p{p}_whole_{k}"]=(d,v)
  for k,v in manual.items():s[f"p{p}_whole_{k}"]=("torch.bfloat16",v)
  for k in ("whole_layer_output","manual_layer_output"):s[f"p{p}_whole_{k}"]=("torch.bfloat16",[1,16,2048])
  for n in range(1,17):
   s[f"p{p}_prefix{n}_final_output"]=("torch.bfloat16",[1,1,2048])
   s[f"p{p}_s{n}_cache_conv"]=("torch.bfloat16",[1,8192,4]);s[f"p{p}_s{n}_cache_recurrent"]=("torch.float32",[1,32,128,128])
 return s

def parse_bank(result):
 manifest=result["record_artifact"]; entries=manifest["records"]; identities=result["source_tensor_sha256"]
 ok=len(entries)==513 and BANK.stat().st_size==BANK_BYTES; full=hashlib.sha256(); parsed=0
 with BANK.open("rb") as f, safe_open(SHARD,framework="pt",device="cpu") as source:
  for expert in range(513):
   e=entries[expert]; ok &= e["expert"]==expert and e["layer"]==0 and e["shared"]==(expert==512) and len(e["projections"])==3
   for proj in range(3):
    rec=f.read(MATRIX_BYTES); full.update(rec); parsed+=len(rec); m=e["projections"]; meta=m[proj]
    h=struct.unpack(HEADER_FORMAT,rec[:HEADER_BYTES]); codes=rec[HEADER_BYTES:HEADER_BYTES+CODE_BYTES]; scales=rec[HEADER_BYTES+CODE_BYTES:HEADER_BYTES+CODE_BYTES+SCALE_BYTES]; pad=rec[-PAD_BYTES:]
    magic,version,layer,hexpert,hproj,bits,rows,cols,group,cbytes,sbytes,crc,reserved=h
    crc2=zlib.crc32(scales,zlib.crc32(codes))&0xffffffff
    chunks=np.frombuffer(codes,dtype=np.uint8).reshape(-1,5).astype(np.uint64);words=chunks[:,0]|chunks[:,1]<<8|chunks[:,2]<<16|chunks[:,3]<<24|chunks[:,4]<<32;fields=np.stack([(words>>(5*j))&31 for j in range(8)],axis=-1)
    source_value=source.get_tensor(meta["source_key"]); expected_codes,expected_scales=quantize_source(source_value)
    expected_shape=(512,2048) if proj<2 else (2048,512)
    ok &= (magic==b"SQ5M" and version==1 and layer==0 and hexpert==expert and hproj==proj and bits==5 and (rows,cols)==expected_shape and group==128 and cbytes==CODE_BYTES and sbytes==SCALE_BYTES and crc==crc2 and reserved==bytes(28) and pad==bytes(PAD_BYTES))
    ok &= (fields.max()<=30 and bool(torch.isfinite(torch.frombuffer(bytearray(scales),dtype=torch.bfloat16).float()).all()) and codes==expected_codes and scales==expected_scales)
    ok &= (source_value.dtype==torch.bfloat16 and list(source_value.shape)==[rows,cols] and tsha(source_value)==meta["source_sha256"]==identities[meta["source_key"]])
    ok &= (meta["revision"]=="a19358a7659bd1f564300250ee189120c49a562f" and meta["layer"]==0 and meta["expert"]==expert and meta["shared"]==(expert==512) and meta["projection"]==proj and meta["total_record_bytes"]==MATRIX_BYTES and meta["codes_sha256"]==hashlib.sha256(codes).hexdigest() and meta["scales_sha256"]==hashlib.sha256(scales).hexdigest() and meta["record_sha256"]==hashlib.sha256(rec).hexdigest() and meta["crc32"]==crc2)
  ok &= not f.read(1)
 return ok and parsed==BANK_BYTES and full.hexdigest()==manifest["sha256"]

def preflight():
 l=json.loads(LOCK.read_text());v=json.loads(VLOCK.read_text());c={"runner_bound":sha(RUNNER)==l["runner_sha256"],"verifier_bound":sha(__file__)==l["verifier_sha256"]==v["verifier_sha256"],"verifier_lock_bound":sha(VLOCK)==l["verifier_lock_sha256"],"outputs_absent":not RUN_DIR.exists() or not any(RUN_DIR.glob("*_result.json")),"schema":v["schema_version"]=="PORT80B_T0R4R4_REF_V1"};return {"kind":"port80b_t0r4r4_verifier_preflight","pass":all(c.values()),"checks":c,"checks_passed":sum(c.values()),"checks_total":len(c)}

def verify(i):
 rp=RUN_DIR/f"t0r4r4_run_{i}_result.json"; ap=RUN_DIR/f"t0r4r4_run_{i}_raw.safetensors"; r=json.loads(rp.read_text()); vals={}
 with safe_open(ap,framework="pt",device="cpu") as h:
  for k in h.keys(): vals[k]=h.get_tensor(k)
 schema=expected_schema(); c={"expected_keys_exact":set(vals)==set(schema)}
 c["schema_exact"]=all(str(vals[k].dtype)==d and list(vals[k].shape)==shape for k,(d,shape) in schema.items());c["all_finite"]=all(bool(torch.isfinite(v.float()).all()) for v in vals.values())
 independent_manifest={k:{"semantic_key":k,"dtype":str(v.dtype),"shape":list(v.shape),"bytes":v.numel()*v.element_size(),"sha256":tsha(v)} for k,v in sorted(vals.items())}
 c["raw_manifest_recomputed"]=independent_manifest==r["raw_tensor_manifest"]
 c["raw_sha"]=sha(ap)==r["raw_artifact_sha256"];c["result_provenance"]=r["kind"]=="port80b_t0r4r4_official_layer0_reference_and_bank_stage" and r["run_index"]==i and r["runner_sha256"]==sha(RUNNER) and r["runner_lock_sha256"]==sha(LOCK) and r["verifier_sha256"]==sha(__file__) and r["verifier_lock_sha256"]==sha(VLOCK)
 manual=[];prefix=True;route=True;margins=[]
 for p in range(4):
  manual.append(ulp(vals[f"p{p}_whole_manual_layer_output"],vals[f"p{p}_whole_whole_layer_output"]))
  prefix &= all(torch.equal(vals[f"p{p}_prefix{n}_final_output"],vals[f"p{p}_whole_whole_layer_output"][:,n-1:n]) for n in range(1,17))
  native=vals[f"p{p}_whole_router_logits_native_bf16"]; logits=native.float(); probs=torch.softmax(logits,dim=-1); w,ids=torch.topk(probs,10,dim=-1);w=w/w.sum(-1,keepdim=True);top11=torch.topk(probs,11,dim=-1).values;margin=top11[:,9]-top11[:,10];nw=w.to(torch.bfloat16)
  route &= torch.equal(native,vals[f"p{p}_whole_router"]) and torch.equal(logits,vals[f"p{p}_whole_router_logits_fp32"]) and torch.equal(probs,vals[f"p{p}_whole_router_probs_fp32"]) and torch.equal(ids,vals[f"p{p}_whole_router_ids"]) and torch.equal(w,vals[f"p{p}_whole_router_weights_precast_fp32"]) and torch.equal(nw,vals[f"p{p}_whole_router_weights_native_bf16"]) and torch.equal(margin,vals[f"p{p}_whole_router_top10_top11_margin_fp32"]) and bool((w>0).all() and (w[:,:-1]>=w[:,1:]).all() and (nw>0).all() and (nw[:,:-1]>=nw[:,1:]).all() and ((w.sum(-1)-1).abs().max()<=2**-20) and ((nw.float().sum(-1)-1).abs().max()<=0.00390720367431640625) and (margin>0).all() and all(torch.unique(x).numel()==10 for x in ids))
  margins.append(float(margin.min()))
 c["manual_ulp_recomputed"]=max(manual)<=1;c["prefix_bytes_recomputed"]=prefix;c["routes_recomputed"]=route and min(margins)==r["minimum_top10_top11_margin_fp32"]
 cache={(x["prompt"],x["step"]):x for x in r["cache_state_schema"]};c["cache_recomputed"]=len(cache)==64
 for p in range(4):
  for n in range(1,17):
   row=cache[(p,n)];cv=vals[f"p{p}_s{n}_cache_conv"];rv=vals[f"p{p}_s{n}_cache_recurrent"]
   c["cache_recomputed"] &= row["conv"]=={"dtype":str(cv.dtype),"shape":list(cv.shape),"bytes":cv.numel()*cv.element_size(),"sha256":tsha(cv)} and row["recurrent"]=={"dtype":str(rv.dtype),"shape":list(rv.shape),"bytes":rv.numel()*rv.element_size(),"sha256":tsha(rv)}
 c["bank_fully_reconstructed"]=parse_bank(r);c["resources"]=r["resources"]["windows_peak_working_set_bytes"]<=12*2**30 and r["resources"]["minimum_available_ram_bytes"]>=2*2**30 and r["resources"]["projected_steady_working_set_bytes"]<=int(10.5*2**30)
 dep=json.loads(DEP.read_text());base=ROOT/".venv-next-ref/Lib/site-packages/transformers";current_sources={n:sha(base/n) for n in dep["transformers_sources"]};versions={"transformers":transformers.__version__,"torch":torch.__version__,"safetensors":safetensors.__version__,"tokenizers":tokenizers.__version__,"numpy":np.__version__}
 c["dependencies_rehashed"]=current_sources==dep["transformers_sources"]==r["actual_dependencies"]["source_sha256"] and versions==r["actual_dependencies"]["package_versions"]
 rt=r["runtime_contract"]; expected_names={"causal_conv1d_fn":"causal_conv1d_fn","causal_conv1d_update":"causal_conv1d_update","torch_chunk_gated_delta_rule":"torch_chunk_gated_delta_rule","torch_recurrent_gated_delta_rule":"torch_recurrent_gated_delta_rule"}
 c["runtime_actual"]=rt["affinity"]==list(range(16)) and rt["torch_threads"]==1 and rt["torch_interop_threads"]==1 and rt["deterministic_algorithms"] and rt["float32_matmul_precision"]=="highest" and rt["mkldnn_enabled"] and rt["flush_denormal"] is False and rt["flush_denormal_nonzero_subnormal_probe"] and rt["cpu_identity"]==dep["runtime"]["cpu_identity"] and rt["torch_cpu_capability"]==dep["runtime"]["torch_cpu_capability"] and rt["inference_mode_inside_compute"] and rt["autocast_cpu_inside_compute"] is False and all(rt["resolved_callables"][k]["name"]==v for k,v in expected_names.items()) and rt["rmsnorm_gated_callable"]["name"]=="forward" and rt["experts_callable"]["name"]=="forward" and r["cuda_initialized_after"] is False
 if i==2:
  r1=json.loads((RUN_DIR/"t0r4r4_run_1_result.json").read_text());a1=RUN_DIR/"t0r4r4_run_1_raw.safetensors";equal=True
  with safe_open(a1,framework="pt",device="cpu") as x, safe_open(ap,framework="pt",device="cpu") as y:
   equal=list(x.keys())==list(y.keys()) and all(torch.equal(x.get_tensor(k),y.get_tensor(k)) for k in x.keys())
  c["clean_replay_independent"]=equal and r1["raw_tensor_manifest"]==independent_manifest and {k:v for k,v in r1["record_artifact"].items() if k!="second_run_compared_bytes"}=={k:v for k,v in r["record_artifact"].items() if k!="second_run_compared_bytes"}
 return {"kind":"port80b_t0r4r4_independent_reconstruction","run_index":i,"pass":all(c.values()),"checks":c,"checks_passed":sum(c.values()),"checks_total":len(c),"recomputed_manual_ulp":manual,"claim_boundary":"Independent R4-REF bank/reference verification only; Q5 execution and T0-P4 closed."}

def failure(i):
 p=RUN_DIR/f"t0r4r4_run_{i}_failure.json";r=json.loads(p.read_text());c={"identity":r["kind"]=="port80b_t0r4r4_official_layer0_failure" and r["status"]=="valid_negative_or_blocked_not_pass" and r["run_index"]==i,"provenance":r["runner_sha256"]==sha(RUNNER) and r["runner_lock_sha256"]==sha(LOCK) and r["verifier_sha256"]==sha(__file__) and r["verifier_lock_sha256"]==sha(VLOCK),"never_pass":r["claim_boundary"]=="Failure evidence only; never a scientific pass.","cpu_only":r["cuda_initialized"] is False};return {"kind":"port80b_t0r4r4_failure_adjudication","pass":False,"valid_failure_evidence":all(c.values()),"checks":c,"status":"valid_negative_or_blocked_not_pass" if all(c.values()) else "invalid_failure_evidence"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--phase",choices=("preflight","verify","failure"),required=True);p.add_argument("--run-index",type=int,choices=(1,2));a=p.parse_args();r=preflight() if a.phase=="preflight" else (failure(a.run_index) if a.phase=="failure" else verify(a.run_index));print(json.dumps(r,indent=2));return 0 if r.get("pass") else (3 if r.get("valid_failure_evidence") else 2)
if __name__=="__main__":raise SystemExit(main())
