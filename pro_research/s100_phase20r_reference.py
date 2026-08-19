from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = Path(os.environ["LS_MODEL_DIR"])
OUT = Path(os.environ["PHASE20R_REFERENCE_OUT"])
HARD = "e8f3c7c4de75ad84fe1bcef95d38eca76214480b"
PROMPTS = (
    "The history of computing and artificial intelligence",
    "Explain why the sky appears blue during the day.",
    "Write a Python function that returns the first n Fibonacci numbers.",
)
STEPS = 32


def top_record(logits):
    x=logits.float().detach()
    logp=torch.log_softmax(x,dim=-1)
    vals,ids=torch.topk(logp,64)
    return {
        "target":int(ids[0].item()),
        "top64_ids":[int(v) for v in ids.cpu().tolist()],
        "top64_logprob":[float(v) for v in vals.cpu().tolist()],
        "rest_prob":float(max(1.0-float(vals.exp().sum().item()),1e-30)),
    }


def main():
    p={"kind":"s100_phase20r_reference","status":"started",
       "transformers_version":transformers.__version__,"model_dir":str(MODEL),"steps":STEPS}
    try:
        if transformers.__version__!="5.14.1":
            raise RuntimeError(f"wrong transformers {transformers.__version__}")
        if MODEL.name!=HARD:
            raise RuntimeError(f"wrong snapshot {MODEL.name}")
        cfg=AutoConfig.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=True)
        if getattr(cfg,"model_type",None)!="nemotron_h":
            raise RuntimeError(f"wrong config type {getattr(cfg,'model_type',None)}")
        cfg.use_mamba_kernels=False
        # On the 8 GB laptop, keep only what Accelerate can fit on GPU and
        # place the rest in system RAM. This is an independent correctness
        # reference, not a performance run.
        max_memory={"cpu":"50GiB"}
        if torch.cuda.is_available(): max_memory[0]="7GiB"
        p["config_available"]=True
        p["config_class"]=type(cfg).__name__
        p["max_memory"]=max_memory

        model=AutoModelForCausalLM.from_pretrained(
            str(MODEL),config=cfg,local_files_only=True,trust_remote_code=True,
            device_map="auto",max_memory=max_memory,low_cpu_mem_usage=True,
            dtype="auto",
        ).eval()
        tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=True,use_fast=True)
        p["full_model_loaded"]=True
        p["model_class"]=type(model).__name__
        p["hf_device_map"]={str(k):str(v) for k,v in getattr(model,"hf_device_map",{}).items()}

        # Choose the input embedding's device; Accelerate hooks move later
        # modules according to hf_device_map.
        try:
            input_device=next(model.get_input_embeddings().parameters()).device
        except Exception:
            input_device=next(model.parameters()).device

        traces=[]
        with torch.inference_mode():
            for prompt in PROMPTS:
                ids=tok.encode(prompt,add_special_tokens=False)
                input_ids=torch.tensor([ids],dtype=torch.long,device=input_device)
                outputs=model(input_ids=input_ids,use_cache=True,return_dict=True)
                past=outputs.past_key_values
                recs=[]
                logits=outputs.logits[0,-1]
                for step in range(STEPS):
                    rec=top_record(logits); rec["step"]=step; recs.append(rec)
                    target=rec["target"]
                    if step+1<STEPS:
                        nxt=torch.tensor([[target]],dtype=torch.long,device=input_device)
                        outputs=model(input_ids=nxt,past_key_values=past,use_cache=True,return_dict=True)
                        past=outputs.past_key_values
                        logits=outputs.logits[0,-1]
                traces.append({"prompt":prompt,"prompt_ids":ids,"records":recs})
                print(f"reference complete: {prompt[:40]}...",flush=True)
        p.update({"status":"measured","full_reference_available":True,"traces":traces,
                  "PHASE20R_REFERENCE_EXECUTED":True})
    except Exception as exc:
        p.update({"status":"technical_block","full_reference_available":False,
                  "PHASE20R_REFERENCE_EXECUTED":False,
                  "error":{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:p.get(k) for k in ("status","transformers_version","config_available","config_class","full_model_loaded","full_reference_available","model_class","error")},indent=2))
    return 0 if p.get("status")=="measured" else 2

if __name__=="__main__": raise SystemExit(main())
