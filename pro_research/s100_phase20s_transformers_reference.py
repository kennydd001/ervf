from __future__ import annotations
import json,os,traceback
from pathlib import Path

OUT=Path(os.environ["PHASE20S_REFERENCE_OUT"])
MODEL=Path(os.environ["LS_MODEL_DIR"])

def main():
    payload={"kind":"s100_phase20s_transformers_reference","status":"started"}
    try:
        import torch,transformers
        from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
        payload["transformers_version"]=transformers.__version__
        cfg=AutoConfig.from_pretrained(str(MODEL),local_files_only=True,
                                       trust_remote_code=False)
        payload["config_class"]=type(cfg).__name__
        payload["architecture"]=getattr(cfg,"architectures",None)
        if hasattr(cfg,"use_mamba_kernels"):cfg.use_mamba_kernels=False

        # Load quantized storage on CPU first; if successful Accelerate moves
        # modules to CUDA only for execution. This is intentionally slow.
        model=AutoModelForCausalLM.from_pretrained(
            str(MODEL),config=cfg,local_files_only=True,trust_remote_code=False,
            device_map="cpu",low_cpu_mem_usage=True,
        )
        payload["model_class"]=type(model).__name__
        payload["full_model_loaded"]=True
        try:
            from accelerate import cpu_offload
            cpu_offload(model,execution_device=torch.device("cuda:0"))
            payload["cpu_offload_enabled"]=True
        except Exception as exc:
            payload["cpu_offload_enabled"]=False
            payload["cpu_offload_error"]=f"{type(exc).__name__}: {exc}"

        tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,
                                          trust_remote_code=False,use_fast=True)
        prompts=[
          "Explain why the sky is blue in three concise sentences.",
          "Write a Python function for binary search and state its invariant.",
          "A shop discounts a 120 euro item by 15 percent. Explain the calculation.",
        ]
        traces=[]
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for text in prompts:
            ids=tok.encode(text,add_special_tokens=False)
            input_ids=torch.tensor([ids],dtype=torch.long,device=device)
            rows=[]
            past=None
            # Slow independent smoke: 8 greedy positions is enough to prove
            # actual model execution; layer-oracle supplies deeper parity.
            for _ in range(8):
                with torch.inference_mode():
                    out=model(input_ids=input_ids,use_cache=True,past_key_values=past)
                logits=out.logits[0,-1].float().cpu()
                top=torch.topk(logits,64)
                nxt=int(top.indices[0])
                rows.append({"target":nxt,
                             "top64":top.indices.tolist(),
                             "top64_logits":top.values.tolist()})
                past=out.past_key_values
                input_ids=torch.tensor([[nxt]],dtype=torch.long,device=device)
            traces.append({"prompt":text,"prompt_ids":ids,"rows":rows})
        payload.update({"status":"measured","full_model_executed":True,"traces":traces})
    except Exception as exc:
        payload.update({"status":"technical_block",
            "full_model_loaded":payload.get("full_model_loaded",False),
            "full_model_executed":False,
            "error":{"type":type(exc).__name__,"message":str(exc),
                     "traceback":traceback.format_exc()}})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:payload.get(k) for k in
      ("status","transformers_version","config_class","model_class",
       "full_model_loaded","full_model_executed","error")},indent=2))
    return 0
if __name__=="__main__":raise SystemExit(main())
