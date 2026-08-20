from __future__ import annotations

from collections import Counter
import json
import traceback
import types

import numpy as np

from common import require_model_dir,utc_now,write_json_atomic
from s100_phase21_common import release
from s100_phase22_common import make_v6
from s100_phase24_common import RESULTS

OUT=RESULTS/"S100_PHASE24_GENERALIZATION.json"

PROMPTS=[
  ("factual","Explain how photosynthesis works and why it matters."),
  ("code","Write a Python implementation of Dijkstra's algorithm with type hints."),
  ("reasoning","A tank is filled by two pipes and drained by a third. Derive the total time step by step."),
  ("conversation","I had a difficult day at work. Respond naturally and help me think it through."),
  ("technical","Explain CUDA memory coalescing, occupancy, register pressure, and tensor-core scheduling."),
  ("dutch","Leg in het Nederlands uit waarom de hemel blauw is en een zonsondergang rood."),
  ("translation","Translate this paragraph from English to Dutch and preserve the technical terminology."),
  ("json","Return a valid JSON object describing a small software project with tasks, owners, risks, and dates."),
]

def block_stats(routes):
    # routes [T, layers, topk]
    T,L,K=routes.shape
    layers=[];all_counts=[]
    for li in range(L):
        c=Counter(int(x) for x in routes[:,li,:].reshape(-1))
        hist=Counter(c.values())
        all_counts.extend(c.values())
        layers.append({
          "layer_index":li,
          "route_slots":T*K,
          "unique_experts":len(c),
          "repeat_rate":1.0-len(c)/float(T*K),
          "m_histogram":{str(k):int(v) for k,v in sorted(hist.items())},
          "max_m":max(c.values()),
        })
    return {
      "T":T,"layers":layers,
      "mean_unique_experts":float(np.mean([x["unique_experts"] for x in layers])),
      "median_unique_experts":float(np.median([x["unique_experts"] for x in layers])),
      "mean_repeat_rate":float(np.mean([x["repeat_rate"] for x in layers])),
      "median_repeat_rate":float(np.median([x["repeat_rate"] for x in layers])),
      "global_m_histogram":{
        str(k):int(v) for k,v in sorted(Counter(all_counts).items())
      },
      "max_m":max(all_counts),
    }

def main():
    payload={"kind":"s100_phase24_generalization","status":"started",
      "started_utc":utc_now(),
      "claim_boundary":"route-reuse generalization; no latency promotion"}
    rt=None
    try:
        import cupy as cp
        from transformers import AutoTokenizer

        rt,keep=make_v6(512)
        tok=AutoTokenizer.from_pretrained(
          str(require_model_dir()),local_files_only=True,
          trust_remote_code=True,use_fast=True
        )
        layers=[int(x) for x in rt.moe_layers]
        original=rt._moe_dev
        marker={"active":False,"current":{}}

        def wrapped(self,i,out):
            result=original(i,out)
            if marker["active"]:
                dev=self._dev_cache[int(i)]
                marker["current"][int(i)]=cp.asnumpy(
                  dev["ids"]
                ).astype(np.int32,copy=True)
            return result

        rt._moe_dev=types.MethodType(wrapped,rt)
        records=[]
        for label,text in PROMPTS:
            ids=tok.encode(text,add_special_tokens=False)
            rt.reset();nxt=None;marker["active"]=False
            for token in ids:
                nxt=int(rt.step(int(token)))
            if nxt is None:
                raise RuntimeError(f"empty prompt {label}")

            steps=[];cur=int(nxt)
            for step in range(8):
                marker["current"]={}
                marker["active"]=True
                cur=int(rt.step(cur))
                marker["active"]=False
                if sorted(marker["current"])!=layers:
                    raise RuntimeError(
                      f"{label} step {step}: captured "
                      f"{sorted(marker['current'])}, expected {layers}"
                    )
                steps.append(np.stack([
                  marker["current"][i] for i in layers
                ]))
            routes=np.stack(steps)  # [8,23,6]
            h4a=block_stats(routes[:4])
            h4b=block_stats(routes[4:])
            h8=block_stats(routes)
            two_h4=(
              sum(x["unique_experts"] for x in h4a["layers"])
              +sum(x["unique_experts"] for x in h4b["layers"])
            )
            h8u=sum(x["unique_experts"] for x in h8["layers"])
            records.append({
              "label":label,"prompt":text,
              "prompt_token_count":len(ids),
              "h4_first":h4a,"h4_second":h4b,"h8":h8,
              "h8_over_two_h4_streams":h8u/float(two_h4),
              "h8_additional_stream_reduction_fraction":1.0-h8u/float(two_h4),
            })
            print(
              f"P24G {label}: H4 repeat="
              f"{(h4a['median_repeat_rate']+h4b['median_repeat_rate'])/2:.3f} "
              f"H8/2H4={h8u/float(two_h4):.3f}",
              flush=True,
            )

        h4_rep=[
          (r["h4_first"]["median_repeat_rate"]
           +r["h4_second"]["median_repeat_rate"])/2.0
          for r in records
        ]
        h8_ratio=[r["h8_over_two_h4_streams"] for r in records]
        valid=all(
          r["h8"]["max_m"]<=8
          and r["h4_first"]["max_m"]<=4
          and r["h4_second"]["max_m"]<=4
          for r in records
        )
        payload.update({
          "status":"measured","records":records,
          "summary":{
            "prompt_count":len(records),
            "h4_repeat_median_across_prompts":float(np.median(h4_rep)),
            "h4_repeat_min_across_prompts":float(np.min(h4_rep)),
            "h4_repeat_max_across_prompts":float(np.max(h4_rep)),
            "h8_over_two_h4_median":float(np.median(h8_ratio)),
            "h8_over_two_h4_worst":float(np.max(h8_ratio)),
            "h8_additional_reuse_median":float(
              1.0-np.median(h8_ratio)
            ),
            "route_multiplicity_valid":valid,
          },
          "PROMPT_ROUTE_GENERALIZATION_GREEN":bool(
            valid and np.median(h4_rep)>=0.15
          ),
          "completed_utc":utc_now(),
        })
        rt._moe_dev=original
    except Exception as exc:
        payload.update({"status":"technical_failure",
          "error":{"type":type(exc).__name__,"message":str(exc),
                   "traceback":traceback.format_exc()},
          "completed_utc":utc_now()})
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass

    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),"summary":payload.get("summary"),
      "PROMPT_ROUTE_GENERALIZATION_GREEN":payload.get(
        "PROMPT_ROUTE_GENERALIZATION_GREEN"
      ),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
