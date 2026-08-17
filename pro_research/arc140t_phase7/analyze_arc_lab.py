from __future__ import annotations
import json, math, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"results"/"arc140t_phase7"
OUT=R/"S100_PHASE7_ARC_SUMMARY.json"
TXT=R/"S100_PHASE7_ARC_SUMMARY.txt"

def load(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return None

def collect_bench():
    rows=[]
    bdir=R/"llama"
    if not bdir.exists(): return rows
    for p in bdir.rglob("*.json"):
        try:
            d=json.loads(p.read_text(encoding="utf-8"))
            seq=d if isinstance(d,list) else d.get("results",[]) if isinstance(d,dict) else []
            for x in seq:
                if not isinstance(x,dict): continue
                ts=x.get("avg_ts")
                if ts is None: continue
                rows.append({"file":str(p.relative_to(R)),"label":p.stem,"avg_ts":float(ts),
                             "n_prompt":x.get("n_prompt"),"n_gen":x.get("n_gen"),
                             "n_depth":x.get("n_depth"),"backend":x.get("backends") or x.get("backend"),
                             "gpu_info":x.get("gpu_info"),"split_mode":x.get("split_mode"),
                             "tensor_split":x.get("tensor_split"),"type_k":x.get("type_k"),"type_v":x.get("type_v")})
        except Exception: pass
    return rows

def best(rows,pred):
    x=[r for r in rows if pred(r)]
    return max(x,key=lambda r:r["avg_ts"]) if x else None

def interp_transfer(transfer,n):
    if not transfer:return None
    rows=transfer.get("rows",[])
    if not rows:return None
    q=min(rows,key=lambda r:abs(int(r["bytes"])-n))
    return q

def main():
    transfer=load(R/"cuda_transfer.json")
    ram=load(R/"ram_bandwidth.json")
    ov=load(R/"openvino_arc_probe.json")
    inv=load(R/"inventory.json")
    llama=collect_bench()
    current=load(R/"s100_current.json") or {}
    qfast_ms=float(current.get("qfast_ms",18.75165))
    qfast_tps=1000/qfast_ms
    best_tg=best(llama,lambda r:(r.get("n_gen") or 0)>0 and (r.get("n_prompt") or 0)==0)
    best_pp=best(llama,lambda r:(r.get("n_prompt") or 0)>0 and (r.get("n_gen") or 0)==0)

    ov_records={r["label"]:r for r in (ov or {}).get("records",[]) if r.get("status")=="measured"}
    expert_arc=ov_records.get("expert_down_f16")
    expert_arc_m6=ov_records.get("expert_down_M6_f16")
    small_xfer=interp_transfer(transfer,65536)
    large_xfer=interp_transfer(transfer,16777216)

    hypotheses=[]
    def add(name,status,evidence,priority):
        hypotheses.append({"name":name,"status":status,"evidence":evidence,"priority":priority})

    if expert_arc_m6:
        # Geometry proxy only: 23 MoE layers if used everywhere.
        per=float(expert_arc_m6["median_ms"])
        add("Arc cold-expert compute","promising" if per<0.35 else "weak_or_unclear",
            {"openvino_fp16_M6_ms":per,"openvino_i8_M6_ms":(ov_records.get("expert_down_M6_i8") or {}).get("median_ms"),"note":"FP16 geometry proxy, not NVFP4/current-model speed"},1)
    else:
        add("Arc cold-expert compute","unmeasured",{"reason":"OpenVINO GPU projection probe unavailable"},1)

    if large_xfer:
        add("Arc DRAM-side sparse coalescer","measure_next",
            {"rtx_h2d_16MiB_ms":large_xfer["h2d"]["median_ms"],
             "note":"Win condition is Arc pack time + larger H2D < current sparse gather/sync"},2)
    else:
        add("Arc DRAM-side sparse coalescer","unmeasured",{},2)

    add("Hot RTX + cold Arc KV","long_context_only",
        {"note":"Requires split online-softmax attention; likely capacity/long-context role, not short-context S100 primary"},4)

    arc_tg=best(llama,lambda r:(r.get("n_gen") or 0)>0 and "arc" in (r.get("label","").lower()))
    if arc_tg:
        add("Arc draft engine + RTX verifier","measure_acceptance",
            {"arc_endpoint_tg":arc_tg["avg_ts"],"target_qfast_tg":qfast_tps,
             "note":"Useful only with high acceptance and cheap multi-token target verification"},3)
    else:
        add("Arc draft engine + RTX verifier","unmeasured",{"reason":"No Arc GGUF endpoint result"},3)

    if best_tg:
        add("CUDA+Arc layer split","measured",{"best_tg":best_tg},5)
    else:
        add("CUDA+Arc layer split","unmeasured",{"reason":"No compatible GGUF/binary"},5)

    add("Cross-vendor tensor parallel","negative_control",
        {"note":"Not primary; current Nemotron-H-MoE tensor mode unsupported upstream and reductions are interconnect-sensitive"},9)

    add("ATSInfer-style tensor scheduler","research_priority",
        {"tiers":["RTX_VRAM","ARC_UMA","CPU_RAM"],"requires":["per-op timing","transfer timing","sync timing","cache-miss rates"],
         "note":"Different from llama.cpp split-mode tensor"},2)

    payload={
        "kind":"s100_phase7_arc_summary","qfast_reference":{"ms":qfast_ms,"tok_s":qfast_tps},
        "inventory":inv,"cuda_transfer":transfer,"ram":ram,"openvino":ov,
        "llama_rows":llama,"best_tg":best_tg,"best_pp":best_pp,
        "hypotheses":sorted(hypotheses,key=lambda x:x["priority"]),
        "next_implementation_trigger":(
            "If Arc expert M6 geometry and boundary economics beat current downflow, implement Arc cold-miss engine; "
            "otherwise use Arc as coalescer/draft/long-context tier and keep short-context decode on RTX."
        )
    }
    R.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    lines=["S100 PHASE 7 — ARC 140T HETEROGENEOUS LAB",
           f"QFAST reference: {qfast_ms:.5f} ms = {qfast_tps:.3f} tok/s",
           f"Best llama TG row: {best_tg}",
           f"Best llama PP row: {best_pp}","",
           "HYPOTHESES"]
    for h in payload["hypotheses"]:
        lines.append(f"- P{h['priority']} {h['name']}: {h['status']} :: {h['evidence']}")
    lines+=["", "NEXT:", payload["next_implementation_trigger"]]
    TXT.write_text("\n".join(lines)+"\n",encoding="utf-8");print("\n".join(lines))
if __name__=="__main__": main()
