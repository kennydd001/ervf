from __future__ import annotations

from collections import defaultdict
import json
import statistics
import traceback

import numpy as np

from common import utc_now,write_json_atomic
from s100_phase20b_kernels import Phase20BKernels
from s100_phase21_common import load_trace,prefill_to,expected_for_block,release
from s100_phase23_common import GPUGroupedMoEH4
from s100_phase24_common import RESULTS,make_v6
from s100_phase24_dense_kernels import DenseM4Kernels

OUT=RESULTS/"S100_PHASE24_COMPONENTS.json"
CTX=1024

def timing(cp,fn,reps=10):
    for _ in range(2):fn()
    cp.cuda.get_current_stream().synchronize()
    vals=[]
    for _ in range(reps):
        a=cp.cuda.Event();b=cp.cuda.Event()
        a.record();fn();b.record();b.synchronize()
        vals.append(float(cp.cuda.get_elapsed_time(a,b)))
    return {
      "median_ms":float(statistics.median(vals)),
      "p10_ms":float(np.percentile(vals,10)),
      "p90_ms":float(np.percentile(vals,90)),
      "raw_ms":vals,
    }

def equal_info(cp,a,b):
    cp.cuda.get_current_stream().synchronize()
    eq=bool(cp.array_equal(a,b).item())
    an=cp.asnumpy(a).astype(np.float64)
    bn=cp.asnumpy(b).astype(np.float64)
    return {
      "bit_exact":eq,
      "nrmse":float(np.linalg.norm(an-bn)/max(np.linalg.norm(an),1e-30)),
      "max_abs":float(np.max(np.abs(an-bn))),
      "finite":bool(np.isfinite(an).all() and np.isfinite(bn).all()),
    }

class CaptureMoE:
    def __init__(self,inner,captured):
        self.inner=inner;self.captured=captured
        self.k=getattr(inner,"k",None)
    def __call__(self,layer,normed,out,collect_stats=False):
        self.captured[int(layer)]=normed.copy()
        return self.inner(layer,normed,out,collect_stats)

def main():
    payload={"kind":"s100_phase24_components","status":"started",
      "context":CTX,"started_utc":utc_now(),
      "claim_boundary":"real H4 exact component screens"}
    rt=None
    try:
        import cupy as cp
        from s100_phase22_common import eager_verifier,selected_head_mode

        tr=load_trace();tokens=tr["tokens"]
        rt,keep=make_v6(CTX)
        v=eager_verifier(rt,selected_head_mode())
        base_moe=GPUGroupedMoEH4(rt)
        captured_moe={}
        v.moeb=CaptureMoE(base_moe,captured_moe)

        # Map exact production matrices to readable cases.
        bf_cases={}
        f_cases={}
        for li in rt.attn_layers:
            i=int(li);d=rt.layer[i]
            hq=int(rt.n_heads*rt.head_dim)
            specs=(
              ("q",d["q_proj"],hq,int(rt.hidden)),
              ("k",d["k_proj"],int(rt.kv_dim),int(rt.hidden)),
              ("v",d["v_proj"],int(rt.kv_dim),int(rt.hidden)),
              ("o",d["o_proj"],int(rt.hidden),hq),
            )
            for side,W,rows,cols in specs:
                bf_cases[int(W.data.ptr)]={
                  "name":f"attention_{i}_{side}","layer":i,"side":side,
                  "W":W,"rows":rows,"cols":cols,"x":[],"out":[],
                }
        for li in rt.moe_layers:
            i=int(li);d=rt.layer[i];W=d["gate_w"]
            f_cases[int(W.data.ptr)]={
              "name":f"router_{i}","layer":i,"W":W,
              "rows":int(rt.n_experts),"cols":int(rt.hidden),
              "x":[],"out":[],
            }

        orig_bf=rt.k.mv_bf16
        orig_f=rt.k.mv_f32

        def cap_bf(out,W,x,rows,cols):
            result=orig_bf(out,W,x,rows,cols)
            rec=bf_cases.get(int(W.data.ptr))
            if rec is not None:
                rec["x"].append(x.copy());rec["out"].append(out.copy())
            return result

        def cap_f(out,W,x,rows,cols):
            result=orig_f(out,W,x,rows,cols)
            rec=f_cases.get(int(W.data.ptr))
            if rec is not None:
                rec["x"].append(x.copy());rec["out"].append(out.copy())
            return result

        # Canonical prefill must not be captured: only the one real H4 block
        # supplies the four rows per matrix.
        prefill_to(rt,tokens,CTX)
        rt.k.mv_bf16=cap_bf
        rt.k.mv_f32=cap_f
        draft,expected=expected_for_block(tokens,CTX)
        got,_=v.block(draft.tolist(),False)
        cp.cuda.get_current_stream().synchronize()
        rt.k.mv_bf16=orig_bf
        rt.k.mv_f32=orig_f
        if not np.array_equal(got,expected):
            raise RuntimeError("component capture block diverged")

        dense=DenseM4Kernels()
        bk=Phase20BKernels()

        # --------------------------- BF16 attention M4
        bf_rows=[]
        bf_outputs={}
        for rec in bf_cases.values():
            if len(rec["x"])!=4:
                raise RuntimeError(
                  f"{rec['name']} captured {len(rec['x'])}, expected 4"
                )
            x=cp.stack(rec["x"])
            ref=cp.stack(rec["out"])
            cand=cp.empty_like(ref)
            dense.bf16(
              rec["W"],x,cand,rec["rows"],rec["cols"]
            )
            info=equal_info(cp,ref,cand)
            bf_rows.append({
              "name":rec["name"],"layer":rec["layer"],"side":rec["side"],
              "shape":[rec["rows"],rec["cols"]],"correctness":info,
            })
            bf_outputs[rec["name"]]=(x,ref,cand,rec)

        def bf_baseline():
            for x,ref,cand,rec in bf_outputs.values():
                for t in range(4):
                    orig_bf(
                      cand[t],rec["W"],x[t],rec["rows"],rec["cols"]
                    )
        def bf_candidate():
            for x,ref,cand,rec in bf_outputs.values():
                dense.bf16(
                  rec["W"],x,cand,rec["rows"],rec["cols"]
                )
        bft0=timing(cp,bf_baseline)
        bft1=timing(cp,bf_candidate)
        bf_exact=all(x["correctness"]["bit_exact"] for x in bf_rows)
        bf_speed=bft0["median_ms"]/bft1["median_ms"]
        bf_open=bool(bf_exact and bf_speed>=1.05)

        # --------------------------- FP32 router M4
        fr_rows=[];fr_outputs={}
        for rec in f_cases.values():
            if len(rec["x"])!=4:
                raise RuntimeError(
                  f"{rec['name']} captured {len(rec['x'])}, expected 4"
                )
            x=cp.stack(rec["x"]);ref=cp.stack(rec["out"])
            cand=cp.empty_like(ref)
            dense.f32(rec["W"],x,cand,rec["rows"],rec["cols"])
            info=equal_info(cp,ref,cand)

            # Route ids/weights from exact reference and candidate.
            rid_ref=cp.empty((4,rt.top_k),cp.int32)
            rw_ref=cp.empty((4,rt.top_k),cp.float32)
            rid_c=cp.empty_like(rid_ref);rw_c=cp.empty_like(rw_ref)
            d=rt.layer[rec["layer"]]
            for t in range(4):
                rt.fused.route_topk(
                  ref[t],d["gate_b"],rid_ref[t],rw_ref[t],
                  rt.n_experts,rt.top_k,rt.scaling,bad_pick=rt._bad_pick
                )
                rt.fused.route_topk(
                  cand[t],d["gate_b"],rid_c[t],rw_c[t],
                  rt.n_experts,rt.top_k,rt.scaling,bad_pick=rt._bad_pick
                )
            cp.cuda.get_current_stream().synchronize()
            info["route_ids_exact"]=bool(cp.array_equal(rid_ref,rid_c).item())
            info["route_weights_exact"]=bool(cp.array_equal(rw_ref,rw_c).item())
            fr_rows.append({
              "name":rec["name"],"layer":rec["layer"],
              "shape":[rec["rows"],rec["cols"]],"correctness":info,
            })
            fr_outputs[rec["name"]]=(x,ref,cand,rec)

        def fr_baseline():
            for x,ref,cand,rec in fr_outputs.values():
                for t in range(4):
                    orig_f(cand[t],rec["W"],x[t],rec["rows"],rec["cols"])
        def fr_candidate():
            for x,ref,cand,rec in fr_outputs.values():
                dense.f32(rec["W"],x,cand,rec["rows"],rec["cols"])
        frt0=timing(cp,fr_baseline)
        frt1=timing(cp,fr_candidate)
        fr_exact=all(
          x["correctness"]["bit_exact"]
          and x["correctness"]["route_ids_exact"]
          and x["correctness"]["route_weights_exact"]
          for x in fr_rows
        )
        fr_speed=frt0["median_ms"]/frt1["median_ms"]
        fr_open=bool(fr_exact and fr_speed>=1.05)

        # --------------------------- shared expert NVFP4 M4
        sh_rows=[];sh_data={}
        for li in rt.moe_layers:
            i=int(li);d=rt.layer[i]
            x=captured_moe[i]
            ref_up=cp.empty((4,rt.shared_inter),cp.float32)
            ref_out=cp.empty((4,rt.hidden),cp.float32)
            cand_up=cp.empty_like(ref_up);cand_out=cp.empty_like(ref_out)
            for t in range(4):
                rt.fused.gemv_into(
                  ref_up[t],d["sh_up_c"],d["sh_up_s"],x[t],
                  d["sh_up_g"],rt.shared_inter,rt.hidden,apply_relu2=True
                )
                rt.fused.gemv_into(
                  ref_out[t],d["sh_dn_c"],d["sh_dn_s"],ref_up[t],
                  d["sh_dn_g"],rt.hidden,rt.shared_inter
                )
            bk.nvfp4(
              d["sh_up_c"],d["sh_up_s"],rt.fused.e2m1,rt.fused.e4m3,
              x,cand_up,d["sh_up_g"],rt.shared_inter,rt.hidden,4,True
            )
            bk.nvfp4(
              d["sh_dn_c"],d["sh_dn_s"],rt.fused.e2m1,rt.fused.e4m3,
              cand_up,cand_out,d["sh_dn_g"],rt.hidden,rt.shared_inter,4,False
            )
            info=equal_info(cp,ref_out,cand_out)
            up_info=equal_info(cp,ref_up,cand_up)
            info["up_bit_exact"]=up_info["bit_exact"]
            info["up_nrmse"]=up_info["nrmse"]
            sh_rows.append({"layer":i,"correctness":info})
            sh_data[i]=(d,x,ref_up,ref_out,cand_up,cand_out)

        def sh_baseline():
            for d,x,ru,ro,cu,co in sh_data.values():
                for t in range(4):
                    rt.fused.gemv_into(
                      cu[t],d["sh_up_c"],d["sh_up_s"],x[t],
                      d["sh_up_g"],rt.shared_inter,rt.hidden,
                      apply_relu2=True
                    )
                    rt.fused.gemv_into(
                      co[t],d["sh_dn_c"],d["sh_dn_s"],cu[t],
                      d["sh_dn_g"],rt.hidden,rt.shared_inter
                    )
        def sh_candidate():
            for d,x,ru,ro,cu,co in sh_data.values():
                bk.nvfp4(
                  d["sh_up_c"],d["sh_up_s"],rt.fused.e2m1,rt.fused.e4m3,
                  x,cu,d["sh_up_g"],rt.shared_inter,rt.hidden,4,True
                )
                bk.nvfp4(
                  d["sh_dn_c"],d["sh_dn_s"],rt.fused.e2m1,rt.fused.e4m3,
                  cu,co,d["sh_dn_g"],rt.hidden,rt.shared_inter,4,False
                )
        sht0=timing(cp,sh_baseline)
        sht1=timing(cp,sh_candidate)
        sh_exact=all(
          x["correctness"]["bit_exact"]
          and x["correctness"]["up_bit_exact"]
          for x in sh_rows
        )
        sh_speed=sht0["median_ms"]/sht1["median_ms"]
        sh_open=bool(sh_exact and sh_speed>=1.05)

        payload.update({
          "status":"measured",
          "attention_bf16":{
            "cases":bf_rows,"baseline":bft0,"candidate":bft1,
            "aggregate_speedup":bf_speed,"all_bit_exact":bf_exact,
          },
          "router_f32":{
            "cases":fr_rows,"baseline":frt0,"candidate":frt1,
            "aggregate_speedup":fr_speed,"all_bit_exact":fr_exact,
          },
          "shared_nvfp4":{
            "cases":sh_rows,"baseline":sht0,"candidate":sht1,
            "aggregate_speedup":sh_speed,"all_bit_exact":sh_exact,
          },
          "ATTENTION_BF16_M4_OPEN":bf_open,
          "ROUTER_F32_M4_OPEN":fr_open,
          "SHARED_NVFP4_M4_OPEN":sh_open,
          "completed_utc":utc_now(),
        })
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
      "status":payload.get("status"),
      "attention":{
        "speedup":(payload.get("attention_bf16") or {}).get("aggregate_speedup"),
        "exact":(payload.get("attention_bf16") or {}).get("all_bit_exact"),
        "open":payload.get("ATTENTION_BF16_M4_OPEN"),
      },
      "router":{
        "speedup":(payload.get("router_f32") or {}).get("aggregate_speedup"),
        "exact":(payload.get("router_f32") or {}).get("all_bit_exact"),
        "open":payload.get("ROUTER_F32_M4_OPEN"),
      },
      "shared":{
        "speedup":(payload.get("shared_nvfp4") or {}).get("aggregate_speedup"),
        "exact":(payload.get("shared_nvfp4") or {}).get("all_bit_exact"),
        "open":payload.get("SHARED_NVFP4_M4_OPEN"),
      },
      "error":(payload.get("error") or {}).get("message"),
      "output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
