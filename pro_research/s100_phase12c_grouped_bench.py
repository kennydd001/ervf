from __future__ import annotations

from collections import Counter
import json
import statistics
import traceback

import numpy as np

from common import REPO, write_json_atomic, utc_now
from down_proj_batch_kernels import DownProjBatchKernels
from scale_resident_kernels import (
    ScaleResidentKernels, DOWN_PANEL_BYTES, PLANE_BYTES,
    PANEL_STRIDE, HIDDEN, INTER, NPANEL,
)
from s100_phase10a_runtime import build
from moe_dev_batched import UP_CODE, UP_SCALE
from s100_phase12c_ervfm_kernels import ERVFM
from s100_phase12c_grouped_kernels import GroupedDown, MS

OUT = (
    REPO / "pro_research" / "results" / "s100_phase12c"
    / "S100_PHASE12C_GROUPED_MOE.json"
)
TRACE = (
    REPO / "pro_research" / "results" / "s100_phase9"
    / "S100_PHASE9_TRACE.npz"
)
CENSUS = (
    REPO / "pro_research" / "results" / "s100_phase12"
    / "S100_PHASE12B_CENSUS.json"
)
N_RECORDS = 48
DENSITY = 0.09

def pick_pairs():
    with np.load(TRACE) as d:
        ids=d["ids"].astype(np.int32)
        counted=d["counted"].astype(bool)
        layers=[int(x) for x in d["layers"]]
    counts=Counter()
    for token in np.nonzero(counted)[0]:
        for li,layer in enumerate(layers):
            for expert in ids[token,li]:
                counts[(layer,int(expert))]+=1
    return [pair for pair,_ in sorted(
        counts.items(), key=lambda kv:(-kv[1],kv[0])
    )[:N_RECORDS]]

def extract_plane(record: np.ndarray) -> np.ndarray:
    out=np.empty(PLANE_BYTES,dtype=np.uint8)
    for p in range(NPANEL):
        out[p*HIDDEN:(p+1)*HIDDEN] = \
            record[p*PANEL_STRIDE:p*PANEL_STRIDE+HIDDEN]
    return out

def main() -> int:
    payload={
        "kind":"s100_phase12c_grouped_moe",
        "status":"started",
        "records":N_RECORDS,
        "activation_density":DENSITY,
        "started_utc":utc_now(),
    }
    try:
        import cupy as cp

        parent=build()
        rt=parent.rt
        ervfm=ERVFM()
        grouped_down=GroupedDown()
        down_ref=ScaleResidentKernels()
        batch=DownProjBatchKernels()

        pairs=pick_pairs()
        records=[]
        for layer,expert in pairs:
            bank=rt.bank[int(layer)]
            up_c=np.ascontiguousarray(
                bank["up_codes"][expert*UP_CODE:(expert+1)*UP_CODE]
            )
            up_s=np.ascontiguousarray(
                bank["up_scales"][expert*UP_SCALE:(expert+1)*UP_SCALE]
            )
            down_h=np.ascontiguousarray(
                bank["down_pm"][expert*DOWN_PANEL_BYTES:
                                (expert+1)*DOWN_PANEL_BYTES]
            )
            records.append({
                "layer":int(layer),"expert":int(expert),
                "up_codes":cp.asarray(up_c),
                "up_scales":cp.asarray(up_s),
                "down_record":cp.asarray(down_h),
                "plane":cp.asarray(extract_plane(down_h)),
                "g_up":float(bank["globals"][expert,1]),
                "g_down":float(bank["globals"][expert,0]),
            })

        props=cp.cuda.runtime.getDeviceProperties(0)
        l2=int(props.get("l2CacheSize",32*1024**2))
        up_rotation=sum(
            int(r["up_codes"].nbytes+r["up_scales"].nbytes)
            for r in records
        )
        down_rotation=sum(int(r["down_record"].nbytes) for r in records)

        rng=np.random.default_rng(20260818)
        census=json.loads(CENSUS.read_text(encoding="utf-8"))
        results={}

        for m in MS:
            x=cp.asarray(
                rng.standard_normal((m,rt.hidden)).astype(np.float32)
            )
            act_h=np.zeros((m,INTER),np.float32)
            for mm in range(m):
                nz=rng.choice(
                    INTER, size=max(1,int(round(INTER*DENSITY))),
                    replace=False,
                )
                act_h[mm,nz]=np.square(
                    np.abs(rng.standard_normal(nz.size).astype(np.float32))
                )
            acts=cp.asarray(act_h)
            masks,plist,pcount,nzlist,nzcount = \
                batch.run_panel_scan_batched(acts.reshape(-1),INTER,m)

            up_ref_outputs=[
                cp.empty((m,INTER),cp.float32) for _ in records
            ]
            up_cand_outputs=[
                cp.empty((m,INTER),cp.float32) for _ in records
            ]
            down_ref_outputs=[
                cp.empty((m,HIDDEN),cp.float32) for _ in records
            ]
            down_cand_outputs=[
                cp.empty((m,HIDDEN),cp.float32) for _ in records
            ]

            slot=cp.asarray([0],dtype=cp.int32)
            eid=cp.asarray([0],dtype=cp.int32)
            nchunks=int(rt.fused.nchunks)
            down_ref_partials=[
                cp.empty((m,nchunks,HIDDEN),cp.float32)
                for _ in records
            ]
            down_cand_partials=[
                cp.empty((m,nchunks,HIDDEN),cp.float32)
                for _ in records
            ]

            def up_ref_one(rec, out):
                for mm in range(m):
                    rt.fused.gemv_into(
                        out[mm],rec["up_codes"],rec["up_scales"],
                        x[mm],rec["g_up"],INTER,rt.hidden,
                        apply_relu2=True,
                    )

            def up_cand_one(rec,out):
                ervfm.run(
                    "nvfp4_relu2",m,out,rec["up_codes"],x,
                    INTER,rt.hidden,scale=rec["g_up"],
                    scales=rec["up_scales"],
                    e2=rt.fused.e2m1,e4=rt.fused.e4m3,
                )

            def down_ref_one(rec,out,index):
                partial=down_ref_partials[index]
                for mm in range(m):
                    down_ref.down_masked_sres(
                        ((HIDDEN+127)//128,nchunks),
                        rec["down_record"],rec["plane"],slot,eid,
                        rec["globals_dev"],acts[mm],
                        plist[mm*NPANEL:(mm+1)*NPANEL],
                        masks[mm*NPANEL:(mm+1)*NPANEL],
                        pcount[mm:mm+1],
                        rt.fused.e2m1,rt.fused.e4m3,
                        partial[mm].reshape(-1),HIDDEN,INTER,
                    )
                    batch.reduce_partials_ref(
                        ((HIDDEN+255)//256,), (256,),
                        (
                            partial[mm].reshape(-1), out[mm],
                            np.int32(HIDDEN), np.int32(nchunks),
                        ),
                    )

            def down_cand_one(rec,out,index):
                partial=down_cand_partials[index]
                grouped_down.run(
                    m,rec["down_record"],rec["plane"],acts,
                    masks,rt.fused.e2m1,rt.fused.e4m3,
                    rec["g_down"],partial.reshape(-1),nchunks,
                )
                batch.reduce_partials_batched(
                    ((HIDDEN+255)//256,m), (256,),
                    (
                        partial.reshape(-1), out.reshape(-1),
                        np.int32(HIDDEN), np.int32(nchunks),
                    ),
                )

            exact_fail=[]
            for i,rec in enumerate(records):
                up_ref_one(rec,up_ref_outputs[i])
                up_cand_one(rec,up_cand_outputs[i])
                down_ref_one(rec,down_ref_outputs[i],i)
                down_cand_one(rec,down_cand_outputs[i],i)
                cp.cuda.Stream.null.synchronize()
                if not bool(cp.array_equal(
                    up_ref_outputs[i],up_cand_outputs[i]
                )):
                    d=cp.abs(up_ref_outputs[i]-up_cand_outputs[i])
                    exact_fail.append({
                        "pair":[rec["layer"],rec["expert"]],
                        "part":"up",
                        "max_abs":float(cp.max(d).item()),
                        "count":int(cp.count_nonzero(d).item()),
                    })
                if not bool(cp.array_equal(
                    down_ref_outputs[i],down_cand_outputs[i]
                )):
                    d=cp.abs(down_ref_outputs[i]-down_cand_outputs[i])
                    exact_fail.append({
                        "pair":[rec["layer"],rec["expert"]],
                        "part":"down",
                        "max_abs":float(cp.max(d).item()),
                        "count":int(cp.count_nonzero(d).item()),
                    })

            def measure(fn,reps=14):
                for _ in range(2):
                    for i,rec in enumerate(records):
                        fn(rec,i)
                cp.cuda.Stream.null.synchronize()
                vals=[]
                for _ in range(reps):
                    a=cp.cuda.Event();b=cp.cuda.Event();a.record()
                    for i,rec in enumerate(records):
                        fn(rec,i)
                    b.record();b.synchronize()
                    vals.append(float(cp.cuda.get_elapsed_time(a,b)))
                return {
                    "median_ms":statistics.median(vals),
                    "p10_ms":float(np.percentile(vals,10)),
                    "p90_ms":float(np.percentile(vals,90)),
                    "raw_ms":vals,
                }

            ur=measure(lambda r,i:up_ref_one(r,up_ref_outputs[i]))
            uc=measure(lambda r,i:up_cand_one(r,up_cand_outputs[i]))
            dr=measure(lambda r,i:down_ref_one(r,down_ref_outputs[i],i))
            dc=measure(lambda r,i:down_cand_one(r,down_cand_outputs[i],i))

            results[str(m)]={
                "exact":not exact_fail,
                "exact_failures":exact_fail,
                "up":{
                    "independent_m1":ur,"grouped":uc,
                    "speedup":ur["median_ms"]/uc["median_ms"],
                },
                "down":{
                    "independent_m1":dr,"grouped":dc,
                    "speedup":dr["median_ms"]/dc["median_ms"],
                },
                "combined":{
                    "independent_m1_ms":ur["median_ms"]+dr["median_ms"],
                    "grouped_ms":uc["median_ms"]+dc["median_ms"],
                    "speedup":(
                        (ur["median_ms"]+dr["median_ms"])
                        /(uc["median_ms"]+dc["median_ms"])
                    ),
                },
            }

        weighted={}
        for B in (4,8):
            hist=census["per_B"][str(B)]["rows_per_expert_hist"]
            current=candidate=0.0
            groups=0
            for m,count in enumerate(hist,start=1):
                if count<=0:
                    continue
                key=str(m)
                if key not in results:
                    continue
                current += count*results[key]["combined"]["independent_m1_ms"]
                candidate += count*results[key]["combined"]["grouped_ms"]
                groups += count
            weighted[str(B)]={
                "histogram":hist,
                "covered_groups":groups,
                "weighted_speedup":current/candidate if candidate else None,
                "current_weighted_units_ms":current,
                "candidate_weighted_units_ms":candidate,
            }

        exact_all=all(v["exact"] for v in results.values())
        m1_penalty=(
            results["1"]["combined"]["grouped_ms"]
            /results["1"]["combined"]["independent_m1_ms"]-1.0
        )
        b4_speed=weighted["4"]["weighted_speedup"]
        gate=bool(
            exact_all
            and up_rotation>=4*l2
            and down_rotation>=4*l2
            and m1_penalty<=0.15
            and b4_speed is not None
            and b4_speed>=1.20
        )

        payload.update({
            "status":"measured",
            "pairs":[[r["layer"],r["expert"]] for r in records],
            "l2_bytes":l2,
            "up_rotation_bytes":up_rotation,
            "down_rotation_bytes":down_rotation,
            "up_rotation_over_l2":up_rotation/l2,
            "down_rotation_over_l2":down_rotation/l2,
            "per_m":results,
            "weighted":weighted,
            "m1_penalty_fraction":m1_penalty,
            "grouped_b4_gate_pass":gate,
            "completed_utc":utc_now(),
        })
        parent.restore_combined()
        parent.restore_sel()
    except Exception as exc:
        payload.update({
            "status":"technical_failure",
            "error":{
                "type":type(exc).__name__,
                "message":str(exc),
                "traceback":traceback.format_exc(),
            },
            "completed_utc":utc_now(),
        })

    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({
        "status":payload.get("status"),
        "rotation":{
            "up":payload.get("up_rotation_over_l2"),
            "down":payload.get("down_rotation_over_l2"),
        },
        "per_m":{
            k:{
                "exact":v.get("exact"),
                "up_speedup":(v.get("up") or {}).get("speedup"),
                "down_speedup":(v.get("down") or {}).get("speedup"),
                "combined_speedup":(v.get("combined") or {}).get("speedup"),
            }
            for k,v in payload.get("per_m",{}).items()
        },
        "weighted":payload.get("weighted"),
        "m1_penalty":payload.get("m1_penalty_fraction"),
        "gate":payload.get("grouped_b4_gate_pass"),
        "error":(payload.get("error") or {}).get("message"),
        "output":str(OUT),
    },indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
