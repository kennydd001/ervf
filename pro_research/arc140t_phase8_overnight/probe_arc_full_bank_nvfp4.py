"""Full-bank Arc NVFP4 cold/warm routed-down pressure benchmark."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import traceback
from pathlib import Path

import numpy as np

KERNEL = r"""
__kernel void routed_down_bank_nvfp4(
    __global const uchar* bank,
    __global const int* expert_ids,
    __global const float* act,
    __global const uint* masks,
    __global const float* globals_all,
    __global const float* route_w,
    __global const float* e2,
    __global const float* e4,
    __global float* out,
    const int nexpert,
    const int rows,
    const int inter,
    const ulong panel_bytes)
{
    int row=(int)get_global_id(0);
    if(row>=rows) return;
    int rowhalf=rows>>1;
    int npanel=inter>>4;
    ulong panel_stride=(ulong)rows + (ulong)16*(ulong)rowhalf;
    int hb=row>>1;
    int hi=row&1;
    float dst=0.0f;
    for(int s=0;s<nexpert;++s){
        int eid=expert_ids[s];
        __global const uchar* rec=bank + (ulong)eid*panel_bytes;
        __global const float* a=act + (ulong)s*inter;
        __global const uint* mkbase=masks + (ulong)s*npanel;
        float g=globals_all[eid];
        float chunks[8]={0,0,0,0,0,0,0,0};
        for(int chunk=0;chunk<8;++chunk){
            float acc=0.0f;
            for(int p=chunk;p<npanel;p+=8){
                uint mk=mkbase[p];
                if(!mk) continue;
                __global const uchar* pbase=rec+(ulong)p*panel_stride;
                float sc=e4[(uint)pbase[row]]*g;
                __global const uchar* pcodes=pbase+rows;
                for(int c=0;c<16;++c){
                    if(!(mk&(1u<<c))) continue;
                    uchar byte=pcodes[(ulong)c*rowhalf+hb];
                    uint q=hi ? ((uint)byte>>4) : ((uint)byte&15u);
                    acc=fma(e2[q]*sc,a[(p<<4)+c],acc);
                }
            }
            chunks[chunk]=acc;
        }
        float v=0.0f;
        for(int c=0;c<8;++c) v+=chunks[c];
        dst=fma(v,route_w[s],dst);
    }
    out[row]=dst;
}
__kernel void cache_scrub(__global uint* p,const uint n){
    uint i=(uint)get_global_id(0); if(i<n){uint v=p[i]; p[i]=v*1664525u+1013904223u;}
}
"""


def cpu_ref(records, ids, act, masks, globals_all, route, e2, e4, rows, inter, panel_bytes):
    rowhalf = rows // 2
    npanel = inter // 16
    stride = rows + 16 * rowhalf
    ridx = np.arange(rows, dtype=np.int32)
    hb = ridx >> 1
    hi = ridx & 1
    dst = np.zeros(rows, dtype=np.float32)
    for s, eid in enumerate(ids):
        rec = records[int(eid)]
        contrib = np.zeros(rows, dtype=np.float32)
        for chunk in range(8):
            acc = np.zeros(rows, dtype=np.float32)
            for p in range(chunk, npanel, 8):
                mk = int(masks[s, p])
                if not mk:
                    continue
                off = p * stride
                scale = e4[rec[off:off+rows].astype(np.int32)] * np.float32(globals_all[int(eid)])
                codes = rec[off+rows:off+stride]
                for c in range(16):
                    if not (mk & (1 << c)):
                        continue
                    by = codes[c*rowhalf:(c+1)*rowhalf]
                    q = np.where(hi == 0, by[hb] & 15, by[hb] >> 4).astype(np.int32)
                    acc = np.add(acc, e2[q] * scale * np.float32(act[s, p*16+c]), dtype=np.float32)
            contrib = np.add(contrib, acc, dtype=np.float32)
        dst = np.add(dst, contrib * np.float32(route[s]), dtype=np.float32)
    return dst


def metrics(ref, got):
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(got, dtype=np.float64)
    d = b - a
    rmse = float(np.sqrt(np.mean(d*d)))
    den = float(np.sqrt(np.mean(a*a))) + 1e-30
    return {
        "finite": bool(np.isfinite(b).all()),
        "cosine": float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30)),
        "nrmse": rmse/den,
        "max_abs": float(np.max(np.abs(d))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--route-sets", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    d = Path(args.bank_dir)
    out = Path(args.out)
    p = {"kind":"s100_p8_overnight_arc_full_bank","status":"started","sets":[]}
    try:
        import pyopencl as cl
        meta = json.loads((d/"meta.json").read_text())
        records = np.load(d/"records.npy", mmap_mode="r")
        globals_all = np.load(d/"globals.npy")
        act = np.load(d/"act.npy")
        masks = np.load(d/"masks.npy")
        route = np.load(d/"route_w.npy")
        e2 = np.load(d/"e2m1.npy")
        e4 = np.load(d/"e4m3.npy")
        rows=int(meta["hidden"]); inter=int(meta["moe_inter"]); ne=int(meta["top_k"])
        n_experts=int(meta["n_experts"]); panel_bytes=int(meta["down_panel_bytes"])
        devs=[]
        for plat in cl.get_platforms():
            for dev in plat.get_devices():
                if (dev.type & cl.device_type.GPU) and "intel" in (dev.vendor+" "+dev.name).lower():
                    devs.append((plat,dev))
        if not devs: raise RuntimeError("Intel OpenCL GPU not found")
        plat,dev=devs[0];ctx=cl.Context([dev]);q=cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
        prg=cl.Program(ctx,KERNEL).build(options=["-cl-std=CL2.0"]);krn=prg.routed_down_bank_nvfp4
        mf=cl.mem_flags
        # One persistent full bank: this is the proposed steady-state Arc residency tier.
        t0=time.perf_counter();bankb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.asarray(records));q.finish();
        p["bank_upload_ms"]=(time.perf_counter()-t0)*1e3
        p["bank_bytes"]=int(records.nbytes);p["device"]={"name":dev.name,"vendor":dev.vendor,"global_mem":int(dev.global_mem_size)}
        ab=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.asarray(act,dtype=np.float32));mb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.asarray(masks,dtype=np.uint32));gb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.asarray(globals_all,dtype=np.float32));wb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.asarray(route,dtype=np.float32));e2b=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.asarray(e2,dtype=np.float32));e4b=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.asarray(e4,dtype=np.float32));outb=cl.Buffer(ctx,mf.WRITE_ONLY,rows*4);idb=cl.Buffer(ctx,mf.READ_ONLY,ne*4)
        scrub_n=(128*1024*1024)//4;scrub=np.arange(scrub_n,dtype=np.uint32);scrubb=cl.Buffer(ctx,mf.READ_WRITE|mf.COPY_HOST_PTR,hostbuf=scrub);scrub_g=((scrub_n+255)//256*256,)

        # Tune workgroup on rotating routes, not a cache-hot single expert set.
        test_ids=np.array([(7+j*17)%n_experts for j in range(ne)],dtype=np.int32);cl.enqueue_copy(q,idb,test_ids).wait()
        tune=[]
        for local in (64,128,256):
            gs=((rows+local-1)//local*local,);vals=[]
            for _ in range(8):
                prg.cache_scrub(q,scrub_g,(256,),scrubb,np.uint32(scrub_n)).wait()
                ev=krn(q,gs,(local,),bankb,idb,ab,mb,gb,wb,e2b,e4b,outb,np.int32(ne),np.int32(rows),np.int32(inter),np.uint64(panel_bytes));ev.wait();vals.append((ev.profile.end-ev.profile.start)/1e6)
            tune.append({"local":local,"median_ms":statistics.median(vals)})
        local=min(tune,key=lambda x:x["median_ms"])["local"];p["local_tuning"]=tune;p["selected_local"]=local;gs=((rows+local-1)//local*local,)

        rng=np.random.default_rng(20260817)
        route_sets=[]
        # First set is the actual model route; remaining sets rotate across the full bank.
        route_sets.append(np.asarray(meta["actual_ids"],dtype=np.int32))
        while len(route_sets)<max(8,int(args.route_sets)):
            route_sets.append(rng.choice(n_experts,size=ne,replace=False).astype(np.int32))

        # Correctness on several widely separated expert sets.
        correct_indices=sorted(set([0,1,len(route_sets)//3,2*len(route_sets)//3,len(route_sets)-1]))
        correctness=[]
        for idx in correct_indices:
            ids=route_sets[idx];cl.enqueue_copy(q,idb,ids).wait();ev=krn(q,gs,(local,),bankb,idb,ab,mb,gb,wb,e2b,e4b,outb,np.int32(ne),np.int32(rows),np.int32(inter),np.uint64(panel_bytes));ev.wait();got=np.empty(rows,np.float32);cl.enqueue_copy(q,got,outb).wait();ref=cpu_ref(records,ids,act,masks,globals_all,route,e2,e4,rows,inter,panel_bytes);correctness.append({"set_index":idx,"ids":ids.tolist(),**metrics(ref,got)})
        p["correctness"]=correctness

        cold=[]
        reps=max(3,int(args.reps))
        for si,ids in enumerate(route_sets):
            cl.enqueue_copy(q,idb,ids).wait();vals=[]
            for _ in range(reps):
                # Evict device caches; scrub event excluded from routed-kernel timing.
                prg.cache_scrub(q,scrub_g,(256,),scrubb,np.uint32(scrub_n)).wait()
                ev=krn(q,gs,(local,),bankb,idb,ab,mb,gb,wb,e2b,e4b,outb,np.int32(ne),np.int32(rows),np.int32(inter),np.uint64(panel_bytes));ev.wait();vals.append((ev.profile.end-ev.profile.start)/1e6)
            med=statistics.median(vals);cold.append(med)
            p["sets"].append({"index":si,"ids":ids.tolist(),"median_ms":med,"p95_ms":float(np.percentile(vals,95))})
            if (si+1)%16==0: print(f"full-bank cold sets {si+1}/{len(route_sets)} median={statistics.median(cold):.4f} ms",flush=True)

        # Warm repeated actual route to quantify cache optimism directly.
        ids=route_sets[0];cl.enqueue_copy(q,idb,ids).wait();warm=[]
        for _ in range(100):
            ev=krn(q,gs,(local,),bankb,idb,ab,mb,gb,wb,e2b,e4b,outb,np.int32(ne),np.int32(rows),np.int32(inter),np.uint64(panel_bytes));ev.wait();warm.append((ev.profile.end-ev.profile.start)/1e6)
        p["cold_summary"]={"sets":len(cold),"median_ms":statistics.median(cold),"p95_ms":float(np.percentile(cold,95)),"min_ms":min(cold),"max_ms":max(cold)}
        p["warm_actual_route"]={"median_ms":statistics.median(warm),"p95_ms":float(np.percentile(warm,95))}
        p["cold_over_warm"]=p["cold_summary"]["median_ms"]/p["warm_actual_route"]["median_ms"]
        p["status"]="measured"
    except Exception as exc:
        p.update({"status":"technical_failure","error":{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}})
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(p,indent=2,allow_nan=False)+"\n",encoding="utf-8");print(json.dumps({"status":p.get("status"),"bank_bytes":p.get("bank_bytes"),"cold_summary":p.get("cold_summary"),"warm":p.get("warm_actual_route"),"cold_over_warm":p.get("cold_over_warm"),"error":(p.get("error") or {}).get("message"),"output":str(out)},indent=2));return 0 if p.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
