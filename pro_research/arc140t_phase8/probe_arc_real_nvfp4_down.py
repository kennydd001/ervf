from __future__ import annotations
import argparse,json,math,statistics,time,traceback
from pathlib import Path
import numpy as np

KERNEL=r"""
__kernel void routed_down_nvfp4(
    __global const uchar* records,
    __global const float* act,
    __global const uint* masks,
    __global const float* globals,
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
        __global const uchar* rec=records + (ulong)s*panel_bytes;
        __global const float* a=act + (ulong)s*inter;
        __global const uint* mkbase=masks + (ulong)s*npanel;
        float g=globals[s];
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
"""

def cpu_ref(records,act,masks,globals_,route,e2,e4,rows,inter,panel_bytes,nexpert):
    rowhalf=rows//2;npanel=inter//16;stride=rows+16*rowhalf
    dst=np.zeros(rows,dtype=np.float32)
    ridx=np.arange(rows,dtype=np.int32);hb=ridx>>1;hi=ridx&1
    for s in range(nexpert):
        rec=records[s]; contrib=np.zeros(rows,dtype=np.float32)
        for chunk in range(8):
            acc=np.zeros(rows,dtype=np.float32)
            for p in range(chunk,npanel,8):
                mk=int(masks[s,p])
                if not mk: continue
                off=p*stride
                scale=e4[rec[off:off+rows].astype(np.int32)]*np.float32(globals_[s])
                codes=rec[off+rows:off+stride]
                for c in range(16):
                    if not (mk&(1<<c)): continue
                    by=codes[c*rowhalf:(c+1)*rowhalf]
                    q=np.where(hi==0,by[hb]&15,by[hb]>>4).astype(np.int32)
                    w=e2[q]*scale
                    acc=np.add(acc,w*np.float32(act[s,p*16+c]),dtype=np.float32)
            contrib=np.add(contrib,acc,dtype=np.float32)
        dst=np.add(dst,contrib*np.float32(route[s]),dtype=np.float32)
    return dst

def metrics(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    err=b-a
    rmse=float(np.sqrt(np.mean(err*err)))
    denom=float(np.sqrt(np.mean(a*a)))+1e-30
    cos=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
    return {"cosine":cos,"nrmse":rmse/denom,"max_abs":float(np.max(np.abs(err))),
            "finite":bool(np.isfinite(b).all())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sample',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();sample=Path(a.sample);out=Path(a.out)
    p={"kind":"s100_phase8_real_nvfp4_arc_down","status":"started","records":[]}
    try:
        import pyopencl as cl
        z=np.load(sample,allow_pickle=False)
        meta=json.loads(str(z["meta_json"]))
        rows=int(meta["hidden"]);inter=int(meta["moe_inter"]);panel_bytes=int(meta["down_panel_bytes"])
        p["shape_contract"]=meta
        devices=[]
        for plat in cl.get_platforms():
            for d in plat.get_devices():
                if d.type & cl.device_type.GPU:
                    devices.append((plat,d))
        cand=[x for x in devices if "intel" in (x[1].vendor+" "+x[1].name).lower()]
        if not cand: raise RuntimeError("Intel OpenCL GPU not found")
        plat,dev=cand[0];ctx=cl.Context([dev])
        p["device"]={"platform":plat.name,"name":dev.name,"vendor":dev.vendor,
                     "version":dev.version,"driver":dev.driver_version,
                     "global_mem":int(dev.global_mem_size),"local_mem":int(dev.local_mem_size)}
        queue=cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
        e2=np.asarray(z["e2m1"],dtype=np.float32);e4=np.asarray(z["e4m3"],dtype=np.float32)

        for fast in (False,True):
            opts=["-cl-std=CL2.0"]
            if fast: opts.append("-cl-fast-relaxed-math")
            program=cl.Program(ctx,KERNEL).build(options=opts)
            krn=program.routed_down_nvfp4
            for lm in meta["layers"]:
                layer=int(lm["layer"]);pref=f"L{layer}"
                records=np.ascontiguousarray(z[pref+"_records"],dtype=np.uint8)
                act=np.ascontiguousarray(z[pref+"_act"],dtype=np.float32)
                masks=np.ascontiguousarray(z[pref+"_masks"],dtype=np.uint32)
                globals_=np.ascontiguousarray(z[pref+"_globals"],dtype=np.float32)
                route=np.ascontiguousarray(z[pref+"_route_w"],dtype=np.float32)
                for ne in (1,2,4,6):
                    ref=cpu_ref(records,act,masks,globals_,route,e2,e4,rows,inter,panel_bytes,ne)
                    mf=cl.mem_flags
                    rb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=records[:ne])
                    ab=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=act[:ne])
                    mb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=masks[:ne])
                    gb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=globals_[:ne])
                    wb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=route[:ne])
                    e2b=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=e2)
                    e4b=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=e4)
                    ob=cl.Buffer(ctx,mf.WRITE_ONLY,rows*4)
                    best=None
                    for local in (64,128,256):
                        global_size=((rows+local-1)//local*local,)
                        # warmup
                        for _ in range(5):
                            ev=krn(queue,global_size,(local,),rb,ab,mb,gb,wb,e2b,e4b,ob,
                                   np.int32(ne),np.int32(rows),np.int32(inter),np.uint64(panel_bytes))
                        queue.finish()
                        evms=[];wall=[]
                        for _ in range(30):
                            t=time.perf_counter()
                            ev=krn(queue,global_size,(local,),rb,ab,mb,gb,wb,e2b,e4b,ob,
                                   np.int32(ne),np.int32(rows),np.int32(inter),np.uint64(panel_bytes))
                            ev.wait()
                            wall.append((time.perf_counter()-t)*1e3)
                            evms.append((ev.profile.end-ev.profile.start)/1e6)
                        row={"local":local,"event_median_ms":statistics.median(evms),
                             "event_p95_ms":float(np.percentile(evms,95)),
                             "wall_median_ms":statistics.median(wall),
                             "wall_p95_ms":float(np.percentile(wall,95))}
                        if best is None or row["wall_median_ms"]<best["wall_median_ms"]:best=row
                    got=np.empty(rows,dtype=np.float32)
                    cl.enqueue_copy(queue,got,ob).wait()
                    met=metrics(ref,got)
                    p["records"].append({"layer":layer,"nexperts":ne,"fast_math":fast,
                                         "best":best,"correctness":met,
                                         "nonzero_fraction":lm["nonzero_fraction"][:ne]})
        p["status"]="measured"
    except Exception as e:
        p.update({"status":"technical_failure","error":{"type":type(e).__name__,
                  "message":str(e),"traceback":traceback.format_exc()}})
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(p,indent=2,allow_nan=False)+"\n")
    print(json.dumps(p,indent=2,allow_nan=False))
    return 0 if p["status"]=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
