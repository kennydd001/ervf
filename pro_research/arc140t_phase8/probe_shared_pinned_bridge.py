from __future__ import annotations
import argparse,json,statistics,sys,time,traceback
from pathlib import Path
import numpy as np

SRC=r"""
__kernel void bridge_copy(__global const uchar* src,__global uchar* dst,const int n){
    int i=(int)get_global_id(0);
    if(i<n) dst[i]=(uchar)(src[i]^0);
}
"""

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--s100-repo',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out)
    p={"kind":"s100_phase8_pinned_opencl_bridge","status":"started","rows":[]}
    try:
        # Reuse the already working CuPy binary package from the Nemotron venv.
        site=Path(a.s100_repo)/'.venv-nemotron'/'Lib'/'site-packages'
        sys.path.insert(0,str(site))
        import cupy as cp
        import pyopencl as cl
        devs=[]
        for plat in cl.get_platforms():
            for d in plat.get_devices():
                if (d.type & cl.device_type.GPU) and "intel" in (d.vendor+" "+d.name).lower():
                    devs.append((plat,d))
        if not devs: raise RuntimeError("Intel OpenCL GPU not found")
        plat,dev=devs[0];ctx=cl.Context([dev]);q=cl.CommandQueue(ctx)
        prg=cl.Program(ctx,SRC).build()
        p["arc_device"]=dev.name;p["cuda_device"]=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(errors='ignore')
        mf=cl.mem_flags
        sizes=[5376,6*1856*4,6*1856*4+2688*4,65536,262144]
        for n in sizes:
            pin=cp.cuda.alloc_pinned_memory(n);hout=cp.cuda.alloc_pinned_memory(n)
            hin=np.frombuffer(pin,dtype=np.uint8,count=n);ho=np.frombuffer(hout,dtype=np.uint8,count=n)
            hin[:]=1;ho[:]=0
            dsrc=cp.ones(n,dtype=np.uint8);ddst=cp.zeros(n,dtype=np.uint8)
            # Intel driver sees the exact CUDA-pinned host pages.
            ib=cl.Buffer(ctx,mf.READ_ONLY|mf.USE_HOST_PTR,hostbuf=hin)
            ob=cl.Buffer(ctx,mf.WRITE_ONLY|mf.USE_HOST_PTR,hostbuf=ho)
            gs=((n+255)//256*256,)
            vals=[]
            for _ in range(6):
                dsrc.get(out=hin);q.finish();prg.bridge_copy(q,gs,(256,),ib,ob,np.int32(n));q.finish();ddst.set(ho);cp.cuda.Stream.null.synchronize()
            for _ in range(50):
                t=time.perf_counter()
                dsrc.get(out=hin);cp.cuda.Stream.null.synchronize()
                prg.bridge_copy(q,gs,(256,),ib,ob,np.int32(n));q.finish()
                ddst.set(ho);cp.cuda.Stream.null.synchronize()
                vals.append((time.perf_counter()-t)*1e3)
            ok=bool(cp.all(ddst==1).item())
            p["rows"].append({"bytes":n,"median_ms":statistics.median(vals),
                              "p95_ms":float(np.percentile(vals,95)),
                              "min_ms":min(vals),"max_ms":max(vals),
                              "correct":ok})
        p["status"]="measured"
    except Exception as e:
        p.update({"status":"technical_failure","error":{"type":type(e).__name__,
                  "message":str(e),"traceback":traceback.format_exc()}})
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(p,indent=2,allow_nan=False)+"\n")
    print(json.dumps(p,indent=2,allow_nan=False))
    return 0 if p["status"]=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
