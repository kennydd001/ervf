from __future__ import annotations
import json, statistics, time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "arc140t_phase7" / "cuda_transfer.json"

def main():
    import cupy as cp
    import numpy as np
    sizes = [4096,16384,65536,262144,1048576,4194304,16777216,67108864]
    rows=[]
    for n in sizes:
        pmem=cp.cuda.alloc_pinned_memory(n)
        h=np.frombuffer(pmem,dtype=np.uint8,count=n)
        h[:] = 1
        d=cp.empty(n,dtype=cp.uint8)
        outmem=cp.cuda.alloc_pinned_memory(n)
        hout=np.frombuffer(outmem,dtype=np.uint8,count=n)
        reps=max(8,min(200, 268435456//n))
        for _ in range(4):
            d.set(h); d.get(out=hout)
        cp.cuda.Stream.null.synchronize()
        h2d=[]; d2h=[]
        for _ in range(reps):
            a=cp.cuda.Event(); b=cp.cuda.Event()
            a.record(); d.set(h); b.record(); b.synchronize()
            h2d.append(float(cp.cuda.get_elapsed_time(a,b)))
            a=cp.cuda.Event(); b=cp.cuda.Event()
            a.record(); d.get(out=hout); b.record(); b.synchronize()
            d2h.append(float(cp.cuda.get_elapsed_time(a,b)))
        def rec(vals):
            med=statistics.median(vals)
            return {"median_ms":med,"min_ms":min(vals),"max_ms":max(vals),
                    "gb_s":(n/1e9)/(med/1e3) if med>0 else None,"reps":len(vals)}
        rows.append({"bytes":n,"h2d":rec(h2d),"d2h":rec(d2h)})
        del d, pmem, outmem
        cp.get_default_memory_pool().free_all_blocks()
    payload={"kind":"arc140t_phase7_cuda_transfer","gpu":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(errors="ignore"),"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
