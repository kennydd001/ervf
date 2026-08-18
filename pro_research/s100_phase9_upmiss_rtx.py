from __future__ import annotations
import argparse,json,statistics,time,traceback
from pathlib import Path
import numpy as np
UP_CODE=2494464;UP_SCALE=311808

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--sample',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);p={'kind':'s100_phase9_rtx_upmiss','status':'started','rows':[]}
 try:
  import cupy as cp
  from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4
  from up_proj_batch_kernels import UpProjBatchKernels
  z=np.load(a.sample);codes=z['codes'];scales=z['scales'];g=z['globals'];x=cp.asarray(z['x']);rows=int(z['inter']);cols=int(z['hidden']);ne=len(codes)
  pc=cp.cuda.alloc_pinned_memory(ne*UP_CODE);ps=cp.cuda.alloc_pinned_memory(ne*UP_SCALE);hc=np.frombuffer(pc,np.uint8).reshape(ne,UP_CODE);hs=np.frombuffer(ps,np.uint8).reshape(ne,UP_SCALE);hc[:]=codes;hs[:]=scales
  fused=FusedNVFP4();up=UpProjBatchKernels();globals_host=np.zeros((ne,2),np.float32);globals_host[:,1]=g;dev=fused.alloc_device_cache(ne,ne,ne,globals_host);dev['ids'].set(np.arange(ne,dtype=np.int32));dev['slots'].set(np.arange(ne,dtype=np.int32));dev['need'].fill(1);cc=cp.zeros(ne*UP_CODE,dtype=cp.uint8);cs=cp.zeros(ne*UP_SCALE,dtype=cp.uint8);outbuf=cp.zeros(ne*rows,dtype=cp.float32)
  def timed(fn,reps=80):
   for _ in range(8):fn();cp.cuda.Stream.null.synchronize()
   vals=[]
   for _ in range(reps):
    t=time.perf_counter_ns();fn();cp.cuda.Stream.null.synchronize();vals.append((time.perf_counter_ns()-t)/1e6)
   return {'median_ms':statistics.median(vals),'p95_ms':float(np.percentile(vals,95)),'min_ms':min(vals),'max_ms':max(vals)}
  for n in (1,2,3):
   if n>ne:continue
   dev['need'][:n].fill(1)
   def fetch():fused.cache_fetch(hc.ctypes.data,hs.ctypes.data,cc,cs,dev,UP_CODE,UP_SCALE,n)
   def warmupgemv():up.run_batched(outbuf,cc,cs,dev['slots'],dev['ids'],dev['globals'],1,fused.e2m1,fused.e4m3,x,rows,cols,True,UP_CODE,UP_SCALE,n)
   def staged():fetch();warmupgemv()
   fetch();cp.cuda.Stream.null.synchronize();warmupgemv();cp.cuda.Stream.null.synchronize();ref=cp.asnumpy(outbuf[:n*rows]).copy()
   # Direct mapped-host: same batched ERVF kernel, same arithmetic, codes/scales pointer is mapped pinned host.
   def direct():up.run_batched(outbuf,np.uint64(hc.ctypes.data),np.uint64(hs.ctypes.data),dev['slots'],dev['ids'],dev['globals'],1,fused.e2m1,fused.e4m3,x,rows,cols,True,UP_CODE,UP_SCALE,n)
   direct();cp.cuda.Stream.null.synchronize();direct_out=cp.asnumpy(outbuf[:n*rows]).copy();exact=bool(np.array_equal(ref,direct_out));mx=float(np.max(np.abs(ref-direct_out)))
   row={'nexperts':n,'fetch_only':timed(fetch),'warm_up_only':timed(warmupgemv),'staged_fetch_plus_up':timed(staged),'direct_host_up':timed(direct),'direct_bitexact':exact,'direct_max_abs':mx};p['rows'].append(row)
  np.savez_compressed(out.with_suffix('.ref.npz'),ref=ref,rows=np.int32(rows),cols=np.int32(cols));p['status']='measured'
 except Exception as e:p.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}})
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(p,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(p,indent=2));return 0 if p.get('status')=='measured' else 2
if __name__=='__main__':raise SystemExit(main())
