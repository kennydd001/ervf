from __future__ import annotations
import argparse,json,statistics,time,traceback
from pathlib import Path
import numpy as np
KERNEL=r"""
__kernel void up_nvfp4(__global const uchar* codes,__global const uchar* scales,__global const float* g,__global const float* x,__global const float* e2,__global const float* e4,__global float* out,const int ne,const int rows,const int cols,const ulong cb,const ulong sb){
 int row=get_global_id(0),s=get_global_id(1);if(row>=rows||s>=ne)return;int nb=cols>>1,ng=cols>>4;__global const uchar* cr=codes+(ulong)s*cb+(ulong)row*nb;__global const uchar* sr=scales+(ulong)s*sb+(ulong)row*ng;float acc=0.0f;
 for(int q=0;q<ng;++q){float sc=e4[(uint)sr[q]]*g[s];int k=q<<4;__global const uchar* p=cr+(q<<3);
  #pragma unroll
  for(int b=0;b<8;++b){uchar v=p[b];acc=fma(e2[(uint)(v&15)]*sc,x[k+(b<<1)],acc);acc=fma(e2[(uint)(v>>4)]*sc,x[k+(b<<1)+1],acc);} }
 float r=fmax(acc,0.0f);out[(ulong)s*rows+row]=r*r;}
"""
E2=np.asarray([0,0.5,1,1.5,2,3,4,6, -0,-0.5,-1,-1.5,-2,-3,-4,-6],np.float32)
def e4_table():
 a=np.empty(256,np.float32)
 for u in range(256):
  sign=-1.0 if u&128 else 1.0;exp=(u>>3)&15;mant=u&7
  if exp==0:a[u]=sign*(mant/8.0)*(2.0**-6)
  elif exp==15 and mant==7:a[u]=0.0
  else:a[u]=sign*(1+mant/8.0)*(2.0**(exp-7))
 return a

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--sample',required=True);ap.add_argument('--rtx-ref',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);p={'kind':'s100_phase9_arc_upmiss','status':'started','rows':[]}
 try:
  import pyopencl as cl
  z=np.load(a.sample);rr=np.load(a.rtx_ref);codes=z['codes'];scales=z['scales'];g=z['globals'].astype(np.float32);x=z['x'].astype(np.float32);rows=int(z['inter']);cols=int(z['hidden']);ne=len(codes);ref=rr['ref'];e4=e4_table()
  ds=[]
  for plat in cl.get_platforms():
   for d in plat.get_devices():
    if (d.type&cl.device_type.GPU) and 'intel' in (d.vendor+' '+d.name).lower():ds.append(d)
  if not ds:raise RuntimeError('Intel Arc OpenCL GPU not found')
  dev=ds[0];ctx=cl.Context([dev]);q=cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE);prg=cl.Program(ctx,KERNEL).build();kn=cl.Kernel(prg,'up_nvfp4');mf=cl.mem_flags
  cb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.ascontiguousarray(codes));sb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.ascontiguousarray(scales));gb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=g);xb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=x);e2b=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=E2);e4b=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=e4);ob=cl.Buffer(ctx,mf.WRITE_ONLY,size=ne*rows*4)
  for n in (1,2,3):
   if n>ne:continue
   best=None
   for local in (64,128,256):
    gs=(((rows+local-1)//local)*local,n);kn.set_args(cb,sb,gb,xb,e2b,e4b,ob,np.int32(n),np.int32(rows),np.int32(cols),np.uint64(codes.shape[1]),np.uint64(scales.shape[1]))
    for _ in range(8):cl.enqueue_nd_range_kernel(q,kn,gs,(local,1)).wait()
    wall=[];evt=[]
    for _ in range(60):
     t=time.perf_counter_ns();ev=cl.enqueue_nd_range_kernel(q,kn,gs,(local,1));ev.wait();wall.append((time.perf_counter_ns()-t)/1e6);evt.append((ev.profile.end-ev.profile.start)/1e6)
    row={'local':local,'wall_median_ms':statistics.median(wall),'event_median_ms':statistics.median(evt),'wall_p95_ms':float(np.percentile(wall,95))}
    if best is None or row['wall_median_ms']<best['wall_median_ms']:best=row
   host=np.empty(ne*rows,np.float32);cl.enqueue_copy(q,host,ob).wait();a0=ref[:n*rows].astype(np.float64);b0=host[:n*rows].astype(np.float64);err=b0-a0;nr=float(np.sqrt(np.mean(err*err))/(np.sqrt(np.mean(a0*a0))+1e-30));cos=float(np.dot(a0,b0)/(np.linalg.norm(a0)*np.linalg.norm(b0)+1e-30));p['rows'].append({'nexperts':n,'best':best,'correctness':{'cosine':cos,'nrmse':nr,'max_abs':float(np.max(np.abs(err))),'finite':bool(np.isfinite(b0).all())}})
  p.update({'status':'measured','device':dev.name})
 except Exception as e:p.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}})
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(p,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(p,indent=2));return 0 if p.get('status')=='measured' else 2
if __name__=='__main__':raise SystemExit(main())
