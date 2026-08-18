"""Saturate Arc with the real full-bank NVFP4 routed-down kernel for contention tests."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from probe_arc_full_bank_nvfp4 import KERNEL

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bank-dir',required=True);ap.add_argument('--seconds',type=int,default=90);a=ap.parse_args();d=Path(a.bank_dir)
 import pyopencl as cl
 meta=json.loads((d/'meta.json').read_text());records=np.load(d/'records.npy',mmap_mode='r');globals_=np.load(d/'globals.npy');act=np.load(d/'act.npy');masks=np.load(d/'masks.npy');route=np.load(d/'route_w.npy');e2=np.load(d/'e2m1.npy');e4=np.load(d/'e4m3.npy')
 devs=[]
 for plat in cl.get_platforms():
  for dev in plat.get_devices():
   if (dev.type&cl.device_type.GPU) and 'intel' in (dev.vendor+' '+dev.name).lower():devs.append((plat,dev))
 if not devs:raise RuntimeError('Intel OpenCL GPU not found')
 ctx=cl.Context([devs[0][1]]);q=cl.CommandQueue(ctx);prg=cl.Program(ctx,KERNEL).build(options=['-cl-std=CL2.0']);mf=cl.mem_flags
 rows=int(meta['hidden']);inter=int(meta['moe_inter']);ne=int(meta['top_k']);nexp=int(meta['n_experts']);pb=int(meta['down_panel_bytes']);bankb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.asarray(records));ab=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=act);mb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=masks);gb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=globals_);wb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=route);e2b=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=e2);e4b=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=e4);ob=cl.Buffer(ctx,mf.WRITE_ONLY,rows*4);idb=cl.Buffer(ctx,mf.READ_ONLY,ne*4);local=128;gs=((rows+local-1)//local*local,);rng=np.random.default_rng(7714);deadline=time.monotonic()+max(5,a.seconds);count=0
 while time.monotonic()<deadline:
  ids=rng.choice(nexp,size=ne,replace=False).astype(np.int32);cl.enqueue_copy(q,idb,ids);prg.routed_down_bank_nvfp4(q,gs,(local,),bankb,idb,ab,mb,gb,wb,e2b,e4b,ob,np.int32(ne),np.int32(rows),np.int32(inter),np.uint64(pb));count+=1
  if count%128==0:q.finish()
 q.finish();print(json.dumps({'status':'complete','seconds':a.seconds,'kernels':count,'kernels_per_s':count/max(1,a.seconds)}));return 0
if __name__=='__main__':raise SystemExit(main())
