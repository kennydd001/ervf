from __future__ import annotations
import argparse,json,os,sys,types,traceback
from pathlib import Path
import numpy as np
TARGETS=(43,38,10,29,22)
UP_CODE=2494464;UP_SCALE=311808

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();repo=Path(a.repo).resolve();od=Path(a.outdir).resolve();od.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(repo/'pro_research'));sys.path.insert(0,str(repo/'src'));os.chdir(repo);p={'status':'started','captures':[]}
 try:
  import cupy as cp
  from graph_e1f22 import _new_runtime,_load_prompt_set
  from s100_phase3_profiles import apply_phase3_profile
  from ervf_dense import DenseERVF
  from down_proj_batch_kernels import DownProjBatchKernels
  from up_proj_batch_kernels import UpProjBatchKernels
  from layer_capacity import apply_nonuniform_capacity
  from selective_ervf_v3 import _install_selective
  from moe_dev_batched import install_batched_moe_dev
  _p,_e,_n,cap=_load_prompt_set('full');rt=_new_runtime(int(cap));apply_phase3_profile(rt,'qfast');rt.enable_cache(int(cap));apply_nonuniform_capacity(rt);rt.device_cache=True;rt.deterministic_accum=True;restore_sel,_=_install_selective(rt,DenseERVF());install_batched_moe_dev(rt,DownProjBatchKernels(),UpProjBatchKernels());orig=rt._moe_dev;capt={}
  def wrap(self,i,out):
   x=cp.asnumpy(self.normed).astype(np.float32,copy=True) if int(i) in TARGETS and int(i) not in capt else None
   r=orig(i,out)
   if x is not None:
    dev=self._dev_cache[i];need=cp.asnumpy(dev['need'][:self.top_k]).astype(np.int32);ids=cp.asnumpy(dev['ids'][:self.top_k]).astype(np.int32);ww=cp.asnumpy(dev['w'][:self.top_k]).astype(np.float32)
    if int(need.sum())>0:
     order=np.argsort(-need,kind='stable');ids=ids[order];need=need[order];ww=ww[order];bank=self.bank[i];codes=np.stack([bank['up_code_view'][int(e)].copy() for e in ids]);scales=np.stack([bank['up_scale_view'][int(e)].copy() for e in ids]);g=np.asarray([bank['g_up'][int(e)] for e in ids],np.float32)
     fn=od/f'UPMISS_LAYER_{int(i)}.npz';np.savez_compressed(fn,layer=np.int32(i),x=x,ids=ids,need=need,route_w=ww,codes=codes,scales=scales,globals=g,hidden=np.int32(self.hidden),inter=np.int32(self.moe_inter));capt[int(i)]={'file':str(fn),'ids':ids.tolist(),'need':need.tolist(),'misses':int(need.sum())};print(f'captured layer {i} misses={int(need.sum())}',flush=True)
   return r
  rt._moe_dev=types.MethodType(wrap,rt);prompts,_e,_n,_c=_load_prompt_set('full')
  for pr in prompts:
   rt.reset();cur=None
   for tok in pr['prompt_ids']:cur=int(rt.step(int(tok)))
   for _ in range(512):
    cur=int(rt.step(cur))
    if all(x in capt for x in TARGETS):break
   if all(x in capt for x in TARGETS):break
  rt._moe_dev=orig;restore_sel();p.update({'status':'measured','captures':[capt[k] for k in sorted(capt)],'target_layers':list(TARGETS)})
 except Exception as e:p.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}})
 (od/'S100_PHASE9_UPMISS_SAMPLES.json').write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8');print(json.dumps(p,indent=2));return 0 if p.get('status')=='measured' and len(p.get('captures',[]))>=3 else 2
if __name__=='__main__':raise SystemExit(main())
