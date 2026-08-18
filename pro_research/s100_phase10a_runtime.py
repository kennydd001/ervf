from __future__ import annotations
import json,types
from dataclasses import dataclass
import numpy as np
from common import REPO
from diag_component_marginals_graph import _recapture
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _new_runtime
from layer_capacity import reallocate_layer
from moe_dev_batched import install_batched_moe_dev,DOWN_PANEL_BYTES,UP_CODE,UP_SCALE
from scale_resident_kernels import ScaleResidentKernels
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels
from s100_phase3_profiles import apply_phase3_profile
from s100_phase5_combined import install_phase5_combined
from s100_phase5_threshold_kernels import Phase5ThresholdKernels
from s100_phase10a_panel_kernels import PanelCacheKernels,NPANEL,PANEL_STRIDE,CODE_PANEL_BYTES

PROF=REPO/"pro_research"/"results"/"s100_phase9"/"S100_PHASE9_CAPACITY_PROFILES.json"
@dataclass
class Bundle:
 rt:object;restore_sel:object;restore_combined:object;state:dict;panel_cache:object
def cmap():
 d=json.loads(PROF.read_text())["profiles"]["current"];return {int(k):int(v) for k,v in d.items()}
def shell():
 rt=_new_runtime(72);apply_phase3_profile(rt,"qfast");mp=cmap();rt.enable_cache(0)
 for l in rt.moe_layers:reallocate_layer(rt,int(l),mp[int(l)])
 rt.device_cache=True;rt.deterministic_accum=True
 dense,down,up=DenseERVF(),DownProjBatchKernels(),UpProjBatchKernels()
 rs,_=_install_selective(rt,dense);install_batched_moe_dev(rt,down,up);rt.setup_graph()
 return rt,rs,down,up
def mkcache(rt,items):
 import cupy as cp
 by={}
 for x in items:by.setdefault(int(x["layer"]),[]).append(x)
 out={}
 for l in rt.moe_layers:
  hm=np.full((rt.n_experts,NPANEL),-1,np.int32);chunks=[]
  bank=rt.bank[int(l)]["down_pm"]
  for s,x in enumerate(by.get(int(l),[])):
   e,p=int(x["expert"]),int(x["panel"]);hm[e,p]=s
   off=e*DOWN_PANEL_BYTES+p*PANEL_STRIDE+rt.hidden
   chunks.append(np.asarray(bank[off:off+CODE_PANEL_BYTES],np.uint8).copy())
  data=np.concatenate(chunks) if chunks else np.zeros(1,np.uint8)
  out[int(l)]={"map":cp.asarray(hm.reshape(-1)),"data":cp.asarray(data)}
 return out
def install(rt,down,up,sres,thr,alpha,pc=None,expose=False):
 cp=rt.cp;pk=PanelCacheKernels() if pc else None;kmax=int(rt.top_k)
 inter,hidden=rt.moe_inter,rt.hidden;npanel=inter//16;nc=rt.fused.nchunks
 orig=rt._moe_dev;state={};mir=[rt.mstate["mirror"],cp.zeros(DOWN_PANEL_BYTES,cp.uint8)]
 gs=cp.cuda.Stream(non_blocking=True)
 gd=[cp.cuda.Event(block=False,disable_timing=True) for _ in range(kmax+1)]
 md=[cp.cuda.Event(block=False,disable_timing=True) for _ in range(kmax)]
 blocks=(inter*32+255)//256
 def alloc():
  return {"act":cp.zeros(kmax*inter,cp.float32),"masks":cp.zeros(kmax*npanel,cp.uint32),
  "plist":cp.zeros(kmax*npanel,cp.int32),"pcount":cp.zeros(kmax,cp.int32),
  "nz":cp.zeros(kmax*inter,cp.int32),"nzc":cp.zeros(kmax,cp.int32),
  "partials":cp.zeros(kmax*nc*hidden,cp.float32),"max_act":cp.zeros(kmax,cp.float32)}
 def moe(self,i,out):
  k,d,f=self.k,self.layer[i],self.fused;bank,c=self.bank[i],self.cache[i]
  if not hasattr(self,"_dev_cache"):self._dev_cache={}
  if i not in self._dev_cache:self._dev_cache[i]=f.alloc_device_cache(self.n_experts,c["cap"],kmax,bank["globals"])
  dev=self._dev_cache[i]
  if i not in state:state[i]=alloc()
  bs=state[i]
  if i not in sres.planes:sres.alloc_planes(i,int(c["cap"]))
  planes=sres.planes[i]
  k.mv_f32(self.rlog,d["gate_w"],self.normed,self.n_experts,self.hidden)
  f.route_topk(self.rlog,d["gate_b"],dev["ids"],dev["w"],self.n_experts,kmax,self.scaling,bad_pick=self._bad_pick)
  f.cache_assign(dev,dev["ids"],c["cap"],kmax);self.evt[0].record()
  with self.copy_stream:
   self.copy_stream.wait_event(self.evt[0]);f.cache_fetch(bank["up_codes"].ctypes.data,bank["up_scales"].ctypes.data,c["codes"],c["scales"],dev,UP_CODE,UP_SCALE,kmax)
   sres.fetch_planes(bank["down_base_ptr"],planes,dev,kmax);self.evt[1].record(self.copy_stream)
  out.fill(0);f.gemv_into(self._act_shared,d["sh_up_c"],d["sh_up_s"],self.normed,d["sh_up_g"],self.shared_inter,self.hidden,apply_relu2=True)
  f.gemv_into(out,d["sh_dn_c"],d["sh_dn_s"],self._act_shared,d["sh_dn_g"],hidden,self.shared_inter)
  main=cp.cuda.get_current_stream();main.wait_event(self.evt[1])
  up.run_batched(bs["act"],c["codes"],c["scales"],dev["slots"],dev["ids"],dev["globals"],1,f.e2m1,f.e4m3,self.normed,inter,hidden,True,UP_CODE,UP_SCALE,kmax)
  thr.panel_scan_threshold_batched((kmax,),(256,),(bs["act"],np.int32(inter),np.float32(alpha),bs["masks"],bs["plist"],bs["pcount"],bs["nz"],bs["nzc"],bs["max_act"]))
  grid=((hidden+127)//128,nc)
  def issue(s):
   if pc:pk.gather(blocks,bank["down_base_ptr"],dev["ids"][s:],pc[int(i)]["map"],mir[s&1],bs["nz"][s*inter:(s+1)*inter],bs["nzc"][s:s+1],hidden)
   else:sres.gather_cols(blocks,bank["down_base_ptr"],dev["ids"][s:],mir[s&1],bs["nz"][s*inter:(s+1)*inter],bs["nzc"][s:s+1],hidden)
  main.record(gd[kmax]);gs.wait_event(gd[kmax])
  with gs:issue(0);gd[0].record(gs)
  for s in range(kmax):
   if s+1<kmax:
    with gs:
     if s>=1:gs.wait_event(md[s-1])
     issue(s+1);gd[s+1].record(gs)
   main.wait_event(gd[s]);sl=s*inter
   if pc:pk.down(grid,mir[s&1],planes,dev["slots"][s:],dev["ids"][s:],dev["globals"],pc[int(i)]["map"],pc[int(i)]["data"],bs["act"][sl:sl+inter],bs["plist"][s*npanel:(s+1)*npanel],bs["masks"][s*npanel:(s+1)*npanel],bs["pcount"][s:s+1],f.e2m1,f.e4m3,bs["partials"][s*nc*hidden:(s+1)*nc*hidden],hidden,inter)
   else:sres.down_masked_sres(grid,mir[s&1],planes,dev["slots"][s:],dev["ids"][s:],dev["globals"],bs["act"][sl:sl+inter],bs["plist"][s*npanel:(s+1)*npanel],bs["masks"][s*npanel:(s+1)*npanel],bs["pcount"][s:s+1],f.e2m1,f.e4m3,bs["partials"][s*nc*hidden:(s+1)*nc*hidden],hidden,inter)
   md[s].record(main)
  rb=(hidden+255)//256;down.reduce_partials_batched((rb,kmax),(256,),(bs["partials"],self.contrib,np.int32(hidden),np.int32(nc)));down.run_accumulate_batched(out,self.contrib,dev["w"],hidden,kmax)
 rt._moe_dev=types.MethodType(moe,rt)
 def restore():rt._moe_dev=orig;sres.planes.clear()
 return restore,state
def build(alpha=.0003,items=None,expose=False,bad=False):
 rt,rs,down,up=shell();sres=ScaleResidentKernels();thr=Phase5ThresholdKernels()
 if items is None and not expose:
  rc=install_phase5_combined(rt,down,up,sres,thr,{"layer_k":{},"alpha":float(alpha)});state={};pc=None
 else:
  pc=mkcache(rt,items or []) if items is not None else None;rc,state=install(rt,down,up,sres,thr,alpha,pc,expose)
 rt._bad_pick=1 if bad else 0;_recapture(rt);return Bundle(rt,rs,rc,state,pc)
