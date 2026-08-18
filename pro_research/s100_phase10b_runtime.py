from __future__ import annotations
import json
from common import REPO
from diag_component_marginals_graph import _recapture
from s100_phase9_capacity_runtime import build as build9
from s100_phase10b_mamba_kernels import MambaERVF2
P=REPO/"pro_research"/"results"/"s100_phase9"/"S100_PHASE9_CAPACITY_PROFILES.json"
def build(variant=None):
 mp={int(k):int(v) for k,v in json.loads(P.read_text())["profiles"]["current"].items()}
 b=build9(mp)
 if variant:
  rt=b.rt;old=rt.k.mv_fp8_tensor;k=MambaERVF2()
  def dispatch(out,W,x,sc,rows,cols):
   shp=(int(rows),int(cols))
   if shp in ((int(rt.proj.size),rt.hidden),(rt.hidden,rt.d_inner)):
    return k.run(variant,out,W,x,float(sc),int(rows),int(cols))
   return old(out,W,x,sc,rows,cols)
  rt.k.mv_fp8_tensor=dispatch;_recapture(rt);b._phase10b_kernel=k;b._phase10b_old=old
 return b
