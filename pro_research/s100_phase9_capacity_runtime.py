from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from diag_component_marginals_graph import _recapture
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _new_runtime
from layer_capacity import reallocate_layer
from moe_dev_batched import install_batched_moe_dev
from moe_dev_scale_resident import planned_plane_bytes
from scale_resident_kernels import ScaleResidentKernels
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels
from s100_phase3_profiles import apply_phase3_profile,public_profile_record
from s100_phase5_combined import install_phase5_combined
from s100_phase5_threshold_kernels import Phase5ThresholdKernels

@dataclass
class Bundle:
    rt:Any;capmap:dict[int,int];profile:dict[str,Any];sres:Any;p5thr:Any;restore_sel:Any;restore_combined:Any;planned:int

def build(capmap):
    import cupy as cp
    rt=_new_runtime(72);prof=apply_phase3_profile(rt,'qfast');rt.enable_cache(72)
    for l in rt.moe_layers:reallocate_layer(rt,int(l),int(capmap[int(l)]))
    rt.device_cache=True;rt.deterministic_accum=True
    dense,down,up=DenseERVF(),DownProjBatchKernels(),UpProjBatchKernels();restore_sel,_=_install_selective(rt,dense);install_batched_moe_dev(rt,down,up);rt.setup_graph();cp.get_default_memory_pool().free_all_blocks()
    planned=int(planned_plane_bytes(rt));free=int(cp.cuda.Device(0).mem_info[0])
    if planned>free:raise RuntimeError(f'phase9 scale planes do not fit: {planned}>{free}')
    sres=ScaleResidentKernels();thr=Phase5ThresholdKernels();cfg={'layer_k':{},'alpha':0.0003};restore=install_phase5_combined(rt,down,up,sres,thr,cfg);_recapture(rt)
    return Bundle(rt,{int(k):int(v) for k,v in capmap.items()},prof,sres,thr,restore_sel,restore,planned)

def record(b):return {'profile':'qfast+thr_0003','capacity_map':{str(k):v for k,v in sorted(b.capmap.items())},'total_slots':sum(b.capmap.values()),'planned_plane_bytes':b.planned,'profile_record':public_profile_record(b.profile)}
