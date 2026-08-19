from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path

import numpy as np

from common import REPO, require_model_dir

RESULTS = REPO / "pro_research" / "results" / "s100_phase21"
TRACE = (
    REPO / "pro_research" / "results" / "s100_phase20b"
    / "S100_PHASE20B_CANONICAL_TRACE.json"
)
SNAPSHOT = "e8f3c7c4de75ad84fe1bcef95d38eca76214480b"
ARMS = (
    "current_grouped",
    "selective_grouped",
    "v6_device_rows",
    "v18_device_rows",
)

def identity_gate():
    model=require_model_dir()
    cfg=json.loads((model/"config.json").read_text(encoding="utf-8"))
    if model.name != SNAPSHOT:
        raise RuntimeError(f"wrong snapshot: {model.name}")
    if cfg.get("model_type")!="nemotron_h" or int(cfg.get("num_hidden_layers",0))!=52:
        raise RuntimeError("not the 52-layer NemotronH Lightning target")
    p20s=json.loads(
        (REPO/"pro_research"/"results"/"s100_phase20s"/"S100_PHASE20S_SUMMARY.json")
        .read_text(encoding="utf-8")
    )
    if not p20s.get("PHASE20B_FULL_VERIFIER_OPEN"):
        raise RuntimeError("Phase20S has not opened Phase20B")
    p20b=json.loads(
        (REPO/"pro_research"/"results"/"s100_phase20b"/"S100_PHASE20B_SUMMARY.json")
        .read_text(encoding="utf-8")
    )
    if not p20b.get("FULL_VERIFIER_CORRECTNESS_GREEN"):
        raise RuntimeError("Phase20B correctness is not green")
    return model,cfg,p20s,p20b

def load_trace():
    if not TRACE.exists():
        raise FileNotFoundError(TRACE)
    tr=json.loads(TRACE.read_text(encoding="utf-8"))
    if tr.get("status")!="measured" or len(tr.get("tokens",[]))<4145:
        raise RuntimeError("Phase20B canonical trace incomplete")
    return tr

def release(rt):
    import cupy as cp
    try:
        rt.bank={};rt.cache={};rt._dev_cache={}
    except Exception:
        pass
    del rt
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def make_rt(context:int, arm:str):
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    model=require_model_dir()
    rt=LightningRuntime(
        model, contexts_max=max(int(context)+128,512),
        embed_on_host=True, fp8_kv=False, verbose=False,
    )
    rt.load_routed_bank()
    rt.deterministic_accum=True

    keep=[]
    if arm=="current_grouped":
        rt.enable_cache(48)
        return rt,keep

    # Every Phase21 parent arm beyond current_grouped uses the frozen V6/V18
    # capacity and selective dense policy.
    rt.enable_cache(72)
    from layer_capacity import apply_nonuniform_capacity
    apply_nonuniform_capacity(rt)

    from ervf_dense import DenseERVF
    from selective_ervf_v3 import _install_selective
    dense=DenseERVF()
    restore_sel,counters=_install_selective(rt,dense)
    keep += [dense,restore_sel,counters]

    if arm=="selective_grouped":
        return rt,keep

    rt.device_cache=True
    from down_proj_batch_kernels import DownProjBatchKernels
    from up_proj_batch_kernels import UpProjBatchKernels
    down=DownProjBatchKernels();up=UpProjBatchKernels()
    keep += [down,up]

    if arm=="v6_device_rows":
        from moe_dev_batched import install_batched_moe_dev
        restore=install_batched_moe_dev(rt,down,up)
        keep.append(restore)
        return rt,keep

    if arm=="v18_device_rows":
        from moe_dev_scale_resident import planned_plane_bytes
        from scale_resident_kernels import ScaleResidentKernels
        from moe_dev_combined import install_combined_moe_dev
        # FullH4 buffers are allocated after this function, so reserve 96 MiB
        # beyond the known scale-plane requirement.
        planned=int(planned_plane_bytes(rt))
        free=int(cp.cuda.Device(0).mem_info[0])
        if planned+96*1024*1024 > free:
            raise RuntimeError(
                f"v18 scale plane does not fit with H4 reserve: "
                f"planned={planned/2**20:.1f}MiB free={free/2**20:.1f}MiB"
            )
        sres=ScaleResidentKernels()
        restore=install_combined_moe_dev(rt,down,up,sres)
        keep += [sres,restore]
        return rt,keep

    raise ValueError(arm)

class DeviceRowsMoEH4:
    """Four exact single-row device-MoE calls at one model layer.

    This intentionally gives up cross-token expert reuse. It reuses the
    mature V6/V18 no-host-route dataplane so Phase21 can isolate whether the
    Phase20B Python grouped scheduler caused the regression.
    """
    def __init__(self,rt):
        self.rt=rt
        self.k=None  # FullH4Verifier keeps its existing Phase20B kernel for head.

    def __call__(self,layer,normed,out,collect_stats=False):
        import cupy as cp
        rt=self.rt
        for t in range(4):
            cp.copyto(rt.normed,normed[t])
            rt._moe(int(layer),out[t])
        return None,None,None

def verifier_for(rt,arm):
    from s100_phase20b_verifier import FullH4Verifier,GroupedMoEH4
    v=FullH4Verifier(rt)
    if arm in ("v6_device_rows","v18_device_rows"):
        # Preserve v.bk (lm_head batch kernel) then release GroupedMoEH4's
        # large scratch. The device-row adapter has no host routing.
        v.moeb=DeviceRowsMoEH4(rt)
    elif arm=="selective_grouped":
        # Rebuild grouped state after cap72/nonuniform cache was installed.
        v.moeb=GroupedMoEH4(rt)
        v.bk=v.moeb.k
    return v

def prefill_to(rt, trace_tokens, context:int):
    # context counts total input tokens in the canonical trace.
    rt.reset()
    for token in trace_tokens[:int(context)]:
        rt.step(int(token))
    if int(rt.pos)!=int(context):
        raise RuntimeError(f"prefill pos {rt.pos} != {context}")

def expected_for_block(trace_tokens,pos):
    drafts=np.asarray(trace_tokens[pos:pos+4],np.int32)
    expected=np.asarray(trace_tokens[pos+1:pos+5],np.int32)
    if len(drafts)!=4 or len(expected)!=4:
        raise RuntimeError("trace too short")
    return drafts,expected

def measure_blocks(rt,v,trace_tokens,context:int,blocks:int,warmup:int=1):
    import cupy as cp
    import time
    records=[]
    # Prefill is outside timing.
    prefill_to(rt,trace_tokens,context)

    for bi in range(warmup+blocks):
        pos=int(rt.pos)
        drafts,expected=expected_for_block(trace_tokens,pos)
        cp.cuda.get_current_stream().synchronize()
        t0=time.perf_counter_ns()
        got,_=v.block(drafts.tolist(),collect_census=False)
        cp.cuda.get_current_stream().synchronize()
        ms=(time.perf_counter_ns()-t0)/1e6
        ok=bool(np.array_equal(got,expected))
        if not ok:
            raise RuntimeError(
                f"token mismatch block={bi} pos={pos} "
                f"got={got.tolist()} expected={expected.tolist()}"
            )
        if bi>=warmup:
            records.append({"block":bi-warmup,"pos":pos,"ms":ms,
                            "got":got.tolist(),"expected":expected.tolist()})
    vals=np.asarray([r["ms"] for r in records],np.float64)
    return records,{
        "count":len(records),
        "median_ms":float(np.median(vals)),
        "p10_ms":float(np.percentile(vals,10)),
        "p90_ms":float(np.percentile(vals,90)),
        "mean_ms":float(vals.mean()),
        "ms_per_useful_token":float(np.median(vals)/4.0),
        "target_only_tok_s":float(4000.0/np.median(vals)),
        "all_token_exact":True,
    }
