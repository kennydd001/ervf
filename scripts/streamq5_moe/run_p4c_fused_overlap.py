from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p3a_integrated_expert import (
    BANK_BYTES, BANK_RESULT_PATH, CACHE_BYTES, DOMAINS, EXPERT_BYTES, KV_BYTES,
    LAYERS, TRUNK_BYTES, bases, copy_expert, kernels as serial_kernels,
    launch_layer, pin_bank,
)
from scripts.streamq5_moe.run_p4a_causal_async import (
    CUDA_SOURCE, initialize_static, percentile, result_row, sha256,
)
from scripts.streamq5_moe.run_p4b_rendezvous_async import (
    load_routes, plan_layer, positive_slope, static_sets,
)


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P4C_FUSED_OVERLAP_PREREGISTRATION.md"
LOCK_PATH = R / "p4c_fused_overlap_input_lock.json"
EVALUATOR_LOCK = R / "p4c_fused_overlap_evaluator_lock.json"
P1D_VERIFY = R / "p1d_physical_bank_verification.json"
P1C_CAPTURE = R / "p1c_route_capture_result.json"
P1C_VALIDATION = R / "p1c_cache_validation.json"
P1C_TEST = R / "p1c_cache_test.json"


FUSED_SOURCE = CUDA_SOURCE + r'''
__device__ __forceinline__ float block_dot(
    const float* x, const unsigned char* packed, const unsigned short* scales,
    int row, int cols, float* reduction) {
    int tid = (int)threadIdx.x;
    float local = q5_dot(x, packed, scales, row, cols, tid);
    reduction[tid] = local; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    return reduction[0];
}
extern "C" __global__ void q5_swiglu_n(
    const float* x, const unsigned char* cache, const int* slots,
    const int* positions, float* activation) {
    int local_expert = (int)blockIdx.x / 768;
    int row = (int)blockIdx.x - local_expert * 768;
    int output_expert = positions[local_expert];
    long long expert_base = (long long)slots[local_expert] * 3035136LL;
    const unsigned char* gate_packed = cache + expert_base + 64;
    const unsigned short* gate_scales = (const unsigned short*)(cache + expert_base + 64 + 983040);
    const unsigned char* up_packed = cache + expert_base + 1011712LL + 64;
    const unsigned short* up_scales = (const unsigned short*)(cache + expert_base + 1011712LL + 64 + 983040);
    __shared__ float reduction[256];
    float gate = block_dot(x, gate_packed, gate_scales, row, 2048, reduction);
    float up = block_dot(x, up_packed, up_scales, row, 2048, reduction);
    if (threadIdx.x == 0) {
        activation[output_expert * 768 + row] =
            (gate / (1.0f + expf(-gate))) * up;
    }
}
'''


def fused_kernels():
    return (
        cp.RawKernel(FUSED_SOURCE, "q5_swiglu_n", options=("--std=c++11",)),
        cp.RawKernel(FUSED_SOURCE, "q5_down_n", options=("--std=c++11",)),
        cp.RawKernel(FUSED_SOURCE, "reduce_experts", options=("--std=c++11",)),
    )


def run_domain_serial(domain, route, fixed_ids, begin, end, pinned, cache,
                      cache_view, layer_bases, stream, kernel_set, seed):
    initialize_static(stream, pinned, cache, layer_bases, fixed_ids)
    fixed_set=[frozenset(row) for row in fixed_ids]
    dynamic=[OrderedDict() for _ in range(LAYERS)]
    state=cp.empty(2048,dtype=cp.float32); gate=cp.empty(6144,dtype=cp.float32)
    up=cp.empty(6144,dtype=cp.float32); down=cp.empty(16384,dtype=cp.float32)
    slots_d=cp.empty(8,dtype=cp.int32); rng=np.random.default_rng(seed)
    times=np.empty(end-begin); misses=np.zeros(end-begin,dtype=np.int64)
    outputs=np.empty((end-begin,2048),dtype=np.float32)
    for local,token in enumerate(range(begin,end)):
        state.set(rng.standard_normal(2048,dtype=np.float32),stream=stream)
        started=time.perf_counter_ns()
        for layer in range(LAYERS):
            slots,_hits,miss_pos,miss_copies=plan_layer(route[token,layer],layer,layer_bases,fixed_ids,fixed_set,dynamic)
            misses[local]+=len(miss_pos)
            for expert,slot in miss_copies: copy_expert(stream,pinned,cache,layer_bases,layer,expert,slot)
            slots_d.set(slots,stream=stream)
            launch_layer(kernel_set,stream,state,cache_view,slots_d,gate,up,down)
        stream.synchronize(); times[local]=(time.perf_counter_ns()-started)/1e6
        outputs[local]=cp.asnumpy(state)
        if local%64==0: print(json.dumps({"mode":"serial","domain":domain,"token":token,"misses":int(misses[local]),"ms":float(times[local])}),flush=True)
    return times,misses,outputs


def run_domain_fused(domain,route,fixed_ids,begin,end,pinned,cache,cache_view,
                     layer_bases,compute_stream,copy_stream,kernel_set,seed):
    initialize_static(compute_stream,pinned,cache,layer_bases,fixed_ids)
    fixed_set=[frozenset(row) for row in fixed_ids]
    dynamic=[OrderedDict() for _ in range(LAYERS)]
    state=cp.empty(2048,dtype=cp.float32); activation=cp.empty(6144,dtype=cp.float32)
    down=cp.empty(16384,dtype=cp.float32)
    metadata=cp.empty(40,dtype=cp.int32)
    hit_slots_d=metadata[8:16]; hit_positions_d=metadata[16:24]
    miss_slots_d=metadata[24:32]; miss_positions_d=metadata[32:40]
    pinned_meta=cp.cuda.alloc_pinned_memory(LAYERS*160)
    host_metadata=np.frombuffer(pinned_meta,dtype=np.int32,count=LAYERS*40).reshape(LAYERS,40)
    ready=[cp.cuda.Event() for _ in range(LAYERS)]
    swiglu,down_kernel,reduce=kernel_set; rng=np.random.default_rng(seed)
    times=np.empty(end-begin); misses=np.zeros(end-begin,dtype=np.int64)
    outputs=np.empty((end-begin,2048),dtype=np.float32); records=0; mixed=0
    for local,token in enumerate(range(begin,end)):
        state.set(rng.standard_normal(2048,dtype=np.float32),stream=compute_stream)
        started=time.perf_counter_ns()
        for layer in range(LAYERS):
            slots,hits,miss_pos,miss_copies=plan_layer(route[token,layer],layer,layer_bases,fixed_ids,fixed_set,dynamic)
            hc=len(hits); mc=len(miss_pos); misses[local]+=mc; records+=mc; mixed+=int(hc>0 and mc>0)
            host=host_metadata[layer]; host.fill(0)
            if hc:
                hp=np.asarray(hits,dtype=np.int32); host[8:8+hc]=slots[hp]; host[16:16+hc]=hp
            if mc:
                mp=np.asarray(miss_pos,dtype=np.int32); host[24:24+mc]=slots[mp]; host[32:32+mc]=mp
            cp.cuda.runtime.memcpyAsync(metadata.data.ptr,pinned_meta.ptr+layer*160,160,cp.cuda.runtime.memcpyHostToDevice,compute_stream.ptr)
            for expert,slot in miss_copies: copy_expert(copy_stream,pinned,cache,layer_bases,layer,expert,slot)
            if mc: ready[layer].record(copy_stream)
            if hc:
                swiglu((hc*768,),(256,),(state,cache_view,hit_slots_d,hit_positions_d,activation),stream=compute_stream)
                down_kernel((hc*2048,),(256,),(activation,cache_view,hit_slots_d,hit_positions_d,down),stream=compute_stream)
            if mc:
                compute_stream.wait_event(ready[layer])
                swiglu((mc*768,),(256,),(state,cache_view,miss_slots_d,miss_positions_d,activation),stream=compute_stream)
                down_kernel((mc*2048,),(256,),(activation,cache_view,miss_slots_d,miss_positions_d,down),stream=compute_stream)
            reduce((8,),(256,),(down,state),stream=compute_stream)
        compute_stream.synchronize(); copy_stream.synchronize()
        times[local]=(time.perf_counter_ns()-started)/1e6; outputs[local]=cp.asnumpy(state)
        if local%64==0: print(json.dumps({"mode":"fused_async","domain":domain,"token":token,"misses":int(misses[local]),"ms":float(times[local])}),flush=True)
    return times,misses,outputs,records,mixed


def run(phase):
    lock=json.loads(LOCK_PATH.read_text(encoding="utf-8")); evaluator=json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"))
    provenance={PREREG:lock["preregistration_sha256"],BANK_RESULT_PATH:lock["p1d_bank_result_sha256"],P1D_VERIFY:lock["p1d_verification_sha256"],P1C_CAPTURE:lock["p1c_route_capture_sha256"],P1C_VALIDATION:lock["p1c_cache_validation_sha256"],P1C_TEST:lock["p1c_cache_test_sha256"]}
    if any(sha256(p)!=h for p,h in provenance.items()): raise ValueError("P4C provenance mismatch")
    if sha256(LOCK_PATH)!=evaluator["input_lock_sha256"] or sha256(Path(__file__))!=evaluator["evaluator_sha256"]: raise ValueError("P4C evaluator mismatch")
    output=R/f"p4c_fused_overlap_{phase}.json"; report=R/f"P4C_FUSED_OVERLAP_{phase.upper()}.md"
    if output.exists() or report.exists(): raise FileExistsError("refusing to overwrite P4C")
    if phase=="test":
        vp=R/"p4c_fused_overlap_validation.json"
        if not vp.exists() or json.loads(vp.read_text(encoding="utf-8"))["status"]!="p4c_validation_pass_test_authorized": raise RuntimeError("P4C test not authorized")
    routes,route_hashes=load_routes(ROOT/lock["route_dir"]); fixed=static_sets(routes)
    bank=json.loads(BANK_RESULT_PATH.read_text(encoding="utf-8")); pinned,pinned_hashes,pin_ms=pin_bank(bank); layer_bases=bases()
    serial_set=serial_kernels(); fused_set=fused_kernels(); cp.get_default_memory_pool().free_all_blocks()
    free_before,total=cp.cuda.runtime.memGetInfo(); cache=cp.cuda.alloc(CACHE_BYTES); cache_view=cp.ndarray((CACHE_BYTES,),dtype=cp.uint8,memptr=cache); trunk=cp.cuda.alloc(TRUNK_BYTES); kv=cp.cuda.alloc(KV_BYTES)
    compute=cp.cuda.Stream(non_blocking=True); copy=cp.cuda.Stream(non_blocking=True)
    with compute:
        cp.cuda.runtime.memsetAsync(cache.ptr,0,CACHE_BYTES,compute.ptr); cp.cuda.runtime.memsetAsync(trunk.ptr,0,TRUNK_BYTES,compute.ptr); cp.cuda.runtime.memsetAsync(kv.ptr,0,KV_BYTES,compute.ptr)
    compute.synchronize(); free_after,_=cp.cuda.runtime.memGetInfo()
    begin,end=lock["partitions"][phase]; domains=("general",) if phase=="smoke" else DOMAINS
    sr={}; ar={}; correctness={}; all_st=[];all_sm=[];all_at=[];all_am=[];records=0;mixed=0
    for domain in domains:
        seed=lock["initial_state_seed"]+DOMAINS.index(domain)*100000+begin
        st,sm,so=run_domain_serial(domain,routes[domain],fixed[domain],begin,end,pinned,cache,cache_view,layer_bases,compute,serial_set,seed)
        at,am,ao,rec,mix=run_domain_fused(domain,routes[domain],fixed[domain],begin,end,pinned,cache,cache_view,layer_bases,compute,copy,fused_set,seed)
        delta=ao-so; correctness[domain]={"exact":bool(np.array_equal(ao,so)),"max_abs":float(np.abs(delta).max()),"relative_l2":float(np.linalg.norm(delta)/max(np.linalg.norm(so),1e-30)),"finite":bool(np.isfinite(ao).all()) and bool(np.isfinite(so).all()),"values":int(ao.size)}
        sr[domain]=result_row(st,sm);ar[domain]=result_row(at,am);all_st.append(st);all_sm.append(sm);all_at.append(at);all_am.append(am);records+=rec;mixed+=mix
    st=np.concatenate(all_st);sm=np.concatenate(all_sm);at=np.concatenate(all_at);am=np.concatenate(all_am)
    agg={"serial":{"tokens":int(st.size),"wall_ms":percentile(st),"misses":percentile(sm.astype(float)),"miss_slope_ms":positive_slope(st,sm)},"fused_async":{"tokens":int(at.size),"wall_ms":percentile(at),"misses":percentile(am.astype(float)),"miss_slope_ms":positive_slope(at,am)},"speedup":float(st.mean()/at.mean()),"prediction_relative_error":float(abs(at.mean()-lock["prediction_ms"]["mean"])/lock["prediction_ms"]["mean"])}
    gates={"full_bank_pinned_and_hashed":len(pinned)==LAYERS and len(pinned)*388497408==BANK_BYTES,"pinned_hashes_match":pinned_hashes=={str(i):bank["manifests"][str(i)]["artifact_sha256"] for i in range(LAYERS)},"device_co_resident_and_scratch":free_after>=lock["gates"]["minimum_scratch_bytes"],"async_misses_match_serial":bool(np.array_equal(am,sm)),"copy_records_exact":records==int(am.sum()),"copy_bytes_exact":records*EXPERT_BYTES==int(am.sum())*EXPERT_BYTES,"outputs_exact":all(x["exact"] for x in correctness.values()),"outputs_within_limits":all(x["max_abs"]<=lock["gates"]["max_abs_output_error"] and x["relative_l2"]<=lock["gates"]["relative_l2_output_error"] for x in correctness.values()),"finite":all(x["finite"] for x in correctness.values()) and bool(np.isfinite(st).all()) and bool(np.isfinite(at).all()),"async_mean_le_20":at.mean()<=lock["gates"]["aggregate_mean_ms_max"],"async_p95_le_25":np.percentile(at,95)<=lock["gates"]["aggregate_p95_ms_max"],"all_domain_mean_le_22":all(x["wall_ms_stats"]["mean"]<=lock["gates"]["all_domain_mean_ms_max"] for x in ar.values()),"all_domain_p95_le_30":all(x["wall_ms_stats"]["p95"]<=lock["gates"]["all_domain_p95_ms_max"] for x in ar.values()),"serial_mean_le_35":st.mean()<=lock["gates"]["serial_mean_ms_max"],"serial_miss_slope_positive":agg["serial"]["miss_slope_ms"]>0,"prediction_within_15pct":agg["prediction_relative_error"]<=lock["prediction_ms"]["tolerance_fraction"]}
    gates={k:bool(v) for k,v in gates.items()};passed=all(gates.values())
    status=("p4c_smoke_pass" if all(x["exact"] for x in correctness.values()) else "p4c_smoke_fail") if phase=="smoke" else (("p4c_validation_pass_test_authorized" if passed else "p4c_validation_closed_test_unopened") if phase=="validation" else ("p4c_fused_overlap_pass" if passed else "p4c_fused_overlap_closed"))
    result={"kind":"streamq5_moe_p4c_fused_overlap","completed_utc":datetime.now(timezone.utc).isoformat(),"phase":phase,"status":status,"inputs":{"preregistration_sha256":sha256(PREREG),"input_lock_sha256":sha256(LOCK_PATH),"evaluator_lock_sha256":sha256(EVALUATOR_LOCK),"evaluator_sha256":sha256(Path(__file__)),"route_capture_sha256":sha256(P1C_CAPTURE),"route_artifact_sha256":route_hashes},"physical":{"pinned_bank_bytes":len(pinned)*388497408,"pin_and_hash_ms":pin_ms,"cache_bytes":CACHE_BYTES,"trunk_reservation_bytes":TRUNK_BYTES,"kv_reservation_bytes":KV_BYTES,"free_before_bytes":int(free_before),"free_after_bytes":int(free_after),"total_vram_bytes":int(total),"copied_records":int(records),"copied_bytes":int(records*EXPERT_BYTES),"mixed_layer_events":int(mixed)},"policy":{"static":20,"dynamic_layers_0_7":15,"dynamic_layers_8_47":14,"lookahead_layers":0,"mixed_layer_kernel_launches":5,"metadata_h2d_bytes_per_layer":160},"serial":sr,"fused_async":ar,"correctness":correctness,"aggregate":agg,"gates":gates,"claim_boundary":"Physical causal fused full-overlap expert dataplane on independent routes only; no full-model decode."}
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8");report.write_text(f"# STREAMQ5-MoE P4C — fused overlap {phase}\n\nStatus: **{status}**. Serial mean/p95 {agg['serial']['wall_ms']['mean']:.3f}/{agg['serial']['wall_ms']['p95']:.3f} ms; fused async {agg['fused_async']['wall_ms']['mean']:.3f}/{agg['fused_async']['wall_ms']['p95']:.3f} ms.\n",encoding="utf-8");return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("smoke","validation","test"),required=True);a=p.parse_args();r=run(a.phase);print(json.dumps({"status":r["status"],"aggregate":r["aggregate"],"gates":r["gates"]},indent=2));
    if r["status"].endswith("fail") or "closed" in r["status"]: raise SystemExit(1)


if __name__=="__main__": main()
