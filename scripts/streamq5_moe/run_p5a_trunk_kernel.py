from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P5A_PHYSICAL_TRUNK_PREREGISTRATION.md"
BANK_RESULT = R / "p5a_trunk_bank_result.json"
BANK_VERIFY = R / "p5a_trunk_bank_verification.json"
LOCK_PATH = R / "p5a_trunk_kernel_input_lock.json"
EVALUATOR_LOCK = R / "p5a_trunk_kernel_evaluator_lock.json"


CUDA_SOURCE = r'''
__device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}
extern "C" __global__ void q8_gemv(
    const float* x, const unsigned char* bank, long long base,
    long long code_bytes, int rows, int cols, float* output) {
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    const signed char* codes = (const signed char*)(bank + base);
    const unsigned short* scales = (const unsigned short*)(bank + base + code_bytes);
    int groups_per_row = cols >> 7;
    float sum = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += blockDim.x) {
        float scale = bf16_to_float(scales[row * groups_per_row + (col >> 7)]);
        sum += ((float)codes[(long long)row * cols + col]) * scale * x[col];
    }
    __shared__ float reduction[256];
    reduction[threadIdx.x] = sum; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = reduction[0];
}
'''


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*2**20),b""):digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    x=np.asarray(values,dtype=np.float64)
    return {"mean":float(x.mean()),"p50":float(np.percentile(x,50)),"p95":float(np.percentile(x,95)),"p99":float(np.percentile(x,99)),"max":float(x.max())}


def pin_bank(bank):
    total=bank["aggregate"]["bytes"]
    memory=cp.cuda.alloc_pinned_memory(total)
    host=np.frombuffer(memory,dtype=np.uint8,count=total)
    offsets={};cursor=0;digest=hashlib.sha256()
    for index,record in enumerate(bank["records"]):
        path=ROOT/record["artifact"]
        with path.open("rb") as handle:
            view=memoryview(host[cursor:cursor+record["bytes"]])
            if handle.readinto(view)!=record["bytes"]:raise RuntimeError("short trunk record read")
        raw=memoryview(host[cursor:cursor+record["bytes"]]); digest.update(raw)
        if hashlib.sha256(raw).hexdigest()!=record["artifact_sha256"]:raise ValueError("trunk record hash mismatch")
        offsets[index]=cursor;cursor+=record["bytes"]
    if cursor!=total:raise RuntimeError("trunk bank byte mismatch")
    return memory,host,offsets,digest.hexdigest()


def launch_record(kernel,stream,bank_gpu,record,base,x,out):
    kernel((record["rows"],),(256,),(x,bank_gpu,np.int64(base),np.int64(record["code_bytes"]),np.int32(record["rows"]),np.int32(record["cols"]),out),stream=stream)


def correctness(kernel,stream,bank_gpu,host,offsets,bank,indices,seed):
    checks=[]
    for index in indices:
        record=bank["records"][index];rng=np.random.default_rng(seed+index)
        x_host=rng.standard_normal(record["cols"],dtype=np.float32)
        x=cp.asarray(x_host);out=cp.empty(record["rows"],dtype=cp.float32)
        launch_record(kernel,stream,bank_gpu,record,offsets[index],x,out);stream.synchronize()
        rows=sorted(set((0,record["rows"]//3,record["rows"]//2,(2*record["rows"])//3,record["rows"]-1)))
        observed=cp.asnumpy(out)[rows]
        base=offsets[index];codes=np.frombuffer(host[base:base+record["code_bytes"]],dtype=np.int8).reshape(record["rows"],record["cols"])
        scales=np.frombuffer(host[base+record["code_bytes"]:base+record["bytes"]],dtype="<u2").astype(np.uint32)
        scales=(scales<<16).view(np.float32).reshape(record["rows"],record["cols"]//128)
        expected=[]
        for row in rows:
            deq=codes[row].astype(np.float32)*np.repeat(scales[row],128)
            expected.append(float(deq@x_host))
        expected=np.asarray(expected,dtype=np.float32);delta=observed-expected
        max_abs=float(np.abs(delta).max());rel=float(np.linalg.norm(delta)/max(np.linalg.norm(expected),1e-30))
        checks.append({"record_index":index,"layer":record["layer"],"name":record["name"],"rows_checked":rows,"max_abs":max_abs,"relative_l2":rel,"finite":bool(np.isfinite(observed).all())})
    return checks


def build_plane(bank):
    by={(r["layer"],r["name"]):(i,r) for i,r in enumerate(bank["records"])}
    return [[by[(layer,name)] for name in ("q","k","v","o","router")] for layer in range(48)],by[(48,"head")]


def enqueue_plane(kernel,stream,bank_gpu,offsets,plane,state,initial,q,k,v,router,head):
    cp.cuda.runtime.memcpyAsync(state.data.ptr,initial.data.ptr,state.nbytes,cp.cuda.runtime.memcpyDeviceToDevice,stream.ptr)
    for layer_records in plane[0]:
        (qi,qr),(ki,kr),(vi,vr),(oi,orr),(ri,rr)=layer_records
        launch_record(kernel,stream,bank_gpu,qr,offsets[qi],state,q)
        launch_record(kernel,stream,bank_gpu,kr,offsets[ki],state,k)
        launch_record(kernel,stream,bank_gpu,vr,offsets[vi],state,v)
        launch_record(kernel,stream,bank_gpu,orr,offsets[oi],q,state)
        launch_record(kernel,stream,bank_gpu,rr,offsets[ri],state,router)
    hi,hr=plane[1];launch_record(kernel,stream,bank_gpu,hr,offsets[hi],state,head)


def measure(enqueue,stream,warmups,iterations):
    for _ in range(warmups):enqueue()
    stream.synchronize();host=[];event=[]
    for _ in range(iterations):
        start=cp.cuda.Event();end=cp.cuda.Event();wall=time.perf_counter_ns();start.record(stream);enqueue();end.record(stream);end.synchronize()
        host.append((time.perf_counter_ns()-wall)/1e6);event.append(float(cp.cuda.get_elapsed_time(start,end)))
    return {"iterations":iterations,"host_ms":host,"event_ms":event,"host_stats":stats(host),"event_stats":stats(event)}


def run(phase):
    lock=json.loads(LOCK_PATH.read_text(encoding="utf-8"));evaluator=json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"));bank=json.loads(BANK_RESULT.read_text(encoding="utf-8"));verification=json.loads(BANK_VERIFY.read_text(encoding="utf-8"))
    if sha256(PREREG)!=lock["preregistration_sha256"] or sha256(BANK_RESULT)!=lock["bank_result_sha256"] or sha256(BANK_VERIFY)!=lock["bank_verification_sha256"]:raise ValueError("P5A provenance mismatch")
    if sha256(LOCK_PATH)!=evaluator["input_lock_sha256"] or sha256(Path(__file__))!=evaluator["evaluator_sha256"]:raise ValueError("P5A evaluator mismatch")
    if verification["status"]!="p5a_trunk_bank_verification_pass":raise RuntimeError("verified trunk bank required")
    output=R/f"p5a_trunk_kernel_{phase}.json";report=R/f"P5A_TRUNK_KERNEL_{phase.upper()}.md"
    if output.exists() or report.exists():raise FileExistsError("refusing to overwrite P5A kernel output")
    if phase=="test":
        vp=R/"p5a_trunk_kernel_validation.json"
        if not vp.exists() or json.loads(vp.read_text(encoding="utf-8"))["status"]!="p5a_validation_pass_test_authorized":raise RuntimeError("P5A test not authorized")
    pinned,host,offsets,bank_digest=pin_bank(bank);kernel=cp.RawKernel(CUDA_SOURCE,"q8_gemv",options=("--std=c++11",));plane=build_plane(bank)
    cp.get_default_memory_pool().free_all_blocks();free_before,total=cp.cuda.runtime.memGetInfo()
    expert_cache=cp.cuda.alloc(lock["co_resident_bytes"]["expert_cache"]);bank_gpu=cp.cuda.alloc(bank["aggregate"]["bytes"]);kv=cp.cuda.alloc(lock["co_resident_bytes"]["kv"]);stream=cp.cuda.Stream(non_blocking=True)
    cp.cuda.runtime.memcpyAsync(bank_gpu.ptr,pinned.ptr,bank["aggregate"]["bytes"],cp.cuda.runtime.memcpyHostToDevice,stream.ptr);stream.synchronize();free_after,_=cp.cuda.runtime.memGetInfo()
    checks=correctness(kernel,stream,bank_gpu,host,offsets,bank,lock["correctness_record_indices"],lock["initial_state_seed"])
    rng=np.random.default_rng(lock["initial_state_seed"]);initial=cp.asarray(rng.standard_normal(2048,dtype=np.float32));state=cp.empty(2048,dtype=cp.float32);q=cp.empty(4096,dtype=cp.float32);k=cp.empty(512,dtype=cp.float32);v=cp.empty(512,dtype=cp.float32);router=cp.empty(128,dtype=cp.float32);head=cp.empty(151936,dtype=cp.float32)
    enqueue=lambda:enqueue_plane(kernel,stream,bank_gpu,offsets,plane,state,initial,q,k,v,router,head)
    warmups=2 if phase=="smoke" else lock["warmups"];timing=measure(enqueue,stream,warmups,lock["iterations"][phase]);finite=bool(cp.isfinite(head).all().get()) and bool(cp.isfinite(state).all().get())
    ratio=timing["event_stats"]["p50"]/timing["host_stats"]["p50"]
    gates={"verified_bank":verification["status"]=="p5a_trunk_bank_verification_pass","physical_bytes_exact":bank["aggregate"]["bytes"]==1248931840 and bank["aggregate"]["weights"]==1229717504,"co_resident_and_scratch":free_after>=lock["co_resident_bytes"]["minimum_scratch"],"correctness_15_records":len(checks)==15 and all(c["finite"] and c["max_abs"]<=lock["gates"]["max_abs"] and c["relative_l2"]<=lock["gates"]["relative_l2"] for c in checks),"finite":finite and all(np.isfinite(timing[k]).all() for k in ("host_ms","event_ms")),"host_mean_le_30":timing["host_stats"]["mean"]<=lock["gates"]["host_mean_ms_max"],"host_p95_le_35":timing["host_stats"]["p95"]<=lock["gates"]["host_p95_ms_max"],"event_host_ratio_ge_0_90":ratio>=lock["gates"]["event_host_p50_ratio_min"],"event_host_ratio_le_1_05":ratio<=lock["gates"]["event_host_p50_ratio_max"]};gates={k:bool(v) for k,v in gates.items()};passed=all(gates.values())
    status=("p5a_smoke_pass" if gates["correctness_15_records"] and finite else "p5a_smoke_fail") if phase=="smoke" else (("p5a_validation_pass_test_authorized" if passed else "p5a_validation_closed_test_unopened") if phase=="validation" else ("p5a_physical_trunk_pass" if passed else "p5a_physical_trunk_closed"))
    result={"kind":"streamq5_moe_p5a_physical_int8_projection_plane","completed_utc":datetime.now(timezone.utc).isoformat(),"phase":phase,"status":status,"inputs":{"preregistration_sha256":sha256(PREREG),"bank_result_sha256":sha256(BANK_RESULT),"bank_verification_sha256":sha256(BANK_VERIFY),"input_lock_sha256":sha256(LOCK_PATH),"evaluator_lock_sha256":sha256(EVALUATOR_LOCK),"evaluator_sha256":sha256(Path(__file__))},"environment":{"gpu":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),"cupy":cp.__version__,"cuda_runtime":cp.cuda.runtime.runtimeGetVersion()},"physical":{"trunk_bank_bytes":bank["aggregate"]["bytes"],"expert_cache_bytes":lock["co_resident_bytes"]["expert_cache"],"kv_bytes":lock["co_resident_bytes"]["kv"],"free_before_bytes":int(free_before),"free_after_bytes":int(free_after),"total_vram_bytes":int(total),"pinned_bank_sha256":bank_digest},"workload":{"matrix_records":241,"weights_per_iteration":bank["aggregate"]["weights"],"kernel_launches_per_iteration":241,"warmups":warmups},"correctness":checks,"timing":timing,"ratios":{"event_to_host_p50":ratio,"effective_weight_gs_host_p50":bank["aggregate"]["weights"]/(timing["host_stats"]["p50"]*1e6)},"gates":gates,"claim_boundary":"Physical INT8 attention/router/head GEMV projection plane only; no attention/KV/full-model decode."}
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8");report.write_text(f"# P5A fysieke trunkkernel — {phase}\n\nStatus: **{status}**. Host mean/p95 {timing['host_stats']['mean']:.3f}/{timing['host_stats']['p95']:.3f} ms.\n",encoding="utf-8");return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("smoke","validation","test"),required=True);a=p.parse_args();r=run(a.phase);print(json.dumps({"status":r["status"],"physical":r["physical"],"timing":{"host":r["timing"]["host_stats"],"event":r["timing"]["event_stats"]},"ratios":r["ratios"],"gates":r["gates"]},indent=2));
    if r["status"].endswith("fail") or "closed" in r["status"]:raise SystemExit(1)


if __name__=="__main__":main()
