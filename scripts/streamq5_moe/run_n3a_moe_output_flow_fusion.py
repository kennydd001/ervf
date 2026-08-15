from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from moe_lab.reporting import ROOT as PROJECT_ROOT
from scripts.streamq5_moe.run_p3a_integrated_expert import LAYERS
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7a_kernel_roofline import load_q5
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE, comparison, stats

R = PROJECT_ROOT / "reports/streamq5_moe"
PREREG = R / "N3A_MOE_OUTPUT_FLOW_FUSION_PREREGISTRATION.md"
OUTPUT = R / "n3a_moe_output_flow_fusion.json"
SEED = 120823

SOURCE = r'''
extern "C" __global__ void q5_down_weighted_residual_ervf16(
    const float* activation, const unsigned char* cache, const int* slots,
    const int* positions, const int* order, const float* weights,
    const float* residual, float* state) {
    int group=(int)threadIdx.x>>4; int lane=(int)threadIdx.x&15;
    int row_group=group>>3; int expert=group&7;
    int row=(int)blockIdx.x*2+row_group;
    if(row>=2048)return;
    int output_expert=positions[expert];
    long long base=(long long)slots[expert]*3035136LL+2LL*1011712LL;
    const unsigned char* packed=cache+base+64;
    const unsigned short* scales=(const unsigned short*)(cache+base+64+983040);
    float value=q5_ervf_row<16>(activation+output_expert*768,packed,scales,row,768,lane);
    __shared__ float values[16];
    if(lane==0)values[row_group*8+output_expert]=round_bf16(value);
    __syncthreads();
    if(group==(row_group<<3)&&lane==0){
        float sum=0.0f;
        #pragma unroll
        for(int item=0;item<8;++item){
            int position=order[item];
            float term=round_bf16(values[row_group*8+position]*weights[position]);
            sum=round_bf16(sum+term);
        }
        state[row]=round_bf16(residual[row]+sum);
    }
}
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(stream, launches, warmups, rounds):
    names=list(launches); values={name:[] for name in names}
    for i in range(warmups):
        for name in (names if i%2==0 else list(reversed(names))): launches[name]()
    stream.synchronize()
    for i in range(rounds):
        for name in (names if i%2==0 else list(reversed(names))):
            begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record(stream);launches[name]();end.record(stream);end.synchronize()
            values[name].append(float(cp.cuda.get_elapsed_time(begin,end)))
    return {name:{"event_ms":v,"stats":stats(v)} for name,v in values.items()}


def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    q5_mem,q5=load_q5(); names=("q5_down_ervf16","weighted_residual","q5_down_weighted_residual_ervf16")
    module=cp.RawModule(code=CUDA_SOURCE+ERVF_SOURCE+SOURCE,options=("--std=c++11",),name_expressions=names)
    k={name:module.get_function(name) for name in names};stream=cp.cuda.Stream(non_blocking=True)
    rng=np.random.default_rng(SEED)
    activations=[cp.asarray(rng.standard_normal(8*768,dtype=np.float32)) for _ in range(LAYERS)]
    residuals=[cp.asarray(rng.standard_normal(2048,dtype=np.float32)) for _ in range(LAYERS)]
    slots=[cp.asarray(np.arange(layer*8,layer*8+8,dtype=np.int32)) for layer in range(LAYERS)]
    positions=cp.asarray(np.arange(8,dtype=np.int32));orders=[];weights=[]
    for _ in range(LAYERS):
        order=rng.permutation(8).astype(np.int32); w=rng.random(8,dtype=np.float32);w/=w.sum(dtype=np.float32)
        orders.append(cp.asarray(order));weights.append(cp.asarray(w))
    down=cp.empty(8*2048,dtype=cp.float32);state=cp.empty(2048,dtype=cp.float32)

    def layer(kind,idx):
        if kind=="reference":
            k["q5_down_ervf16"]((1024,),(256,),(activations[idx],q5,slots[idx],positions,down),stream=stream)
            k["weighted_residual"]((8,),(256,),(down,orders[idx],weights[idx],residuals[idx],state),stream=stream)
        else:
            k["q5_down_weighted_residual_ervf16"]((1024,),(256,),(activations[idx],q5,slots[idx],positions,orders[idx],weights[idx],residuals[idx],state),stream=stream)

    def plane(kind,capture=False):
        out=np.empty((LAYERS,2048),dtype=np.float32) if capture else None
        for idx in range(LAYERS):
            layer(kind,idx)
            if capture:stream.synchronize();out[idx]=cp.asnumpy(state)
        return out

    reference=plane("reference",True);candidate=plane("candidate",True);correct=comparison(candidate,reference)
    validation=paired(stream,{"reference":lambda:plane("reference"),"candidate":lambda:plane("candidate")},5,30)
    validation["p50_ratio"]=validation["candidate"]["stats"]["p50"]/validation["reference"]["stats"]["p50"]
    validation["p95_ratio"]=validation["candidate"]["stats"]["p95"]/validation["reference"]["stats"]["p95"]
    opened=correct["bitwise_equal"] and validation["p50_ratio"]<=0.98;test=None
    if opened:
        test=paired(stream,{"reference":lambda:plane("reference"),"candidate":lambda:plane("candidate")},10,120)
        test["p50_ratio"]=test["candidate"]["stats"]["p50"]/test["reference"]["stats"]["p50"]
        test["p95_ratio"]=test["candidate"]["stats"]["p95"]/test["reference"]["stats"]["p95"]
        test["pass"]=test["p50_ratio"]<=0.97 and test["p95_ratio"]<=1.0
    result={"kind":"streamq5_moe_n3a_moe_output_flow_fusion","completed_utc":datetime.now(timezone.utc).isoformat(),
            "inputs":{"preregistration_sha256":sha256(PREREG),"seed":SEED,"layers":LAYERS,"physical_q5_bytes":int(q5.size)},
            "correctness":correct,"validation":validation,"test_opened":opened,"test":test,"overall_pass":bool(test and test["pass"]),
            "claim_boundary":"Isolated resident all-eight MoE output flow; no miss split, full decoder, other fusion chain or SOTA claim."}
    OUTPUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"correctness":correct,"validation_ratios":{"p50":validation["p50_ratio"],"p95":validation["p95_ratio"]},"test_opened":opened,"test_ratios":None if test is None else {"p50":test["p50_ratio"],"p95":test["p95_ratio"]},"overall_pass":result["overall_pass"]},indent=2))


if __name__=="__main__":main()
