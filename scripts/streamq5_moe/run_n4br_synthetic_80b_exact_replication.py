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

from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE, comparison, stats

R=ROOT/"reports/streamq5_moe";PREREG=R/"N4BR_SYNTHETIC_80B_EXACT_REPLICATION_PREREGISTRATION.md"
OUTPUT=R/"n4br_synthetic_80b_exact_replication.json";N4A=R/"n4a_synthetic_80b_shape_capacity.json";N1C=R/"n1c_generalized_exact_reduction_autotuner.json";N4B=R/"n4b_synthetic_80b_gpu_shape.json"
LAYERS=48;ACTIVE=11;HIDDEN=2048;INTER=512;MATRIX_BYTES=675840;EXPERT_BYTES=2027520;SLOTS=LAYERS*ACTIVE
CODE_BYTES=655360;SCALE_BYTES=16384;WIDTHS=(8,16,32);SEED=120825

SOURCE=r'''
#define DEFINE_80B(WIDTH) \
extern "C" __global__ void q5_80b_gate_up_##WIDTH(const float* x,const unsigned char* cache,const int* slots,float* gate,float* up){ \
 int groups=256/WIDTH;int group=(int)threadIdx.x/WIDTH;int lane=(int)threadIdx.x&(WIDTH-1);int gr=(int)blockIdx.x*groups+group;if(gr>=11*1024)return;int expert=gr/1024;int local=gr-expert*1024;int proj=local>=512;int row=local-proj*512;long long base=(long long)slots[expert]*2027520LL+(long long)proj*675840LL;const unsigned char* packed=cache+base+64;const unsigned short* scales=(const unsigned short*)(cache+base+64+655360);float value=q5_ervf_row<WIDTH>(x,packed,scales,row,2048,lane);if(lane==0){if(proj)up[expert*512+row]=round_bf16(value);else gate[expert*512+row]=round_bf16(value);}} \
extern "C" __global__ void q5_80b_down_##WIDTH(const float* activation,const unsigned char* cache,const int* slots,float* down){ \
 int groups=256/WIDTH;int group=(int)threadIdx.x/WIDTH;int lane=(int)threadIdx.x&(WIDTH-1);int gr=(int)blockIdx.x*groups+group;if(gr>=11*2048)return;int expert=gr/2048;int row=gr-expert*2048;long long base=(long long)slots[expert]*2027520LL+2LL*675840LL;const unsigned char* packed=cache+base+64;const unsigned short* scales=(const unsigned short*)(cache+base+64+655360);float value=q5_ervf_row<WIDTH>(activation+expert*512,packed,scales,row,512,lane);if(lane==0)down[expert*2048+row]=round_bf16(value);} 
DEFINE_80B(8)
DEFINE_80B(16)
DEFINE_80B(32)
extern "C" __global__ void swiglu_80b(float* gate,const float* up){int i=(int)blockIdx.x*blockDim.x+threadIdx.x;if(i<11*512){float g=round_bf16(gate[i]);float u=round_bf16(up[i]);float silu=round_bf16(g/(1.0f+expf(-g)));gate[i]=round_bf16(silu*u);}}
'''

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def paired(stream,launches,warmups,rounds):
 names=list(launches);values={n:[] for n in names}
 for i in range(warmups):
  for n in (names if i%2==0 else list(reversed(names))):launches[n]()
 stream.synchronize()
 for i in range(rounds):
  for n in (names if i%2==0 else list(reversed(names))):
   a,b=cp.cuda.Event(),cp.cuda.Event();a.record(stream);launches[n]();b.record(stream);b.synchronize();values[n].append(float(cp.cuda.get_elapsed_time(a,b)))
 return {n:{"event_ms":v,"stats":stats(v)}for n,v in values.items()}

def load_bank():
 total=SLOTS*EXPERT_BYTES;mem=cp.cuda.alloc(total);device=cp.ndarray((total,),dtype=cp.uint8,memptr=mem)
 host_mem=cp.cuda.alloc_pinned_memory(EXPERT_BYTES);host=np.frombuffer(host_mem,dtype=np.uint8,count=EXPERT_BYTES);host.fill(0x55)
 for proj in range(3):host[proj*MATRIX_BYTES+64+CODE_BYTES:proj*MATRIX_BYTES+64+CODE_BYTES+SCALE_BYTES].view(np.uint16).fill(0x3c00)
 stream=cp.cuda.Stream(non_blocking=True)
 for slot in range(SLOTS):cp.cuda.runtime.memcpyAsync(mem.ptr+slot*EXPERT_BYTES,host_mem.ptr,EXPERT_BYTES,cp.cuda.runtime.memcpyHostToDevice,stream.ptr)
 stream.synchronize();return mem,device

def main():
 if OUTPUT.exists():raise FileExistsError(OUTPUT)
 n4a=json.loads(N4A.read_text(encoding="utf-8"));n1c=json.loads(N1C.read_text(encoding="utf-8"))
 if not n4a["overall_pass"]:raise RuntimeError("N4A pass required")
 bank_mem,bank=load_bank();names=["swiglu_80b"]
 for w in WIDTHS:names.extend((f"q5_80b_gate_up_{w}",f"q5_80b_down_{w}"))
 module=cp.RawModule(code=CUDA_SOURCE+ERVF_SOURCE+SOURCE,options=("--std=c++11",),name_expressions=tuple(names));k={n:module.get_function(n)for n in names};stream=cp.cuda.Stream(non_blocking=True)
 rng=np.random.default_rng(SEED);x=cp.asarray(rng.standard_normal(HIDDEN,dtype=np.float32));slots=[cp.asarray(np.arange(l*ACTIVE,l*ACTIVE+ACTIVE,dtype=np.int32))for l in range(LAYERS)]
 gate=cp.empty(ACTIVE*INTER,dtype=cp.float32);up=cp.empty_like(gate);down=cp.empty(ACTIVE*HIDDEN,dtype=cp.float32)
 def plane(width,capture=False):
  out=np.empty((LAYERS,ACTIVE*(INTER+INTER+HIDDEN)),dtype=np.float32)if capture else None;groups=256//width
  for layer in range(LAYERS):
   k[f"q5_80b_gate_up_{width}"](((ACTIVE*2*INTER+groups-1)//groups,),(256,),(x,bank,slots[layer],gate,up),stream=stream)
   k["swiglu_80b"](((ACTIVE*INTER+255)//256,),(256,),(gate,up),stream=stream)
   k[f"q5_80b_down_{width}"](((ACTIVE*HIDDEN+groups-1)//groups,),(256,),(gate,bank,slots[layer],down),stream=stream)
   if capture:stream.synchronize();out[layer]=np.concatenate((cp.asnumpy(gate),cp.asnumpy(up),cp.asnumpy(down)))
  return out
 ref=plane(16,True);ref_digest=hashlib.sha256(ref.tobytes()).hexdigest();correct={};output_digests={"16_reference":ref_digest}
 for w in WIDTHS:
  observed=plane(w,True);output_digests[str(w)]=hashlib.sha256(observed.tobytes()).hexdigest();correct[str(w)]=comparison(observed,ref)
 launches={str(w):(lambda width=w:plane(width))for w in WIDTHS};validation=paired(stream,launches,5,30);eligible=[w for w in WIDTHS if correct[str(w)]["bitwise_equal"]];selected=min(eligible,key=lambda w:validation[str(w)]["stats"]["p50"])
 test=paired(stream,{"16":launches["16"],str(selected):launches[str(selected)]},10,120)if selected!=16 else paired(stream,{"selected16":launches["16"]},10,120)
 selected_key=str(selected)if selected!=16 else "selected16";expert=test[selected_key]["stats"]
 n1c_q8=n1c["test"]["q8"]["candidate"];old_bytes=n1c["physical"]["q8_device_bytes"] if "physical"in n1c and "q8_device_bytes"in n1c["physical"]else 1248931840
 shell=n4a["device_budget"]["q8_device_shell_bytes"];dense_p95=n1c_q8["p95"]*shell/old_bytes;dense_conservative=2*dense_p95
 gates={"all_widths_exact":all(v["bitwise_equal"]for v in correct.values()),"expert_p95_le_50ms":expert["p95"]<=50,"dense_byte_linear_p95_le_40ms":dense_p95<=40,"dense_2x_p95_le_40ms":dense_conservative<=40,"projected_total_p95_le_90ms":expert["p95"]+dense_conservative<=90,"n4a_host_le_58g":n4a["gates"]["host_with_1gib_reserve_le_58_gib"],"n4a_4k_cache":n4a["gates"]["4k_cache_at_least_32_per_layer"],"n4a_32k_cache":n4a["gates"]["32k_cache_at_least_32_per_layer"]}
 input_digest=hashlib.sha256(cp.asnumpy(x).tobytes()).hexdigest();gates["all_output_digests_equal"]=len(set(output_digests.values()))==1
 result={"kind":"streamq5_moe_n4br_synthetic_80b_exact_replication","completed_utc":datetime.now(timezone.utc).isoformat(),"inputs":{"preregistration_sha256":sha256(PREREG),"evaluator_sha256":sha256(Path(__file__)),"n4b_sha256":sha256(N4B),"n4a_sha256":sha256(N4A),"n1c_sha256":sha256(N1C),"input_sha256":input_digest,"seed":SEED},"physical":{"gpu":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),"q5_bank_bytes":int(bank.size),"layers":LAYERS,"active_experts":ACTIVE,"hidden":HIDDEN,"intermediate":INTER},"correctness":correct,"output_digests":output_digests,"validation":validation,"selected_width":selected,"test":test,"expert_test_stats":expert,"dense_projection":{"n1c_q8_p95_ms":n1c_q8["p95"],"source_bytes":old_bytes,"official_80b_shell_bytes":shell,"byte_linear_p95_ms":dense_p95,"conservative_2x_p95_ms":dense_conservative,"projected_total_p95_ms":expert["p95"]+dense_conservative},"gates":gates,"overall_pass":all(gates.values()),"claim_boundary":"Synthetic physical Q5 active-expert shape plus byte-linear physical-Q8 projection; no checkpoint payload, quality, routing, DeltaNet timing or end-to-end tok/s."}
 OUTPUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps({"selected_width":selected,"correctness":correct,"expert":expert,"dense_projection":result["dense_projection"],"gates":gates,"overall_pass":result["overall_pass"]},indent=2))
if __name__=="__main__":main()
