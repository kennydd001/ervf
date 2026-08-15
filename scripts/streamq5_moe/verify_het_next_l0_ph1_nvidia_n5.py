#!/usr/bin/env python3
"""Independent PH1 NVIDIA N5 compile/physical verifier.

This file imports no candidate runner, backend, codec, kernel or transaction
module.  All contracts and arithmetic below are independently frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; REPORTS = ROOT / "reports/streamq5_moe"; SCRIPTS = ROOT / "scripts/streamq5_moe"
LOCK = REPORTS / "het_next_l0_ph1_nvidia_n5_verifier_lock.json"; COMPILE = REPORTS / "het_next_l0_ph1_nvidia_n5_compile"; PHYSICAL = REPORTS / "het_next_l0_ph1_nvidia_n5_physical"; OUT = REPORTS / "het_next_l0_ph1_nvidia_n5_independent_verification.json"
SOURCE_LOCK = REPORTS / "het_next_l0_ph1_nvidia_n5_source_lock.json"; CUDA_SOURCE = SCRIPTS / "het_next_l0_ph1_nvidia_n5_kernels.cu"
CUDA_SHA = "9f369ab3621c6d56b2a3597bca59c25be8d15e7ac3a2a150d916d6695623a781"
CPU = REPORTS / "het_next_l0_ph1_cpu_freeze_r2"; SHARD = Path(r"C:/Users/de_do/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors"); D2 = ROOT / "reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors"
HEADER = struct.Struct("<4sHHHBBIIH2xIII28s"); INPUT = (155138788,4096,"5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f"); LUT_SHA="a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed"
SPECS=(("gate",0,(512,2048),(3498051416,3500148568),"05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c","20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b","658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8","e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9"),("up",1,(512,2048),(3500148568,3502245720),"4b36f661a351aaf907be1e041743833bc7a0564e07a6c140917ef1c8d69e4c0d","6b2a3f124c3bc42d584b2816b063801d63244bd2a9e59cb00a32e339591e25cb","c275fd13db6ea41ab8af1563a32a8de188e5fa488f91a6c7c939c4d3ca80a9f9","6da7025af27de06c4f6011ddfc82672263b6f0593b2dcacf77705a443f44fbfb"),("down",2,(2048,512),(3495954264,3498051416),"bdf53c222b88c66b5845fd548ae984c20959231150b2fd34ddccf10d1777e479","3d8782d588d507fea2a2c51ef8a3ea18ce6795d72b4be047b0c123652d77a703","a3cd1a7c827dd9cb64925ad15299adbc18d74e592a1414504c3015e29854977e","bd1a8ef9ae689fefb73408f3985c96a0725670dc0b0f7f46268a5a89d12157"))
STAGE={"gate":"e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867","up":"f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08","silu":"a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8","activation":"762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f","down":"142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc"}
BUFF=(("gate_record",675840),("up_record",675840),("down_record",675840),("natural_input",4096),("silu_lut",131072),("gate",1024),("up",1024),("silu",1024),("activation",1024),("down",4096),("gate_counters",2048),("up_counters",2048),("activation_counters",2048),("down_counters",8192)); ARGS=(("q5_linear:gate",("gate_record","natural_input","gate","gate_counters")),("q5_linear:up",("up_record","natural_input","up","up_counters")),("bf16_lut_activation",("gate","up","silu_lut","silu","activation","activation_counters")),("q5_linear:down",("down_record","activation","down","down_counters"))); LAUNCH=(("q5_linear:gate",[16,1,1],[256,1,1]),("q5_linear:up",[16,1,1],[256,1,1]),("bf16_lut_activation",[2,1,1],[256,1,1]),("q5_linear:down",[64,1,1],[256,1,1])); RES=("process_start","post_authorization","post_cpu_package","post_controls","pre_cuda_init","post_context_push","post_module_stream_preallocation","post_allocations","post_memset_h2d","post_launches_queued","post_d2h_sync","post_ordinary_releases_pre_pop","post_context_release","post_serialization")
OPTIONS=["--std=c++17","--fmad=true","--prec-div=true","--prec-sqrt=true","--ftz=false","--gpu-architecture=sm_120","--device-as-default-execution-space"]
ABI={
"cuInit":["c_uint"],"cuDriverGetVersion":["POINTER(c_int)"],"cuDeviceGetCount":["POINTER(c_int)"],"cuDeviceGet":["POINTER(c_int)","c_int"],
"cuDeviceGetName":["c_char_p","c_int","c_int"],"cuDeviceGetUuid_v2":["POINTER(CUuuid)","c_int"],"cuDeviceGetPCIBusId":["c_char_p","c_int","c_int"],
"cuDeviceGetAttribute":["POINTER(c_int)","c_int","c_int"],"cuDeviceTotalMem_v2":["POINTER(c_uint64)","c_int"],"cuMemGetInfo_v2":["POINTER(c_uint64)","POINTER(c_uint64)"],
"cuCtxGetCurrent":["POINTER(c_void_p)"],"cuDevicePrimaryCtxGetState":["c_int","POINTER(c_uint)","POINTER(c_int)"],"cuDevicePrimaryCtxRetain":["POINTER(c_void_p)","c_int"],
"cuCtxPushCurrent_v2":["c_void_p"],"cuCtxPopCurrent_v2":["POINTER(c_void_p)"],"cuDevicePrimaryCtxRelease_v2":["c_int"],
"cuStreamCreate":["POINTER(c_void_p)","c_uint"],"cuStreamSynchronize":["c_void_p"],"cuStreamDestroy_v2":["c_void_p"],
"cuModuleLoadDataEx":["POINTER(c_void_p)","c_void_p","c_uint","POINTER(c_int)","POINTER(c_void_p)"],"cuModuleGetFunction":["POINTER(c_void_p)","c_void_p","c_char_p"],"cuModuleUnload":["c_void_p"],
"cuMemHostAlloc":["POINTER(c_void_p)","c_uint64","c_uint"],"cuMemFreeHost":["c_void_p"],"cuMemAlloc_v2":["POINTER(c_uint64)","c_uint64"],"cuMemFree_v2":["c_uint64"],
"cuMemcpyHtoDAsync_v2":["c_uint64","c_void_p","c_uint64","c_void_p"],"cuMemcpyDtoHAsync_v2":["c_void_p","c_uint64","c_uint64","c_void_p"],"cuMemsetD8Async":["c_uint64","c_ubyte","c_uint64","c_void_p"],
"cuLaunchKernel":["c_void_p","c_uint","c_uint","c_uint","c_uint","c_uint","c_uint","c_uint","c_void_p","POINTER(c_void_p)","POINTER(c_void_p)"]}

def sha(b): return hashlib.sha256(b).hexdigest()
def fsha(p):
 d=hashlib.sha256()
 with Path(p).open("rb") as h:
  for c in iter(lambda:h.read(1048576),b""):d.update(c)
 return d.hexdigest()
def canonical(v):return (json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True)+"\n").encode()
def provenance():
 try:
  lock=json.loads(LOCK.read_text())
  if lock.get("kind")!="het_next_l0_ph1_nvidia_n5_verifier_lock":return False
  for item in lock.get("bindings",{}).values():
   path=Path(item["path"]);path=path if path.is_absolute() else ROOT/path
   if fsha(path)!=item["sha256"]:return False
  return True
 except Exception:return False
def rr(p,o,n):
 with Path(p).open("rb") as h:h.seek(o);b=h.read(n)
 if len(b)!=n:raise EOFError(p)
 return b
def b2f(w):return (np.asarray(w,np.uint16).astype(np.uint32)<<np.uint32(16)).view(np.float32)
def f2b(v):
 b=np.asarray(v,np.float32).view(np.uint32);return ((b+np.uint32(0x7fff)+((b>>16)&1))>>16).astype(np.uint16)
def record(spec,src):
 name,ordinal,(r,c),_,ss,cs,scs,rs=spec
 if sha(src)!=ss:raise ValueError("source")
 v=b2f(np.frombuffer(src,"<u2")).reshape(r,c);g=v.reshape(r,c//128,128);m=np.max(np.abs(g),axis=-1,keepdims=True);s=np.where(m>0,np.asarray(m/np.float32(15),np.float32),np.float32(1));q=np.where(m>0,np.clip(np.rint(np.asarray(g/s,np.float32)),-15,15),0).astype(np.int16);f=(q+15).astype(np.uint64).reshape(-1,8);w=np.bitwise_or.reduce(f<<(np.arange(8,dtype=np.uint64)*5),axis=1);codes=np.stack([(w>>(8*i))&255 for i in range(5)],axis=1).astype(np.uint8).tobytes();sc=f2b(s.reshape(-1)).astype("<u2").tobytes();head=HEADER.pack(b"SQ5M",1,0,50,ordinal,5,r,c,128,len(codes),len(sc),zlib.crc32(sc,zlib.crc32(codes))&0xffffffff,bytes(28));rec=head+codes+sc+bytes(4032)
 if (sha(codes),sha(sc),sha(rec))!=(cs,scs,rs):raise ValueError("codec")
 return rec
def check_record(data,spec,input_digest):
 if len(data)!=675840:return "size"
 x=HEADER.unpack(data[:64]);name,ordinal,shape,*_=spec
 if x[:9]!=(b"SQ5M",1,0,50,ordinal,5,*shape,128):return "identity"
 codes,sc=data[64:655424],data[655424:671808]
 if zlib.crc32(sc,zlib.crc32(codes))&0xffffffff!=x[11]:return "crc"
 a=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=sum(a[:,i]<<(8*i) for i in range(5))
 if np.any(((w[:,None]>>(5*np.arange(8,dtype=np.uint64)))&31)==31):return "field31"
 if sha(codes)!=spec[5] or sha(sc)!=spec[6]:return "canonical_digest"
 if sha(data)!=spec[7]:return "record_digest"
 return "pass" if input_digest==INPUT[2] else "input_digest"
def controls(records,lut):
 rows=[];zero={x:0 for x in ("nvrtc_load","compile","nvcuda_load","context","module","stream","allocation","launch")}
 for spec in SPECS:
  name=spec[0];base=records[name];cases=[("truncation",base[:-1],INPUT[2],"size")];h=bytearray(base);x=list(HEADER.unpack(h[:64]));x[4]=(x[4]+1)%3;h[:64]=HEADER.pack(*x);cases.append(("wrong_projection",bytes(h),INPUT[2],"identity"));h=bytearray(base);h[64]^=1;cases.append(("stale_crc",bytes(h),INPUT[2],"crc"));h=bytearray(base);w=int.from_bytes(h[64:69],"little");pick=None
  for slot in range(8):
   f=(w>>(5*slot))&31
   if f!=15:pick=(slot,f-1 if f>15 else f+1);break
  slot,new=pick;w=(w&~(31<<(5*slot)))|(new<<(5*slot));h[64:69]=w.to_bytes(5,"little");x=list(HEADER.unpack(h[:64]));x[11]=zlib.crc32(h[655424:671808],zlib.crc32(h[64:655424]))&0xffffffff;h[:64]=HEADER.pack(*x);cases.append(("code_mutation",bytes(h),INPUT[2],"canonical_digest"));h=bytearray(base);h[655424]^=1;x=list(HEADER.unpack(h[:64]));x[11]=zlib.crc32(h[655424:671808],zlib.crc32(h[64:655424]))&0xffffffff;h[:64]=HEADER.pack(*x);cases.append(("scale_mutation",bytes(h),INPUT[2],"canonical_digest"));h=bytearray(base);w=int.from_bytes(h[64:69],"little");h[64:69]=((w&~31)|31).to_bytes(5,"little");x=list(HEADER.unpack(h[:64]));x[11]=zlib.crc32(h[655424:671808],zlib.crc32(h[64:655424]))&0xffffffff;h[:64]=HEADER.pack(*x);cases.append(("field31",bytes(h),INPUT[2],"field31"));cases.append(("wrong_input",base,"0"*64,"input_digest"))
  for c,d,i,e in cases:
   o=check_record(d,spec,i);order=("size","identity","crc","field31","canonical_digest","record_digest","input_digest","pass");stop=order.index(o);h=HEADER.unpack(d[:64]) if len(d)>=64 else None
   req={"record":name,"expert":50,"projection":spec[1],"shape":list(spec[2]),"record_sha256":spec[7],"codes_sha256":spec[5],"scales_sha256":spec[6],"input_sha256":INPUT[2]}
   pres={"bytes":len(d),"record_sha256":sha(d),"input_sha256":i,"header":None if h is None else {"magic":h[0].hex(),"version":h[1],"expert":h[3],"projection":h[4],"bits":h[5],"rows":h[6],"cols":h[7],"group":h[8],"crc32":h[11]},"codes_sha256":None if len(d)<655424 else sha(d[64:655424]),"scales_sha256":None if len(d)<671808 else sha(d[655424:671808])}
   trace=[{"stage":stage,"result":o if j==stop else "pass"} for j,stage in enumerate(order[:stop+1])]
   rows.append({"record":name,"control":c,"expected":e,"observed":o,"pass":o==e,"requested":req,"presented":pres,"checker_trace":trace,"predevice_counts":dict(zero)})
 wrong=bytearray(lut);wrong[0]^=1;o="lut_digest" if sha(wrong)!=LUT_SHA else "pass";rows.append({"record":"global","control":"wrong_lut_digest","expected":"lut_digest","observed":o,"pass":o=="lut_digest","requested":{"lut_sha256":LUT_SHA},"presented":{"lut_sha256":sha(wrong),"bytes":len(wrong)},"checker_trace":[{"stage":"lut_digest","result":o}],"predevice_counts":dict(zero)});return rows
def parts(b):
 s=-1 if b>>31 else 1;e=(b>>23)&255;f=b&0x7fffff
 if e==255:raise ValueError("finite")
 return (s*f,-149) if e==0 else (s*((1<<23)|f),e-150)
def rse(n,s):
 if s<=0:return n<<-s
 q,r=divmod(n,1<<s);h=1<<(s-1);return q+int(r>h or(r==h and q&1))
def pack(n,e):
 if n==0:return 0
 sg=0x80000000 if n<0 else 0;n=abs(n);top=n.bit_length()-1+e
 if top>127:return sg|0x7f800000
 if top>=-126:
  sh=n.bit_length()-24;si=rse(n,sh)
  if si==1<<24:si>>=1;sh+=1
  ue=e+sh+23;return sg|0x7f800000 if ue>127 else sg|((ue+127)<<23)|(si&0x7fffff)
 fr=rse(n,-149-e);return sg if fr==0 else sg|(1<<23) if fr>=1<<23 else sg|fr
def fma(a,b,c):
 an,ae=parts(a);bn,be=parts(b);cn,ce=parts(c);pn,pe=an*bn,ae+be;e=min(pe,ce);return pack((pn<<(pe-e))+(cn<<(ce-e)),e)
def add(a,b):return fma(a,0x3f800000,b)
def rb(b):return ((b+0x7fff+((b>>16)&1))>>16)&0xffff
def mul(a,b):
 if ((a>>7)&255)==255 or ((b>>7)&255)==255:raise ValueError("finite")
 return ((a^b)&0x8000) if (a&0x7fff)==0 or (b&0x7fff)==0 else rb(fma(a<<16,b<<16,0))
def decode(rec,spec):
 r,c=spec[2];a=np.frombuffer(rec[64:655424],np.uint8).reshape(-1,5).astype(np.uint64);w=sum(a[:,i]<<(8*i) for i in range(5));q=(((w[:,None]>>(5*np.arange(8,dtype=np.uint64)))&31).astype(np.int16)-15).reshape(r,c);s=b2f(np.frombuffer(rec[655424:671808],"<u2")).reshape(r,c//128).repeat(128,1);return f2b(q.astype(np.float32)*s).reshape(r,c)
def linear(w,x):
 r,c=w.shape;v=c//64;tree=(16,8,4,2,1) if c==2048 else (4,2,1);out=np.empty(r,np.uint16)
 for row in range(r):
  p=[[0]*v for _ in range(8)]
  for lane in range(8):
   for j in range(v):
    col=(lane+8*j)*8;a=0
    for k in range(8):a=fma(int(w[row,col+k])<<16,int(x[col+k])<<16,a)
    p[lane][j]=a
  for d in tree:
   for lane in range(8):
    old=p[lane].copy()
    for i in range(d):p[lane][i]=add(old[i],old[i+d])
  lanes=[p[i][0] for i in range(8)]
  for d in (4,2,1):
   old=lanes.copy()
   for i in range(d):lanes[i]=add(old[i],old[i+d])
  out[row]=rb(lanes[0])
 return out
def prepare():
 inp=rr(D2,*INPUT[:2]);lut=(CPU/"bf16_silu_lut.bin").read_bytes();records={s[0]:record(s,rr(SHARD,s[3][0],s[3][1]-s[3][0])) for s in SPECS};weights={s[0]:decode(records[s[0]],s) for s in SPECS};x=np.frombuffer(inp,"<u2");g=linear(weights["gate"],x);u=linear(weights["up"],x);si=np.frombuffer(lut,"<u2")[g];a=np.asarray([mul(int(i),int(j)) for i,j in zip(si,u,strict=True)],np.uint16);d=linear(weights["down"],a);stages={"gate":g.astype("<u2").tobytes(),"up":u.astype("<u2").tobytes(),"silu":si.astype("<u2").tobytes(),"activation":a.astype("<u2").tobytes(),"down":d.astype("<u2").tobytes()}
 if sha(inp)!=INPUT[2] or sha(lut)!=LUT_SHA or {k:sha(v) for k,v in stages.items()}!=STAGE:raise ValueError("oracle")
 return records,inp,lut,stages,controls(records,lut)

def bundle(directory,kind):
 try:
  directory=Path(directory);files={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
  if any(not p.is_file() for p in directory.iterdir()) or not {"result.json","manifest.json","commit.json"}<=set(files) or sum(map(len,files.values()))>16*2**20:return False
  m=json.loads(files["manifest.json"]);c=json.loads(files["commit.json"]);payload=set(files)-{"manifest.json","commit.json"};expected={"kind":kind+"_manifest","files":[{"name":n,"bytes":len(files[n]),"sha256":sha(files[n])} for n in sorted(payload)]}
  return m==expected and c=={"kind":kind+"_commit","result_sha256":sha(files["result.json"]),"manifest_sha256":sha(files["manifest.json"])}
 except Exception:return False

def contract_snapshot_valid(s):
 """Pure executable verifier contract used by production and static mutations."""
 try:
  numerical=set(s["numerical_false"]);protocol=s["protocol"]
  terminal=(s["status"]=="nvidia_physical_positive" and not numerical and s["positive"] is True) or (s["status"]=="nvidia_device_numerical_negative" and numerical and numerical<={"stages_exact","counters_exact"} and s["positive"] is False)
  return (all(protocol.values()) and terminal and s["abi_functions"]==30 and s["all_codes_zero"] is True and
          s["allocations"]==[14,14] and s["schedule"]==[9,5,4,9,1,7] and s["release_count"]==30 and
          s["pointer_crosslinks"] is True and s["stream_crosslinks"] is True and s["runtime_forbidden"]==[])
 except Exception:return False

def verify_compile(directory, provenance_fn=provenance):
 kind="het_next_l0_ph1_nvidia_n5_compile";directory=Path(directory);checks={"provenance":provenance_fn(),"bundle":bundle(directory,kind)}
 try:
  r=json.loads((directory/"result.json").read_text());source=(directory/"source.cu").read_bytes();ptx=(directory/"ptx.bin").read_bytes();cubin=(directory/"cubin.bin").read_bytes();log=(directory/"build.log").read_bytes();cu=(directory/"cuobjdump.txt").read_bytes();nv=(directory/"nvdisasm.txt").read_bytes();led=r["compiler"]["ledger"]
  sl=json.loads(SOURCE_LOCK.read_text());bound=sl["bindings"]["cuda_source"];text=source.decode("utf-8");pt=ptx.lower();sass=(cu+b"\n"+nv).lower()
  dag=(text.count('extern "C" __global__ void q5_linear')==1 and text.count('extern "C" __global__ void bf16_lut_activation')==1 and
       all(token in text for token in ("(lane + 8 * virtual_index) * 8","field < 8","fmaf(","cols == 2048 ? 16 : 4","tile.shfl_down(value, distance)","lut[(unsigned)gate_word]","multiply_bf16_exact","atomicAdd(&counters[row], 1U)")))
  ptx_ok=(pt.count(b".entry q5_linear")==1 and pt.count(b".entry bf16_lut_activation")==1 and all(x not in pt for x in (b".ftz",b"approx",b".extern .func",b"call.uni")))
  sass_ok=(cubin.startswith(b"\x7fELF") and all(x in sass for x in (b"q5_linear",b"bf16_lut_activation",b"fma",b"shfl",b"atom")) and all(x not in sass for x in (b"ftz",b"approx",b"unresolved",b"extern")))
  checks.update({"schema":r["kind"]==kind and r["positive"] is True and r["status"]=="compile_positive",
  "candidate_source":sha(source)==CUDA_SHA==fsha(CUDA_SOURCE) and bound=={"path":"scripts/streamq5_moe/het_next_l0_ph1_nvidia_n5_kernels.cu","sha256":CUDA_SHA} and r["artifact_manifest"]["source.cu"]["sha256"]==CUDA_SHA and r["authorization"]["observed"]["cuda_source"]==CUDA_SHA,
  "one_program":[x["op"] for x in led]==["nvrtcVersion","nvrtcCreateProgram","nvrtcCompileProgram","nvrtcGetProgramLogSize","nvrtcGetProgramLog","nvrtcGetPTXSize","nvrtcGetPTX","nvrtcGetCUBINSize","nvrtcGetCUBIN","nvrtcDestroyProgram"] and len({x.get("program_identity") for x in led if x.get("program_identity")})==1 and all(x["attempted"] is True and x["code"]==0 for x in led),
  "options":r["compiler"]["options"]==OPTIONS,"artifacts":r["artifact_manifest"]=={n:{"bytes":len(v),"sha256":sha(v),**({"label":"PTX_from_sm_120_targeted_compile"} if n=="ptx.bin" else {})} for n,v in {"source.cu":source,"ptx.bin":ptx,"cubin.bin":cubin,"build.log":log,"cuobjdump.txt":cu,"nvdisasm.txt":nv}.items()},
  "width8_dag":dag,"ptx_no_ftz_unresolved":ptx_ok,"sass_no_ftz_unresolved":sass_ok,
  "runtime":r["cudart_loaded"] is False and r["runtime_version"]=="not_applicable_driver_api_only"})
 except Exception:checks["parse"]=False
 return checks

def verify_physical(directory, provenance_fn=provenance, prepare_fn=prepare):
 kind="het_next_l0_ph1_nvidia_n5_physical";directory=Path(directory);checks={"provenance":provenance_fn(),"bundle":bundle(directory,kind)}
 try:
  r=json.loads((directory/"result.json").read_text());records,inp,lut,stages,expected_controls=prepare_fn();ev=r["evidence"];outs={k:bytes.fromhex(v) for k,v in ev["outputs"].items()};led=ev["ledger"];ctx=ev["context_ledger"];alloc_h=[x for x in led if x["op"]=="pinned_allocate"];alloc_d=[x for x in led if x["op"]=="device_allocate"];writes=[x for x in led if x["op"]=="pinned_write"];mem=[x for x in led if x["op"]=="memset"];h2d=[x for x in led if x["op"]=="h2d"];launch=[x for x in led if x["op"]=="launch"];d2h=[x for x in led if x["op"]=="d2h"];rel=[x for x in led if x["op"]=="release"];reads=[x for x in led if x["op"]=="pinned_read"]
  stage_exact=all(outs.get(n)==b and sha(b)==STAGE[n] for n,b in stages.items());counter_exact=all(len(outs.get(n,b""))==count*4 and set(struct.unpack("<"+"I"*count,outs[n]))=={1} for n,count in (("gate_counters",512),("up_counters",512),("activation_counters",512),("down_counters",2048)))
  expected_release=["device:"+n for n,_ in reversed(BUFF)]+["pinned:"+n for n,_ in reversed(BUFF)]+["module","stream"]
  pinned={x["name"]:x["returned"] for x in alloc_h};dev={x["name"]:x["returned"] for x in alloc_d};stream_rows=[x for x in led if x["op"]=="stream_create"];stream=stream_rows[0]["returned"] if len(stream_rows)==1 else 0;meminfo=[x for x in led if x["op"]=="meminfo"]
  abi_names=set(ABI)
  loader=[x for x in led if x["op"]=="driver_load"];modules=ev["runtime_modules"]
  protocol={"finite":r["gates"]["finite"],"controls":r["gates"]["controls"],"resources":r["gates"]["resources"],"cleanup":r["gates"]["cleanup"],"operation_codes":r["gates"]["operation_codes"],"schedule":r["gates"]["schedule"],"abi":r["gates"]["abi"],"runtime_surface":r["gates"]["runtime_surface"]}
  snapshot={"protocol":protocol,"numerical_false":[n for n in ("stages_exact","counters_exact") if not r["gates"][n]],"status":r["status"],"positive":r["positive"],"abi_functions":len(ev["abi"]),"all_codes_zero":all(x.get("code")==0 for x in led if "code" in x) and all(x["code"]==0 for x in ctx),"allocations":[len(alloc_h),len(alloc_d)],"schedule":[len(mem),len(h2d),len(launch),len(d2h),sum(x["op"]=="stream_synchronize" for x in led),len(meminfo)],"release_count":len(rel),"pointer_crosslinks":all(x["source"]==pinned[x["name"]] and x["destination"]==dev[x["name"]] for x in h2d) and all(x["source"]==dev[x["name"]] and x["destination"]==pinned[x["name"]] for x in d2h),"stream_crosslinks":stream!=0 and all(x["stream"]==stream for x in mem+h2d+launch+d2h+[x for x in led if x["op"]=="stream_synchronize"]),"runtime_forbidden":modules["forbidden"]}
  checks.update({"schema":r["kind"]==kind and r["terminal_valid"] is True and r["status"] in {"nvidia_physical_positive","nvidia_device_numerical_negative"} and r["positive"] is (r["status"]=="nvidia_physical_positive"),"package":r["cpu_package"]["record_sha256"]=={s[0]:s[7] for s in SPECS} and r["cpu_package"]["input_sha256"]==sha(inp) and r["cpu_package"]["lut_sha256"]==sha(lut) and r["cpu_package"]["oracle_sha256"]==STAGE,"controls":r["controls"]==expected_controls and len(expected_controls)==22 and all(x["requested"] and x["presented"] and x["checker_trace"] for x in expected_controls),"identity":ev["identity"]["name"]=="NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU" and ev["identity"]["pci"]=="0000:01:00.0" and ev["identity"]["driver_version"]==13020 and ev["identity"]["compute_capability"]==[12,0] and ev["identity"]["runtime_version"]=="not_applicable_driver_api_only",
  "loader_abi":len(loader)==1 and loader[0]["path"].casefold()==r"c:\windows\system32\nvcuda.dll" and loader[0]["sha256"]=="86b41599a673f1aa4699ab458dc5c1e02b57da64d17221f45327af0393fd59a5" and loader[0]["bytes"]==4466920 and loader[0]["calling_convention"]=="WinDLL" and loader[0]["winmode"]==0x800 and set(ev["abi"])==abi_names and ev["abi"]=={name:{"argtypes":args,"restype":"c_int"} for name,args in ABI.items()},
  "return_codes":all(x.get("code")==0 for x in led if "code" in x) and all(x["code"]==0 for x in ctx),
  "context":[x["op"] for x in ctx]==["prior_current","primary_state","retain","push","post_push_current","pop","restored_current","primary_release"] and ctx[0]["pointer"]==0 and ctx[4]["pointer"]==ctx[4]["owned"] and ctx[5]["popped"]==ctx[5]["owned"] and ctx[6]["pointer"]==0,
  "owner_module_function_crosslinks":ev["owner_tid"]>0 and all(x.get("owner_tid")==ev["owner_tid"] for x in ctx+led if x["op"]!="driver_load") and len([x for x in led if x["op"]=="module_get_function"])==2 and len({x["pointer"] for x in led if x["op"]=="module_get_function"})==2 and all(x["pointer"]>0 for x in led if x["op"]=="module_get_function") and all(x["function_pointer"]==next(y["pointer"] for y in led if y["op"]=="module_get_function" and y["entry"]==x["function"]) for x in launch),
  "allocations":[(x["name"],x["bytes"],x.get("flags")) for x in alloc_h]==[(n,z,0) for n,z in BUFF] and [(x["name"],x["bytes"]) for x in alloc_d]==list(BUFF) and len(set(pinned.values()))==14 and len(set(dev.values()))==14 and not(set(pinned.values())&set(dev.values())) and all(x["returned"] and x["registered_owned"] for x in alloc_h+alloc_d),
  "pinned_write_crosslinks":[(x["name"],x["bytes"],x["sha256"],x["pointer"]) for x in writes]==[("gate_record",675840,SPECS[0][7],pinned["gate_record"]),("up_record",675840,SPECS[1][7],pinned["up_record"]),("down_record",675840,SPECS[2][7],pinned["down_record"]),("natural_input",4096,sha(inp),pinned["natural_input"]),("silu_lut",131072,sha(lut),pinned["silu_lut"])],
  "pointer_stream_crosslinks":stream!=0 and all(x["stream"]==stream for x in mem+h2d+launch+d2h+[x for x in led if x["op"]=="stream_synchronize"]) and all(x["source"]==pinned[x["name"]] and x["destination"]==dev[x["name"]] for x in h2d) and all(x["source"]==dev[x["name"]] and x["destination"]==pinned[x["name"]] for x in d2h) and all(tuple(x["argument_names"])==dict(ARGS)[x["label"]] and x["argument_values"]==[dev[n] for n in x["argument_names"]] and len(set(x["parameter_slots"]))==len(x["parameter_slots"]) and all(v and v not in set(dev.values())|set(pinned.values()) for v in x["parameter_slots"]) for x in launch),
  "schedule":[(x["name"],x["value"],x["bytes"]) for x in mem]==[(n,255,dict(BUFF)[n]) for n in ("gate","up","silu","activation","down")]+[(n,0,dict(BUFF)[n]) for n in ("gate_counters","up_counters","activation_counters","down_counters")] and [(x["name"],x["bytes"]) for x in h2d]==[(n,dict(BUFF)[n]) for n in ("gate_record","up_record","down_record","natural_input","silu_lut")] and [(x["label"],x["grid"],x["block"]) for x in launch]==list(LAUNCH) and [(x["name"],x["bytes"]) for x in d2h]==[(n,dict(BUFF)[n]) for n in ("gate","up","silu","activation","down","gate_counters","up_counters","activation_counters","down_counters")],
  "meminfo":len(meminfo)==7 and [x["stage"] for x in meminfo]==list(RES[5:12]) and all(x["code"]==0 and x["context"]==ctx[4]["owned"] and x["after_primary_release"] is False and x["free"]>0 and x["total"]>=x["free"] for x in meminfo),
  "sync_read":sum(x["op"]=="stream_synchronize" for x in led)==1 and [x["name"] for x in reads]==["gate","up","silu","activation","down","gate_counters","up_counters","activation_counters","down_counters"] and all(x["after_sync"] is True and x["sha256"]==sha(outs[x["name"]]) for x in reads),"output_integrity":set(outs)=={"gate","up","silu","activation","down","gate_counters","up_counters","activation_counters","down_counters"} and [len(outs[n]) for n in ("gate","up","silu","activation","down","gate_counters","up_counters","activation_counters","down_counters")]==[1024,1024,1024,1024,4096,2048,2048,2048,8192] and all(not any(((w>>7)&255)==255 for w in struct.unpack("<"+"H"*(len(outs[n])//2),outs[n])) for n in ("gate","up","silu","activation","down")),
  "release":[x["name"] for x in rel]==expected_release and [x["attempt_index"] for x in rel]==list(range(30)) and all(x["code"]==0 and x["exception"] is None and x["owned_before"] is True and x["owned_after"] is False for x in rel) and ev["cleanup_errors"]==[] and ev["live_owned_resources"]==0 and ev["primary_released"] is True,
  "resources":len(ev["resources"])==14 and [x["stage"] for x in ev["resources"]]==list(RES) and all(x["telemetry_error"] is None and x["available"]>=(16*2**30 if i==0 else 2*2**30) and x["peak_wset"]<=12*2**30 for i,x in enumerate(ev["resources"])) and all(x["device_query_state"]==("attempted" if 5<=i<=11 else "not_attempted") for i,x in enumerate(ev["resources"])) and ev["resources"][11]["device_free_bytes"]>=ev["resources"][6]["device_free_bytes"]-64*2**20 and ev["resources"][12]["driver_context_calls_after_primary_release"]==0,
  "forbidden_runtime":ev["static_call_surface"]=="hash_bound_backend_ast_allowlist" and modules["forbidden"]==[] and modules["nvcuda_exact_count"]==1 and set(modules)=={"before","after","post_execution","forbidden","nvcuda_exact_count"} and all(Path(x["path"]).is_absolute() and x["bytes"]>0 and len(x["sha256"])==64 and "cudart" not in x["path"].casefold() and "cupy" not in x["path"].casefold() for phase in ("before","after","post_execution") for x in modules[phase]),
  "terminal":contract_snapshot_valid(snapshot) and r["stage_exact"]=={k:outs.get(k)==v for k,v in stages.items()} and r["counter_exact"]=={n:(len(outs.get(n,b""))==c*4 and set(struct.unpack("<"+"I"*c,outs[n]))=={1}) for n,c in (("gate_counters",512),("up_counters",512),("activation_counters",512),("down_counters",2048))}})
 except Exception:checks["parse"]=False
 return checks

def main():
 p=argparse.ArgumentParser();p.add_argument("--candidate-bundle",type=Path);p.add_argument("--mode",choices=("compile","physical"),default="physical");p.add_argument("--no-write",action="store_true");p.add_argument("--ack");a=p.parse_args();candidate=(a.candidate_bundle or (COMPILE if a.mode=="compile" else PHYSICAL)).resolve();checks=verify_compile(candidate) if a.mode=="compile" else verify_physical(candidate);result={"kind":"het_next_l0_ph1_nvidia_n5_independent_verification","mode":a.mode,"candidate_bundle":str(candidate),"checks":checks,"pass":all(checks.values()),"passed":sum(checks.values()),"total":len(checks),"claim":"one real expert50/input NVIDIA correctness component only"}
 if a.no_write:sys.stdout.write(json.dumps(result,separators=(",",":")));return 0 if result["pass"] else 3
 lock=json.loads(LOCK.read_text());
 if lock.get("verification_open") is not True or lock.get("verification_token")!=a.ack:raise PermissionError("verification_closed")
 if OUT.exists():raise FileExistsError(OUT)
 if not result["pass"]:return 3
 raw=canonical(result)
 with OUT.open("xb") as h:h.write(raw);h.flush()
 return 0
if __name__=="__main__":raise SystemExit(main())
