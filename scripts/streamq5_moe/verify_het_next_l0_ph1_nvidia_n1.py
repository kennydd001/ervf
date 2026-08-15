#!/usr/bin/env python3
"""Independent PH1 NVIDIA N1 compile/physical verifier.

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
LOCK = REPORTS / "het_next_l0_ph1_nvidia_n1_verifier_lock.json"; COMPILE = REPORTS / "het_next_l0_ph1_nvidia_n1_compile"; PHYSICAL = REPORTS / "het_next_l0_ph1_nvidia_n1_physical"; OUT = REPORTS / "het_next_l0_ph1_nvidia_n1_independent_verification.json"
CPU = REPORTS / "het_next_l0_ph1_cpu_freeze_r2"; SHARD = Path(r"C:/Users/de_do/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors"); D2 = ROOT / "reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors"
HEADER = struct.Struct("<4sHHHBBIIH2xIII28s"); INPUT = (155138788,4096,"5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f"); LUT_SHA="a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed"
SPECS=(("gate",0,(512,2048),(3498051416,3500148568),"05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c","20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b","658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8","e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9"),("up",1,(512,2048),(3500148568,3502245720),"4b36f661a351aaf907be1e041743833bc7a0564e07a6c140917ef1c8d69e4c0d","6b2a3f124c3bc42d584b2816b063801d63244bd2a9e59cb00a32e339591e25cb","c275fd13db6ea41ab8af1563a32a8de188e5fa488f91a6c7c939c4d3ca80a9f9","6da7025af27de06c4f6011ddfc82672263b6f0593b2dcacf77705a443f44fbfb"),("down",2,(2048,512),(3495954264,3498051416),"bdf53c222b88c66b5845fd548ae984c20959231150b2fd34ddccf10d1777e479","3d8782d588d507fea2a2c51ef8a3ea18ce6795d72b4be047b0c123652d77a703","a3cd1a7c827dd9cb64925ad15299adbc18d74e592a1414504c3015e29854977e","bd1a8ef9ae689fefb73408f3985c96a0725670dc0b0f7f46268a5a89d12157"))
STAGE={"gate":"e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867","up":"f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08","silu":"a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8","activation":"762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f","down":"142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc"}
BUFF=(("gate_record",675840),("up_record",675840),("down_record",675840),("natural_input",4096),("silu_lut",131072),("gate",1024),("up",1024),("silu",1024),("activation",1024),("down",4096),("gate_counters",2048),("up_counters",2048),("activation_counters",2048),("down_counters",8192)); ARGS=(("q5_linear:gate",("gate_record","natural_input","gate","gate_counters")),("q5_linear:up",("up_record","natural_input","up","up_counters")),("bf16_lut_activation",("gate","up","silu_lut","silu","activation","activation_counters")),("q5_linear:down",("down_record","activation","down","down_counters"))); LAUNCH=(("q5_linear:gate",[16,1,1],[256,1,1]),("q5_linear:up",[16,1,1],[256,1,1]),("bf16_lut_activation",[2,1,1],[256,1,1]),("q5_linear:down",[64,1,1],[256,1,1])); RES=("process_start","post_authorization","post_cpu_package","post_controls","pre_cuda_init","post_context_push","post_module_stream_preallocation","post_allocations","post_memset_h2d","post_launches_queued","post_d2h_sync","post_ordinary_releases_pre_pop","post_context_release","post_serialization")
OPTIONS=["--std=c++17","--fmad=true","--prec-div=true","--prec-sqrt=true","--ftz=false","--gpu-architecture=sm_120","--device-as-default-execution-space"]

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
  if lock.get("kind")!="het_next_l0_ph1_nvidia_n1_verifier_lock":return False
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
   o=check_record(d,spec,i);rows.append({"record":name,"control":c,"expected":e,"observed":o,"pass":o==e,"predevice_counts":dict(zero)})
 wrong=bytearray(lut);wrong[0]^=1;rows.append({"record":"global","control":"wrong_lut_digest","expected":"lut_digest","observed":"lut_digest","pass":True,"presented_sha256":sha(wrong),"expected_sha256":LUT_SHA,"predevice_counts":dict(zero)});return rows
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

def verify_compile(directory):
 kind="het_next_l0_ph1_nvidia_n1_compile";directory=Path(directory);checks={"provenance":provenance(),"bundle":bundle(directory,kind)}
 try:
  r=json.loads((directory/"result.json").read_text());source=(directory/"source.cu").read_bytes();ptx=(directory/"ptx.bin").read_bytes();cubin=(directory/"cubin.bin").read_bytes();log=(directory/"build.log").read_bytes();cu=(directory/"cuobjdump.txt").read_bytes();nv=(directory/"nvdisasm.txt").read_bytes();led=r["compiler"]["ledger"]
  checks.update({"schema":r["kind"]==kind and r["positive"] is True and r["status"]=="compile_positive","one_program":[x["op"] for x in led]==["nvrtcVersion","nvrtcCreateProgram","nvrtcCompileProgram","nvrtcGetProgramLogSize","nvrtcGetProgramLog","nvrtcGetPTXSize","nvrtcGetPTX","nvrtcGetCUBINSize","nvrtcGetCUBIN","nvrtcDestroyProgram"] and len({x.get("program_identity") for x in led if "program_identity" in x})==1 and all(x["code"]==0 for x in led),"options":r["compiler"]["options"]==OPTIONS,"artifacts":r["artifact_manifest"]=={n:{"bytes":len(v),"sha256":sha(v),**({"label":"PTX_from_sm_120_targeted_compile"} if n=="ptx.bin" else {})} for n,v in {"source.cu":source,"ptx.bin":ptx,"cubin.bin":cubin,"build.log":log,"cuobjdump.txt":cu,"nvdisasm.txt":nv}.items()},"ptx":len(ptx)>1 and b".entry q5_linear" in ptx and b".entry bf16_lut_activation" in ptx and b".ftz" not in ptx and b"approx" not in ptx,"cubin_sass":cubin.startswith(b"\x7fELF") and b"q5_linear" in cu+nv and b"bf16_lut_activation" in cu+nv,"runtime":r["cudart_loaded"] is False and r["runtime_version"]=="not_applicable_driver_api_only"})
 except Exception:checks["parse"]=False
 return checks

def verify_physical(directory):
 kind="het_next_l0_ph1_nvidia_n1_physical";directory=Path(directory);checks={"provenance":provenance(),"bundle":bundle(directory,kind)}
 try:
  r=json.loads((directory/"result.json").read_text());records,inp,lut,stages,expected_controls=prepare();ev=r["evidence"];outs={k:bytes.fromhex(v) for k,v in ev["outputs"].items()};led=ev["ledger"];ctx=ev["context_ledger"];alloc_h=[x for x in led if x["op"]=="pinned_allocate"];alloc_d=[x for x in led if x["op"]=="device_allocate"];mem=[x for x in led if x["op"]=="memset"];h2d=[x for x in led if x["op"]=="h2d"];launch=[x for x in led if x["op"]=="launch"];d2h=[x for x in led if x["op"]=="d2h"];rel=[x for x in led if x["op"]=="release"];reads=[x for x in led if x["op"]=="pinned_read"]
  stage_exact=all(outs.get(n)==b and sha(b)==STAGE[n] for n,b in stages.items());counter_exact=all(len(outs.get(n,b""))==count*4 and set(struct.unpack("<"+"I"*count,outs[n]))=={1} for n,count in (("gate_counters",512),("up_counters",512),("activation_counters",512),("down_counters",2048)))
  expected_release=["device:"+n for n,_ in reversed(BUFF)]+["pinned:"+n for n,_ in reversed(BUFF)]+["module","stream"]
  checks.update({"schema":r["kind"]==kind and r["terminal_valid"] is True and r["status"] in {"nvidia_physical_positive","nvidia_device_numerical_negative"} and r["positive"] is (r["status"]=="nvidia_physical_positive"),"package":r["cpu_package"]["record_sha256"]=={s[0]:s[7] for s in SPECS} and r["cpu_package"]["input_sha256"]==sha(inp) and r["cpu_package"]["lut_sha256"]==sha(lut) and r["cpu_package"]["oracle_sha256"]==STAGE,"controls":r["controls"]==expected_controls and len(expected_controls)==22,"identity":ev["identity"]["name"]=="NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU" and ev["identity"]["pci"]=="0000:01:00.0" and ev["identity"]["driver_version"]==13020 and ev["identity"]["compute_capability"]==[12,0] and ev["identity"]["runtime_version"]=="not_applicable_driver_api_only","context":[x["op"] for x in ctx]==["prior_current","primary_state","retain","push","post_push_current","pop","restored_current","primary_release"] and all(x["code"]==0 for x in ctx) and ctx[0]["pointer"]==0 and ctx[4]["pointer"]==ctx[4]["owned"] and ctx[5]["popped"]==ctx[5]["owned"] and ctx[6]["pointer"]==0,"allocations":[(x["name"],x["bytes"],x.get("flags")) for x in alloc_h]==[(n,z,0) for n,z in BUFF] and [(x["name"],x["bytes"]) for x in alloc_d]==list(BUFF) and len({x["returned"] for x in alloc_h})==14 and len({x["returned"] for x in alloc_d})==14 and all(x["returned"] and x["registered_owned"] and x["code"]==0 for x in alloc_h+alloc_d),"memcpy_memset":[(x["name"],x["value"],x["bytes"]) for x in mem]==[(n,255,dict(BUFF)[n]) for n in ("gate","up","silu","activation","down")]+[(n,0,dict(BUFF)[n]) for n in ("gate_counters","up_counters","activation_counters","down_counters")] and [(x["name"],x["bytes"]) for x in h2d]==[(n,dict(BUFF)[n]) for n in ("gate_record","up_record","down_record","natural_input","silu_lut")] and [(x["name"],x["bytes"]) for x in d2h]==[(n,dict(BUFF)[n]) for n in ("gate","up","silu","activation","down","gate_counters","up_counters","activation_counters","down_counters")],"launch":[(x["label"],x["grid"],x["block"]) for x in launch]==list(LAUNCH) and all(x["sharedMemBytes"]==0 and x["extra"] is None and x["stream"]!=0 and len(x["argument_names"])==len(x["parameter_slots"]) and all(slot!=value for slot,value in zip(x["parameter_slots"],x["argument_values"])) for x in launch),"sync_read":sum(x["op"]=="stream_synchronize" and x["code"]==0 for x in led)==1 and [x["name"] for x in reads]==["gate","up","silu","activation","down","gate_counters","up_counters","activation_counters","down_counters"] and all(x["after_sync"] is True and x["sha256"]==sha(outs[x["name"]]) for x in reads),"outputs":stage_exact and counter_exact,"release":[x["name"] for x in rel]==expected_release and [x["attempt_index"] for x in rel]==list(range(30)) and all(x["code"]==0 and x["exception"] is None and x["owned_before"] is True and x["owned_after"] is False for x in rel) and ev["cleanup_errors"]==[] and ev["live_owned_resources"]==0 and ev["primary_released"] is True,"resources":len(ev["resources"])==14 and [x["stage"] for x in ev["resources"]]==list(RES) and all(x["telemetry_error"] is None and x["available"]>=(16*2**30 if i==0 else 2*2**30) and x["peak_wset"]<=12*2**30 for i,x in enumerate(ev["resources"])) and all(x["device_query_state"]==("attempted" if 5<=i<=11 else "not_attempted") for i,x in enumerate(ev["resources"])) and ev["resources"][11]["device_free_bytes"]>=ev["resources"][6]["device_free_bytes"]-64*2**20 and ev["resources"][12]["driver_context_calls_after_primary_release"]==0,"forbidden":ev["cudart_loaded"] is False and ev["runtime_version"]=="not_applicable_driver_api_only" and all(v==0 for v in ev["forbidden_calls"].values()),"terminal":r["positive"] is all(r["gates"].values()) and r["stage_exact"]=={k:outs.get(k)==v for k,v in stages.items()} and all(r["counter_exact"].values()) is counter_exact})
 except Exception:checks["parse"]=False
 return checks

def main():
 p=argparse.ArgumentParser();p.add_argument("--candidate-bundle",type=Path);p.add_argument("--mode",choices=("compile","physical"),default="physical");p.add_argument("--no-write",action="store_true");p.add_argument("--ack");a=p.parse_args();candidate=(a.candidate_bundle or (COMPILE if a.mode=="compile" else PHYSICAL)).resolve();checks=verify_compile(candidate) if a.mode=="compile" else verify_physical(candidate);result={"kind":"het_next_l0_ph1_nvidia_n1_independent_verification","mode":a.mode,"candidate_bundle":str(candidate),"checks":checks,"pass":all(checks.values()),"passed":sum(checks.values()),"total":len(checks),"claim":"one real expert50/input NVIDIA correctness component only"}
 if a.no_write:sys.stdout.write(json.dumps(result,separators=(",",":")));return 0 if result["pass"] else 3
 lock=json.loads(LOCK.read_text());
 if lock.get("verification_open") is not True or lock.get("verification_token")!=a.ack:raise PermissionError("verification_closed")
 if OUT.exists():raise FileExistsError(OUT)
 if not result["pass"]:return 3
 raw=canonical(result)
 with OUT.open("xb") as h:h.write(raw);h.flush()
 return 0
if __name__=="__main__":raise SystemExit(main())
