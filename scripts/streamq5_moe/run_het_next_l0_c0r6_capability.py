#!/usr/bin/env python3
"""C0-R6 static contract and real capability-only backend; source/validation closed."""
from __future__ import annotations
import argparse, hashlib, json, math, os, struct, sys, time, traceback, uuid, threading
from types import MappingProxyType
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe'
LOCK=R/'het_next_l0_c0r6_runner_lock.json';VLOCK=R/'het_next_l0_c0r6_verifier_lock.json';VER=S/'verify_het_next_l0_c0r6_capability.py';PREF=S/'preflight_het_next_l0_c0r6_static.py';KERNEL=S/'het_next_l0_c0r6_kernels.py';SYNC=S/'het_next_l0_c0r6_sync.py';PM=R/'het_next_l0_c0r6_recorded_provenance_manifest.json';TM=R/'het_next_l0_c0r6_d2_sealed_tensor_manifest.json'
PR=R/'HET_NEXT_L0_C0R3_WHOLE_EXPERT_HYBRID_PREREGISTRATION_2026-08-13.md';REV=R/'HET_NEXT_L0_C0R4_WORKER_EPOCH_REVISION_2026-08-13.md';DES=R/'HET_NEXT_L0_C0R3_CAPABILITY_PREFLIGHT_DESIGN_2026-08-13.md';ADD=R/'HET_NEXT_L0_C0R4_CAPABILITY_PREFLIGHT_ADDENDUM_2026-08-13.md'
D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors';D2RES=D2.with_name('t0r12d2_result.json');SHARD=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors'
OUT=ROOT/'reports/runs/streamq5_moe/het_next_l0_c0r6_capability';CAP=OUT/'c0r6_capability.json';FAIL=OUT/'c0r6_capability_failure.json';COM=OUT/'c0r6_capability_commit.json'
SEED=2026081302;ROUTES=((50,199,237,474,245,374,239,8,168,12),(42,162,267,299,467,307,326,145,297,182),(474,232,382,80,31,450,103,372,286,206),(26,159,28,176,253,84,431,294,386,356));TEMPLATES=(tuple('ABSBASSABSBA'),tuple('ASBBSAASBBSA'),tuple('SABSBAABSBAS'));REVERSE=((3,2,1,0),(1,0,3,2),(3,2,1,0));LP=(0,2,4,6);NAMES=('gate','up','down')
ACTIVATION_FIXTURE={'input_bf16_words':[49152,49024,48896,0,16128,16256,16384],'sigmoid_bf16_words':[15860,16010,16065,16128,16159,16187,16225],'silu_bf16_words':[48756,48778,48705,0,16031,16187,16353]}
PROVENANCE={
 'd2_raw':('reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors',171696126,'f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd'),
 'd2_result':('reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_result.json',1043105,'694b45004c9dea6827e201c80198d7f63a8fa7b90deea97198879d17162d2acb'),
 'd2_audit':('reports/streamq5_moe/PORT80B_T0R12D2R3_INDEPENDENT_ARTIFACT_AUDIT_2026-08-13.json',6624,'a048450b10c9ab2a06fa00629eb5089bb67333c36879da814afcaafac4538c33'),
 'd2_interpretation':('reports/streamq5_moe/PORT80B_T0R12D2R3_INDEPENDENT_INTERPRETATION_2026-08-13.md',5243,'be603f4edc648939aa86b2fcec16df802f4e778c6ab14256aecdc48f347da7f0'),
 'combined_json':('reports/streamq5_moe/PORT80B_T0Q5_S0R5_C1R2A_COMBINED_ADJUDICATION_2026-08-13.json',6504,'a8b41382b68488f393eafd9f057ddc5140ee8cb2c04ab3c3833a07345311f265'),
 'combined_audit':('reports/streamq5_moe/port80b_t0q5_s0r5_c1r2a_combined_independent_verification.json',4916,'8dedb139fb9aed8c33015aa19bf489dc0fac2f0bb5a5469af429d5f65ca9822d'),
 'combined_report':('reports/streamq5_moe/PORT80B_T0Q5_S0R5_C1R2A_COMBINED_REPORT_2026-08-13.md',2101,'cd91e9226f99e1177caff83a62a438c29651af462a53d95891fdbc95b9477e06'),
 'combined_verifier':('scripts/streamq5_moe/verify_port80b_t0q5_s0r5_c1r2a_adjudication.py',6296,'34f75d5209027be00cee4ee2bb931b18e192107f4ffc538e43174ab4c49d7fc3'),
 'c1_verifier':('scripts/streamq5_moe/verify_port80b_t0q5s0c1r2a_control_only.py',12894,'ccf7bda4dcc13135a5e43a9c8ee35d79182a61b2b1a0192a6a37337365e99a11'),
 'st2_report':('reports/streamq5_moe/ST2_MINI_HOST_USM_Q5_REPORT_2026-08-12.md',8057,'af23a4fbffb18028ce1a88b3c73f21546cae7ee397d118c186094f985ee4ac49'),
 'st2_audit':('reports/streamq5_moe/st2_mini_independent_verification.json',13857,'79f53e525ef7893345484ac39fbb253d7599f39572a86eda309a98ac6f36ecac'),
 'st2_source':('scripts/streamq5_moe/run_st2_mini_ergv_w8.py',7391,'6472de274fa68a9f577b1483ef1225607f8425ac8587cb348d0c328cff7126ca'),
 'd7_report':('reports/streamq5_moe/PORT80B_D7_STAGED_EXACT_Q5_PLANE_REPORT_2026-08-12.md',256,'8fce019bb4d51ff5e04a0c91d1cfa43679caf54e86f884acb4fb6a5225df5e86'),
 'd7_audit':('reports/streamq5_moe/port80b_d6_d7_exact_planes_independent_verification.json',8782,'02d1cbbe5a18ff983c1a868bb159fd8e355832135bb3fd1c28613c00dfdcb46f'),
 'd7_source':('scripts/streamq5_moe/run_port80b_d7_staged_exact_q5_plane.py',14260,'26d4daba81d5f132857f9b584dfb12f3634874a2e9ee2290f9221c486ef4059a'),
 'c0_audit':('reports/streamq5_moe/HET_NEXT_L0_C0_INDEPENDENT_DESIGN_AUDIT_2026-08-13.md',5551,'d2d33e0131b56fee2432c6945226998058495ec06bc44639bf42cba1d9767fed'),
 'dependency':('reports/streamq5_moe/port80b_t0r4_dependency_execution_lock.json',3091,'1d08457aded09f139d25af84ba778d8e275ab5ff71967a3dc8b9a7452e6d2fae'),
}
D2_KEY_SHA={
0:{'post_norm':'d82286fac9616cdf8b03b8eddb8347acd3679afb639c8db696daf3f643084853','official_router_ids':'c183be31d947f3a74865eb58f874a0ffd2289adbe455d4689d38239c5a6be2ca','official_router_weights':'d048f9eddc9f3e358d59383557da8f3fc3b91ab84baddb6b412c82164b2e3be2','shared_gate':'3630e2b1cb0ad297f0efd2f029140f5befd810c3520c4dc7eeb0ce746ed49fc0','experts':'a74a8a9ef47df5a43ff6ca3ecd28a14650c6275586b21ef6c0fc9f1c3559477c','shared':'3e1f0052460430ca03c19f7a312a80c68034d86b387d3981ae0cce3224e67125'},
1:{'post_norm':'3fc5c730553bb2344e0cfb4ea09adb4710600615481ad456e1d6adbc7fbf8b22','official_router_ids':'884697be4c218856f6b414d0aff8a942ecc80d571331c27a5c364af2ed2b9800','official_router_weights':'584be7a37aec57393a0aa775895f24e1861111bf5cc2c3eaf7b7cf9d96c58f8f','shared_gate':'7ac9aa43acc27b02341e02975687541564508d5270fcb03c76431ae8fcb0c2d5','experts':'01f6a1ce7dd5751b55d1e2aefa972d170807717c99c48d9857cb63c944c7b5a6','shared':'840ac191b0f8a5151d3afa33514f15c660d4802135da57f982495e721e9dc9f5'},
2:{'post_norm':'598bbba0f452ea34c46dc30f316241abdf654a1df7c19454f8f72f16c1ad7341','official_router_ids':'cc23db5cc0c124e74558b9f3e23f0c480f361cfe0ed3c0b03ec53efbd531b556','official_router_weights':'8924d5302108de798196317b1a32aa7b0039a7a8c051086b9fcec8ce04bd56d6','shared_gate':'52dda580bffefb23670fe224fa0257d4159d6f8efcda8b645dd979d35c3e4c06','experts':'162b2253ad8f2ac951a031feb207b8fef63b6b5f3e86d5a8a20365d6e756be84','shared':'abde8ea1e3e4c5f5ecbd30fad6466cc87a767d380b0989a95e91a11af37dc034'},
3:{'post_norm':'3d415335f078aea3d8d72808f29d80752874af71903902e0fd7e41c4d02188c9','official_router_ids':'24ab6969a9738d0733e1233aac66e66e712ce4a214c2475ef43cf149471a26c4','official_router_weights':'0b6290f644a57caec6981d9a6a2030ee971f81983d508fb0d294f469b45fac31','shared_gate':'e5bc7e6692a9c54ebc2142e130073d278546424330753aeaece0017fcfd38e7b','experts':'2846ca0a436963d70d737b8866abfdb867d711ff51ea5347f86a98e030c44f6a','shared':'c6caa0ffde957efea74eeb4ad2832412e6d9321f466e8bb47862bc1efa41a912'}}
DTYPES={'post_norm':('BF16',(1,16,2048)),'official_router_ids':('I64',(16,10)),'official_router_weights':('BF16',(16,10)),'shared_gate':('BF16',(16,1)),'experts':('BF16',(16,2048)),'shared':('BF16',(16,2048))}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def safetensor_header(p):
 with Path(p).open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];raw=f.read(n)
 if len(raw)!=n or n>64*2**20:raise ValueError('header')
 h=json.loads(raw);base=8+n
 return {k:{**v,'absolute':(base+v['data_offsets'][0],base+v['data_offsets'][1])} for k,v in h.items() if k!='__metadata__'}
def d2_specs(header):
 out={}
 for row in range(4):
  for sem,(dtype,shape) in DTYPES.items():
   key=f'p{row}_whole_{sem}';v=header[key]
   if v['dtype']!=dtype or tuple(v['shape'])!=shape:raise ValueError(key)
   out[key]={'row':row,'dtype':dtype,'shape':shape,'absolute':tuple(v['absolute']),'bytes':v['absolute'][1]-v['absolute'][0],'sha256':D2_KEY_SHA[row][sem]}
 return out
_VALIDATION_TOKEN=object();_TEST_TOKEN=object()
class SealedReader:
 def __init__(self,path,specs):
  self.__path=Path(path);copy={k:MappingProxyType(dict(v)) for k,v in specs.items()};ranges=[]
  for k,v in copy.items():
   if v['row'] not in (0,1,2,3) or not (0<=v['absolute'][0]<v['absolute'][1]):raise ValueError('bad_spec')
   ranges.append((v['absolute'][0],v['absolute'][1],k))
  ranges.sort()
  if any(ranges[i][1]>ranges[i+1][0] for i in range(len(ranges)-1)):raise ValueError('overlapping_allowlist')
  self.__specs=MappingProxyType(copy);self.__tests_open=False;self.__ledger=[]
 @property
 def ledger(self):return tuple(dict(x) for x in self.__ledger)
 def open_tests(self,commit):
  if not isinstance(commit,dict) or commit.get('status')!='validation_pass' or commit.get('verified') is not True:raise PermissionError('invalid_open_commit')
  self.__tests_open=True
 def read(self,key,token):
  if key not in self.__specs:raise PermissionError('key_not_allowlisted')
  s=self.__specs[key];row=s['row']
  if row>0 and not self.__tests_open:raise PermissionError('test_payload_sealed')
  if token is _VALIDATION_TOKEN and row!=0:raise PermissionError('phase_row_mismatch')
  if token is _TEST_TOKEN and row==0:raise PermissionError('phase_row_mismatch')
  if token not in (_VALIDATION_TOKEN,_TEST_TOKEN):raise PermissionError('opaque_phase_token')
  a,b=s['absolute'];entry={'key':key,'row':row,'phase':'validation' if token is _VALIDATION_TOKEN else 'test','absolute':[a,b],'completed':False};self.__ledger.append(entry)
  with self.__path.open('rb') as f:f.seek(a);data=f.read(b-a)
  if len(data)!=s['bytes'] or hashlib.sha256(data).hexdigest()!=s['sha256']:raise ValueError('payload_binding')
  entry.update(completed=True,sha256=s['sha256'],bytes=len(data));return data
def official_shard_key(e,j):
 owner='experts.'+str(e) if e<512 else 'shared_expert'
 return f'model.layers.0.mlp.{owner}.{NAMES[j]}_proj.weight'
def validate_shard_header(h):
 required=[]
 for e in sorted(set(ROUTES[0]))+[512]:
  for j,shape in enumerate(((512,2048),(512,2048),(2048,512))):
   k=official_shard_key(e,j);v=h[k]
   if v['dtype']!='BF16' or tuple(v['shape'])!=shape:raise ValueError(k)
   required.append(k)
 return required
def schedule():
 rows=[]
 for b in range(30):
  t=(SEED+b)%3
  for k,a in enumerate(TEMPLATES[t]):
   g=k//3;rows.append((b,t,g,a,min(g,REVERSE[t][g]),max(g,REVERSE[t][g])))
 return rows
class EpochMachine:
 def __init__(self):self.last={'intel':0,'nvidia':0};self.ack=dict(self.last);self.start=False;self.epoch=0;self.log=[]
 def execute(self,active):
  if self.start or any(self.ack[i]!=self.last[i] for i in active):raise RuntimeError('reuse')
  self.epoch+=1
  for i in active:self.last[i]=self.epoch
  self.log.append({'epoch':self.epoch,'active':list(active),'prior_inactive':{i:self.last[i] for i in self.last if i not in active}});self.start=True
  for i in active:self.ack[i]=self.last[i]
  if any(self.ack[i]!=self.last[i] for i in active):raise RuntimeError('ack')
  self.start=False;return self.epoch
def simulate_sync():
 m=EpochMachine()
 for a in (('nvidia',),('intel','nvidia'),('intel',),('nvidia',),('nvidia',)):m.execute(a)
 before=(m.last.copy(),m.ack.copy());neg=False;m.ack['intel']=0
 try:m.execute(('intel',))
 except RuntimeError:neg=True
 return {'rows':m.log,'final_last':before[0],'final_ack':before[1],'stale_rejected':neg}
def lockcheck():
 l=json.loads(LOCK.read_text());expected={'runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'preflight_sha256':sha(PREF),'kernel_sha256':sha(KERNEL),'verifier_lock_sha256':sha(VLOCK),'prereg_sha256':sha(PR),'revision_sha256':sha(REV),'design_sha256':sha(DES),'addendum_sha256':sha(ADD)}
 return {'pass':all(l.get(k)==v for k,v in expected.items()) and l.get('capability_open') is False and l.get('source_build_open') is False and l.get('execution_open') is False,'expected':expected}
def atomic_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.inprogress')
 with tmp.open('xb') as f:f.write(canon(obj)+b'\n');f.flush();os.fsync(f.fileno())
 if path.exists():raise FileExistsError(path)
 os.rename(tmp,path)
 if os.name!='nt':
  fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def recover_output():
 if not OUT.exists():return []
 if CAP.exists() and COM.exists():
  try:
   m=json.loads(COM.read_text());ok=m.get('files')=={CAP.name:{'bytes':CAP.stat().st_size,'sha256':sha(CAP)}}
   if ok:return ['valid_commit']
  except Exception:pass
 bad=OUT/'failed_attempts';bad.mkdir(exist_ok=True);moved=[]
 for x in list(OUT.glob('*.inprogress'))+[x for x in (CAP,COM,FAIL) if x.exists()]:
  q=bad/f'{uuid.uuid4().hex}_{x.name}';os.rename(x,q);moved.append({'from':x.name,'to':q.name,'sha256':sha(q),'bytes':q.stat().st_size})
 return moved

def set_affinity(logical_processor):
 import ctypes as C
 from ctypes import wintypes
 k=C.WinDLL('kernel32',use_last_error=True);k.SetThreadAffinityMask.argtypes=[wintypes.HANDLE,C.c_size_t];k.SetThreadAffinityMask.restype=C.c_size_t;k.GetCurrentThread.restype=wintypes.HANDLE
 if not k.SetThreadAffinityMask(k.GetCurrentThread(),C.c_size_t(1<<logical_processor)):raise OSError(C.get_last_error(),'SetThreadAffinityMask')
 return {'requested_lp':logical_processor,'thread_id':threading.get_native_id()}
def bf16_word(x):
 b=struct.unpack('<I',struct.pack('<f',float(x)))[0];b=(b+0x7fff+((b>>16)&1))&0xffffffff;return b>>16
def ergv_expected():return [bf16_word(sum(((r*17+c*13)%31-15) for c in range(256))) for r in range(32)]
def intel_capability(kernel_source,ergv_source,coexist_barrier=None):
 import ctypes as C
 from ctypes import wintypes
 CL_SUCCESS=0;CL_DEVICE_TYPE_GPU=4;CL_PLATFORM_NAME=0x0902;CL_PLATFORM_VENDOR=0x0903;CL_PLATFORM_VERSION=0x0901;CL_DEVICE_NAME=0x102B;CL_DEVICE_VENDOR=0x102C;CL_DRIVER_VERSION=0x102D;CL_DEVICE_VERSION=0x102F;CL_DEVICE_EXTENSIONS=0x1030;CL_DEVICE_GLOBAL_MEM_SIZE=0x101F;CL_DEVICE_MAX_MEM_ALLOC_SIZE=0x1010;CL_CONTEXT_PLATFORM=0x1084;CL_TRUE=1
 lib=C.WinDLL('OpenCL.dll');cleanup=[];counts={'context':0,'queue':0,'program':0,'kernel':0,'ergv_kernel':0,'usm':0,'event':0,'ergv_event':0}
 lib.clGetPlatformIDs.argtypes=[C.c_uint,C.POINTER(C.c_void_p),C.POINTER(C.c_uint)];lib.clGetPlatformIDs.restype=C.c_int
 lib.clGetPlatformInfo.argtypes=[C.c_void_p,C.c_uint,C.c_size_t,C.c_void_p,C.POINTER(C.c_size_t)];lib.clGetPlatformInfo.restype=C.c_int
 lib.clGetDeviceIDs.argtypes=[C.c_void_p,C.c_ulonglong,C.c_uint,C.POINTER(C.c_void_p),C.POINTER(C.c_uint)];lib.clGetDeviceIDs.restype=C.c_int
 lib.clGetDeviceInfo.argtypes=[C.c_void_p,C.c_uint,C.c_size_t,C.c_void_p,C.POINTER(C.c_size_t)];lib.clGetDeviceInfo.restype=C.c_int
 lib.clCreateContext.argtypes=[C.POINTER(C.c_ssize_t),C.c_uint,C.POINTER(C.c_void_p),C.c_void_p,C.c_void_p,C.POINTER(C.c_int)];lib.clCreateContext.restype=C.c_void_p
 lib.clCreateCommandQueue.argtypes=[C.c_void_p,C.c_void_p,C.c_ulonglong,C.POINTER(C.c_int)];lib.clCreateCommandQueue.restype=C.c_void_p
 lib.clCreateProgramWithSource.argtypes=[C.c_void_p,C.c_uint,C.POINTER(C.c_char_p),C.POINTER(C.c_size_t),C.POINTER(C.c_int)];lib.clCreateProgramWithSource.restype=C.c_void_p
 lib.clBuildProgram.argtypes=[C.c_void_p,C.c_uint,C.POINTER(C.c_void_p),C.c_char_p,C.c_void_p,C.c_void_p];lib.clBuildProgram.restype=C.c_int
 lib.clCreateKernel.argtypes=[C.c_void_p,C.c_char_p,C.POINTER(C.c_int)];lib.clCreateKernel.restype=C.c_void_p
 lib.clSetKernelArg.argtypes=[C.c_void_p,C.c_uint,C.c_size_t,C.c_void_p];lib.clSetKernelArg.restype=C.c_int
 lib.clEnqueueNDRangeKernel.argtypes=[C.c_void_p,C.c_void_p,C.c_uint,C.c_void_p,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t),C.c_uint,C.c_void_p,C.POINTER(C.c_void_p)];lib.clEnqueueNDRangeKernel.restype=C.c_int
 lib.clFinish.argtypes=[C.c_void_p];lib.clFinish.restype=C.c_int
 lib.clGetExtensionFunctionAddressForPlatform.argtypes=[C.c_void_p,C.c_char_p];lib.clGetExtensionFunctionAddressForPlatform.restype=C.c_void_p
 lib.clGetEventProfilingInfo.argtypes=[C.c_void_p,C.c_uint,C.c_size_t,C.c_void_p,C.POINTER(C.c_size_t)];lib.clGetEventProfilingInfo.restype=C.c_int
 for fn in ('clReleaseEvent','clReleaseKernel','clReleaseProgram','clReleaseCommandQueue','clReleaseContext'):getattr(lib,fn).argtypes=[C.c_void_p];getattr(lib,fn).restype=C.c_int
 def ck(x,n):
  if x!=0:raise RuntimeError(f'{n}:{x}')
 def info(obj,fn,param):
  n=C.c_size_t();ck(fn(obj,param,0,None,C.byref(n)),'info_size');b=C.create_string_buffer(n.value);ck(fn(obj,param,n.value,b,None),'info');return b.value.decode(errors='replace')
 def uint64info(obj,param):
  v=C.c_ulonglong();ck(lib.clGetDeviceInfo(obj,param,C.sizeof(v),C.byref(v),None),'uint64info');return int(v.value)
 np=C.c_uint();ck(lib.clGetPlatformIDs(0,None,C.byref(np)),'platform_count');plats=(C.c_void_p*np.value)();ck(lib.clGetPlatformIDs(np,plats,None),'platforms');chosen=None;identity={}
 for p in plats:
  nd=C.c_uint();code=lib.clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,0,None,C.byref(nd))
  if code or not nd.value:continue
  ds=(C.c_void_p*nd.value)();ck(lib.clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,nd,ds,None),'devices')
  for d in ds:
   name=info(d,lib.clGetDeviceInfo,CL_DEVICE_NAME);vendor=info(d,lib.clGetDeviceInfo,CL_DEVICE_VENDOR)
   if 'Intel' in vendor and ('Arc' in name or '140T' in name):chosen=(p,d);identity={'platform':info(p,lib.clGetPlatformInfo,CL_PLATFORM_NAME),'platform_vendor':info(p,lib.clGetPlatformInfo,CL_PLATFORM_VENDOR),'platform_version':info(p,lib.clGetPlatformInfo,CL_PLATFORM_VERSION),'device':name,'vendor':vendor,'driver_version':info(d,lib.clGetDeviceInfo,CL_DRIVER_VERSION),'device_version':info(d,lib.clGetDeviceInfo,CL_DEVICE_VERSION),'global_mem':uint64info(d,CL_DEVICE_GLOBAL_MEM_SIZE),'max_alloc':uint64info(d,CL_DEVICE_MAX_MEM_ALLOC_SIZE)};break
  if chosen:break
 if not chosen:raise RuntimeError('intel_gpu_not_found')
 p,d=chosen;ext=info(d,lib.clGetDeviceInfo,CL_DEVICE_EXTENSIONS).split();required=('cl_intel_unified_shared_memory','cl_intel_subgroups','cl_intel_required_subgroup_size')
 if any(x not in ext for x in required):raise RuntimeError('intel_extensions')
 props=(C.c_ssize_t*3)(CL_CONTEXT_PLATFORM,int(p),0);err=C.c_int();devices=(C.c_void_p*1)(d);ctx=q=prog=k=ergv=None;ptr=None;event=C.c_void_p();ergv_event=C.c_void_p()
 def extfn(name,restype,*args):
  a=lib.clGetExtensionFunctionAddressForPlatform(p,name.encode());
  if not a:raise RuntimeError(name)
  return C.WINFUNCTYPE(restype,*args)(a)
 alloc=free=setptr=None;build_log='';profile={}
 try:
  ctx=lib.clCreateContext(props,1,devices,None,None,C.byref(err));ck(err.value,'context');counts['context']=1
  q=lib.clCreateCommandQueue(ctx,d,2,C.byref(err));ck(err.value,'queue');counts['queue']=1
  combined=(kernel_source+'\n'+ergv_source).encode();sp=C.c_char_p(combined);sl=C.c_size_t(len(combined));prog=lib.clCreateProgramWithSource(ctx,1,C.byref(sp),C.byref(sl),C.byref(err));ck(err.value,'program');counts['program']=1;opts=b'-cl-std=CL2.0 -cl-fp32-correctly-rounded-divide-sqrt';code=lib.clBuildProgram(prog,1,devices,opts,None,None)
  if code:
   size=C.c_size_t();lib.clGetProgramBuildInfo(prog,d,0x1183,0,None,C.byref(size));buf=C.create_string_buffer(size.value);lib.clGetProgramBuildInfo(prog,d,0x1183,size.value,buf,None);raise RuntimeError(f'build:{code}:{buf.value.decode(errors="replace")}')
  build_log='success'
  k=lib.clCreateKernel(prog,b'c0r6_usm_copyless',C.byref(err));ck(err.value,'kernel');counts['kernel']=1;ergv=lib.clCreateKernel(prog,b'c0r6_ergv8_sentinel',C.byref(err));ck(err.value,'ergv_kernel');counts['ergv_kernel']=1
  if coexist_barrier is not None:coexist_barrier.wait(timeout=30)
  alloc=extfn('clHostMemAllocINTEL',C.c_void_p,C.c_void_p,C.POINTER(C.c_longlong),C.c_size_t,C.c_uint,C.POINTER(C.c_int));free=extfn('clMemFreeINTEL',C.c_int,C.c_void_p,C.c_void_p);setptr=extfn('clSetKernelArgMemPointerINTEL',C.c_int,C.c_void_p,C.c_uint,C.c_void_p)
  ptr=alloc(ctx,None,4096,64,C.byref(err));ck(err.value,'usm_alloc');counts['usm']=1;arr=(C.c_ubyte*4096).from_address(ptr);before=bytes((i*17+3)&255 for i in range(4096));arr[:]=before;ck(setptr(k,0,ptr),'setptr');n=C.c_uint(4096);ck(lib.clSetKernelArg(k,1,C.sizeof(n),C.byref(n)),'set_n');gs=C.c_size_t(4096);ls=C.c_size_t(64);ck(lib.clEnqueueNDRangeKernel(q,k,1,None,C.byref(gs),C.byref(ls),0,None,C.byref(event)),'enqueue');counts['event']=1;ck(lib.clFinish(q),'finish');after=bytes(arr);expected=bytes((x^0x5a) for x in before);ok=after==expected
  started=C.c_ulonglong();ended=C.c_ulonglong();ck(lib.clGetEventProfilingInfo(event,0x1282,C.sizeof(started),C.byref(started),None),'profile_start');ck(lib.clGetEventProfilingInfo(event,0x1283,C.sizeof(ended),C.byref(ended),None),'profile_end');profile={'start_ns':started.value,'end_ns':ended.value}
  if not ok:raise RuntimeError('intel_sentinel')
  xptr=ptr;yptr=ptr+2048;xarr=(C.c_float*256).from_address(xptr);yarr=(C.c_ushort*32).from_address(yptr)
  for i in range(256):xarr[i]=1.0
  for i in range(32):yarr[i]=0
  ck(setptr(ergv,0,C.c_void_p(xptr)),'ergv_x');ck(setptr(ergv,1,C.c_void_p(yptr)),'ergv_y');rows=C.c_int(32);cols=C.c_int(256);ck(lib.clSetKernelArg(ergv,2,C.sizeof(rows),C.byref(rows)),'ergv_rows');ck(lib.clSetKernelArg(ergv,3,C.sizeof(cols),C.byref(cols)),'ergv_cols');gs2=C.c_size_t(256);ls2=C.c_size_t(256);ck(lib.clEnqueueNDRangeKernel(q,ergv,1,None,C.byref(gs2),C.byref(ls2),0,None,C.byref(ergv_event)),'ergv_enqueue');counts['ergv_event']=1;ck(lib.clFinish(q),'ergv_finish');ergv_words=list(yarr);expected_ergv=ergv_expected()
  if ergv_words!=expected_ergv:raise RuntimeError('intel_ergv_words')
 finally:
  for h,nm,release in ((ergv_event.value,'ergv_event',lib.clReleaseEvent),(ergv,'ergv_kernel',lib.clReleaseKernel),(event.value,'event',lib.clReleaseEvent),(k,'kernel',lib.clReleaseKernel)):
   if h:
    try:ck(release(C.c_void_p(h)) if isinstance(h,int) else release(h),f'release_{nm}')
    finally:
     if nm in counts:counts[nm]=0
  if ptr and free:
   try:ck(free(ctx,ptr),'free_usm')
   finally:counts['usm']=0
  for h,nm,release in ((prog,'program',lib.clReleaseProgram),(q,'queue',lib.clReleaseCommandQueue),(ctx,'context',lib.clReleaseContext)):
   if h:
    try:ck(release(h),f'release_{nm}')
    finally:counts[nm]=0
 return {**identity,'extensions':ext,'required_extensions':list(required),'buffer_bytes':4096,'alignment':64,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),'expected_after_sha256':hashlib.sha256(expected).hexdigest(),'ergv_words':ergv_words,'ergv_expected_words':expected_ergv,'copyless_host_usm':True,'used_cl_mem':False,'used_enqueue_write':False,'used_migrate':False,'compiler_options':opts.decode(),'build_log':build_log,'profiling':profile,'cleanup_counts':counts}
def nvidia_capability(kernel_source,coexist_barrier=None):
 import cupy as cp
 dev=cp.cuda.Device();props=cp.cuda.runtime.getDeviceProperties(dev.id);before=bytes((i*29+7)&255 for i in range(4096));x=cp.asarray(bytearray(before),dtype=cp.uint8);mod=cp.RawModule(code=kernel_source,options=('--std=c++14','--fmad=false'),name_expressions=('c0r6_cuda_sentinel','c0r6_ergv8_sentinel'));fn=mod.get_function('c0r6_cuda_sentinel');ergv=mod.get_function('c0r6_ergv8_sentinel');
 if coexist_barrier is not None:coexist_barrier.wait(timeout=30)
 fn((16,),(256,),(x,4096));cp.cuda.runtime.deviceSynchronize();after=bytes(cp.asnumpy(x));expected=bytes(((v+0x33)&255) for v in before);ok=after==expected;ex=cp.ones((256,),dtype=cp.float32);ey=cp.zeros((32,),dtype=cp.uint16);ergv((1,),(256,),(ex,ey,32,256));cp.cuda.runtime.deviceSynchronize();ergv_words=[int(v) for v in cp.asnumpy(ey)];expected_ergv=ergv_expected();del x,ex,ey,fn,ergv,mod;cp.get_default_memory_pool().free_all_blocks()
 if not ok:raise RuntimeError('nvidia_sentinel')
 if ergv_words!=expected_ergv:raise RuntimeError('nvidia_ergv_words')
 try:pci=cp.cuda.runtime.deviceGetPCIBusId(dev.id).decode()
 except Exception:pci=str(props.get('pciBusID',''))
 return {'device_id':dev.id,'name':props['name'].decode() if isinstance(props['name'],bytes) else str(props['name']),'pci_bus_id':pci,'compute_capability':[int(props['major']),int(props['minor'])],'total_global_mem':int(props['totalGlobalMem']),'buffer_bytes':4096,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),'expected_after_sha256':hashlib.sha256(expected).hexdigest(),'ergv_words':ergv_words,'ergv_expected_words':expected_ergv,'compiler_options':['--std=c++14','--fmad=false'],'cleanup_pool_used_bytes':int(cp.get_default_memory_pool().used_bytes())}
def pdh_monitor(stop):
 import ctypes as C
 from ctypes import wintypes
 class U(C.Union):_fields_=[('long',wintypes.LONG),('double',C.c_double),('large',C.c_longlong)]
 class V(C.Structure):_anonymous_=('value',);_fields_=[('status',wintypes.DWORD),('value',U)]
 pdh=C.WinDLL('pdh',use_last_error=True);pdh.PdhOpenQueryW.argtypes=[wintypes.LPCWSTR,C.c_void_p,C.POINTER(C.c_void_p)];pdh.PdhAddEnglishCounterW.argtypes=[C.c_void_p,wintypes.LPCWSTR,C.c_void_p,C.POINTER(C.c_void_p)];pdh.PdhCollectQueryData.argtypes=[C.c_void_p];pdh.PdhGetFormattedCounterValue.argtypes=[C.c_void_p,wintypes.DWORD,C.c_void_p,C.POINTER(V)];pdh.PdhCloseQuery.argtypes=[C.c_void_p];query=C.c_void_p();counters=[];paths=(r'\Memory\Page Reads/sec',r'\Memory\Pages Input/sec',r'\Paging File(_Total)\% Usage');samples=[]
 if pdh.PdhOpenQueryW(None,None,C.byref(query)):raise RuntimeError('PdhOpenQueryW')
 try:
  for path in paths:
   c=C.c_void_p()
   if pdh.PdhAddEnglishCounterW(query,path,None,C.byref(c)):raise RuntimeError(path)
   counters.append(c)
  if pdh.PdhCollectQueryData(query):raise RuntimeError('pdh_initial')
  while not stop.wait(.1):
   scheduled=time.perf_counter_ns()
   if pdh.PdhCollectQueryData(query):raise RuntimeError('pdh_collect')
   row={'scheduled_ns':scheduled,'actual_ns':time.perf_counter_ns(),'values':{}}
   for path,c in zip(paths,counters):
    v=V();code=pdh.PdhGetFormattedCounterValue(c,0x200,None,C.byref(v))
    if code or v.status:raise RuntimeError(f'pdh_value:{path}:{code}:{v.status}')
    row['values'][path]=float(v.double)
   samples.append(row)
 finally:pdh.PdhCloseQuery(query)
 return {'paths':list(paths),'samples':samples,'sample_count':len(samples),'closed':True}
def capability():
 l=json.loads(LOCK.read_text())
 if not (l.get('capability_open') is True and ARGS.ack==l.get('capability_audit_token') and not str(ARGS.ack).startswith('PENDING')):raise PermissionError('capability_closed')
 if OUT.exists():
  state=recover_output()
  if state==['valid_commit']:raise FileExistsError('capability_already_complete')
  if any(x for x in OUT.iterdir() if x.name!='failed_attempts'):raise FileExistsError('capability_output_not_clean')
 from het_next_l0_c0r6_kernels import INTEL_SOURCE,NVIDIA_SOURCE
 from het_next_l0_c0r6_sync import Protocol,WinPrimitives,simulate_protocol
 started=time.time_ns();p=WinPrimitives();proto=Protocol(p);coexist=threading.Barrier(2);monitor_stop=threading.Event();results={};errors={};threads={};thread_rows={}
 def monitor():
  try:thread_rows['monitor']=set_affinity(6);results['monitor']=pdh_monitor(monitor_stop)
  except Exception as e:errors['monitor']=f'{type(e).__name__}: {e}'
 def worker(name,lp,fn,source,ergv):
  try:
   thread_rows[name]=set_affinity(lp);d=proto.worker_descriptor(name);proto.worker_started(name,d['epoch']);o=fn(source,ergv,coexist) if name=='intel' else fn(source,coexist);results[name]=o;proto.worker_finish(name,d['epoch'],o,{'finite':True,'thread_id':threading.get_native_id()})
  except Exception as e:errors[name]=f'{type(e).__name__}: {e}'
 threads['monitor']=threading.Thread(target=monitor,name='c0r6-pdh-monitor');threads['monitor'].start();epoch=proto.publish(('intel','nvidia'),'CAPABILITY_COHABITATION')
 threads['intel']=threading.Thread(target=worker,args=('intel',2,intel_capability,INTEL_SOURCE,INTEL_SOURCE),name='c0r6-intel');threads['nvidia']=threading.Thread(target=worker,args=('nvidia',4,nvidia_capability,NVIDIA_SOURCE,None),name='c0r6-nvidia');threads['intel'].start();threads['nvidia'].start();thread_rows['coordinator']=set_affinity(0);t0=proto.wait_ready_release(('intel','nvidia'))
 for n in ('intel','nvidia'):threads[n].join(timeout=30)
 if errors:raise RuntimeError(errors)
 outputs,t1=proto.collect(('intel','nvidia'),epoch);monitor_stop.set();threads['monitor'].join(timeout=5);proto.close();sync=simulate_protocol();intel=results['intel'];nvidia=results['nvidia'];payload={'kind':'het_next_l0_c0r6_capability','status':'capability_positive','runner_sha256':sha(__file__),'runner_lock_sha256':sha(LOCK),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VLOCK),'kernel_sha256':sha(KERNEL),'sync_sha256':sha(SYNC),'provenance_manifest_sha256':sha(PM),'tensor_manifest_sha256':sha(TM),'started_ns':started,'ended_ns':time.time_ns(),'device_payload_reads':0,'weight_payload_reads':0,'d2_shard_files_opened':0,'sentinel_total_bytes':8192,'intel':intel,'nvidia':nvidia,'sync_simulation':sync,'physical_protocol_log':proto.log,'primitive_calls':p.calls,'physical_epoch':epoch,'t0':t0,'t1':t1,'thread_rows':thread_rows,'monitor':results['monitor'],'topology_required_lps':list(LP),'claim_boundary':'capability-only; no weights, benchmark, source-build, validation or component result'};atomic_json(CAP,payload);commit={'kind':'het_next_l0_c0r6_capability_commit','files':{CAP.name:{'bytes':CAP.stat().st_size,'sha256':sha(CAP)}}};atomic_json(COM,commit);return payload
def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('contract','capability','source_build','validation'),required=True);p.add_argument('--ack');ARGS=p.parse_args()
 if ARGS.phase=='contract':print(json.dumps({'kind':'c0r6_contract','lockcheck':lockcheck(),'schedule_count':len(schedule()),'sync':simulate_sync()}));return 0
 if ARGS.phase in ('source_build','validation'):raise PermissionError(f'{ARGS.phase}_not_implemented_and_closed')
 try:o=capability();print(o['status']);return 0
 except Exception as e:
  dispositions=recover_output()
  if FAIL.exists():raise
  atomic_json(FAIL,{'kind':'het_next_l0_c0r6_capability_failure','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'runner_sha256':sha(__file__),'runner_lock_sha256':sha(LOCK),'no_weight_payload':True,'dispositions':dispositions});raise
if __name__=='__main__':raise SystemExit(main())
