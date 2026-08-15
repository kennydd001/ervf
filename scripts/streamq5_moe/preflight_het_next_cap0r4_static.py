#!/usr/bin/env python3
"""Static/no-device CAP0 contract preflight; execution intentionally closed."""
from __future__ import annotations
import ast,hashlib,json,struct,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';OUT=ROOT/'reports/runs/streamq5_moe/het_next_cap0r4_dual_device_cohabitation'
RUN=S/'run_het_next_cap0r4_dual_device_cohabitation.py';PRO=S/'het_next_cap0r4_protocol.py';KER=S/'het_next_cap0r4_kernels.py';VER=S/'verify_het_next_cap0r4_dual_device_cohabitation.py';LOCK=R/'het_next_cap0r4_runner_lock.json';VLOCK=R/'het_next_cap0r4_verifier_lock.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixture():
 x=[0x45585430]
 for _ in range(1,1024):x.append((1664525*x[-1]+1013904223)&0xffffffff)
 yi=[(((((v^0xa5a5a5a5)<<7)|((v^0xa5a5a5a5)>>25))+0x3c6ef372)&0xffffffff) for v in x];yn=[((((((v+0x9e3779b9)&0xffffffff)>>11)|(((v+0x9e3779b9)&0xffffffff)<<21))&0xffffffff)^0xc3c3c3c3 for v in x];h=lambda a:hashlib.sha256(struct.pack('<1024I',*a)).hexdigest();return h(x),h(yi),h(yn)
def main():
 l=json.loads(LOCK.read_text());vl=json.loads(VLOCK.read_text());checks={};checks['closed']=l['execution_open'] is False and vl['execution_open'] is False and l['audit_token']==vl['audit_token']=='PENDING_INDEPENDENT_SOURCE_AUDIT';checks['output_absent']=not OUT.exists();checks['fixtures']=fixture()==('a9d32afd712f6ac80ef7739b11c2baa59e4f84c2067e20307f175de4e8a1acca','c83e434be87333bc6bf15d3f0ee492c3e3f9d65b847902bea55310165a42923f','f07c3d87d952d1dc82c65d90f467af87426c1658267b7d94f359122e73eafd5f')
 files={**{k:v for k,v in l.items() if k.endswith('_sha256')},**{f'v_{k}':v for k,v in vl.get('files',{}).items()}};checks['runner_bindings']=all(l[k]==sha(p) for k,p in {'runner_sha256':RUN,'protocol_sha256':PRO,'kernel_sha256':KER,'verifier_sha256':VER,'verifier_lock_sha256':VLOCK,'preflight_sha256':Path(__file__),'prereg_sha256':R/'HET_NEXT_CAP0R4_DUAL_DEVICE_COHABITATION_PREREGISTRATION_2026-08-13.md','design_sha256':R/'HET_NEXT_CAP0R4_STATIC_PREFLIGHT_DESIGN_2026-08-13.md'}.items());checks['verifier_bindings']=all(sha(ROOT/v['path'])==v['sha256'] and (ROOT/v['path']).stat().st_size==v['bytes'] for v in vl['files'].values())
 tree=ast.parse(RUN.read_text());imports={n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)}|{n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module};forbidden=('torch','transformers','safetensors','model','checkpoint','q5','shard','d2','benchmark','throughput','percentile');all_source='\n'.join((RUN.read_text(),PRO.read_text(),KER.read_text(),VER.read_text()));checks['scope']=not any(x and any(f in x.lower() for f in forbidden) for x in imports) and not any(token in all_source.lower() for token in ('from torch','import torch','safetensors','transformers','checkpoint_path','shard_path','q5_','throughput_','percentile(')) and all(x in RUN.read_text() for x in ('WORD_COUNT = 1024','REPETITIONS = 3','BUFFER_BYTES = 4096','THREAD_LPS = {"coordinator": 0, "intel": 2, "nvidia": 4, "monitor": 6}'))
 src=RUN.read_text();required=('clHostMemAllocINTEL','clGetMemAllocInfoINTEL','clSetKernelArgMemPointerINTEL','clGetExtensionFunctionAddressForPlatform','PdhAddEnglishCounterW','PdhRemoveCounter.argtypes','CreateWaitableTimerW.argtypes','WaitForSingleObject.argtypes','CloseHandle.argtypes','SetThreadGroupAffinity.argtypes','cuInit','cuDeviceGetUuid_v2','cuCtxCreate_v2','cuCtxDestroy_v2','cuModuleLoadDataEx','cuModuleUnload','cuStreamCreate','cuStreamDestroy_v2','cuMemHostAlloc','cuMemFreeHost','cuMemAlloc_v2','cuMemFree_v2','cuMemcpyHtoDAsync_v2','cuMemcpyDtoHAsync_v2','cuLaunchKernel','cuEventCreate','cuEventDestroy_v2','nvrtcCreateProgram','nvrtcDestroyProgram','CL_DEVICE_PCI_BUS_INFO_KHR');checks['abi_contract']=all(x in src for x in required) and not any(x in src for x in ('clCreateBuffer','clEnqueueWriteBuffer','clEnqueueReadBuffer','clEnqueueMigrateMemObjects','import cupy','from cupy','cuda.runtime','default_stream'))
 vt=ast.parse(VER.read_text());vimports={n.names[0].name for n in ast.walk(vt) if isinstance(n,ast.Import)}|{n.module for n in ast.walk(vt) if isinstance(n,ast.ImportFrom) and n.module};checks['independent_verifier']=not any(x and x.startswith('het_next_cap0') for x in vimports) and all(x in VER.read_text() for x in ("len(life)==3","len(ledger)==sum(expected.values())","len(samples)>=11"))
 ns={};exec(compile(PRO.read_text(),str(PRO),'exec'),ns);sim=ns['simulate_protocol']();checks['actual_protocol_simulation']=sim['pass'] and len(sim['outputs'])==3 and all(sim['negative'].values())
 with tempfile.TemporaryDirectory(prefix='cap0_static_') as td:
  td=Path(td);checks['actual_atomic_failure_simulation']=all(ns['simulate_atomic_failure'](td).values());tx=td/'tx';tx.mkdir();(tx/'cap0r4_result.json.x.inprogress').write_bytes(b'x');rec=ns['recover_transaction'](tx);checks['actual_transaction_recovery']=len(rec)==1 and (tx/'failed_attempts').exists()
 checks['actual_release_failure_simulation']=all(ns['simulate_release_failure']().values());checks['cacheline_cells']=ns['C'].sizeof(ns['CachelineCellStorage'])==128 and ns['CachelineCellStorage'].value.offset==0
 out={'kind':'het_next_cap0r4_static_preflight','pass':all(checks.values()),'checks':checks,'device_calls':0,'model_or_weight_opens':0};print(json.dumps(out,sort_keys=True));return 0 if out['pass'] else 1
if __name__=='__main__':raise SystemExit(main())






