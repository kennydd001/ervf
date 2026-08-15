from __future__ import annotations
import hashlib,json,platform
from pathlib import Path
import numpy as np,torch,cupy as cp
from safetensors import safe_open
from safetensors.torch import save_file
import run_port80b_t0y_canonical_router_accumulation as base
ROOT=base.ROOT;REPORTS=base.REPORTS;RUN=ROOT/'reports/runs/streamq5_moe/port80b_t0yr_canonical_router';PREREG=REPORTS/'PORT80B_T0YR_CANONICAL_ROUTER_INCLUDE_REVISION_PREREGISTRATION_2026-08-13.md';LOCK=REPORTS/'port80b_t0yr_canonical_router_lock.json';VERIFIER=ROOT/'scripts/streamq5_moe/verify_port80b_t0yr_canonical_router_include_revision.py'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def main():
 l=json.loads(LOCK.read_text());src={'runner':sha(__file__),'verifier':sha(VERIFIER),'prereg':sha(PREREG)}
 if src!=l['source_sha256']:raise RuntimeError('source lock mismatch pre-CUDA')
 inp={'raw':sha(base.RAW6),'shard':sha(base.SHARD)}
 if inp!=base.EXPECTED:raise RuntimeError('input hash mismatch pre-CUDA')
 inc=ROOT/'.venv/Lib/site-packages/nvidia/cu13/include'
 if not inc.is_dir():raise RuntimeError('bundled CUDA include missing')
 with safe_open(str(base.RAW6),framework='pt',device='cpu') as f:x=f.get_tensor('official_gate_input').contiguous();official_cpu_ids=f.get_tensor('official_ids').contiguous()
 with safe_open(str(base.SHARD),framework='pt',device='cpu') as f:w=f.get_tensor('model.layers.0.mlp.gate.weight').contiguous()
 xn=base.bf16_np(x);wn=base.bf16_np(w);acc=np.zeros((16,512),np.float32)
 for k in range(2048):acc=np.add(acc,np.multiply(xn[:,k,None],wn[None,:,k],dtype=np.float32),dtype=np.float32)
 cpu_ids=np.empty((16,10),np.int64);experts=np.arange(512)
 for r in range(16):cpu_ids[r]=np.lexsort((experts,-acc[r]))[:10]
 free,total=cp.cuda.runtime.memGetInfo()
 if free<1<<30:raise RuntimeError('less than 1GiB free VRAM')
 mod=cp.RawModule(code=base.CUDA,options=('--std=c++11',f'--include-path={inc}'),name_expressions=('canonical_logits','stable_top10'));kl=mod.get_function('canonical_logits');kt=mod.get_function('stable_top10');xg=cp.asarray(x.view(torch.uint16).numpy());wg=cp.asarray(w.view(torch.uint16).numpy());calls=[]
 for _ in range(2):
  lg=cp.empty((16,512),cp.float32);ig=cp.empty((16,10),cp.int64);kl((2,16),(256,),(xg,wg,lg));kt((16,),(1,),(lg,ig));cp.cuda.runtime.deviceSynchronize();calls.append((cp.asnumpy(lg),cp.asnumpy(ig)))
 peak=int(xg.nbytes+wg.nbytes+sum(a.nbytes+b.nbytes for a,b in calls));elog=np.array_equal(acc.view(np.uint32),calls[0][0].view(np.uint32));eids=np.array_equal(cpu_ids,calls[0][1]);repeat=np.array_equal(calls[0][0].view(np.uint32),calls[1][0].view(np.uint32)) and np.array_equal(calls[0][1],calls[1][1]);finite=bool(np.isfinite(acc).all() and np.isfinite(calls[0][0]).all());valid=all(len(set(map(int,row)))==10 and min(row)>=0 and max(row)<512 for row in calls[0][1]);passed=elog and eids and repeat and finite and valid and peak<256<<20
 offgpu=ROOT/'reports/runs/streamq5_moe/port80b_t0xr_router_portability/t0xr_cpu_cuda_router_raw.safetensors';go=None
 if offgpu.is_file():
  with safe_open(str(offgpu),framework='pt',device='cpu') as f:go=f.get_tensor('gpu1_ids').numpy()
 RUN.mkdir(parents=True,exist_ok=True);raw=RUN/'t0yr_canonical_router_raw.safetensors';res=RUN/'t0yr_canonical_router_result.json';save_file({'cpu_logits':torch.from_numpy(acc.copy()),'cpu_ids':torch.from_numpy(cpu_ids.copy()),'gpu1_logits':torch.from_numpy(calls[0][0].copy()),'gpu1_ids':torch.from_numpy(calls[0][1].copy()),'gpu2_logits':torch.from_numpy(calls[1][0].copy()),'gpu2_ids':torch.from_numpy(calls[1][1].copy()),'official_cpu_ids':official_cpu_ids},str(raw))
 out={'kind':'port80b_t0yr_canonical_router_include_revision','status':'canonical_router_pass' if passed else 'canonical_router_negative','overall_pass':passed,'gates':{'cpu_cuda_logits_bitexact':elog,'cpu_cuda_ids_exact':eids,'cuda_repeat':repeat,'finite':finite,'ids_valid':valid,'resource':peak<256<<20},'different_logit_bits':int(np.count_nonzero(acc.view(np.uint32)!=calls[0][0].view(np.uint32))),'different_id_values':int(np.count_nonzero(cpu_ids!=calls[0][1])),'official_cpu_ordered_rows':int(sum(np.array_equal(cpu_ids[r],official_cpu_ids.numpy()[r]) for r in range(16))),'official_cuda_ordered_rows':None if go is None else int(sum(np.array_equal(cpu_ids[r],go[r]) for r in range(16))),'raw_sha256':sha(raw),'inputs':inp,'sources':src,'resources':{'free_vram_before':int(free),'total_vram':int(total),'estimated_peak_bytes':peak},'environment':{'python':platform.python_version(),'numpy':np.__version__,'cupy':cp.__version__,'cuda_runtime':cp.cuda.runtime.runtimeGetVersion(),'cuda_include':str(inc)},'bank_built':False,'claim_boundary':'Canonical router primitive on 16 real rows only; no quality, weight, speed or model pass.'};res.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
