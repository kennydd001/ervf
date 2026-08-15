from __future__ import annotations

import hashlib, json, platform
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import run_port80b_t0x_cpu_cuda_router_portability as base

ROOT=base.ROOT; REPORTS=base.REPORTS
RUN_DIR=ROOT/"reports/runs/streamq5_moe/port80b_t0xr_router_portability"
PREREG=REPORTS/"PORT80B_T0XR_CPU_CUDA_ROUTER_PORTABILITY_REVISION_PREREGISTRATION_2026-08-13.md"
LOCK=REPORTS/"port80b_t0xr_router_portability_lock.json"
VERIFIER=ROOT/"scripts/streamq5_moe/verify_port80b_t0xr_cpu_cuda_router_portability_revision.py"

def sha256(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(8<<20),b""): h.update(c)
 return h.hexdigest()

def main()->int:
 lock=json.loads(LOCK.read_text())
 sources={"runner":sha256(Path(__file__)),"verifier":sha256(VERIFIER),"prereg":sha256(PREREG)}
 if sources!=lock["source_sha256"]: raise RuntimeError("source lock mismatch before CUDA initialization")
 inputs={"r6_raw_sha256":sha256(base.R6_RAW),"r6_json_sha256":sha256(base.R6_JSON),"shard_sha256":sha256(base.SHARD)}
 if inputs!=base.EXPECTED: raise RuntimeError("input hash mismatch before CUDA initialization")
 with safe_open(str(base.R6_RAW),framework="pt",device="cpu") as f:
  hidden=f.get_tensor("official_gate_input").contiguous(); archived_logits=f.get_tensor("official_logits").contiguous(); archived_ids=f.get_tensor("official_ids").contiguous(); archived_weights=f.get_tensor("official_weights").contiguous()
 with safe_open(str(base.SHARD),framework="pt",device="cpu") as f: weight=f.get_tensor("model.layers.0.mlp.gate.weight").contiguous()
 if hidden.shape!=(16,2048) or hidden.dtype!=torch.bfloat16: raise RuntimeError("repaired hidden contract mismatch")
 if weight.shape!=(512,2048) or weight.dtype!=torch.bfloat16: raise RuntimeError("weight contract mismatch")
 torch.set_grad_enabled(False); torch.use_deterministic_algorithms(True)
 cpu=tuple(x.detach().cpu().contiguous() for x in base.route(hidden,weight))
 replay={"logits":base.bits_equal(cpu[0],archived_logits),"ids":torch.equal(cpu[2],archived_ids),"weights":base.bits_equal(cpu[3],archived_weights)}
 if not all(replay.values()): raise RuntimeError(f"CPU archive replay mismatch: {replay}")
 if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
 free,total=torch.cuda.mem_get_info()
 if free<1<<30: raise RuntimeError("less than 1 GiB free VRAM")
 dev=torch.device("cuda:0"); torch.cuda.reset_peak_memory_stats(dev)
 hg=hidden.to(dev); wg=weight.to(dev); calls=[]
 for _ in range(2):
  o=base.route(hg,wg); torch.cuda.synchronize(dev); calls.append(tuple(x.detach().cpu().contiguous() for x in o))
 peak=int(torch.cuda.max_memory_allocated(dev))
 if peak>=256<<20: raise RuntimeError("CUDA allocation exceeded 256 MiB")
 r6=json.loads(base.R6_JSON.read_text()); rows=[]
 for i in range(16):
  ci=[int(x) for x in cpu[2][i].tolist()]; gi=[int(x) for x in calls[0][2][i].tolist()]
  rows.append({"row":i,"ordered_ids_equal":ci==gi,"id_set_equal":set(ci)==set(gi),"cpu_ids":ci,"gpu_ids":gi,"symmetric_difference":sorted(set(ci)^set(gi)),"cpu_probability_margin":float(r6["rows"][i]["probability_margin"]),"cpu_boundary_tie_expert_ids":r6["rows"][i]["boundary_tie_expert_ids"],"selected_weights_bit_equal":base.bits_equal(cpu[3][i],calls[0][3][i]),"logit_max_abs":float((cpu[0][i].float()-calls[0][0][i].float()).abs().max())})
 repeat={"logits":base.bits_equal(calls[0][0],calls[1][0]),"probs":torch.equal(calls[0][1],calls[1][1]),"ids":torch.equal(calls[0][2],calls[1][2]),"weights":base.bits_equal(calls[0][3],calls[1][3])}
 finite=all(torch.isfinite(x.float()).all().item() for call in (cpu,calls[0],calls[1]) for x in (call[0],call[1],call[3]))
 exact=all(r["ordered_ids_equal"] and r["selected_weights_bit_equal"] for r in rows) and all(repeat.values()) and finite
 verdict="exact_cross_backend_pass" if exact else "cross_backend_negative"
 RUN_DIR.mkdir(parents=True,exist_ok=True); raw=RUN_DIR/"t0xr_cpu_cuda_router_raw.safetensors"; result=RUN_DIR/"t0xr_cpu_cuda_router_result.json"
 save_file({"hidden":hidden,"gate_weight":weight,"cpu_logits":cpu[0],"cpu_probs":cpu[1],"cpu_ids":cpu[2],"cpu_weights":cpu[3],"gpu1_logits":calls[0][0],"gpu1_probs":calls[0][1],"gpu1_ids":calls[0][2],"gpu1_weights":calls[0][3],"gpu2_logits":calls[1][0],"gpu2_probs":calls[1][1],"gpu2_ids":calls[1][2],"gpu2_weights":calls[1][3]},str(raw))
 props=torch.cuda.get_device_properties(dev)
 out={"kind":"port80b_t0xr_cpu_cuda_router_portability_revision","status":verdict,"overall_pass":exact,"verdict":verdict,"rows":rows,"cpu_archive_replay":replay,"cuda_repeat":repeat,"all_finite":finite,"ordered_id_equal_rows":sum(r["ordered_ids_equal"] for r in rows),"id_set_equal_rows":sum(r["id_set_equal"] for r in rows),"weight_bit_equal_rows":sum(r["selected_weights_bit_equal"] for r in rows),"zero_cpu_margin_rows":sum(r["cpu_probability_margin"]==0 for r in rows),"raw_artifact":str(raw.relative_to(ROOT)).replace("\\","/"),"raw_sha256":sha256(raw),"input_sha256":inputs,"source_sha256":sources,"resources":{"free_vram_before":int(free),"total_vram":int(total),"peak_cuda_allocated":peak},"environment":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"device":props.name,"capability":[props.major,props.minor]},"bank_built":False,"host_registered":False,"claim_boundary":"Diagnostic CPU/CUDA router portability for 16 archived layer-0 rows only."}
 result.write_text(json.dumps(out,indent=2)); print(json.dumps({"status":verdict,"ordered_id_equal_rows":out["ordered_id_equal_rows"],"id_set_equal_rows":out["id_set_equal_rows"],"weight_bit_equal_rows":out["weight_bit_equal_rows"],"cuda_repeat":repeat,"peak_cuda_allocated":peak},indent=2)); return 0 if exact else 1

if __name__=="__main__": raise SystemExit(main())
