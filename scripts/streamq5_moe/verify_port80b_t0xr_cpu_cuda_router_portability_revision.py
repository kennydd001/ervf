from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2]; REPORTS=ROOT/"reports/streamq5_moe"; RUN=ROOT/"reports/runs/streamq5_moe/port80b_t0xr_router_portability"
LOCK=REPORTS/"port80b_t0xr_router_portability_lock.json"; RESULT=RUN/"t0xr_cpu_cuda_router_result.json"; RAW=RUN/"t0xr_cpu_cuda_router_raw.safetensors"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for c in iter(lambda:f.read(8<<20),b""): h.update(c)
 return h.hexdigest()
def eq(a,b): return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.view(torch.uint16),b.view(torch.uint16))
def preflight():
 l=json.loads(LOCK.read_text()); c={"runner":sha(ROOT/l["runner"])==l["source_sha256"]["runner"],"verifier":sha(Path(__file__))==l["source_sha256"]["verifier"],"prereg":sha(ROOT/l["prereg"])==l["source_sha256"]["prereg"],"outputs_closed":not RESULT.exists() and not RAW.exists()}; print(json.dumps({"kind":"t0xr_preflight","pass":all(c.values()),"checks":c},indent=2)); return 0 if all(c.values()) else 1
def verify():
 j=json.loads(RESULT.read_text());
 with safe_open(str(RAW),framework="pt",device="cpu") as f: t={k:f.get_tensor(k).contiguous() for k in f.keys()}
 rows=[]
 for i in range(16):
  ci=t["cpu_ids"][i];gi=t["gpu1_ids"][i];rows.append((torch.equal(ci,gi),set(map(int,ci.tolist()))==set(map(int,gi.tolist())),eq(t["cpu_weights"][i],t["gpu1_weights"][i])))
 repeat={"logits":eq(t["gpu1_logits"],t["gpu2_logits"]),"probs":torch.equal(t["gpu1_probs"],t["gpu2_probs"]),"ids":torch.equal(t["gpu1_ids"],t["gpu2_ids"]),"weights":eq(t["gpu1_weights"],t["gpu2_weights"])}
 finite=all(torch.isfinite(t[k].float()).all().item() for k in ("cpu_logits","cpu_probs","cpu_weights","gpu1_logits","gpu1_probs","gpu1_weights","gpu2_logits","gpu2_probs","gpu2_weights")); exact=all(a and c for a,b,c in rows) and all(repeat.values()) and finite; verdict="exact_cross_backend_pass" if exact else "cross_backend_negative"
 checks={"raw_sha":sha(RAW)==j["raw_sha256"],"row_counts":sum(a for a,b,c in rows)==j["ordered_id_equal_rows"] and sum(b for a,b,c in rows)==j["id_set_equal_rows"] and sum(c for a,b,c in rows)==j["weight_bit_equal_rows"],"cuda_repeat":repeat==j["cuda_repeat"],"finite":finite==j["all_finite"],"verdict":j["verdict"]==verdict and j["overall_pass"]==exact,"diagnostic_boundary":j["bank_built"] is False and j["host_registered"] is False}
 out={"kind":"t0xr_independent_verification","verification_pass":all(checks.values()),"scientific_pass":exact,"verdict":verdict,"checks":checks,"ordered_id_equal_rows":sum(a for a,b,c in rows),"id_set_equal_rows":sum(b for a,b,c in rows),"weight_bit_equal_rows":sum(c for a,b,c in rows),"claim_boundary":"Independent stored-tensor replay, no CUDA rerun."}; (REPORTS/"port80b_t0xr_cpu_cuda_router_portability_independent_verification.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2)); return 0 if all(checks.values()) else 1
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--phase",choices=("preflight","verify"),required=True);a=p.parse_args();raise SystemExit(preflight() if a.phase=="preflight" else verify())
