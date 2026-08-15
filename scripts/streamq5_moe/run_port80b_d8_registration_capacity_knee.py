from __future__ import annotations

import hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
import psutil

ROOT_PATH=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT_PATH))
from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_port80b_d2_registered_scatter import EXPECTED_BANK_SHA256,REGISTER_FLAGS,record_offset,unregister_ranges
from scripts.streamq5_moe.run_port80b_p0_physical_host_bank import BANK,BANK_BYTES,EXPERT_BYTES,LAYERS,MANIFEST

R=ROOT/"reports/streamq5_moe";PREREG=R/"PORT80B_D8_REGISTRATION_CAPACITY_KNEE_PREREGISTRATION.md";OUTPUT=R/"port80b_d8_registration_capacity_knee.json";REPORT=R/"PORT80B_D8_REGISTRATION_CAPACITY_KNEE_REPORT_2026-08-12.md"
PREFIXES=(435,461,486,499,512);ENTROPY_PIN_GIB=41.441;MIN_AVAILABLE=2*2**30

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 if OUTPUT.exists()or REPORT.exists():raise FileExistsError("refusing overwrite")
 manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
 if not BANK.is_file()or BANK.stat().st_size!=BANK_BYTES or manifest.get("bank_sha256")!=EXPECTED_BANK_SHA256:raise RuntimeError("bank contract")
 mapped=np.memmap(BANK,dtype=np.uint8,mode="r",shape=(BANK_BYTES,));rows=[];largest=0;started=time.perf_counter()
 for experts in PREFIXES:
  pointers=[];row={"experts_per_layer":experts,"fraction":experts/512,"registered_bytes":LAYERS*experts*EXPERT_BYTES,"registered_gib":LAYERS*experts*EXPERT_BYTES/2**30,"available_before":int(psutil.virtual_memory().available)}
  try:
   begin=time.perf_counter()
   for layer in range(LAYERS):
    pointer=int(mapped.ctypes.data)+record_offset(layer,0);cp.cuda.runtime.hostRegister(pointer,experts*EXPERT_BYTES,REGISTER_FLAGS);pointers.append(pointer)
   row["registration_seconds"]=time.perf_counter()-begin;row["registered_ranges"]=len(pointers);row["available_after_registration"]=int(psutil.virtual_memory().available);row["success"]=len(pointers)==LAYERS
   if row["available_after_registration"]<MIN_AVAILABLE:row["safety_stop"]=True
  except Exception as exc:
   row.update({"success":False,"error":f"{type(exc).__name__}: {exc}"})
  finally:
   row["unregister_failures"]=unregister_ranges(pointers);row["clean_success"]=bool(row.get("success")) and not row["unregister_failures"];row["available_after_unregister"]=int(psutil.virtual_memory().available);rows.append(row)
   if row["clean_success"]:largest=max(largest,row["registered_bytes"])
  if not row.get("success")or row.get("safety_stop")or row["unregister_failures"]:break
 entropy_bytes=int(ENTROPY_PIN_GIB*2**30);result={"kind":"port80b_d8_registration_capacity_knee","completed_utc":datetime.now(timezone.utc).isoformat(),"status":"capacity_knee_measured","inputs":{"preregistration_sha256":sha(PREREG),"evaluator_sha256":sha(Path(__file__)),"manifest_sha256":sha(MANIFEST),"bank_sha256_from_manifest":manifest["bank_sha256"]},"sweep":rows,"largest_successful_registered_bytes":largest,"largest_successful_registered_gib":largest/2**30,"entropy_pin_theoretical_gib":ENTROPY_PIN_GIB,"entropy_pin_below_largest_successful_capacity":entropy_bytes<=largest,"wall_seconds":time.perf_counter()-started,"claim_boundary":"Registration capacity only; no compressed bank, entropy codec, payload decode, transfer timing, model or quality result."}
 OUTPUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");REPORT.write_text("# PORT80B-D8 — registration capacity knee\n\n"+f"Largest successful prefix: **{result['largest_successful_registered_gib']:.3f} GiB**. EntropyPin theoretical 41.441 GiB below measured capacity: **{result['entropy_pin_below_largest_successful_capacity']}**.\n",encoding="utf-8");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
