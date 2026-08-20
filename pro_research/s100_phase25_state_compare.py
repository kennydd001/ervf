from __future__ import annotations

import argparse
import json
import numpy as np

from common import utc_now,write_json_atomic
from s100_phase25_common import RESULTS,VARIANTS

def nrmse(a,b):
    aa=np.asarray(a,np.float64);bb=np.asarray(b,np.float64)
    return float(np.linalg.norm(aa-bb)/max(np.linalg.norm(aa),1e-30))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--variant",choices=tuple(VARIANTS),required=True);args=ap.parse_args()
    vtag=args.variant.upper();out=RESULTS/f"S100_PHASE25_STATE_CHECK_{vtag}.json"
    pm=json.loads((RESULTS/"S100_PHASE25_STATE_PARENT.json").read_text(encoding="utf-8"))
    cm=json.loads((RESULTS/f"S100_PHASE25_STATE_{vtag}.json").read_text(encoding="utf-8"))
    if pm.get("status")!="measured" or cm.get("status")!="measured":raise RuntimeError("state captures incomplete")
    with np.load(RESULTS/"S100_PHASE25_STATE_PARENT.npz") as p, np.load(RESULTS/f"S100_PHASE25_STATE_{vtag}.npz") as c:
        keys=sorted(set(p.files)&set(c.files));ssm=max(nrmse(p[k],c[k]) for k in keys if k.startswith("ssm_"))
        conv=max(nrmse(p[k],c[k]) for k in keys if k.startswith("conv_"));kv=max([nrmse(p[k],c[k]) for k in keys if k.startswith("k_") or k.startswith("v_")] or [0.0])
        logits=nrmse(p["logits"],c["logits"]);ids=np.array_equal(p["ids"],c["ids"]);det=np.array_equal(c["ids"],c["ids_repeat"])
        finite=all(np.isfinite(c[k]).all() for k in keys)
    gates={"ids_exact":bool(ids),"candidate_deterministic_ids":bool(det),"ssm":ssm<=5e-5,
      "conv":conv<=1e-5,"kv":kv<=5e-6,"logits":logits<=5e-4,"finite":bool(finite)}
    obj={"kind":"s100_phase25_state_check","status":"measured","variant":args.variant,"created_utc":utc_now(),
      "state":{"max_ssm_nrmse":ssm,"max_conv_nrmse":conv,"max_kv_nrmse":kv,"logits_nrmse":logits},
      "gates":gates,"H8_STATE_GREEN":all(gates.values())}
    write_json_atomic(out,obj,archive=True);print(json.dumps(obj,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
