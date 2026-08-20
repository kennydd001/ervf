from __future__ import annotations

import json
import numpy as np

from common import utc_now,write_json_atomic
from s100_phase24_common import RESULTS

OUT=RESULTS/"S100_PHASE24_STATE_CHECK.json"

def nrmse(a,b):
    aa=np.asarray(a,np.float64);bb=np.asarray(b,np.float64)
    return float(
      np.linalg.norm(aa-bb)/max(np.linalg.norm(aa),1e-30)
    )

def main():
    bmeta=json.loads(
      (RESULTS/"S100_PHASE24_STATE_BASELINE.json").read_text(encoding="utf-8")
    )
    smeta=json.loads(
      (RESULTS/"S100_PHASE24_STATE_SELECTED.json").read_text(encoding="utf-8")
    )
    if bmeta.get("status")!="measured" or smeta.get("status")!="measured":
        raise RuntimeError("state captures incomplete")

    with np.load(RESULTS/"S100_PHASE24_STATE_BASELINE.npz") as b,\
         np.load(RESULTS/"S100_PHASE24_STATE_SELECTED.npz") as s:
        keys=sorted(set(b.files)&set(s.files))
        ssm=max(nrmse(b[k],s[k]) for k in keys if k.startswith("ssm_"))
        conv=max(nrmse(b[k],s[k]) for k in keys if k.startswith("conv_"))
        kv=max(
          [nrmse(b[k],s[k]) for k in keys
           if k.startswith("k_") or k.startswith("v_")] or [0.0]
        )
        logits=nrmse(b["logits"],s["logits"])
        ids=np.array_equal(b["ids"],s["ids"])
        det=np.array_equal(s["ids"],s["ids_repeat"])
        finite=all(np.isfinite(s[k]).all() for k in keys)

    gates={
      "ids_exact":bool(ids),
      "selected_deterministic_ids":bool(det),
      "ssm":ssm<=5e-5,
      "conv":conv<=1e-5,
      "kv":kv<=5e-6,
      "logits":logits<=5e-4,
      "finite":bool(finite),
    }
    out={"kind":"s100_phase24_state_check","status":"measured",
      "created_utc":utc_now(),
      "selected_config":smeta.get("config"),
      "state":{
        "max_ssm_nrmse":ssm,"max_conv_nrmse":conv,
        "max_kv_nrmse":kv,"logits_nrmse":logits,
      },
      "gates":gates,
      "BEST_OF_ALL_STATE_GREEN":all(gates.values()),
    }
    write_json_atomic(OUT,out,archive=True)
    print(json.dumps(out,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
