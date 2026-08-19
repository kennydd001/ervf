from __future__ import annotations
import json,traceback
from common import REPO,require_model_dir,write_json_atomic,utc_now
from s100_phase20a_identity import _schema_payload
OUT=REPO/"pro_research"/"results"/"s100_phase20r"/"S100_PHASE20R_CONSUMPTION.json"

def main():
    p={"kind":"s100_phase20r_consumption","status":"started","started_utc":utc_now()}
    try:
        _,audit=_schema_payload(require_model_dir())
        unknown=audit.get("UNKNOWN_UNUSED_WEIGHTS",[])
        p.update({"status":"measured","audit":audit,
                  "KVSCALE_UNKNOWN_REMAINING":[x for x in unknown if x.endswith("k_scale") or x.endswith("v_scale")],
                  "TARGET_CONSUMPTION_GREEN":bool(audit.get("target_consumption_gate")),
                  "completed_utc":utc_now()})
    except Exception as e:
        p.update({"status":"technical_failure","error":{"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()},"completed_utc":utc_now()})
    OUT.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,p,archive=True)
    print(json.dumps({"status":p.get("status"),"TARGET_CONSUMPTION_GREEN":p.get("TARGET_CONSUMPTION_GREEN"),
                      "unknown":(p.get("audit") or {}).get("UNKNOWN_UNUSED_WEIGHTS"),
                      "missing":(p.get("audit") or {}).get("EXPECTED_BUT_MISSING_WEIGHTS"),
                      "error":(p.get("error") or {}).get("message")},indent=2))
    return 0 if p.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
