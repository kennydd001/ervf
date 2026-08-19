from __future__ import annotations
import json
from common import write_json_atomic,utc_now
from s100_phase16_common import RESULTS,phase14_savings
OUT=RESULTS/"S100_PHASE16D_SELECTED_SAVINGS.json"

def main():
    b=json.loads((RESULTS/"S100_PHASE16B_SUBSET_VALIDATION.json").read_text(encoding="utf-8"))
    names=(b.get("selected_strict_subset") or {}).get("names",[])
    sav=phase14_savings();rows=[{"name":n,**sav.get(n,{})} for n in names]
    base=sum(float(x.get("B1_baseline_ms",0)) for x in rows)
    native=sum(float(x.get("B1_native_ms",0)) for x in rows)
    out={"kind":"s100_phase16d_selected_savings","created_utc":utc_now(),
      "selected_names":names,"selected_count":len(names),
      "phase14_B1_baseline_component_ms":base,
      "phase14_B1_native_component_ms":native,
      "phase14_B1_component_saving_ms":base-native,"rows":rows,
      "claim_boundary":"Phase14 cold component accounting, not integrated latency"}
    write_json_atomic(OUT,out,archive=True);print(json.dumps(out,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
