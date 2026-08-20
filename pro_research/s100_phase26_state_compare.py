from __future__ import annotations

import json
from common import utc_now, write_json_atomic
from s100_phase26_common import RESULTS, compare_npz

OUT=RESULTS/"S100_PHASE26_STATE_CHECK.json"

def meta(name):
    return json.loads(
        (RESULTS/f"S100_PHASE26_STATE_{name}.json").read_text(encoding="utf-8")
    )

def main():
    pairs={}
    all_green=True
    for horizon,parent,cand in (
        ("h4","H4_PARENT","H4_OVERLAP"),
        ("h8","H8_PARENT","H8_OVERLAP"),
    ):
        pm=meta(parent);cm=meta(cand)
        if pm.get("status")!="measured" or cm.get("status")!="measured":
            raise RuntimeError(f"{horizon} state captures incomplete")
        state,gates=compare_npz(
            RESULTS/f"S100_PHASE26_STATE_{parent}.npz",
            RESULTS/f"S100_PHASE26_STATE_{cand}.npz",
        )
        green=all(gates.values())
        all_green &= green
        pairs[horizon]={
          "state":state,"gates":gates,
          "green":green,
          "parent_ids":pm.get("ids"),
          "candidate_ids":cm.get("ids"),
          "candidate_repeat_ids":cm.get("ids_repeat"),
        }

    out={
      "kind":"s100_phase26_state_check","status":"measured",
      "created_utc":utc_now(),"pairs":pairs,
      "H4_OVERLAP_STATE_GREEN":bool(pairs["h4"]["green"]),
      "H8_OVERLAP_STATE_GREEN":bool(pairs["h8"]["green"]),
      "ALL_OVERLAP_STATE_GREEN":bool(all_green),
    }
    write_json_atomic(OUT,out,archive=True)
    print(json.dumps(out,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
