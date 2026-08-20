from __future__ import annotations

import json
from common import utc_now,write_json_atomic
from s100_phase25_common import RESULTS,OFFICIAL_PARENT_H8_MS,ADOPTION_ABS_MS,STRONG_MS,BREAKTHROUGH_MS,S100_MS
OUT=RESULTS/"S100_PHASE25_SUMMARY.json"

def load(name):
    try:return json.loads((RESULTS/name).read_text(encoding="utf-8"))
    except Exception:return {}
def main():
    pre=load("S100_PHASE25_PREFLIGHT.json");sel=load("S100_PHASE25_SELECTION.json");th=load("S100_PHASE25_THERMAL_ADJUDICATION.json");prof=load("S100_PHASE25_PROFILE.json")
    selected=sel.get("selected") or {};variant=selected.get("variant");screen_ms=selected.get("median_ms")
    st=load(f"S100_PHASE25_STATE_CHECK_{str(variant).upper()}.json") if variant else {}
    c128=load(f"S100_PHASE25_PROMOTED_{str(variant).upper()}_CTX128.json") if variant else {}
    c4096=load(f"S100_PHASE25_PROMOTED_{str(variant).upper()}_CTX4096.json") if variant else {}
    thermal_adopt=bool(th.get("H8_ADOPTED"));thermal_ms=th.get("selected_median_of_rounds_ms")
    target_ms=float(thermal_ms) if thermal_adopt and thermal_ms is not None else (float(screen_ms) if screen_ms is not None else None)
    target_only_s100=bool(thermal_adopt and thermal_ms is not None and float(thermal_ms)<=S100_MS)
    if not pre.get("PREFLIGHT_GREEN"):next_route="REPAIR_PHASE25_TECHNICAL_PREFLIGHT"
    elif not sel.get("H8_STATE_GREEN_ANY"):next_route="REPAIR_OR_CLOSE_H8_NUMERICS"
    elif not sel.get("THERMAL_ADOPTION_OPEN"):next_route="PROFILE_H8_ECONOMICS_AND_REDUCE_DOMINANT_STAGE"
    elif not thermal_adopt:next_route="H8_SCREEN_WIN_NOT_THERMALLY_ADOPTED_PROFILE_OR_CLOSE"
    elif target_only_s100:next_route="OPEN_DRAFTER_INTEGRATION_AND_TRUE_E2E_S100"
    elif target_ms is not None and target_ms<=BREAKTHROUGH_MS:next_route="BUILD_H8_DRAFTER_AWARE_E2E_AND_CLOSE_REMAINING_20MS"
    else:next_route="USE_H8_PARENT_AND_ATTACK_PROFILED_DOMINANT_STAGE"
    out={"kind":"s100_phase25_summary","created_utc":utc_now(),"instrumentation_complete":bool(pre and sel),
      "official_parent_h8_ms":OFFICIAL_PARENT_H8_MS,"adoption_abs_ms":ADOPTION_ABS_MS,
      "selected_variant":variant,"screen_selected":selected,"state":st.get("state"),"state_green":bool(st.get("H8_STATE_GREEN")),
      "thermal":th,"profile_summary":{"status":prof.get("status"),"stage_totals_ms_per_h8":prof.get("stage_totals_ms_per_h8"),
        "weight_streams_per_h8":prof.get("weight_streams_per_h8"),"max_m":prof.get("max_m")},
      "contexts":{"128":(c128.get("summary") or {}),"1024_screen_ms":screen_ms,"4096":(c4096.get("summary") or {})},
      "gates":{"PREFLIGHT_GREEN":bool(pre.get("PREFLIGHT_GREEN")),"H8_STATE_GREEN":bool(st.get("H8_STATE_GREEN")),
        "THERMAL_ADOPTION_OPEN":bool(sel.get("THERMAL_ADOPTION_OPEN")),"H8_ADOPTED":thermal_adopt,
        "STRONG_LE_120MS":bool(target_ms is not None and target_ms<=STRONG_MS),
        "BREAKTHROUGH_LE_100MS":bool(target_ms is not None and target_ms<=BREAKTHROUGH_MS),
        "S100_TARGET_ONLY_LE_80MS":target_only_s100},
      "H8_ACTIVE_PARENT":bool(thermal_adopt),"S100_TARGET_ONLY_ACHIEVED":target_only_s100,
      "S100_SINGLE_ACHIEVED":False,
      "claim_boundary":"Even <=80 ms/H8 is target-verifier-only. True single-stream S100 remains false until drafter/rejection/fallback costs are included.",
      "NEXT_ROUTE":next_route}
    write_json_atomic(OUT,out,archive=True);print(json.dumps(out,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
