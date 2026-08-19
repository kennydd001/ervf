from __future__ import annotations
import json
from common import REPO,utc_now,write_json_atomic

R=REPO/"pro_research"/"results"/"s100_phase20s"
OUT=R/"S100_PHASE20S_SUMMARY.json"

def load(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return {}

def main():
    reclass=load(R/"S100_PHASE20S_RECLASSIFY.json")
    kv=load(R/"S100_PHASE20S_FP8KV_FIDELITY.json")
    oracle=load(R/"S100_PHASE20S_LAYER_ORACLE.json")
    tref=load(R/"S100_PHASE20S_TRANSFORMERS_REFERENCE.json")
    p20=load(REPO/"pro_research"/"results"/"s100_phase20a_identity.json")
    sabotage=((p20.get("all_23_mamba_screen") or {}).get("sabotage_control") or {})
    sabotage_green=bool(sabotage.get("observable_change"))

    target_math=bool(reclass.get("TARGET_MATH_CONSUMPTION_GREEN"))
    oracle_green=bool(oracle.get("INDEPENDENT_LAYER_ORACLE_GREEN"))
    full_ref=bool(tref.get("status")=="measured" and tref.get("full_model_executed"))
    parity=bool(target_math and oracle_green and sabotage_green)
    kv_open=bool(kv.get("FP8_KV_SERVING_OPEN"))
    policy="fp8_kv=True" if kv_open else "fp8_kv=False"

    if parity:
        route="BUILD_PHASE20B_FULL_H4_VERIFIER_FP32KV" if not kv_open else "BUILD_PHASE20B_FULL_H4_VERIFIER_FP8KV"
    elif not target_math:
        route="REPAIR_TARGET_CONSUMPTION_CLASSIFICATION"
    elif not oracle_green:
        route="REPAIR_INDEPENDENT_LAYER_PARITY"
    else:
        route="REPAIR_SABOTAGE_OR_INCOMPLETE_EVIDENCE"

    out={
      "kind":"s100_phase20s_summary","created_utc":utc_now(),
      "TARGET_MATH_CONSUMPTION_GREEN":target_math,
      "FP8_KV_SERVING_OPEN":kv_open,
      "phase20b_kv_policy":policy,
      "INDEPENDENT_LAYER_ORACLE_GREEN":oracle_green,
      "TRANSFORMERS_FULL_REFERENCE_EXECUTED":full_ref,
      "transformers_reference_status":tref.get("status"),
      "sabotage_green":sabotage_green,
      "PHASE20A_OFFICIAL_PARITY_GREEN":parity,
      "PHASE20B_FULL_VERIFIER_OPEN":parity,
      "NEXT_ROUTE":route,
      "S100_SINGLE_ACHIEVED":False,
      "claim_boundary":"target-math parity adjudication; no full block timing",
    }
    R.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,out,archive=True)
    text=(
      "S100 PHASE 20S — TARGET MATH / SERVING METADATA\n"
      f"TARGET_MATH_CONSUMPTION_GREEN: {target_math}\n"
      f"INDEPENDENT_LAYER_ORACLE_GREEN: {oracle_green}\n"
      f"TRANSFORMERS_FULL_REFERENCE_EXECUTED: {full_ref}\n"
      f"FP8_KV_SERVING_OPEN: {kv_open}\n"
      f"Phase20B KV policy: {policy}\n"
      f"Sabotage green: {sabotage_green}\n"
      f"PHASE20A_OFFICIAL_PARITY_GREEN: {parity}\n"
      f"PHASE20B_FULL_VERIFIER_OPEN: {parity}\n"
      f"NEXT_ROUTE: {route}\n"
      "S100 SINGLE ACHIEVED: False\n"
    )
    (R/"S100_PHASE20S_SUMMARY.txt").write_text(text,encoding="utf-8")
    print(text);return 0
if __name__=="__main__":raise SystemExit(main())
