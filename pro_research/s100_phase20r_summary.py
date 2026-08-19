from __future__ import annotations
import json
from common import REPO,write_json_atomic,write_text_atomic,utc_now
R=REPO/"pro_research"/"results"/"s100_phase20r"
REPORT=REPO/"reports"/"S100_PHASE20R_RUN_REPORT.md"

def load(name):
    try:return json.loads((R/name).read_text(encoding="utf-8"))
    except Exception:return {}

def main():
    pf=load("S100_PHASE20R_PREFLIGHT.json")
    kv=load("S100_PHASE20R_KVSCALE_AUDIT.json")
    patch=load("S100_PHASE20R_PATCH.json")
    cons=load("S100_PHASE20R_CONSUMPTION.json")
    ref=load("S100_PHASE20R_REFERENCE.json")
    par=load("S100_PHASE20R_CANDIDATE_PARITY.json")
    green=bool(
        pf.get("PREFLIGHT_GREEN") is True
        and kv.get("KVSCALE_SEMANTICS_GREEN") is True
        and patch.get("PATCH_APPLIED") is True
        and cons.get("TARGET_CONSUMPTION_GREEN") is True
        and ref.get("full_reference_available") is True
        and par.get("PARITY_GREEN") is True
    )
    if green: route="RUN_PHASE20B_FULL_PERFECT_DRAFT_VERIFIER"
    elif pf.get("PREFLIGHT_GREEN") is not True: route="REPAIR_PHASE20R_PREFLIGHT"
    elif kv.get("KVSCALE_SEMANTICS_GREEN") is not True: route="REPAIR_KVSCALE_SEMANTICS"
    elif patch.get("PATCH_APPLIED") is not True: route="REPAIR_GUARDED_RUNTIME_PATCH"
    elif cons.get("TARGET_CONSUMPTION_GREEN") is not True: route="REPAIR_TARGET_CONSUMPTION"
    elif ref.get("full_reference_available") is not True: route="REPAIR_OR_REPLACE_INDEPENDENT_REFERENCE_ONLY"
    else: route="DEBUG_OFFICIAL_PARITY_BEFORE_20B"
    out={
        "kind":"s100_phase20r_summary","created_utc":utc_now(),
        "PREFLIGHT_GREEN":pf.get("PREFLIGHT_GREEN"),
        "KVSCALE_SEMANTICS_GREEN":kv.get("KVSCALE_SEMANTICS_GREEN"),
        "PATCH_APPLIED":patch.get("PATCH_APPLIED"),
        "TARGET_CONSUMPTION_GREEN":cons.get("TARGET_CONSUMPTION_GREEN"),
        "FULL_REFERENCE_AVAILABLE":ref.get("full_reference_available"),
        "REFERENCE_TRANSFORMERS_VERSION":ref.get("transformers_version"),
        "REFERENCE_MODEL_CLASS":ref.get("model_class"),
        "PARITY_GREEN":par.get("PARITY_GREEN"),
        "parity_summary":par.get("summary"),
        "PHASE20A_OFFICIAL_PARITY_GREEN":green,
        "PHASE20B_FULL_VERIFIER_OPEN":green,
        "NEXT_ROUTE":route,"S100_SINGLE_ACHIEVED":False,
        "claim_boundary":"Phase20A repair only; no block-verifier timing",
    }
    R.mkdir(parents=True,exist_ok=True)
    write_json_atomic(R/"S100_PHASE20R_SUMMARY.json",out,archive=True)
    text=("S100 PHASE 20R — KV SCALE + REFERENCE\n"
          f"PREFLIGHT_GREEN: {out['PREFLIGHT_GREEN']}\n"
          f"KVSCALE_SEMANTICS_GREEN: {out['KVSCALE_SEMANTICS_GREEN']}\n"
          f"PATCH_APPLIED: {out['PATCH_APPLIED']}\n"
          f"TARGET_CONSUMPTION_GREEN: {out['TARGET_CONSUMPTION_GREEN']}\n"
          f"FULL_REFERENCE_AVAILABLE: {out['FULL_REFERENCE_AVAILABLE']}\n"
          f"REFERENCE_TRANSFORMERS_VERSION: {out['REFERENCE_TRANSFORMERS_VERSION']}\n"
          f"PARITY_GREEN: {out['PARITY_GREEN']}\n"
          f"PHASE20A_OFFICIAL_PARITY_GREEN: {green}\n"
          f"PHASE20B_FULL_VERIFIER_OPEN: {green}\n"
          f"NEXT_ROUTE: {route}\nS100 SINGLE ACHIEVED: False\n")
    (R/"S100_PHASE20R_SUMMARY.txt").write_text(text,encoding="utf-8")

    kv_rows=[]
    for layer,row in sorted((kv.get("per_layer") or {}).items(),key=lambda x:int(x[0])):
        kv_rows.append(
            f"| {layer} | {row.get('k_scale')} | {row.get('v_scale')} | "
            f"{row.get('mean_unit_nrmse')} | {row.get('mean_scaled_nrmse')} | {row.get('pass')} |"
        )
    if not kv_rows: kv_rows=["| — | — | — | — | — | — |"]
    ps=par.get("summary") or {}
    reference_error=(ref.get("error") or {}).get("message")
    report="\n".join([
        "# S100 Phase 20R — KV-scale + independent-reference repair",
        "",
        "Target: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`",
        "Snapshot: `e8f3c7c4de75ad84fe1bcef95d38eca76214480b`",
        "",
        "## Adjudication",
        "",
        f"- Preflight exact-12 blocker: **{out['PREFLIGHT_GREEN']}**",
        f"- KV-scale semantics: **{out['KVSCALE_SEMANTICS_GREEN']}**",
        f"- Guarded production patch: **{out['PATCH_APPLIED']}**",
        f"- Target consumption: **{out['TARGET_CONSUMPTION_GREEN']}**",
        f"- Independent full reference: **{out['FULL_REFERENCE_AVAILABLE']}**",
        f"- Candidate/reference parity: **{out['PARITY_GREEN']}**",
        f"- `PHASE20A_OFFICIAL_PARITY_GREEN`: **{green}**",
        f"- `PHASE20B_FULL_VERIFIER_OPEN`: **{green}**",
        f"- Next route: `{route}`",
        "",
        "## Attention KV scales",
        "",
        "| layer | k_scale | v_scale | unit NRMSE | scaled NRMSE | pass |",
        "|---:|---:|---:|---:|---:|---|",
        *kv_rows,
        "",
        "## Independent reference",
        "",
        f"- Transformers: `{ref.get('transformers_version')}`",
        f"- Config class: `{ref.get('config_class')}`",
        f"- Model class: `{ref.get('model_class')}`",
        f"- Full model loaded: `{ref.get('full_model_loaded')}`",
        f"- Technical blocker: `{reference_error}`",
        "",
        "## Parity",
        "",
        f"- tokens: {ps.get('tokens')}",
        f"- top1: {ps.get('top1_agreement')}",
        f"- top5: {ps.get('target_in_top5')}",
        f"- mean CE delta: {ps.get('mean_ce_delta')}",
        f"- mean coarse KL: {ps.get('mean_coarse_kl')}",
        f"- p95 coarse KL: {ps.get('p95_coarse_kl')}",
        "",
        "## Claim boundary",
        "",
        "20R only repairs and adjudicates Phase20A. It contains no full H=4 block-verifier timing and cannot claim S100.",
        "",
    ])
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    write_text_atomic(REPORT,report,archive=True)
    print(text);return 0
if __name__=="__main__":raise SystemExit(main())
