from __future__ import annotations
import json
from common import REPO, write_json_atomic, utc_now

R=REPO/"pro_research"/"results"/"s100_phase17"

def load(name):
    try:return json.loads((R/name).read_text(encoding="utf-8"))
    except Exception:return {}

def main():
    pf=load("S100_PHASE17_PREFLIGHT.json")
    d=load("S100_PHASE17_MAMBA_BLOCK.json")
    complete=bool(
        pf.get("status")=="measured" and pf.get("PREFLIGHT_GREEN")
        and d.get("status")=="measured"
    )
    ssm=d.get("SSM_SCAN_MICROKERNEL_OPEN") if d.get("status")=="measured" else None
    core=d.get("MAMBA_CORE_BLOCK_OPEN") if d.get("status")=="measured" else None
    layer=d.get("MAMBA_LAYER_B4_CEILING_OPEN") if d.get("status")=="measured" else None
    p18=d.get("PHASE18_FULL_BLOCK_VERIFIER_OPEN") if d.get("status")=="measured" else None

    if p18 is True:
        route="BUILD_PHASE18_FULL_PERFECT_INPUT_BLOCK_VERIFIER"
    elif core is True:
        route="OPTIMIZE_PROJECTION_BLOCKING_THEN_RETEST_FULL_LAYER"
    elif ssm is True:
        route="SCAN_KERNEL_WORKS_BUT_CORE_NOT_ENOUGH_OPTIMIZE_CONV_NORM_FUSION"
    elif complete:
        route="AFFINE_SCAN_ALGEBRA_GREEN_BUT_GPU_ECONOMICS_CLOSED"
    else:
        route="REPAIR_INCOMPLETE_PHASE17_EVIDENCE"

    out={
        "kind":"s100_phase17_summary",
        "created_utc":utc_now(),
        "instrumentation_complete":complete,
        "SSM_SCAN_MICROKERNEL_OPEN":ssm,
        "MAMBA_CORE_BLOCK_OPEN":core,
        "MAMBA_LAYER_B4_CEILING_OPEN":layer,
        "PHASE18_FULL_BLOCK_VERIFIER_OPEN":p18,
        "NEXT_ROUTE":route,
        "S100_SINGLE_ACHIEVED":False,
        "claim_boundary":"Phase17 Mamba-layer ceiling, not end-to-end decode",
    }
    R.mkdir(parents=True,exist_ok=True)
    write_json_atomic(R/"S100_PHASE17_SUMMARY.json",out,archive=True)
    text=(
        "S100 PHASE 17 — MAMBA BLOCK SCAN\n"
        f"Instrumentation complete: {complete}\n"
        f"SSM_SCAN_MICROKERNEL_OPEN: {ssm}\n"
        f"MAMBA_CORE_BLOCK_OPEN: {core}\n"
        f"MAMBA_LAYER_B4_CEILING_OPEN: {layer}\n"
        f"PHASE18_FULL_BLOCK_VERIFIER_OPEN: {p18}\n"
        f"NEXT_ROUTE: {route}\n"
        "S100 SINGLE ACHIEVED: False\n"
    )
    (R/"S100_PHASE17_SUMMARY.txt").write_text(text,encoding="utf-8")
    print(text);return 0
if __name__=="__main__":
    raise SystemExit(main())
