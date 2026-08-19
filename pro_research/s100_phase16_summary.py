from __future__ import annotations
import json
from common import write_json_atomic,utc_now
from s100_phase16_common import RESULTS

def load(name):
    try:return json.loads((RESULTS/name).read_text(encoding="utf-8"))
    except Exception:return {}

def main():
    a=load("S100_PHASE16A_LOCAL_SENSITIVITY.json")
    b=load("S100_PHASE16B_SUBSET_VALIDATION.json")
    c=load("S100_PHASE16C_EXACT_STATE_HORIZON.json")
    d=load("S100_PHASE16D_SELECTED_SAVINGS.json")
    e=load("S100_PHASE16E_MAMBA_AFFINE_SCAN.json")
    subset=bool(b.get("selected_strict_subset"))
    horizon=bool(c.get("ANY_H4_BLOCK_RESEARCH_GO"))
    scan=bool(e.get("MAMBA_AFFINE_SCAN_BUILD_OPEN"))
    if subset:route="BUILD_GRAPH_INTEGRATED_SAFE_BF16_SUBSET"
    elif horizon and scan:route="BUILD_PARALLEL_MAMBA_SCAN_BLOCK_VERIFIER"
    elif horizon:route="BUILD_SHORT_HORIZON_DRAFT_PROTOTYPE"
    elif scan:route="BUILD_MAMBA_SCAN_MICROKERNEL_THEN_RETEST_BLOCK"
    elif int(a.get("safe_matrix_count") or 0)>0:route="TEST_NONCUMULATIVE_SAFE_MATRIX_SCHEDULING"
    else:route="CLOSE_NATIVE_BF16_SUBSTITUTION_KEEP_SCAN_RESEARCH"
    complete=all(x.get("status")=="measured" for x in (a,b,c,e))
    out={"kind":"s100_phase16_summary","created_utc":utc_now(),
      "instrumentation_complete":complete,"safe_matrix_count":a.get("safe_matrix_count"),
      "selected_strict_subset":b.get("selected_strict_subset"),
      "selected_B1_component_saving_ms":d.get("phase14_B1_component_saving_ms"),
      "ANY_H4_BLOCK_RESEARCH_GO":horizon,
      "MAMBA_AFFINE_SCAN_BUILD_OPEN":scan,
      "SAFE_BF16_SUBSET_RUNTIME_BUILD_OPEN":subset,
      "PARALLEL_BLOCK_VERIFIER_RESEARCH_OPEN":bool(horizon or scan),
      "NEXT_ROUTE":route,"S100_SINGLE_ACHIEVED":False,
      "claim_boundary":"Phase16 research authorization only"}
    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(RESULTS/"S100_PHASE16_SUMMARY.json",out,archive=True)
    text=("S100 PHASE 16 — LOCALIZE / HORIZON / SCAN\n"
      f"Instrumentation complete: {complete}\nSafe matrix count: {out['safe_matrix_count']}\n"
      f"Selected strict subset: {out['selected_strict_subset']}\n"
      f"Selected B1 component saving ms: {out['selected_B1_component_saving_ms']}\n"
      f"ANY_H4_BLOCK_RESEARCH_GO: {horizon}\nMAMBA_AFFINE_SCAN_BUILD_OPEN: {scan}\n"
      f"SAFE_BF16_SUBSET_RUNTIME_BUILD_OPEN: {subset}\n"
      f"PARALLEL_BLOCK_VERIFIER_RESEARCH_OPEN: {out['PARALLEL_BLOCK_VERIFIER_RESEARCH_OPEN']}\n"
      f"NEXT_ROUTE: {route}\nS100 SINGLE ACHIEVED: False\n")
    (RESULTS/"S100_PHASE16_SUMMARY.txt").write_text(text,encoding="utf-8")
    print(text);return 0
if __name__=="__main__":raise SystemExit(main())
