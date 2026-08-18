from __future__ import annotations

import json
from common import REPO, write_json_atomic, utc_now

R=REPO/"pro_research"/"results"/"s100_phase12c"

def load(name):
    try:
        return json.loads((R/name).read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    dense=load("S100_PHASE12C_DENSE.json") or {}
    grouped=load("S100_PHASE12C_GROUPED_MOE.json") or {}
    econ=load("S100_PHASE12C_ECONOMICS.json") or {}
    complete=(
        dense.get("status")=="measured"
        and grouped.get("status")=="measured"
    )
    out={
        "kind":"s100_phase12c_summary",
        "created_utc":utc_now(),
        "instrumentation_complete":complete,
        "dense_b4_gate_pass":bool(dense.get("dense_b4_gate_pass")),
        "grouped_b4_gate_pass":bool(grouped.get("grouped_b4_gate_pass")),
        "INTEGRATED_VERIFIER_BUILD_OPEN":bool(
            econ.get("INTEGRATED_VERIFIER_BUILD_OPEN")
        ),
        "projection":econ.get("projection"),
        "s100_single_achieved":False,
        "claim_boundary":"microkernel phase; no end-to-end S100 claim",
    }
    write_json_atomic(R/"S100_PHASE12C_SUMMARY.json",out,archive=True)
    text=(
        "S100 PHASE 12C — ERVF-M + GROUPED MOE\n"
        f"Instrumentation complete: {complete}\n"
        f"Dense B=4 gate: {out['dense_b4_gate_pass']}\n"
        f"Grouped B=4 gate: {out['grouped_b4_gate_pass']}\n"
        f"INTEGRATED VERIFIER BUILD OPEN: {out['INTEGRATED_VERIFIER_BUILD_OPEN']}\n"
        f"Projection: {out['projection']}\n"
        "S100 SINGLE ACHIEVED: False (microkernel phase)\n"
    )
    (R/"S100_PHASE12C_SUMMARY.txt").write_text(text,encoding="utf-8")
    print(text)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
