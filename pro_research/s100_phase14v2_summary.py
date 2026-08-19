from __future__ import annotations
import json
from common import REPO, write_json_atomic, utc_now

R = REPO / "pro_research" / "results"

def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    d = load(R/"s100_phase14d2"/"S100_PHASE14D2_SUMMARY.json") or {}
    n = load(R/"s100_phase14n2"/"S100_PHASE14N2_SUMMARY.json") or {}
    k = load(R/"s100_phase14k2"/"S100_PHASE14K2_REAL_WITNESS.json") or {}

    b1 = d.get("NATIVE_BF16_B1_DIRECT_OPEN")
    b4 = d.get("NATIVE_BF16_BLOCK_BUILD_OPEN")
    fp4 = n.get("NATIVE_NVFP4_C3_RUNTIME_BUILD_OPEN")
    witness = k.get("REAL_K16_SHORTLIST_GREEN")
    margin = k.get("REAL_MARGIN_GATE_GREEN")

    if b1 is True:
        next_route = "BUILD_GRAPH_INTEGRATED_NATIVE_BF16_B1_FIRST"
    elif b4 is True and fp4 is True:
        next_route = "BUILD_FULL_NATIVE_BF16_NVFP4_BLOCK_VERIFIER"
    elif b4 is True:
        next_route = "BUILD_NATIVE_BF16_BLOCK_VERIFIER_KEEP_NVFP4_ERVF"
    elif fp4 is True:
        next_route = "BUILD_REAL_WEIGHT_NATIVE_NVFP4_C3_SINGLE_TOKEN"
    elif any(x is None for x in (b1, b4, fp4)):
        next_route = "REPAIR_INCOMPLETE_EVIDENCE"
    else:
        next_route = "NATIVE_TC_ROUTE_NOT_YET_OPEN_REPROFILE_PARENT"

    out = {
        "kind": "s100_phase14v2_summary",
        "created_utc": utc_now(),
        "NATIVE_BF16_B1_DIRECT_OPEN": b1,
        "NATIVE_BF16_BLOCK_BUILD_OPEN": b4,
        "NATIVE_NVFP4_C3_RUNTIME_BUILD_OPEN": fp4,
        "REAL_K16_SHORTLIST_GREEN": witness,
        "REAL_MARGIN_GATE_GREEN": margin,
        "NEXT_ROUTE": next_route,
        "s100_single_achieved": False,
        "claim_boundary": "combined research adjudication; no S100 claim",
    }
    outdir = R/"s100_phase14v2"
    outdir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(outdir/"S100_PHASE14V2_SUMMARY.json", out, archive=True)
    text = (
        "S100 PHASE 14 V2 — COMBINED\n"
        f"NATIVE_BF16_B1_DIRECT_OPEN: {b1}\n"
        f"NATIVE_BF16_BLOCK_BUILD_OPEN: {b4}\n"
        f"NATIVE_NVFP4_C3_RUNTIME_BUILD_OPEN: {fp4}\n"
        f"REAL_K16_SHORTLIST_GREEN: {witness}\n"
        f"REAL_MARGIN_GATE_GREEN: {margin}\n"
        f"NEXT_ROUTE: {next_route}\n"
        "S100 SINGLE ACHIEVED: False\n"
    )
    (outdir/"S100_PHASE14V2_SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
