from __future__ import annotations

import json

from common import REPO, utc_now, write_json_atomic
from s100_phase26_common import RESULTS

OUT=RESULTS/"S100_PHASE26_SUMMARY.json"

def load(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return {}

def main():
    pre=load(RESULTS/"S100_PHASE26_PREFLIGHT.json")
    state=load(RESULTS/"S100_PHASE26_STATE_CHECK.json")
    screen=load(RESULTS/"S100_PHASE26_SCREEN.json")
    thermal=load(RESULTS/"S100_PHASE26_THERMAL_ADJUDICATION.json")

    adopted=bool(thermal.get("ADOPT_OVERLAP"))
    h=thermal.get("selected_horizon") if adopted else screen.get("selected_horizon")
    promoted=[]
    if adopted and h in (4,8):
        for ctx in (128,1024,4096):
            d=load(RESULTS/f"S100_PHASE26_PROMOTED_CTX{ctx}.json")
            promoted.append({
              "context":ctx,"status":d.get("status"),
              "correctness_green":d.get("correctness_green"),
              "summary":d.get("summary"),
              "ms_per_useful_token":d.get("ms_per_useful_token"),
              "target_only_tok_s":d.get("target_only_tok_s"),
              "telemetry":d.get("telemetry"),
            })

    promoted_green=bool(
      adopted and len(promoted)==3
      and all(x["status"]=="measured" and x["correctness_green"]
              for x in promoted)
    )
    mpt=[
      float(x["ms_per_useful_token"]) for x in promoted
      if x.get("ms_per_useful_token") is not None
    ]
    target100=bool(promoted_green and len(mpt)==3 and max(mpt)<=10.0)
    drafter=bool(promoted_green and len(mpt)==3 and max(mpt)<=8.0)

    if drafter:
        route="OPEN_DSPARK_MTP_DFLASH_SHOOTOUT"
    elif target100:
        route="TARGET_100_CEILING_OPEN_REDUCE_DRAFTER_HEADROOM"
    elif adopted:
        route="PROFILE_ADOPTED_OVERLAP_AND_ATTACK_REMAINING_STAGE"
    else:
        route="BUILD_DOWN_GATHER_TRANSFER_COMPUTE_PIPELINE"

    complete=bool(
      pre.get("PREFLIGHT_GREEN")
      and state.get("ALL_OVERLAP_STATE_GREEN")
      and screen
      and thermal
      and (not adopted or promoted_green)
    )

    out={
      "kind":"s100_phase26_summary","created_utc":utc_now(),
      "instrumentation_complete":complete,
      "PREFLIGHT_GREEN":bool(pre.get("PREFLIGHT_GREEN")),
      "H4_OVERLAP_STATE_GREEN":bool(state.get("H4_OVERLAP_STATE_GREEN")),
      "H8_OVERLAP_STATE_GREEN":bool(state.get("H8_OVERLAP_STATE_GREEN")),
      "screen":screen,
      "thermal_adjudication":thermal,
      "selected_horizon":h,
      "PHASE26_ACTIVE_PARENT_ADOPTED":adopted,
      "promoted_contexts":promoted,
      "TARGET_100_TARGET_ONLY_OPEN":target100,
      "DRAFTER_SHOOTOUT_OPEN":drafter,
      "NEXT_ROUTE":route,
      "S100_SINGLE_ACHIEVED":False,
      "claim_boundary":"exact target-only verifier; no drafter cost",
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,out,archive=True)

    text=(
      "S100 PHASE 26 — SHARED/ROUTED OVERLAP\n"
      f"Instrumentation complete: {complete}\n"
      f"H4 state green: {out['H4_OVERLAP_STATE_GREEN']}\n"
      f"H8 state green: {out['H8_OVERLAP_STATE_GREEN']}\n"
      f"Selected horizon: {h}\n"
      f"Thermally adopted: {adopted}\n"
      f"TARGET_100_TARGET_ONLY_OPEN: {target100}\n"
      f"DRAFTER_SHOOTOUT_OPEN: {drafter}\n"
      f"NEXT_ROUTE: {route}\n"
      "S100 SINGLE ACHIEVED: False\n"
    )
    (RESULTS/"S100_PHASE26_SUMMARY.txt").write_text(text,encoding="utf-8")

    report=REPO/"reports"/"S100_PHASE26_RUN_REPORT.md"
    report.parent.mkdir(parents=True,exist_ok=True)
    lines=[
      "# S100 Phase 26 — Shared/Routed MoE Concurrency",
      "",
      "Phase26 changes scheduling only: the shared expert branch is forked "
      "to a side CUDA stream while the routed branch executes on the main "
      "stream. The original shared-then-slot-order FMA accumulation is "
      "preserved after the join.",
      "",
      f"- Synthetic cross-stream graph preflight: `{out['PREFLIGHT_GREEN']}`",
      f"- H4 full-state parity: `{out['H4_OVERLAP_STATE_GREEN']}`",
      f"- H8 full-state parity: `{out['H8_OVERLAP_STATE_GREEN']}`",
      f"- Selected horizon: `{h}`",
      f"- Thermally adopted: `{adopted}`",
    ]
    if adopted:
        lines += ["", "## Promoted contexts", "",
                  "| Context | ms/useful token | target-only tok/s |",
                  "|---:|---:|---:|"]
        for x in promoted:
            lines.append(
              f"| {x['context']} | "
              f"{x.get('ms_per_useful_token')} | "
              f"{x.get('target_only_tok_s')} |"
            )
    lines += [
      "",
      f"- Target-only 100 tok/s gate: `{target100}`",
      f"- Drafter shootout open: `{drafter}`",
      f"- Next route: `{route}`",
      "- S100 single achieved: `False`",
      "",
    ]
    report.write_text("\n".join(lines),encoding="utf-8")
    print(text)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
