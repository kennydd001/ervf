from __future__ import annotations

import json

from common import REPO,utc_now,write_json_atomic
from s100_phase27_common import RESULTS


def load(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return {}


def main():
    pre=load(RESULTS/"S100_PHASE27_PREFLIGHT.json")
    geom=load(RESULTS/"S100_PHASE27_GEOMETRY_SELECTION.json")
    pipe=load(RESULTS/"S100_PHASE27_PIPELINE_SELECTION.json")
    sel=load(RESULTS/"S100_PHASE27_SELECTION.json")
    state=load(RESULTS/"S100_PHASE27_STATE_CHECK.json")
    screen=load(RESULTS/"S100_PHASE27_FINAL_SCREEN.json")
    thermal=load(RESULTS/"S100_PHASE27_THERMAL_ADJUDICATION.json")

    adopted=bool(thermal.get("ADOPT_PHASE27"))
    variant=sel.get("selected_variant")

    promoted=[]
    if adopted and variant:
        for ctx in (128,1024,4096):
            d=load(RESULTS/f"S100_PHASE27_PROMOTED_CTX{ctx}.json")
            promoted.append({
              "context":ctx,
              "status":d.get("status"),
              "correctness_green":d.get("correctness_green"),
              "summary":d.get("summary"),
              "ms_per_useful_token":d.get("ms_per_useful_token"),
              "target_only_tok_s":d.get("target_only_tok_s"),
              "telemetry":d.get("telemetry"),
            })

    promoted_green=bool(
      adopted and len(promoted)==3
      and all(
        x["status"]=="measured" and x["correctness_green"]
        for x in promoted
      )
    )
    mpt=[
      float(x["ms_per_useful_token"])
      for x in promoted
      if x.get("ms_per_useful_token") is not None
    ]
    target100=bool(
      promoted_green and len(mpt)==3 and max(mpt)<=10.0
    )
    drafter=bool(
      promoted_green and len(mpt)==3 and max(mpt)<=8.0
    )

    screen_gain=screen.get("gain_fraction")
    if drafter:
        route="OPEN_DSPARK_MTP_DFLASH_SHOOTOUT"
    elif target100:
        route="REDUCE_TARGET_VERIFIER_BELOW_8MS_FOR_DRAFTER_HEADROOM"
    elif adopted:
        route="PROFILE_ADOPTED_DOWN_PIPELINE_AND_ATTACK_REMAINING_CRITICAL_STAGE"
    elif screen_gain is not None and float(screen_gain)>0:
        route="FUSE_GATHER_DOWN_AND_ELIMINATE_MIRROR_TRAFFIC"
    else:
        route="DIRECT_SPARSE_TRANSFER_ENGINE_OR_ZERO_COPY_GROUPED_DOWN"

    complete=bool(
      pre.get("PREFLIGHT_GREEN")
      and geom.get("GEOMETRY_SELECTION_GREEN")
      and pipe.get("PIPELINE_SELECTION_GREEN")
      and sel.get("COMBINATION_SELECTION_GREEN")
      and state.get("SELECTED_STATE_GREEN")
      and screen
      and thermal
      and (not adopted or promoted_green)
    )

    out={
      "kind":"s100_phase27_summary",
      "created_utc":utc_now(),
      "instrumentation_complete":complete,
      "PREFLIGHT_GREEN":bool(pre.get("PREFLIGHT_GREEN")),
      "geometry_selection":geom,
      "pipeline_selection":pipe,
      "combination_selection":sel,
      "SELECTED_STATE_GREEN":bool(state.get("SELECTED_STATE_GREEN")),
      "final_screen":screen,
      "thermal_adjudication":thermal,
      "selected_variant":variant,
      "PHASE27_ACTIVE_PARENT_ADOPTED":adopted,
      "promoted_contexts":promoted,
      "TARGET_100_TARGET_ONLY_OPEN":target100,
      "DRAFTER_SHOOTOUT_OPEN":drafter,
      "NEXT_ROUTE":route,
      "S100_SINGLE_ACHIEVED":False,
      "claim_boundary":"exact H4 target-only verifier; no drafter cost",
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(RESULTS/"S100_PHASE27_SUMMARY.json",out,archive=True)

    txt=(
      "S100 PHASE 27 — DOWN-GATHER TRANSFER/COMPUTE PIPELINE\n"
      f"Instrumentation complete: {complete}\n"
      f"Preflight green: {out['PREFLIGHT_GREEN']}\n"
      f"Selected variant: {variant}\n"
      f"State green: {out['SELECTED_STATE_GREEN']}\n"
      f"Final screen gain: {screen_gain}\n"
      f"Thermally adopted: {adopted}\n"
      f"TARGET_100_TARGET_ONLY_OPEN: {target100}\n"
      f"DRAFTER_SHOOTOUT_OPEN: {drafter}\n"
      f"NEXT_ROUTE: {route}\n"
      "S100 SINGLE ACHIEVED: False\n"
    )
    (RESULTS/"S100_PHASE27_SUMMARY.txt").write_text(txt,encoding="utf-8")

    report=REPO/"reports"/"S100_PHASE27_RUN_REPORT.md"
    report.parent.mkdir(parents=True,exist_ok=True)
    lines=[
      "# S100 Phase 27 — Down-Gather Transfer/Compute Pipeline",
      "",
      "Phase27 keeps the Phase24 exact H4 parent frozen and changes only "
      "down-gather scheduling/launch geometry. Range-down writes the same "
      "eight chunk partials per route; the existing reduction and route-slot "
      "accumulation remain unchanged.",
      "",
      f"- Preflight green: `{out['PREFLIGHT_GREEN']}`",
      f"- Selected gather_y: `{geom.get('selected_gather_y')}`",
      f"- Selected pipeline batches: `{pipe.get('selected_batches')}`",
      f"- Selected combined variant: `{variant}`",
      f"- Full-state gate: `{out['SELECTED_STATE_GREEN']}`",
      f"- Final screen gain: `{screen_gain}`",
      f"- Thermal adoption: `{adopted}`",
    ]
    if adopted:
        lines += [
          "",
          "## Promoted contexts",
          "",
          "| Context | ms/useful token | target-only tok/s |",
          "|---:|---:|---:|",
        ]
        for x in promoted:
            lines.append(
              f"| {x['context']} | {x.get('ms_per_useful_token')} | "
              f"{x.get('target_only_tok_s')} |"
            )
    lines += [
      "",
      f"- TARGET_100_TARGET_ONLY_OPEN: `{target100}`",
      f"- DRAFTER_SHOOTOUT_OPEN: `{drafter}`",
      f"- NEXT_ROUTE: `{route}`",
      "- S100 SINGLE ACHIEVED: `False`",
      "",
    ]
    report.write_text("\n".join(lines),encoding="utf-8")
    print(txt)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
