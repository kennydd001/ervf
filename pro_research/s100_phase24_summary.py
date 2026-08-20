from __future__ import annotations

import json

from common import REPO,utc_now,write_json_atomic
from s100_phase24_common import RESULTS

OUT=RESULTS/"S100_PHASE24_SUMMARY.json"
V18_TOKEN_MS=19.5729
V18_H4_EQUIV=4.0*V18_TOKEN_MS

def load(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return {}

def phase24_promoted():
    rows=[]
    for ctx in (128,1024,4096):
        d=load(RESULTS/f"S100_PHASE24_PROMOTED_CTX{ctx}.json")
        rows.append({
          "context":ctx,"status":d.get("status"),
          "correctness_green":d.get("correctness_green"),
          "summary":d.get("summary"),"config":d.get("config"),
          "actual_plane_bytes":d.get("actual_plane_bytes"),
        })
    return rows

def phase23r_promoted():
    d=load(
      REPO/"pro_research"/"results"/"s100_phase23r"/
      "S100_PHASE23R_SUMMARY.json"
    )
    return d.get("promoted_contexts") or []

def ctx_median(rows,ctx):
    for row in rows:
        if int(row.get("context",-1))==int(ctx):
            try:return float(row["summary"]["median_ms"])
            except Exception:return None
    return None

def main():
    prof=load(RESULTS/"S100_PHASE24_PROFILE.json")
    comp=load(RESULTS/"S100_PHASE24_COMPONENTS.json")
    sel=load(RESULTS/"S100_PHASE24_SELECTION.json")
    state=load(RESULTS/"S100_PHASE24_STATE_CHECK.json")
    therm=load(RESULTS/"S100_PHASE24_THERMAL_ADJUDICATION.json")
    gen=load(RESULTS/"S100_PHASE24_GENERALIZATION.json")

    selected=sel.get("selected")
    selected_label=sel.get("selected_label")
    selected_is_new=bool(selected and selected_label!="baseline")
    adopted=bool(
      selected_is_new and therm.get("BEST_OF_ALL_ADOPTED")
    )
    new_promoted=phase24_promoted() if adopted else []
    old_promoted=phase23r_promoted()
    active_promoted=new_promoted if adopted else old_promoted

    correct=bool(
      state.get("BEST_OF_ALL_STATE_GREEN")
      if selected_is_new else True
    )
    p128=ctx_median(active_promoted,128)
    p1024=ctx_median(active_promoted,1024)
    p4096=ctx_median(active_promoted,4096)
    medians=[x for x in (p128,p1024,p4096) if x is not None]
    target40=bool(len(medians)==3 and max(medians)<=40.0 and correct)
    draft32=bool(len(medians)==3 and max(medians)<=32.0 and correct)
    beats_v18=bool(
      p1024 is not None and p1024<V18_H4_EQUIV and correct
    )

    gs=gen.get("summary") or {}
    h8_ratio=gs.get("h8_over_two_h4_median")
    h8_open=bool(
      not target40
      and correct
      and gen.get("PROMPT_ROUTE_GENERALIZATION_GREEN")
      and h8_ratio is not None
      and float(h8_ratio)<=0.925
      and gs.get("route_multiplicity_valid")
    )

    if draft32:
        route="OPEN_DSPARK_MTP_DFLASH_SHOOTOUT"
    elif target40:
        route="TARGET_100_CEILING_OPEN_OPTIMIZE_DRAFTER_HEADROOM"
    elif h8_open:
        route="BUILD_H8_BEST_OF_ALL_FULL_VERIFIER"
    elif adopted:
        route="PROFILE_ADOPTED_SYNTHESIS_AND_ATTACK_NEXT_DOMINANT_STAGE"
    elif selected_is_new:
        route="KEEP_PHASE23R_PARENT_COMPONENT_GAINS_NOT_5PCT_END_TO_END"
    else:
        route="KEEP_PHASE23R_PARENT_BUILD_H8_OR_NEXT_DOMINANT_COMPONENT"

    remaining_factor=(
      p1024/40.0 if p1024 is not None else None
    )
    best_tok_s=(
      4000.0/p1024 if p1024 is not None else None
    )

    complete=bool(
      prof.get("status")=="measured"
      and comp.get("status")=="measured"
      and sel
      and gen.get("status")=="measured"
      and (
        not selected_is_new
        or (
          state.get("status")=="measured"
          and therm
          and (not adopted or len(new_promoted)==3)
        )
      )
    )

    out={
      "kind":"s100_phase24_summary","created_utc":utc_now(),
      "instrumentation_complete":complete,
      "component_flags":{
        "ATTENTION_BF16_M4_OPEN":comp.get("ATTENTION_BF16_M4_OPEN"),
        "ROUTER_F32_M4_OPEN":comp.get("ROUTER_F32_M4_OPEN"),
        "SHARED_NVFP4_M4_OPEN":comp.get("SHARED_NVFP4_M4_OPEN"),
      },
      "component_speedups":{
        "attention":(comp.get("attention_bf16") or {}).get("aggregate_speedup"),
        "router":(comp.get("router_f32") or {}).get("aggregate_speedup"),
        "shared":(comp.get("shared_nvfp4") or {}).get("aggregate_speedup"),
      },
      "selection":sel,
      "selected_state_green":correct,
      "thermal_adjudication":therm,
      "BEST_OF_ALL_ADOPTED":adopted,
      "active_parent":(
        "phase24_best_of_all" if adopted else "phase23r_gpu_grouped"
      ),
      "active_contexts":active_promoted,
      "context_medians_ms":{
        "128":p128,"1024":p1024,"4096":p4096,
      },
      "best_target_only_tok_s_at_1024":best_tok_s,
      "v18_single_token_ms":V18_TOKEN_MS,
      "v18_four_token_equivalent_ms":V18_H4_EQUIV,
      "H4_BEATS_V18_PARENT":beats_v18,
      "remaining_factor_to_40ms_at_1024":remaining_factor,
      "route_generalization":gen,
      "PHASE24_TARGET_40MS_OPEN":target40,
      "DRAFTER_SHOOTOUT_OPEN":draft32,
      "PHASE25_H8_BUILD_OPEN":h8_open,
      "NEXT_ROUTE":route,
      "S100_SINGLE_ACHIEVED":False,
      "claim_boundary":"best-of-all target-only H4; no drafter",
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,out,archive=True)

    text=(
      "S100 PHASE 24 — BEST OF ALL\n"
      f"Instrumentation complete: {complete}\n"
      f"Selected screen arm: {selected_label}\n"
      f"BEST_OF_ALL_ADOPTED: {adopted}\n"
      f"Active parent: {out['active_parent']}\n"
      f"Context 1024 H4 ms: {p1024}\n"
      f"Context 1024 target-only tok/s: {best_tok_s}\n"
      f"H4_BEATS_V18_PARENT: {beats_v18}\n"
      f"Remaining factor to 40 ms: {remaining_factor}\n"
      f"Prompt route generalization green: "
      f"{gen.get('PROMPT_ROUTE_GENERALIZATION_GREEN')}\n"
      f"PHASE24_TARGET_40MS_OPEN: {target40}\n"
      f"DRAFTER_SHOOTOUT_OPEN: {draft32}\n"
      f"PHASE25_H8_BUILD_OPEN: {h8_open}\n"
      f"NEXT_ROUTE: {route}\n"
      "S100 SINGLE ACHIEVED: False\n"
    )
    (RESULTS/"S100_PHASE24_SUMMARY.txt").write_text(
      text,encoding="utf-8"
    )

    # Compact human report.
    report=REPO/"reports"/"S100_PHASE24_RUN_REPORT.md"
    report.parent.mkdir(parents=True,exist_ok=True)
    arms=sel.get("arms") or []
    arm_table=[
      "| Arm | Status | H4 ms @1024 | tok/s | Plane MiB |",
      "|---|---:|---:|---:|---:|",
    ]
    for a in arms:
        plane=(a.get("actual_plane_bytes") or 0)/2**20
        arm_table.append(
          f"| {a.get('label')} | {a.get('status')} | "
          f"{a.get('median_ms')} | {a.get('tok_s')} | {plane:.1f} |"
        )
    prompt_table=[
      "| Prompt | H4 repeat | H8 / two-H4 streams |",
      "|---|---:|---:|",
    ]
    for r in gen.get("records",[]) or []:
        rep=(
          r["h4_first"]["median_repeat_rate"]
          +r["h4_second"]["median_repeat_rate"]
        )/2.0
        prompt_table.append(
          f"| {r['label']} | {rep:.3f} | "
          f"{r['h8_over_two_h4_streams']:.3f} |"
        )
    md=(
      "# S100 Phase 24 — Best-of-All Lightning Synthesis\n\n"
      "## Component screens\n\n"
      f"- Attention BF16 M4: `{out['component_flags']['ATTENTION_BF16_M4_OPEN']}`, "
      f"speedup `{out['component_speedups']['attention']}`\n"
      f"- Router FP32 M4: `{out['component_flags']['ROUTER_F32_M4_OPEN']}`, "
      f"speedup `{out['component_speedups']['router']}`\n"
      f"- Shared NVFP4 M4: `{out['component_flags']['SHARED_NVFP4_M4_OPEN']}`, "
      f"speedup `{out['component_speedups']['shared']}`\n\n"
      "## Scale-resident synthesis screen\n\n"
      +"\n".join(arm_table)+"\n\n"
      "## Adoption and active parent\n\n"
      f"- Selected: `{selected_label}`\n"
      f"- Thermally adopted: `{adopted}`\n"
      f"- Active parent: `{out['active_parent']}`\n"
      f"- Context1024 H4: `{p1024}` ms\n"
      f"- Target-only: `{best_tok_s}` tok/s\n"
      f"- Beats V18 four-token equivalent `{V18_H4_EQUIV:.4f}` ms: `{beats_v18}`\n"
      f"- Remaining factor to 40 ms: `{remaining_factor}`\n\n"
      "## Prompt/H8 route generalization\n\n"
      +"\n".join(prompt_table)+"\n\n"
      f"- Generalization green: `{gen.get('PROMPT_ROUTE_GENERALIZATION_GREEN')}`\n"
      f"- H8 build open: `{h8_open}`\n\n"
      "## Final\n\n"
      f"- Target <=40 ms/H4: `{target40}`\n"
      f"- Drafter shootout open: `{draft32}`\n"
      f"- Next route: `{route}`\n"
      "- S100 single achieved: `False`\n"
    )
    report.write_text(md,encoding="utf-8")
    print(text)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
