from __future__ import annotations
import json
from common import REPO
from s100_phase25_common import RESULTS,OFFICIAL_PARENT_H8_MS,ADOPTION_ABS_MS

def f(x,n=2):
    try:return f"{float(x):.{n}f}"
    except Exception:return "n/a"
def main():
    s=json.loads((RESULTS/"S100_PHASE25_SUMMARY.json").read_text(encoding="utf-8"));sel=s.get("screen_selected") or {};th=s.get("thermal") or {};g=s.get("gates") or {}
    tms=th.get("selected_median_of_rounds_ms");tok=None if tms is None else 8000.0/float(tms)
    lines=["# S100 Phase 25 — H8 Best-of-All Full Verifier","","## Frozen parent","",
      f"- Phase24 H4 + H4 official baseline: `{OFFICIAL_PARENT_H8_MS:.4f} ms / 8 tokens`.",
      f"- Absolute adoption gate: `<= {ADOPTION_ABS_MS:.2f} ms/H8`.","",
      "## Selection","",f"- Selected variant: `{s.get('selected_variant')}`.",
      f"- Screen latency: `{f(sel.get('median_ms'),4)} ms/H8`.",f"- Screen target-only: `{f(sel.get('tok_s'),2)} tok/s`.",
      f"- Full state green: `{s.get('state_green')}`.","","## Thermal adoption","",
      f"- H8 adopted: `{g.get('H8_ADOPTED')}`.",f"- Thermal median-of-rounds: `{f(tms,4)} ms/H8`.",
      f"- Thermal target-only: `{f(tok,2)} tok/s`.",f"- S100 target-only <=80 ms: `{g.get('S100_TARGET_ONLY_LE_80MS')}`.","",
      "## State parity","",f"- `{json.dumps(s.get('state'),sort_keys=True)}`","","## Profile","",
      f"- Stage totals: `{json.dumps((s.get('profile_summary') or {}).get('stage_totals_ms_per_h8'),sort_keys=True)}`",
      f"- Expert streams: `{json.dumps((s.get('profile_summary') or {}).get('weight_streams_per_h8'),sort_keys=True)}`","",
      "## Final","",f"- H8 active parent: `{s.get('H8_ACTIVE_PARENT')}`.",
      f"- S100 target-only achieved: `{s.get('S100_TARGET_ONLY_ACHIEVED')}`.",
      f"- S100 single achieved: `{s.get('S100_SINGLE_ACHIEVED')}`.",f"- Next route: `{s.get('NEXT_ROUTE')}`.","",
      "> Claim boundary: target-verifier timing is not true end-to-end single-stream throughput until drafter, rejection, and fallback costs are included."]
    out=REPO/"reports"/"S100_PHASE25_RUN_REPORT.md";out.parent.mkdir(parents=True,exist_ok=True);out.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(out);return 0
if __name__=="__main__":raise SystemExit(main())
