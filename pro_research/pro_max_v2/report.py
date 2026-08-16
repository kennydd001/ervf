from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "pro_max_v2"
OUT = RESULTS / "PV2_FINAL_REPORT.md"


def read(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, n=4):
    return "—" if x is None else f"{float(x):.{n}f}"


def main():
    prov = read("PV2_00_PROVENANCE.json")
    candidates = [("Add+RMSNorm", read("PV2_10_ADDNORM.json")),
                  ("Q/K/V one-launch", read("PV2_11_QKV.json")),
                  ("LM-head+argmax", read("PV2_12_LMHEAD_ARGMAX.json"))]
    final = read("PV2_13_FINALE.json")
    epoch = read("PV2_20_CHILD_EPOCH.json")
    verify = read("PV2_VERIFICATION.json")
    lines = ["# PRO-MAX V2 eindrapport", "",
             "## Uitgangspunt", ""]
    if prov:
        v = prov.get("v6", {})
        lines += [f"V6: **{fmt(v.get('p50_ms'))} ms/token = {fmt(v.get('tok_s'), 2)} tok/s**.", ""]
    else:
        lines += ["De provenance-run ontbreekt.", ""]
    lines += ["## Exacte final-mile kandidaten", "",
              "| kandidaat | status | adopt | p50 ms | tok/s | winst ms |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, d in candidates:
        if not d:
            lines.append(f"| {label} | missing | — | — | — | — |")
            continue
        s = d.get("summary", {})
        lines.append(f"| {label} | {d.get('status')} | {d.get('adopt')} | "
                     f"{fmt(s.get('candidate_p50_ms'))} | {fmt(s.get('candidate_tok_s'),2)} | "
                     f"{fmt(s.get('gain_ms'))} |")
    lines += ["", "## Gecombineerde V10-run", ""]
    if final:
        s = final.get("summary", {})
        lines += [f"Status: **{final.get('status')}**.",
                  f"P50: **{fmt(s.get('v10_p50_ms'))} ms = {fmt(s.get('v10_tok_s'),2)} tok/s**.",
                  f"Nog tot 50 tok/s: **{fmt(s.get('remaining_to_50_ms'))} ms/token**.",
                  f"Milestones: `{json.dumps(final.get('milestones', {}), sort_keys=True)}`.", ""]
    else:
        lines += ["Nog niet uitgevoerd.", ""]
    lines += ["## Exacte child-graph epochs", ""]
    if epoch:
        lines.append(f"Status: **{epoch.get('status')}**; beste: `{json.dumps(epoch.get('best'))}`.")
    else:
        lines.append("Nog niet uitgevoerd.")
    lines += ["", "## Onafhankelijke verificatie", ""]
    lines.append("Nog niet uitgevoerd." if not verify else f"Verdict: **{verify.get('verdict')}**.")
    lines += ["", "## Interpretatie", "",
              "Een componentmicrobenchmark is geen tok/s-doorbraak. Alleen de gecombineerde "
              "causale V10-run kan E50 openen. Child-graph epochs zijn queued/offline "
              "throughput en veranderen de latency van het eerste token niet. 100 tok/s "
              "single-stream blijft een afzonderlijke, veel zwaardere eis; aggregate batch>1 "
              "blijft de voornaamste post-E50 architecturale route."]
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
