"""Build a concise Dutch status report from PRO result JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common import load_json, result_path, utc_now, write_text_atomic

OUT = result_path("PRO_FINAL_REPORT.md")


def maybe(name: str) -> dict[str, Any] | None:
    path = result_path(name)
    return load_json(path) if path.exists() else None


def f(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def status_icon(status: str | None) -> str:
    if status in {"pass", "verified_pass", "micro_pass"}:
        return "PASS"
    if status in {"technical_failure", "technical_blocked", "verification_error"}:
        return "TECHNISCH GEBLOKKEERD"
    if status is None:
        return "NIET GEDRAAID"
    return "NEGATIEF / POORT GEMIST"


def main() -> int:
    graph = maybe("PRO_G0_E1F22_GRAPH_AB.json")
    dense = maybe("PRO_G1_DENSE_ERVF.json")
    epoch = maybe("PRO_G2_EPOCH_GRAPH.json")
    verify = maybe("PRO_VERIFICATION.json")

    lines = [
        "# PRO research — automatisch eindrapport",
        "",
        f"Gegenereerd: `{utc_now()}`",
        "",
        "## Oordeel in gewone taal",
        "",
    ]

    verified = verify.get("verified_candidates", []) if verify else []
    if verified:
        lines.append(
            "Er is minstens één kandidaat die de onafhankelijke verifier heeft gehaald: "
            + ", ".join(f"`{x}`" for x in verified)
            + ". Dit is nog alleen een doorbraakclaim binnen de vooraf beschreven meetgrens."
        )
    elif any(x is not None for x in (graph, dense, epoch)):
        lines.append(
            "De experimenten leverden nog geen onafhankelijk geverifieerde doorbraakkandidaat op. "
            "De negatieve of technische uitkomst is wel bruikbaar: ze sluit een concreet mechanisme."
        )
    else:
        lines.append("Er zijn nog geen GPU-experimenten uitgevoerd. Start met `INSTALL_AND_RUN.ps1 -Mode smoke`.")

    lines += ["", "## Resultaten", "", "| spoor | status | kerngetal |", "|---|---|---|"]

    if graph:
        e = graph.get("arms", {}).get("EGR", {}).get("timing_ms", {}).get("p50")
        g = graph.get("arms", {}).get("GRAPH", {}).get("timing_ms", {}).get("p50")
        gain = None if e is None or g is None else float(e) - float(g)
        lines.append(f"| G0 full-token graph | {status_icon(graph.get('status'))} | p50 {f(e)} -> {f(g)} ms; winst {f(gain)} ms |")
    else:
        lines.append("| G0 full-token graph | NIET GEDRAAID | — |")

    if dense:
        gm = dense.get("microbench", {}).get("geometric_mean_speedup")
        integ = dense.get("integration", {})
        gain = integ.get("gain_ms")
        pro_ts = integ.get("tok_s", {}).get("pro") if integ else None
        lines.append(f"| G1 ERVF voor BF16/FP8/FP32 | {status_icon(dense.get('status'))} | micro {f(gm)}x; geïntegreerd {f(gain)} ms; {f(pro_ts)} tok/s |")
    else:
        lines.append("| G1 ERVF voor BF16/FP8/FP32 | NIET GEDRAAID | — |")

    if epoch:
        best = epoch.get("best") or {}
        lines.append(f"| G2 K-token epoch graph | {status_icon(epoch.get('status'))} | K={best.get('k', '—')}; {f(best.get('speedup'))}x; {f(best.get('tok_s'))} tok/s |")
    else:
        lines.append("| G2 K-token epoch graph | NIET GEDRAAID | — |")

    lines += ["", "## Detail per spoor", ""]

    lines.append("### G0 — E1F22 graph")
    if not graph:
        lines.append("Niet uitgevoerd.")
    elif graph.get("status") == "technical_failure":
        lines.append(f"Technische fout: `{graph.get('error', {}).get('type')}: {graph.get('error', {}).get('message')}`")
    else:
        for name, gate in graph.get("gates", {}).items():
            lines.append(f"- `{name}`: **{gate.get('passed')}**")
        summary = graph.get("summary", {})
        lines.append(f"- Eager: {f(summary.get('eager_tok_s_from_p50'))} tok/s op basis van p50.")
        lines.append(f"- Graph: {f(summary.get('graph_tok_s_from_p50'))} tok/s op basis van p50.")

    lines += ["", "### G1 — generalized ERVF"]
    if not dense:
        lines.append("Niet uitgevoerd.")
    elif dense.get("status") == "technical_failure":
        lines.append(f"Technische fout: `{dense.get('error', {}).get('type')}: {dense.get('error', {}).get('message')}`")
    else:
        for case in dense.get("microbench", {}).get("cases", []):
            lines.append(
                f"- `{case['name']}` ({case['kind']} {case['rows']}x{case['cols']}): "
                f"{f(case.get('speedup'))}x, bitexact={case.get('bit_equal')}."
            )
        if dense.get("integration"):
            x = dense["integration"]
            lines.append(f"- Geïntegreerde p50-winst: {f(x.get('gain_ms'))} ms ({f(100*(x.get('gain_fraction') or 0), 2)}%).")
            lines.append(f"- Kandidaatdoorvoer: {f(x.get('tok_s', {}).get('pro'))} tok/s.")

    lines += ["", "### G2 — epoch graph"]
    if not epoch:
        lines.append("Niet uitgevoerd.")
    elif epoch.get("status") in {"technical_failure", "technical_blocked"}:
        lines.append("Parent/nested graph-capture is technisch niet bruikbaar in deze vorm. Zie JSON per K voor de exacte CUDA-fout.")
    else:
        for k, rec in epoch.get("epochs", {}).items():
            if rec.get("status") == "measured":
                lines.append(f"- K={k}: {f(rec.get('speedup'))}x, exact={rec.get('identical')}, parent p50={f(rec.get('parent_graph_per_token_ms', {}).get('p50'))} ms/token.")
            else:
                lines.append(f"- K={k}: `{rec.get('status')}` — {rec.get('error', {}).get('message', 'geen meting')}")

    lines += [
        "",
        "## Breakthroughcontrole",
        "",
        "De productdrempel blijft **minstens 50 tok/s in een geïntegreerde causale run**, niet een microbenchmark. "
        "Een snelle component wordt niet automatisch bij andere percentages opgeteld.",
        "",
        "## Bestanden",
        "",
        "- `PRO_G0_E1F22_GRAPH_AB.json`",
        "- `PRO_G1_DENSE_ERVF.json`",
        "- `PRO_G2_EPOCH_GRAPH.json`",
        "- `PRO_VERIFICATION.json`",
    ]
    write_text_atomic(OUT, "\n".join(lines))
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
