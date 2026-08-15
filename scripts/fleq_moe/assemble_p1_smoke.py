from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from moe_lab.reporting import ROOT


LOCK = ROOT / "reports/fleq_moe/p1_smoke_expert_lock.json"
PREREG = ROOT / "reports/fleq_moe/P1_QWEN_EXPERT_STREAMED_PREREGISTRATION.md"
ADDENDA = [
    ROOT / "reports/fleq_moe/P1_PROTOCOL_ADDENDUM_001.md",
    ROOT / "reports/fleq_moe/P1_PROTOCOL_ADDENDUM_002.md",
    ROOT / "reports/fleq_moe/P1_PROTOCOL_ADDENDUM_003.md",
]
RESULT = ROOT / "reports/fleq_moe/p1_smoke_result.json"
REPORT = ROOT / "reports/fleq_moe/P1_QWEN_EXPERT_STREAMED.md"
TWO_DIR = ROOT / "reports/fleq_moe/p1_experts"
TERNARY_DIR = ROOT / "reports/fleq_moe/p1_ternary_experts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(rows: list[dict], kind: str) -> dict:
    if kind == "2bit":
        improvements = [row["heldout_gsq_improvement_over_gptq"] for row in rows]
        baseline = [row["metrics"]["gptq_2bit"]["heldout"]["router_weighted_relative_mse"] for row in rows]
        gsq = [row["metrics"]["gsq_2bit"]["heldout"]["router_weighted_relative_mse"] for row in rows]
        p95_regressions = sum(
            row["metrics"]["gsq_2bit"]["heldout"]["relative_row_p95"]
            > row["metrics"]["gptq_2bit"]["heldout"]["relative_row_p95"]
            for row in rows
        )
    else:
        improvements = [row["heldout_gsq_improvement_over_rtn"] for row in rows]
        baseline = [row["metrics"]["rtn_ternary"]["heldout"]["router_weighted_relative_mse"] for row in rows]
        gsq = [row["metrics"]["gsq_ternary"]["heldout"]["router_weighted_relative_mse"] for row in rows]
        p95_regressions = sum(
            row["metrics"]["gsq_ternary"]["heldout"]["relative_row_p95"]
            > row["metrics"]["rtn_ternary"]["heldout"]["relative_row_p95"]
            for row in rows
        )
    return {
        "experts": len(rows),
        "experts_improved": sum(value > 0 for value in improvements),
        "experts_improved_at_least_20pct": sum(value >= 0.20 for value in improvements),
        "mean_improvement": sum(improvements) / len(improvements),
        "minimum_improvement": min(improvements),
        "maximum_improvement": max(improvements),
        "mean_baseline_heldout_relative_mse": sum(baseline) / len(baseline),
        "mean_gsq_heldout_relative_mse": sum(gsq) / len(gsq),
        "p95_regressions": p95_regressions,
    }


if __name__ == "__main__":
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite P1 aggregate")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    artifacts = {}
    layer_results = {}
    controls = {
        "selection_lock_matches": True,
        "all_expected_reports_present": True,
        "all_artifact_hashes_match": True,
        "all_finite": True,
        "all_fallbacks_bit_exact": True,
        "determinism_checks_pass": True,
        "resource_limits_pass": True,
        "all_codes_in_range": True,
    }
    for layer_text, locked in lock["layers"].items():
        layer = int(layer_text)
        two_rows, ternary_rows = [], []
        for expert in locked["selected_experts"]:
            two_path = TWO_DIR / f"layer_{layer:02d}_expert_{expert:03d}.json"
            ternary_path = TERNARY_DIR / f"layer_{layer:02d}_expert_{expert:03d}.json"
            if not two_path.is_file() or not ternary_path.is_file():
                controls["all_expected_reports_present"] = False
                continue
            two = json.loads(two_path.read_text(encoding="utf-8"))
            ternary = json.loads(ternary_path.read_text(encoding="utf-8"))
            two_rows.append(two)
            ternary_rows.append(ternary)
            for name, row, path in (("2bit", two, two_path), ("ternary", ternary, ternary_path)):
                artifact = ROOT / row["artifact"]
                controls["all_artifact_hashes_match"] &= sha256(artifact) == row["artifact_sha256"]
                controls["all_finite"] &= bool(row["all_finite"])
                controls["resource_limits_pass"] &= (
                    row["peak_cuda_allocated_bytes"] <= 7.5 * 2**30
                    and row["process_rss_peak_observed_bytes"] <= 32 * 2**30
                )
                if row["repeat_required"]:
                    controls["determinism_checks_pass"] &= row["repeat_exact"] is True
                for method in row["code_summaries"].values():
                    controls["all_codes_in_range"] &= all(item["codes_in_range"] for item in method.values())
                key = f"layer_{layer:02d}_expert_{expert:03d}_{name}"
                artifacts[key] = {
                    "report": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "report_sha256": sha256(path),
                    "artifact": row["artifact"],
                    "artifact_sha256": row["artifact_sha256"],
                }
            controls["all_fallbacks_bit_exact"] &= bool(two["fallback_bit_exact"])
            controls["selection_lock_matches"] &= (
                two["selection_lock_sha256"] == sha256(LOCK)
                and ternary["selection_lock_sha256"] == sha256(LOCK)
            )
        layer_results[layer_text] = {
            "selected_experts": locked["selected_experts"],
            "two_bit": summarize(two_rows, "2bit"),
            "ternary": summarize(ternary_rows, "ternary"),
        }

    two_bit_gate = all(
        row["two_bit"]["mean_improvement"] >= 0.20
        and row["two_bit"]["experts_improved_at_least_20pct"] >= 6
        and row["two_bit"]["p95_regressions"] == 0
        for row in layer_results.values()
    )
    controls_pass = all(controls.values())
    verdict = "infrastructure_positive" if controls_pass and two_bit_gate else "smoke_negative"
    payload = {
        "kind": "fleq_moe_p1_expert_streamed_smoke",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "p2_authorized": verdict == "infrastructure_positive",
        "preregistration_sha256": sha256(PREREG),
        "protocol_addenda_sha256": {
            path.name: sha256(path) for path in ADDENDA
        },
        "selection_lock_sha256": sha256(LOCK),
        "controls": controls,
        "all_required_controls_pass": controls_pass,
        "two_bit_gate_pass": two_bit_gate,
        "layers": layer_results,
        "storage": {
            "two_bit_codes_plus_bf16_group128_scales_bpp": 2.125,
            "ternary_ideal_log2_3_plus_bf16_group128_scales_bpp": 1.709962500721156,
            "ternary_two_bit_pack_plus_bf16_group128_scales_bpp": 2.125,
        },
        "claim_boundaries": {
            "full_depth_ce": "not measured",
            "packed_runtime": "not measured",
            "rollouts": "not measured",
            "eureka": False,
            "ternary": "post-lock diagnostic allowed by preregistration; cannot replace failed 2-bit gate",
        },
        "artifacts": artifacts,
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table = []
    for layer in ("0", "47"):
        two = layer_results[layer]["two_bit"]
        ternary = layer_results[layer]["ternary"]
        table.append(
            f"| {layer} | 2-bit GSQ vs GPTQ | {two['experts_improved']}/8 | "
            f"{100*two['mean_improvement']:.2f}% | {two['mean_baseline_heldout_relative_mse']:.4f} → {two['mean_gsq_heldout_relative_mse']:.4f} |"
        )
        table.append(
            f"| {layer} | ternary GSQ vs RTN | {ternary['experts_improved']}/8 | "
            f"{100*ternary['mean_improvement']:.2f}% | {ternary['mean_baseline_heldout_relative_mse']:.4f} → {ternary['mean_gsq_heldout_relative_mse']:.4f} |"
        )
    markdown = "\n".join([
        "# FLEQ-MoE P1 — Qwen expert-streamed GSQ-smoke",
        "",
        f"**Verdict: `{verdict}`. P2 geautoriseerd: `{payload['p2_authorized']}`.**",
        "",
        "## Kernresultaat",
        "",
        "De officiële, hash-gepinde GSQ-codebookoperator draait deterministisch en ruim binnen het laptopgeheugen. De inhoudelijke 2-bitgate faalt echter hard: geen van de zestien vooraf geselecteerde experts verbetert op de ongeziene context tegenover zijn GPTQ-initialisatie.",
        "",
        "| Laag | Vergelijking | Experts verbeterd | Gemiddelde verbetering | Gem. held-out gewogen relatieve MSE |",
        "|---:|---|---:|---:|---:|",
        *table,
        "",
        "Laag 0 heeft 0/8 en laag 47 0/8 2-bitverbeteringen; vereist was minstens 6/8 per laag met minimaal 20% aggregate verbetering en zonder p95-regressie. Ternary verbetert RTN lokaal, vooral in laag 47, maar blijft absoluut veel onnauwkeuriger dan 2-bit GPTQ en mag de gefaalde primaire gate niet vervangen.",
        "",
        "## Opslaggrens",
        "",
        "2-bit codes plus BF16-group128-scales kosten analytisch 2,125 bpp en missen dus al de uiteindelijke ≤2,0-bpp-gate wanneer metadata wordt meegerekend. Ternary heeft een ideale cardinaliteitsbound van 1,710 bpp inclusief die scales, maar een gewone 2-bit pack kost eveneens 2,125 bpp; echte entropycoding en directe kernels zijn niet gebouwd.",
        "",
        "## Controles en claimgrens",
        "",
        f"Alle verplichte uitvoeringscontroles: `{controls_pass}`. BF16-fallbacks zijn bit-exact, beide determinismeherhalingen sluiten, alle codes/scales/outputs zijn eindig en alle artifacthashes sluiten. Full-depth CE, benchmarkkwaliteit, rollouts, bitpacked artifactgrootte en runtime zijn niet gemeten. Dit is geen Eureka.",
        "",
        "Volgens de preregistratie blijft P2 geblokkeerd. De bestaande GSQ-PTQ-reproductielijn sluit als `smoke_negative`; expert-trajectory-QAT mag alleen via een nieuwe, afzonderlijk gemotiveerde near-miss-preregistratie worden geopend.",
        "",
    ])
    REPORT.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "controls_pass": controls_pass,
        "two_bit_gate_pass": two_bit_gate,
        "layers": layer_results,
    }, indent=2))
