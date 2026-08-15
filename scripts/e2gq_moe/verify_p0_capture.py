from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

from moe_lab.reporting import ROOT


CAPTURE = ROOT / "reports/e2gq_moe/p0_capture_result.json"
OUTPUT = ROOT / "reports/e2gq_moe/p0_capture_verification.json"
REPORT = ROOT / "reports/e2gq_moe/P0_COVERAGE_VERDICT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite P0 verification")
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    checks = {
        "capture_declares_coverage_negative": capture["status"] == "coverage_negative" and capture["coverage_pass"] is False,
        "all_48_layers_manifested": len(capture["artifacts"]) == 48 and len(capture["layers"]) == 48,
        "all_artifact_hashes_match": True,
        "all_layer_report_hashes_match": True,
        "all_shapes_match": True,
        "all_route_counts_recompute": True,
        "all_route_totals_equal_262144": True,
        "all_inputs_finite": True,
        "all_hook_routes_exact": True,
        "all_layers_fail_minimum_coverage": True,
        "resource_limits_pass": capture["hardware"]["peak_cuda_allocated_bytes"] <= 7.5 * 2**30 and capture["hardware"]["peak_process_rss_bytes"] <= 32 * 2**30,
        "no_gptq_phase_opened": not (ROOT / "reports/runs/e2gq_moe/p0_gptq").exists(),
    }
    failed_pairs = zero_pairs = 0
    global_maximum = 0
    details = []
    for layer in range(48):
        manifest = capture["artifacts"][str(layer)]
        artifact = ROOT / manifest["artifact"]
        report_path = ROOT / manifest["report"]
        checks["all_artifact_hashes_match"] &= sha256(artifact) == manifest["artifact_sha256"]
        checks["all_layer_report_hashes_match"] &= sha256(report_path) == manifest["report_sha256"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        with safe_open(artifact, framework="pt", device="cpu") as handle:
            x = handle.get_tensor("moe_input")
            ids = handle.get_tensor("router_ids")
        checks["all_shapes_match"] &= tuple(x.shape) == (32768, 2048) and tuple(ids.shape) == (32768, 8)
        counts = torch.bincount(ids.long().reshape(-1), minlength=128)
        checks["all_route_counts_recompute"] &= counts.tolist() == report["router_counts"]
        checks["all_route_totals_equal_262144"] &= int(counts.sum()) == 262144
        checks["all_inputs_finite"] &= bool(torch.isfinite(x.float()).all()) and report["finite_moe_input"] is True
        checks["all_hook_routes_exact"] &= report["route_ids_exact"] is True and report["router_logit_maximum_absolute_error"] == 0.0
        below = int((counts < 128).sum())
        zero = int((counts == 0).sum())
        checks["all_layers_fail_minimum_coverage"] &= below > 0
        checks["all_route_counts_recompute"] &= (
            below == report["experts_below_128"] == capture["layers"][str(layer)]["experts_below_128"]
        )
        failed_pairs += below
        zero_pairs += zero
        global_maximum = max(global_maximum, int(counts.max()))
        details.append({"layer": layer, "minimum_rows": int(counts.min()), "maximum_rows": int(counts.max()), "experts_below_128": below, "zero_experts": zero})
        del x, ids, counts

    payload = {
        "kind": "e2gq_p0_capture_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "capture_sha256": sha256(CAPTURE), "checks": checks,
        "checks_passed": sum(checks.values()), "checks_total": len(checks),
        "verification_pass": all(checks.values()),
        "layers_failing_coverage": sum(row["experts_below_128"] > 0 for row in details),
        "layer_expert_pairs_below_128": failed_pairs,
        "zero_invocation_layer_expert_pairs": zero_pairs,
        "global_minimum_rows": min(row["minimum_rows"] for row in details),
        "global_maximum_rows": global_maximum, "layers": details,
        "verdict": "coverage_negative_p0_stops_before_gptq",
        "claim_boundary": "The frozen WikiText calibration is insufficient; entropy-GPTQ itself is not falsified.",
    }
    if not payload["verification_pass"]:
        raise AssertionError([name for name, passed in checks.items() if not passed])
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E2GQ-MoE P0 — coverageverdict", "",
        f"Uitkomst: **coverage_negative**, onafhankelijk geverifieerd met {payload['checks_passed']}/{payload['checks_total']} controles.", "",
        "De vooraf vastgelegde 32.768 WikiText-train-tokens dekken de volledige expertbank niet voldoende:", "",
        f"- alle **48/48 lagen** bevatten experts met minder dan 128 routed rijen;",
        f"- **{failed_pairs:,}/6.144** laag-expertparen zitten onder 128;",
        f"- **{zero_pairs}** laag-expertparen hebben exact nul invocaties;",
        f"- de verdeling loopt van 0 tot {global_maximum:,} invocaties.", "",
        "Volgens de preregistratie zijn geen GPTQ-codes voor ondergedekte experts geconstrueerd en is P1 niet geopend. Dit falsifieert de gekozen monolinguale calibratieprocedure, niet de reeds bevestigde 16-expert entropy-precondition en niet entropy-GPTQ in het algemeen.", "",
        "Een vervolg vereist een nieuwe, vooraf vastgelegde coveragehypothese met een representatieve meertalige/multidomeincalibratie of een principiële activation-agnostic quantizer. De huidige registry mag die keuze niet post-hoc maken.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("verification_pass", "checks_passed", "checks_total", "layers_failing_coverage", "layer_expert_pairs_below_128", "zero_invocation_layer_expert_pairs")}, indent=2))
