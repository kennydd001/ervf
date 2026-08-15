from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from moe_lab.reporting import ROOT


RESULT = ROOT / "reports/offload_roofline/p_c_h2d_result.json"
PREREG = ROOT / "reports/offload_roofline/P_C_H2D_ROOFLINE_PREREGISTRATION.md"
OUT_JSON = ROOT / "reports/offload_roofline/p_c_h2d_verification.json"
OUT_MD = ROOT / "reports/offload_roofline/P_C_H2D_VERIFICATION.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("refusing to overwrite P-C verification")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = {}
    checks["preregistration_hash"] = sha256(PREREG) == result["preregistration_sha256"]
    checks["cuda_currently_available"] = torch.cuda.is_available()
    checks["device_identity"] = torch.cuda.is_available() and torch.cuda.get_device_name(0) == result["hardware"]["device"]
    checks["all_sizes_present"] = [row["size_mib"] for row in result["measurements"]] == [64, 256, 512]
    checks["no_allocation_failures"] = not result["failures"]
    trial_counts = stats_exact = inversion_exact = finite_positive = sizes_exact = True
    for row in result["measurements"]:
        milliseconds = np.asarray(row["milliseconds"], dtype=np.float64)
        bandwidth = np.asarray(row["bandwidth_gb_s"], dtype=np.float64)
        trial_counts &= milliseconds.size == 50 and bandwidth.size == 50 and row["warmups"] == 10 and row["trials"] == 50
        finite_positive &= bool(np.isfinite(milliseconds).all() and np.isfinite(bandwidth).all() and (milliseconds > 0).all() and (bandwidth > 0).all())
        sizes_exact &= row["size_bytes"] == row["size_mib"] * 2**20
        reconstructed = row["size_bytes"] / (milliseconds / 1000.0) / 1e9
        inversion_exact &= bool(np.allclose(reconstructed, bandwidth, rtol=0, atol=1e-12))
        latency_stats = row["latency_ms"]
        bandwidth_stats = row["effective_bandwidth_gb_s"]
        stats_exact &= abs(statistics.median(milliseconds) - latency_stats["median"]) < 1e-12
        stats_exact &= abs(float(milliseconds.mean()) - latency_stats["mean"]) < 1e-12
        stats_exact &= abs(float(np.quantile(milliseconds, 0.05, method="linear")) - latency_stats["p05"]) < 1e-12
        stats_exact &= abs(float(np.quantile(milliseconds, 0.95, method="linear")) - latency_stats["p95"]) < 1e-12
        stats_exact &= abs(statistics.median(bandwidth) - bandwidth_stats["median"]) < 1e-12
        stats_exact &= abs(float(bandwidth.mean()) - bandwidth_stats["mean"]) < 1e-12
        stats_exact &= abs(float(np.quantile(bandwidth, 0.05, method="linear")) - bandwidth_stats["p05"]) < 1e-12
        stats_exact &= abs(float(np.quantile(bandwidth, 0.95, method="linear")) - bandwidth_stats["p95"]) < 1e-12
    checks["trial_counts"] = trial_counts
    checks["finite_positive_measurements"] = finite_positive
    checks["size_bytes"] = sizes_exact
    checks["latency_bandwidth_inversion"] = inversion_exact
    checks["summary_statistics"] = stats_exact
    primary_row = max(result["measurements"], key=lambda row: row["size_mib"])
    primary_bw = primary_row["effective_bandwidth_gb_s"]["median"]
    ceiling = primary_bw / 27.28
    checks["primary_selection"] = result["primary"]["largest_successful_size_mib"] == 512 and abs(primary_bw - result["primary"]["median_bandwidth_gb_s"]) < 1e-12
    checks["roofline_arithmetic"] = abs(ceiling - result["primary"]["conditional_trunk_ceiling_tokens_per_second"]) < 1e-12
    checks["conditional_gate"] = (ceiling <= 1.0) == result["gates"]["hardware_leg_supports_conditional_le_1_tps"]
    checks["full_claim_not_promoted"] = not result["gates"]["actual_k3_trunk_bytes_measured"] and not result["gates"]["actual_64_token_k3_decode_measured"] and not result["gates"]["full_p_c_proven"]
    checks["verdict"] = result["verdict"] == "hardware_leg_supports_conditional_k3_le_1"
    checks = {name: bool(value) for name, value in checks.items()}
    passed = sum(checks.values())
    verification = {
        "kind": "offload_roofline_p_c_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "checks_passed": passed, "checks_total": len(checks),
        "all_pass": passed == len(checks),
        "verdict": "p_c_hardware_leg_verified" if passed == len(checks) else "verification_failed",
        "conditional_ceiling_tokens_per_second": ceiling,
    }
    OUT_JSON.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join([
        "# P-C onafhankelijke verificatie", "",
        f"**{verification['verdict']}** — {passed}/{len(checks)} controles geslaagd.", "",
        f"De 150 ruwe eventmetingen, byte/tijd-inversie, distributiestatistieken, hardware-identiteit en roofline zijn herberekend. Conditioneel plafond: {ceiling:.6f} tok/s.", "",
        "De verifier bevestigt expliciet niet de externe 27,28-GB-input of een K3-decode; de volledige P-C-claim blijft open/geblokkeerd.", "",
    ]), encoding="utf-8")
    print(json.dumps({"verdict": verification["verdict"], "checks": f"{passed}/{len(checks)}", "conditional_ceiling": ceiling}, indent=2))
