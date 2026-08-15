from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from moe_lab.reporting import ROOT


OUT_JSON = ROOT / "reports/three_pack_campaign_verification.json"
OUT_MD = ROOT / "reports/THREE_PACK_CAMPAIGN_VERIFICATION.md"
REPORT = ROOT / "reports/THREE_PACK_CAMPAIGN_2026-08-11.md"


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def load_yaml(relative: str):
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("refusing to overwrite campaign verification")
    bitflow_v = load_json("reports/bitflow_moe/p0_c1_q4_verification.json")
    coretail_v = load_json("reports/coretail_moe/p0a_locked16_format_verification.json")
    p_b_v = load_json("reports/offload_roofline/p_b_lfu_verification.json")
    p_c_v = load_json("reports/offload_roofline/p_c_h2d_verification.json")
    p_e_v = load_json("reports/offload_roofline/p_e_permutation_verification.json")
    audit_v = load_json("reports/offload_roofline/remaining_proposal_audit_verification.json")
    regressions = load_json("reports/three_pack_regression_tests.json")
    bitflow_r = load_yaml("reports/bitflow_moe/EXPERIMENT_REGISTRY.yaml")
    coretail_r = load_yaml("reports/coretail_moe/EXPERIMENT_REGISTRY.yaml")
    offload_r = load_yaml("reports/offload_roofline/EXPERIMENT_REGISTRY.yaml")
    checks = {
        "bitflow_verifier": bitflow_v["checks_passed"] == bitflow_v["checks_total"] == 23 and bitflow_v["final_verdict"] == "p0_linear_branch_negative_verified",
        "coretail_verifier": coretail_v["checks_passed"] == coretail_v["checks_total"] == 28 and coretail_v["verdict"] == "locked16_mechanics_verified_full_p0_still_blocked",
        "p_b_verifier": p_b_v["checks_passed"] == p_b_v["checks_total"] == 14 and p_b_v["verdict"] == "p_b_negative_verified",
        "p_c_verifier": p_c_v["checks_passed"] == p_c_v["checks_total"] == 15 and p_c_v["verdict"] == "p_c_hardware_leg_verified",
        "p_e_verifier": p_e_v["checks_passed"] == p_e_v["checks_total"] == 18 and p_e_v["verdict"] == "p_e_negative_verified",
        "remaining_audit_verifier": audit_v["checks_passed"] == audit_v["checks_total"] == 15 and audit_v["verdict"] == "remaining_proposal_audit_verified",
        "regression_tests": regressions["passed"] and regressions["returncode"] == 0 and "153 passed" in regressions["stdout"],
        "bitflow_registry_closed": bitflow_r["status"] == "closed_p0_linear_branch_negative_verified",
        "coretail_registry_bounded": coretail_r["status"] == "p0_blocked_missing_full_bank_gptq_codes" and coretail_r["outcome"]["official_full_bank_p0_pass"] is False,
        "offload_registry_closed": offload_r["status"] == "closed_mixed_results_no_eureka",
        "p_a_not_overclaimed": offload_r["proposals"]["P_A_QWEN_WALLCLOCK"]["status"] == "blocked_missing_full_bank_gptq_and_runtime",
        "p_c_not_overclaimed": offload_r["proposals"]["P_C_K3_ROOFLINE"]["status"] == "hardware_leg_verified_full_claim_blocked",
        "consolidated_report_exists": REPORT.is_file() and REPORT.stat().st_size > 1000,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = sum(checks.values())
    result = {
        "kind": "three_pack_campaign_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "checks_passed": passed, "checks_total": len(checks),
        "all_pass": passed == len(checks),
        "verdict": "campaign_verified_no_eureka" if passed == len(checks) else "campaign_verification_failed",
        "consolidated_report_sha256": sha256(REPORT),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join([
        "# Drie-packcampagne — integriteitscontrole", "",
        f"**{result['verdict']}** — {passed}/{len(checks)} controles geslaagd.", "",
        "Alle onafhankelijke verifiers, registry-eindstatussen, bewijsgrenzen en de 153-test-regressierun zijn gecontroleerd.", "",
    ]), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "checks": f"{passed}/{len(checks)}"}, indent=2))
