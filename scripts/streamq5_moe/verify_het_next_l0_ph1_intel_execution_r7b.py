#!/usr/bin/env python3
"""Independent authorization-chain verifier for the R7B-gated R7A result."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))

import verify_het_next_l0_ph1_intel_execution_r7a as physical_verifier
import run_het_next_l0_ph1_intel_execution_r7b as r7b

OUT = R / "het_next_l0_ph1_intel_execution_r7a"
VERIFY = R / "het_next_l0_ph1_intel_execution_r7b_independent_verification.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result_path = OUT / "result.json"
    manifest_path = OUT / "manifest.json"
    commit_path = OUT / "commit.json"
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    checks = physical_verifier.verify_dict(result)
    manifest = json.loads(manifest_path.read_text())
    checks["bundle"] = physical_verifier.verify_bundle_contract(
        result_bytes,
        manifest,
        json.loads(commit_path.read_text()),
        {path.name for path in OUT.iterdir()},
        sum(path.stat().st_size for path in OUT.iterdir()),
    )

    extension = result.get("authorization", {}).get("r7b_authorization", {})
    observed = {name: sha256(path) for name, path in r7b.CHAIN.items()}
    auth_result = json.loads(r7b.AUTH_RESULT.read_text())
    checks["r7b_chain"] = (
        extension.get("lock_sha256") == sha256(r7b.LOCK)
        and extension.get("observed") == observed
        and extension.get("authorization_result_sha256") == r7b.AUTH_RESULT_SHA
        and extension.get("authorization_result") == auth_result
        and extension.get("audit_token") == r7b.ACK
        and observed["authorization_result_sha256"] == r7b.AUTH_RESULT_SHA
        and observed["r7a_audit_sha256"] == "cbcbd1a861fc54e0dd529de22eb8fd3658a7fa81292e2c0ae0b188366055a5cd"
        and observed["r7p_result_sha256"] == r7b.R7P_SHA
    )
    try:
        r7b.validate_auth_result()
        checks["authorization_preflight_pass7"] = True
    except Exception:
        checks["authorization_preflight_pass7"] = False

    output = {
        "kind": "ph1_intel_execution_r7b_independent_verification",
        "checks": checks,
        "pass": all(checks.values()),
        "passed": sum(value is True for value in checks.values()),
        "total": len(checks),
        "claim": "one real expert/input Intel correctness component only",
    }
    if VERIFY.exists():
        raise FileExistsError(VERIFY)
    VERIFY.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if output["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
