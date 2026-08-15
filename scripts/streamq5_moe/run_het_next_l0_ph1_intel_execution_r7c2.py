#!/usr/bin/env python3
"""PH1 Intel R7C2: exact inherited device-state retention repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))

import run_het_next_l0_ph1_intel_execution_r7a as physical
import run_het_next_l0_ph1_intel_execution_r7c1 as prior

LOCK = R / "het_next_l0_ph1_intel_execution_r7c2_lock.json"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C2_PREREGISTRATION_2026-08-14.md"
PREFLIGHT = S / "preflight_het_next_l0_ph1_intel_execution_r7c2.py"
VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r7c2.py"
AUTH_RESULT = prior.AUTH_RESULT
OUTER_FAILED = R / "het_next_l0_ph1_intel_execution_r7c2_failed_attempts"
OUTER_QUARANTINE = R / "het_next_l0_ph1_intel_execution_r7c2_quarantine"
REVISION_OUT = R / "het_next_l0_ph1_intel_execution_r7c2"
ACK = "PH1_INTEL_EXECUTION_R7C2_AFTER_DEVICE_STATE_AND_CLEAN_STATE_AUDIT_GO"
R7A_ACK = prior.R7A_ACK
R7P_SHA = prior.R7P_SHA
AUTH_RESULT_SHA = prior.AUTH_RESULT_SHA
MAX_FAILURE_BYTES = prior.MAX_FAILURE_BYTES
AUTH_CHECKS = prior.AUTH_CHECKS
OUTER_STAGES = prior.OUTER_STAGES
ALLOWED_INHERITED_DISPOSITIONS = prior.ALLOWED_INHERITED_DISPOSITIONS
CHAIN = {
    "runner_sha256": Path(__file__),
    "verifier_sha256": VERIFIER,
    "preflight_sha256": PREFLIGHT,
    "prereg_sha256": PREREG,
    "r7c1_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r7c1.py",
    "r7c1_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7c1.py",
    "r7c1_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r7c1.py",
    "r7c1_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C1_PREREGISTRATION_2026-08-14.md",
    "r7c1_lock_sha256": R / "het_next_l0_ph1_intel_execution_r7c1_lock.json",
    "r7c1_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7C1_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
    **{name: path for name, path in prior.CHAIN.items() if name not in {"runner_sha256", "verifier_sha256", "preflight_sha256", "prereg_sha256"}},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_auth_result() -> dict:
    return prior.validate_auth_result()


def authorize() -> dict:
    result = validate_auth_result(); observed = {name: sha256(path) for name, path in CHAIN.items()}; lock = json.loads(LOCK.read_text())
    if not (
        set(lock) == {"kind", "execution_open", "audit_token", "physical_output", "physical_verifier", *observed.keys()}
        and lock.get("kind") == "ph1_intel_execution_r7c2_lock"
        and lock.get("execution_open") is True and lock.get("audit_token") == ACK
        and lock.get("physical_output") == "het_next_l0_ph1_intel_execution_r7a"
        and lock.get("physical_verifier") == "verify_het_next_l0_ph1_intel_execution_r7c2.py"
        and all(lock.get(name) == digest for name, digest in observed.items())
        and observed["authorization_result_sha256"] == AUTH_RESULT_SHA
        and observed["r7c1_audit_sha256"] == "e75ae1897d4ce73664c1225ca499e41a029660f0506b2fc72ee4cc65ddfadeb2"
        and observed["r7p_result_sha256"] == R7P_SHA
    ):
        raise RuntimeError("r7c2_authorization")
    inherited = physical.authorize()
    inherited["r7c2_authorization"] = {"lock_sha256": sha256(LOCK), "observed": observed, "authorization_result_sha256": AUTH_RESULT_SHA, "authorization_result": result, "audit_token": ACK, "outer_failure_stages": list(OUTER_STAGES)}
    return inherited


def canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def bounded_payload(payload: dict) -> bytes:
    encoded = canonical(payload)
    if len(encoded) <= MAX_FAILURE_BYTES:
        return encoded
    return canonical({"kind": "ph1_intel_execution_r7c2_failure", "status": "valid_negative_failure", "stage": payload.get("stage", "outer_boundary"), "error": "failure_evidence_oversize", "original_bytes": len(encoded), "original_sha256": hashlib.sha256(encoded).hexdigest(), "device_opened": bool(payload.get("device_opened", False)), "disposition": "bounded_summary_only"})


def atomic_summary(payload: dict) -> Path:
    encoded = bounded_payload(payload)
    if len(encoded) > MAX_FAILURE_BYTES:
        raise RuntimeError("failure_cap")
    OUTER_FAILED.mkdir(parents=True, exist_ok=True); nonce = uuid.uuid4().hex
    temporary = R / f"{OUTER_FAILED.name}.{nonce}.inprogress"; destination = OUTER_FAILED / f"attempt_{nonce}"; temporary.mkdir(parents=False, exist_ok=False)
    try:
        physical.base.write(temporary / "failure.json", encoded); physical.base.move(temporary, destination)
    except Exception:
        if temporary.exists():
            OUTER_QUARANTINE.mkdir(parents=True, exist_ok=True); physical.base.move(temporary, OUTER_QUARANTINE / f"failed_commit_{nonce}")
        raise
    return destination


def quarantine_stale() -> None:
    stale = sorted(R.glob(OUTER_FAILED.name + ".*.inprogress"))
    if not stale:
        return
    OUTER_QUARANTINE.mkdir(parents=True, exist_ok=True)
    for path in stale:
        physical.base.move(path, OUTER_QUARANTINE / ("stale_" + path.name.rsplit(".", 2)[-2]))
    raise RuntimeError("stale_outer_failure_quarantined")


failure_paths = prior.failure_paths
bundle_digest = prior.bundle_digest
committed_result = prior.committed_result


def inherited_evidence(path: Path) -> tuple[bool, dict]:
    directory = path.parent; total, digest, rows = bundle_digest(directory)
    record = {"relative_path": path.relative_to(physical.FAILED).as_posix(), "failure_sha256": sha256(path), "bundle_sha256": digest, "bundle_bytes": total, "file_count": len(rows), "files": rows[:32]}
    if total > MAX_FAILURE_BYTES or path.stat().st_size > MAX_FAILURE_BYTES or len(rows) > 32:
        return False, {**record, "adjudication": "oversize_or_cardinality", "inherited_device_opened": None}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return False, {**record, "adjudication": f"parse:{type(exc).__name__}", "inherited_device_opened": None}
    device_opened = payload.get("device_opened")
    valid = isinstance(payload, dict) and payload.get("kind") == "ph1_intel_execution_r7a_failure" and payload.get("status") == "valid_negative_failure" and isinstance(payload.get("error"), str) and isinstance(device_opened, bool) and payload.get("disposition") in ALLOWED_INHERITED_DISPOSITIONS
    return valid, {**record, "adjudication": "valid" if valid else "schema_or_disposition", "inherited_disposition": payload.get("disposition"), "inherited_device_opened": device_opened if isinstance(device_opened, bool) else None}


def delegated_summary(return_code, before: set[Path], after: set[Path]) -> dict:
    new = sorted(after - before); observations = []
    for path in new[:32]:
        try: valid, evidence = inherited_evidence(path)
        except Exception as exc: valid, evidence = False, {"relative_path": str(path), "adjudication": f"inspection:{type(exc).__name__}", "inherited_device_opened": None}
        observations.append({"valid": valid, **evidence})
    exactly_one_valid = len(new) == 1 and len(observations) == 1 and observations[0]["valid"]
    adjudication = "one_valid_inherited_failure" if exactly_one_valid else "missing_inherited_failure" if not new else "multiple_inherited_failures" if len(new) > 1 else observations[0].get("adjudication", "invalid_inherited_failure")
    return {"kind": "ph1_intel_execution_r7c2_failure", "status": "valid_negative_failure", "stage": "delegated_nonzero", "error": "delegated_execution_nonzero", "delegated_return": return_code, "device_opened": any(row.get("inherited_device_opened") is True for row in observations), "new_inherited_failure_count": len(new), "inherited_evidence_valid": exactly_one_valid, "adjudication": adjudication, "inherited": observations, "disposition": "atomic_bounded_digest_summary"}


def outer_execute(authorization: dict, executor=physical.execute_authorized) -> int:
    before = failure_paths()
    try:
        quarantine_stale(); return_code = executor(authorization)
    except Exception as exc:
        committed = committed_result()
        if committed is not None: return 0 if committed.get("positive") is True else 3
        atomic_summary({"kind": "ph1_intel_execution_r7c2_failure", "status": "valid_negative_failure", "stage": getattr(exc, "stage", "r7a_outer_boundary"), "error": f"{type(exc).__name__}:{exc}"[:2048], "traceback": traceback.format_exc()[-32768:], "device_opened": bool(getattr(exc, "device_opened", False)), "covered_stages": list(OUTER_STAGES), "disposition": "atomic_create_new_bounded_outer_failure"}); return 3
    committed = committed_result()
    if committed is not None: return 0 if committed.get("positive") is True else 3
    after = failure_paths(); payload = delegated_summary(return_code, before, after)
    if return_code == 0: payload["error"] = "delegated_success_without_valid_commit"; payload["stage"] = "delegated_invalid_success"
    atomic_summary(payload); return 3


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--ack", required=True); args = parser.parse_args()
    if args.ack != ACK: return 3
    try: authorization = authorize()
    except Exception: return 3
    return outer_execute(authorization)


if __name__ == "__main__": raise SystemExit(main())
