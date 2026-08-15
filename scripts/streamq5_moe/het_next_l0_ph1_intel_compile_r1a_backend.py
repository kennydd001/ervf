#!/usr/bin/env python3
"""Authorization-only PH1-R1A wrapper around the frozen R1 compile backend."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import het_next_l0_ph1_intel_compile_r1_backend as frozen_r1


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1a_lock.json"
R1_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1_lock.json"
PASS_RESULT = REPORTS / "het_next_l0_ph1_intel_compile_r1_preflight_result.json"
ACK = "PH1_INTEL_COMPILE_R1A_AFTER_PREFLIGHT_PASS_AND_INDEPENDENT_FINAL_AUDIT_GO"
SRC = frozen_r1.SRC
SOURCE_SHA256 = frozen_r1.SOURCE_SHA256
OPTIONS = frozen_r1.OPTIONS
CompileFailure = frozen_r1.CompileFailure


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def authorize(expected: dict) -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    passed = json.loads(PASS_RESULT.read_text(encoding="utf-8"))
    if not (
        lock.get("kind") == "het_next_l0_ph1_intel_compile_r1a_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == ACK
        and lock.get("backend_sha256") == file_sha256(Path(__file__))
        and lock.get("r1_backend_sha256") == file_sha256(Path(frozen_r1.__file__))
        and lock.get("r1_closed_lock_sha256") == file_sha256(R1_LOCK)
        and lock.get("r1_preflight_pass_sha256") == file_sha256(PASS_RESULT)
        and lock.get("source_sha256") == SOURCE_SHA256
        and passed.get("kind") == "het_next_l0_ph1_intel_compile_r1_static_preflight"
        and passed.get("pass") is True
        and passed.get("passed") == passed.get("total") == 8
        and passed.get("device_calls") == passed.get("compiler_calls") == passed.get("payload_reads") == 0
        and passed.get("source_lock_sha256") == lock.get("r1_closed_lock_sha256")
        and passed.get("preflight_sha256") == lock.get("r1_preflight_sha256")
        and lock.get("cpu_commit_sha256") == expected.get("cpu_commit_sha256")
        and lock.get("cpu_verification_sha256") == expected.get("cpu_verification_sha256")
        and lock.get("prior_audit_sha256") == expected.get("prior_audit_sha256")
    ):
        raise RuntimeError("compile_r1a_authorization")
    return {
        "lock_sha256": file_sha256(LOCK),
        "r1_closed_lock_sha256": file_sha256(R1_LOCK),
        "r1_preflight_pass_sha256": file_sha256(PASS_RESULT),
        "audit_token": ACK,
    }


def compile_only(eligibility: dict) -> dict:
    """Reuse the byte-identical R1 implementation; substitute only its closed authorization gate."""
    authorization = authorize(eligibility)
    original = frozen_r1.authorize

    def authorized_r1_call(_expected: dict) -> dict:
        return authorization

    frozen_r1.authorize = authorized_r1_call
    try:
        evidence = frozen_r1.compile_only(eligibility)
    finally:
        frozen_r1.authorize = original
    if evidence.get("authorization") != authorization:
        raise RuntimeError("r1a_authorization_evidence")
    evidence["authorization_revision"] = "R1A"
    evidence["frozen_r1_backend_sha256"] = file_sha256(Path(frozen_r1.__file__))
    return evidence
