#!/usr/bin/env python3
"""PH1-R2 compile-only backend: frozen R1 implementation with exact R2 source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import het_next_l0_ph1_intel_compile_r1_backend as frozen_r1
import het_next_l0_ph1_intel_compile_r2_source as source_r2


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2_lock.json"
R1B_FAILURE = REPORTS / "het_next_l0_ph1_intel_compile_r1b_failed_attempts/attempt_failure_06df3c72c9c44379a04d39b43d301b53/failure.json"
R1B_FAILURE_SHA256 = "62107b4cee0809fd744bacfe5d6890c7e09ec9002b0b029a6e84c98359f95fbb"
ACK = "PH1_INTEL_COMPILE_R2_AFTER_SOURCE_AUDIT_PREFLIGHT_AND_AUTH_GO"
SRC = source_r2.SRC
SOURCE_SHA256 = source_r2.R2_SOURCE_SHA256
OPTIONS = source_r2.OPTIONS
CompileFailure = frozen_r1.CompileFailure


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def authorize(expected: dict) -> dict:
    """Fail before the frozen backend opens OpenCL.dll unless R2 is separately authorized."""
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if not (
        lock.get("kind") == "het_next_l0_ph1_intel_compile_r2_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == ACK
        and lock.get("backend_sha256") == file_sha(Path(__file__))
        and lock.get("source_module_sha256") == file_sha(Path(source_r2.__file__))
        and lock.get("r1_backend_sha256") == file_sha(Path(frozen_r1.__file__))
        and lock.get("r1b_failure_sha256") == file_sha(R1B_FAILURE) == R1B_FAILURE_SHA256
        and lock.get("source_sha256") == SOURCE_SHA256
        and lock.get("cpu_commit_sha256") == expected.get("cpu_commit_sha256")
        and lock.get("cpu_verification_sha256") == expected.get("cpu_verification_sha256")
        and lock.get("prior_audit_sha256") == expected.get("prior_audit_sha256")
    ):
        raise RuntimeError("compile_r2_authorization")
    return {
        "lock_sha256": file_sha(LOCK),
        "source_sha256": SOURCE_SHA256,
        "r1b_failure_sha256": R1B_FAILURE_SHA256,
        "audit_token": ACK,
    }


def compile_only(eligibility: dict) -> dict:
    """Substitute only R2 source/authorization during one call and always restore R1 globals."""
    authorization = authorize(eligibility)
    original_source = frozen_r1.SRC
    original_source_sha = frozen_r1.SOURCE_SHA256
    original_authorize = frozen_r1.authorize
    frozen_r1.SRC = SRC
    frozen_r1.SOURCE_SHA256 = SOURCE_SHA256
    frozen_r1.authorize = lambda _expected: authorization
    try:
        evidence = frozen_r1.compile_only(eligibility)
    finally:
        frozen_r1.SRC = original_source
        frozen_r1.SOURCE_SHA256 = original_source_sha
        frozen_r1.authorize = original_authorize
    if evidence.get("authorization") != authorization:
        raise RuntimeError("r2_authorization_evidence")
    if evidence.get("source_sha256") != SOURCE_SHA256 or evidence.get("source") != SRC:
        raise RuntimeError("r2_source_evidence")
    evidence["source_revision"] = "R2"
    evidence["frozen_r1_backend_sha256"] = file_sha(Path(frozen_r1.__file__))
    return evidence
