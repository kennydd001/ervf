#!/usr/bin/env python3
"""Final R2A authorization wrapper around the frozen R2 compile backend."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import het_next_l0_ph1_intel_compile_r2_backend as frozen_r2


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2a_lock.json"
R2P1_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2p1_lock.json"
R2P1_PASS = REPORTS / "het_next_l0_ph1_intel_compile_r2p1_static_preflight.json"
ACK = "PH1_INTEL_COMPILE_R2A_AFTER_R2P1_PASS_AND_INDEPENDENT_FINAL_AUDIT_GO"
SRC = frozen_r2.SRC
SOURCE_SHA256 = frozen_r2.SOURCE_SHA256
OPTIONS = frozen_r2.OPTIONS
CompileFailure = frozen_r2.CompileFailure


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def authorize(expected: dict) -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    passed = json.loads(R2P1_PASS.read_text(encoding="utf-8"))
    if not (
        lock.get("kind") == "het_next_l0_ph1_intel_compile_r2a_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == ACK
        and lock.get("backend_sha256") == file_sha(Path(__file__))
        and lock.get("r2_backend_sha256") == file_sha(Path(frozen_r2.__file__))
        and lock.get("r2p1_lock_sha256") == file_sha(R2P1_LOCK)
        and lock.get("r2p1_pass_sha256") == file_sha(R2P1_PASS)
        and lock.get("source_sha256") == SOURCE_SHA256
        and passed.get("kind") == "het_next_l0_ph1_intel_compile_r2p1_static_preflight"
        and passed.get("pass") is True
        and passed.get("passed") == passed.get("total") == 8
        and passed.get("compiler_calls") == passed.get("device_calls") == passed.get("payload_reads") == 0
        and lock.get("cpu_commit_sha256") == expected.get("cpu_commit_sha256")
        and lock.get("cpu_verification_sha256") == expected.get("cpu_verification_sha256")
        and lock.get("prior_audit_sha256") == expected.get("prior_audit_sha256")
    ):
        raise RuntimeError("compile_r2a_authorization")
    return {"lock_sha256": file_sha(LOCK), "r2p1_lock_sha256": file_sha(R2P1_LOCK), "r2p1_pass_sha256": file_sha(R2P1_PASS), "audit_token": ACK}


def compile_only(eligibility: dict) -> dict:
    authorization = authorize(eligibility)
    original = frozen_r2.authorize
    frozen_r2.authorize = lambda _expected: authorization
    try:
        evidence = frozen_r2.compile_only(eligibility)
    finally:
        frozen_r2.authorize = original
    if evidence.get("authorization") != authorization or evidence.get("source_sha256") != SOURCE_SHA256:
        raise RuntimeError("r2a_evidence_binding")
    evidence["authorization_revision"] = "R2A"
    evidence["frozen_r2_backend_sha256"] = file_sha(Path(frozen_r2.__file__))
    return evidence
