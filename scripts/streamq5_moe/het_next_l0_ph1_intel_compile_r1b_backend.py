#!/usr/bin/env python3
"""PH1-R1B authorization-only wrapper; compile implementation remains frozen R1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import het_next_l0_ph1_intel_compile_r1a_backend as r1a


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1b_lock.json"
R1A_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1a_lock.json"
R1_PASS = REPORTS / "het_next_l0_ph1_intel_compile_r1_preflight_result.json"
ACK = "PH1_INTEL_COMPILE_R1B_AFTER_PREFLIGHT_PASS_AND_INDEPENDENT_FINAL_AUDIT_GO"
SRC = r1a.SRC
SOURCE_SHA256 = r1a.SOURCE_SHA256
OPTIONS = r1a.OPTIONS
CompileFailure = r1a.CompileFailure


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def authorize(expected: dict) -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    passed = json.loads(R1_PASS.read_text(encoding="utf-8"))
    if not (
        lock.get("kind") == "het_next_l0_ph1_intel_compile_r1b_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == ACK
        and lock.get("backend_sha256") == file_sha(Path(__file__))
        and lock.get("r1a_backend_sha256") == file_sha(Path(r1a.__file__))
        and lock.get("r1a_lock_sha256") == file_sha(R1A_LOCK)
        and lock.get("r1_preflight_pass_sha256") == file_sha(R1_PASS)
        and lock.get("source_sha256") == SOURCE_SHA256
        and passed.get("pass") is True
        and passed.get("passed") == passed.get("total") == 8
        and passed.get("device_calls") == passed.get("compiler_calls") == passed.get("payload_reads") == 0
        and lock.get("cpu_commit_sha256") == expected.get("cpu_commit_sha256")
        and lock.get("cpu_verification_sha256") == expected.get("cpu_verification_sha256")
        and lock.get("prior_audit_sha256") == expected.get("prior_audit_sha256")
    ):
        raise RuntimeError("compile_r1b_authorization")
    return {"lock_sha256": file_sha(LOCK), "r1a_lock_sha256": file_sha(R1A_LOCK), "r1_preflight_pass_sha256": file_sha(R1_PASS), "audit_token": ACK}


def compile_only(eligibility: dict) -> dict:
    authorization = authorize(eligibility)
    original = r1a.authorize
    r1a.authorize = lambda _expected: authorization
    try:
        evidence = r1a.compile_only(eligibility)
    finally:
        r1a.authorize = original
    if evidence.get("authorization") != authorization:
        raise RuntimeError("r1b_authorization_evidence")
    evidence["authorization_revision"] = "R1B"
    evidence["frozen_r1a_backend_sha256"] = file_sha(Path(r1a.__file__))
    return evidence
