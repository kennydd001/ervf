#!/usr/bin/env python3
"""R2P1 static preflight with complete TEMP redirection and guaranteed restoration."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
REPORTS = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(SCRIPTS))
import preflight_het_next_l0_ph1_intel_compile_r2p as prior

PREFLIGHT = Path(__file__)
REVISION = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2P1_PREFLIGHT_REVISION_2026-08-14.md"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2p1_lock.json"
R2P_PREFLIGHT = SCRIPTS / "preflight_het_next_l0_ph1_intel_compile_r2p.py"
R2P_REVISION = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2P_PREFLIGHT_REVISION_2026-08-14.md"
R2P_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2p_lock.json"
R2P_FAILED_RESULT = REPORTS / "het_next_l0_ph1_intel_compile_r2p_static_preflight.json"
SOURCE_MODULE = SCRIPTS / "het_next_l0_ph1_intel_compile_r2_source.py"
BACKEND = SCRIPTS / "het_next_l0_ph1_intel_compile_r2_backend.py"
RUNNER = SCRIPTS / "run_het_next_l0_ph1_intel_compile_r2.py"
R2_PREREG = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2_PREREGISTRATION_2026-08-14.md"
R2_DESIGN = REPORTS / "HET_NEXT_L0_PH1_INTEL_COMPILE_R2_SOURCE_REVISION_2026-08-14.md"
R2_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r2_lock.json"
R1B_FAILURE = REPORTS / "het_next_l0_ph1_intel_compile_r1b_failed_attempts/attempt_failure_06df3c72c9c44379a04d39b43d301b53/failure.json"
OUT = REPORTS / "het_next_l0_ph1_intel_compile_r2p1"
RESULT = REPORTS / "het_next_l0_ph1_intel_compile_r2p1_static_preflight.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corrected_transaction_simulation() -> bool:
    spec = importlib.util.spec_from_file_location("ph1_r2_runner_static_p1", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    base = runner.base
    runner_globals = {name: getattr(runner, name) for name in ("OUT", "FAILED", "QUARANTINE")}
    base_globals = {name: getattr(base, name) for name in ("REPORTS", "OUT", "FAILED", "QUARANTINE", "verify_bundle")}
    try:
        with tempfile.TemporaryDirectory(prefix="ph1_r2p1_") as temporary:
            root = Path(temporary)
            out, failed, quarantine = root / "out", root / "failed", root / "quarantine"
            runner.OUT, runner.FAILED, runner.QUARANTINE = out, failed, quarantine
            base.REPORTS = root
            runner.configure_base()
            base.verify_bundle = runner.verify_bundle

            stale = root / "out.dead.inprogress"
            stale.mkdir()
            (stale / "partial").write_bytes(b"x")
            try:
                base.recover()
                return False
            except RuntimeError:
                pass
            if stale.exists() or len(list(quarantine.glob("stale_temp_*"))) != 1:
                return False

            attempt = root / "out.fresh.inprogress"
            attempt.mkdir()
            binary = b"R2P1-nonempty"
            compiled = {
                "source": prior.derive_r2(),
                "binary_hex": binary.hex(),
                "build_log_hex": b"static".hex(),
                "binary_nonempty": True,
                "queried_program_devices": 1,
                "declared_binary_bytes": len(binary),
                "read_binary_bytes": len(binary),
                "binary_sha256": hashlib.sha256(binary).hexdigest(),
                "cleanup_errors": [],
                "payload_read": False,
                "queues_created": 0,
                "kernels_created": 0,
                "events_created": 0,
                "memory_objects_created": 0,
                "allocations": 0,
                "kernels_launched": 0,
            }
            runner.build(attempt, {"static": True}, compiled)
            base.durable_move(attempt, out)
            if not runner.verify_bundle(out)["result"]["positive"]:
                return False
            if not base.recover()["already_complete"]:
                return False

            (out / "result.json").write_bytes(b"corrupt")
            try:
                base.recover()
                return False
            except RuntimeError:
                pass
            if out.exists() or len(list(quarantine.glob("corrupt_final_*"))) != 1:
                return False

            partial = root / "failed.inprogress"
            partial.mkdir()
            base.archive(failed, "attempt_failure", {"kind": "injected"}, partial)
            rows = list(failed.glob("attempt_failure_*"))
            return len(rows) == 1 and (rows[0] / "failure.json").is_file() and not partial.exists()
    finally:
        for name, value in runner_globals.items():
            setattr(runner, name, value)
        for name, value in base_globals.items():
            setattr(base, name, value)


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    failed = json.loads(R2P_FAILED_RESULT.read_text(encoding="utf-8"))
    observed = {
        "preflight_sha256": sha(PREFLIGHT),
        "revision_sha256": sha(REVISION),
        "r2p_preflight_sha256": sha(R2P_PREFLIGHT),
        "r2p_revision_sha256": sha(R2P_REVISION),
        "r2p_lock_sha256": sha(R2P_LOCK),
        "r2p_failed_result_sha256": sha(R2P_FAILED_RESULT),
        "source_module_sha256": sha(SOURCE_MODULE),
        "backend_sha256": sha(BACKEND),
        "runner_sha256": sha(RUNNER),
        "r2_prereg_sha256": sha(R2_PREREG),
        "r2_design_sha256": sha(R2_DESIGN),
        "r2_closed_lock_sha256": sha(R2_LOCK),
        "r1b_failure_sha256": sha(R1B_FAILURE),
        "source_sha256": hashlib.sha256(prior.derive_r2().encode()).hexdigest(),
    }
    tests = {
        "closed_self_bound_full_chain": lock.get("execution_open") is False and lock.get("audit_token") == "PENDING" and all(lock.get(key) == value for key, value in observed.items()),
        "prior_failure_exact_single_conjunct": failed.get("pass") is False and failed.get("passed") == 6 and failed.get("total") == 7 and failed.get("tests", {}).get("actual_r2_transaction_simulation") is False and sum(value is False for value in failed.get("tests", {}).values()) == 1,
        "source_contract": prior.source_contract(prior.derive_r2()),
        "executed_emulator_and_mutations": prior.emulator_and_mutations(),
        "ast_compile_transaction_contract": prior.ast_contract(),
        "corrected_actual_transaction_simulation": corrected_transaction_simulation(),
        "output_absent": not OUT.exists(),
        "result_absent": not RESULT.exists(),
    }
    result = {"kind": "het_next_l0_ph1_intel_compile_r2p1_static_preflight", "tests": tests, "pass": all(tests.values()), "passed": sum(tests.values()), "total": len(tests), "compiler_calls": 0, "device_calls": 0, "payload_reads": 0}
    if RESULT.exists():
        raise FileExistsError(RESULT)
    with RESULT.open("xb") as handle:
        handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
