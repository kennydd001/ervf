from __future__ import annotations

import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import het_next_l0_ph0r3_common as common
import run_het_next_l0_ph0x_r5_nvidia_only_real_projection as r5
import run_het_next_l0_ph0x_r6_nvidia_only_ledger_repair as r6
import run_het_next_l0_ph0x_r7_nvidia_only_lifecycle_repair as r7


RUN = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r9_direct_noftz_ptx_nvidia"
RESULT = RUN / "ph0x_r9_result.json"
PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_L0_PH0X_R9_DIRECT_NOFTZ_PTX_NVIDIA_COMPLETION_PREREGISTRATION_2026-08-13.md"
PTX = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r8_direct_nvrtc_noftz.ptx"
R7_RESULT = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r7_nvidia_only_lifecycle_repair/ph0x_r7_result.json"
R8_RESULT = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r8_direct_nvrtc_noftz_diagnostic.json"
R8R1_RESULT = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r8r1_noftz_ptx_parser_correction.json"
EXPECTED_FILES = {
    Path(r7.__file__).resolve(): "1063011521414be1255840e226b40e1a65a9325e3702a7e9b0965488d146a445",
    R7_RESULT: "314e08fc907965cf13b2af110b6a45424a9ac75ec5ec429b8f7bc7bf99fdba53",
    R8_RESULT: "c5df7a09ea13e4c29caa0d9acf40120131ae6a45033e73126f8563180f005ff2",
    PTX: "ec4789735f548123be0df3c2ff20c3e05c7b3741d9ed5f00b7b51eaeaa8ca7ae",
    R8R1_RESULT: "171650e58abef1dd9224e3d2a6db1a0b74f56c99e3a0bf5887299d6d2b3713a0",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def predevice_gate() -> tuple[dict[str, str], dict[str, object]]:
    observed = {str(path): common.file_digest(path) for path in EXPECTED_FILES}
    if any(observed[str(path)] != expected for path, expected in EXPECTED_FILES.items()):
        raise RuntimeError("r9_dependency_hash_drift")
    bindings, prior_intel = r5.predevice_gate()
    r7_result = json.loads(R7_RESULT.read_text(encoding="utf-8"))
    if not (
        r7_result.get("status") == "exploratory_nvidia_completion_negative"
        and r7_result.get("positive") is False
        and r7_result.get("nvidia_comparison", {}).get("different_words") == 122
        and r7_result.get("nvidia_comparison", {}).get("output_sha256") == "6525b36b911003ae7e746e6fea1930af61128adfd3fbc41530b6da08d0689041"
        and r7_result.get("gates", {}).get("nvidia_exact") is False
        and all(value is True for key, value in r7_result.get("gates", {}).items() if key != "nvidia_exact")
        and r7_result.get("nvidia", {}).get("ledger", [])[-1] == {"cleanup_complete": True, "errors": []}
    ):
        raise RuntimeError("r7_negative_evidence_gate")
    r8 = json.loads(R8_RESULT.read_text(encoding="utf-8"))
    correction = json.loads(R8R1_RESULT.read_text(encoding="utf-8"))
    if not (
        r8.get("diagnostic_pass") is False
        and r8.get("compile", {}).get("ftz_modifier_count") == 0
        and r8.get("compile", {}).get("ptx_sha256") == EXPECTED_FILES[PTX]
        and r8.get("kernel_launched") is False
        and correction.get("pass") is True
        and correction.get("ptx_sha256") == EXPECTED_FILES[PTX]
        and [row.get("offset") for row in correction.get("shuffles", [])] == [4, 2, 1]
        and all(row.get("clamp_segment") == 6175 for row in correction.get("shuffles", []))
    ):
        raise RuntimeError("r8_noftz_evidence_gate")
    bindings.update(observed)
    return bindings, prior_intel


def normalize_ledger(evidence: dict[str, object]) -> None:
    ledger = evidence.get("ledger", [])
    expected_compile = {"op": "compile", "source_sha256": r5.EXPECTED_CUDA_SOURCE_SHA, "success": True}
    if ledger and ledger[0] == expected_compile:
        ledger[0] = {"op": "ptx_load", "ptx_sha256": EXPECTED_FILES[PTX], "ptx_bytes": PTX.stat().st_size, "success": True}


def run_nvidia(record: bytes, input_bytes: bytes) -> dict[str, object]:
    import cupy as cp

    original_factory = cp.RawModule
    call_count = 0

    def direct_ptx_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if args or kwargs != {
            "code": r5.r4.CUDA_SOURCE,
            "backend": "nvrtc",
            "options": ("--std=c++17", "--fmad=true", "--prec-div=true", "--prec-sqrt=true", "--ftz=false"),
            "name_expressions": ("ph0",),
        }:
            raise RuntimeError("rawmodule_intercept_contract")
        return original_factory(path=str(PTX))

    cp.RawModule = direct_ptx_factory
    try:
        try:
            evidence = r7.run_nvidia(record, input_bytes)
        except r6.NvidiaRunFailure as exc:
            normalize_ledger(exc.evidence)
            exc.evidence["rawmodule_intercept_calls"] = call_count
            raise
        normalize_ledger(evidence)
        evidence["rawmodule_intercept_calls"] = call_count
        if call_count != 1:
            raise r6.NvidiaRunFailure("rawmodule_intercept_cardinality", evidence)
        return evidence
    finally:
        cp.RawModule = original_factory


def validate_ledger(ledger: list[dict[str, object]]) -> dict[str, object]:
    if not ledger or ledger[0] != {"op": "ptx_load", "ptx_sha256": EXPECTED_FILES[PTX], "ptx_bytes": 133404, "success": True}:
        raise RuntimeError("r9_ptx_ledger")
    surrogate = [{"op": "compile", "source_sha256": r5.EXPECTED_CUDA_SOURCE_SHA, "success": True}] + ledger[1:]
    result = r6.validate_success_ledger(surrogate)
    result.update({"ptx_load_exact": True, "ptx_sha256": EXPECTED_FILES[PTX]})
    return result


def main() -> int:
    if RUN.exists():
        raise FileExistsError(RUN)
    bindings, prior = predevice_gate()
    RUN.mkdir(parents=True)
    result: dict[str, object] = {
        "kind": "het_next_l0_ph0x_r9_direct_noftz_ptx_nvidia",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {"prereg_sha256": common.file_digest(PREREG), "runner_sha256": common.file_digest(Path(__file__)), "dependencies": bindings, "ptx_sha256": EXPECTED_FILES[PTX]},
        "intel_reexecuted": False,
        "source_compilation_performed": False,
    }
    error = None
    try:
        source = common.read_exact(common.SHARD, common.SOURCE_OFFSET, common.SOURCE_BYTES)
        input_bytes = common.read_exact(common.D2, common.INPUT_OFFSET, common.INPUT_BYTES)
        if sha(input_bytes) != r5.EXPECTED_INPUT_SHA:
            raise RuntimeError("input_hash_drift")
        record, evidence = common.build_record(source)
        if evidence["record_sha256"] != r5.EXPECTED_RECORD_SHA:
            raise RuntimeError("record_hash_drift")
        common.safe_check(record, input_bytes)
        oracle = common.cpu_oracle(record, input_bytes)
        if sha(oracle.tobytes()) != r5.EXPECTED_CPU_SHA or oracle.tobytes().hex() != prior.get("cpu_output_hex"):
            raise RuntimeError("cpu_oracle_drift")
        nvidia = run_nvidia(record, input_bytes)
        result["nvidia"] = nvidia
        ledger_validation = validate_ledger(nvidia["ledger"])
        output = np.frombuffer(bytes.fromhex(nvidia["output_hex"]), "<u2")
        counters = np.frombuffer(bytes.fromhex(nvidia["counters_hex"]), "<u4")
        comparison = {"words": int(output.size), "different_words": int(np.count_nonzero(output != oracle)) if output.size == oracle.size else -1, "output_sha256": sha(output.tobytes()), "counters_all_one": bool(counters.size == common.ROWS and np.all(counters == 1)), "sentinel_overwritten": bool(output.size == common.ROWS and np.all(output != 0xFFFF))}
        gates = {"nvidia_exact": comparison["different_words"] == 0 and comparison["output_sha256"] == r5.EXPECTED_CPU_SHA, "nvidia_counters": comparison["counters_all_one"], "nvidia_sentinel": comparison["sentinel_overwritten"], "nvidia_identity": nvidia["identity"].get("count") == 1 and nvidia["identity"].get("name") == "NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU" and nvidia["identity"].get("pci") == "0000:01:00.0", "nvidia_ledger_exact": ledger_validation["ordered_exact"] and ledger_validation["ptx_load_exact"], "rawmodule_intercept_exact": nvidia.get("rawmodule_intercept_calls") == 1, "distinct_from_bound_intel": prior["intel"]["identity"]["pci"] != nvidia["identity"]["pci"]}
        result.update({"record_evidence": evidence, "cpu_output_sha256": r5.EXPECTED_CPU_SHA, "bound_intel_comparison": prior["intel_comparison"], "nvidia_comparison": comparison, "ledger_validation": ledger_validation, "gates": gates, "positive": all(gates.values())})
        result["status"] = "exploratory_direct_noftz_nvidia_positive" if result["positive"] else "exploratory_direct_noftz_nvidia_negative"
    except r6.NvidiaRunFailure as exc:
        error = str(exc)
        result.update({"nvidia_failure_evidence": exc.evidence, "positive": False, "status": "exploratory_direct_noftz_nvidia_failure", "error": error, "traceback": traceback.format_exc()})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result.update({"positive": False, "status": "exploratory_direct_noftz_nvidia_failure", "error": error, "traceback": traceback.format_exc()})
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    result["claim_boundary"] = "Direct-no-FTZ NVIDIA completion for one real projection/input, bound to prior Intel evidence; no full expert/layer/model/performance/concurrency/deployment/novelty/breakthrough claim."
    common.write_atomic_new(RESULT, common.canonical(result))
    print(json.dumps({"status": result["status"], "positive": result["positive"], "gates": result.get("gates"), "error": error}, indent=2))
    return 0 if result.get("positive") is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
