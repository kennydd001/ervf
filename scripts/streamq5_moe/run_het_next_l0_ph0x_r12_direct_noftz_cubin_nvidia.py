from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import het_next_l0_ph0r3_common as common
import run_het_next_l0_ph0x_r9_direct_noftz_ptx_nvidia as r9
import run_het_next_l0_ph0x_r10_direct_noftz_ptx_provenance_repair as r10


RUN = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r12_direct_noftz_cubin_nvidia"
RESULT = RUN / "ph0x_r12_result.json"
PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_L0_PH0X_R12_DIRECT_NOFTZ_CUBIN_NVIDIA_COMPLETION_PREREGISTRATION_2026-08-13.md"
CUBIN = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r11_direct_nvrtc_noftz.cubin"
R10_RESULT = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r10_direct_noftz_ptx_provenance_repair/ph0x_r10_result.json"
R11_RESULT = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r11_direct_nvrtc_noftz_cubin_diagnostic.json"
R11_SCRIPT = ROOT / "scripts/streamq5_moe/diagnose_het_next_l0_ph0x_r11_direct_nvrtc_noftz_cubin.py"
R11_PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_L0_PH0X_R11_DIRECT_NVRTC_NOFTZ_CUBIN_DIAGNOSTIC_PREREGISTRATION_2026-08-13.md"
EXPECTED = {
    Path(r10.__file__).resolve(): "43e1be0427cd9fc689407eb6ab31dfffa59b6543d4f7f7362a63d62e04e6816c",
    r10.PREREG: "888b4c4d5f9d52e6b452fb55d1a3550beacdc589e90bd2973fd07ce3e05b118b",
    R10_RESULT: "14eb1b20b8b3f077fe5bcd73e652fe0aa4b2b6233530b637d33d73388977e51e",
    R11_RESULT: "21e2d57e85a3089cfa1c387827b636560de14c017fe316a8e7f9bf4de45bda25",
    R11_SCRIPT: "7ff376ab67a150a884f7c08cb374cdc1c75aa57c0ea86b31249712bcb60ae946",
    R11_PREREG: "d563f052d3695183670ca73929a168067e093e2c0512920bfce917326a2dd167",
    CUBIN: "660c22aec2574f12c15d8eed757433d0c9a30a1146fd27957adc96dcea6aaf57",
}


def predevice_gate():
    observed = {str(path): common.file_digest(path) for path in EXPECTED}
    if any(observed[str(path)] != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("r12_dependency_hash_drift")
    inherited, prior = r10.expanded_predevice_gate()
    r10_result = json.loads(R10_RESULT.read_text(encoding="utf-8"))
    r11 = json.loads(R11_RESULT.read_text(encoding="utf-8"))
    if not (
        r10_result.get("status") == "exploratory_direct_noftz_nvidia_failure"
        and "CUDA_ERROR_UNSUPPORTED_PTX_VERSION" in r10_result.get("error", "")
        and r10_result.get("intel_reexecuted") is False
        and r10_result.get("source_compilation_performed") is False
        and r11.get("diagnostic_pass") is True
        and r11.get("module_loaded") is False
        and r11.get("kernel_launched") is False
        and r11.get("compile", {}).get("sha256") == EXPECTED[CUBIN]
        and r11.get("compile", {}).get("bytes") == 62319
        and r11.get("compile", {}).get("elf_magic") == "7f454c46"
        and r11.get("cleanup", {}).get("error") is None
    ):
        raise RuntimeError("r10_r11_evidence_gate")
    inherited.update(observed)
    return inherited, prior


def cubin_run(record: bytes, input_bytes: bytes):
    import cupy as cp

    original = cp.RawModule
    call_count = 0

    def factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if args or kwargs != {
            "code": r9.r5.r4.CUDA_SOURCE,
            "backend": "nvrtc",
            "options": ("--std=c++17", "--fmad=true", "--prec-div=true", "--prec-sqrt=true", "--ftz=false"),
            "name_expressions": ("ph0",),
        }:
            raise RuntimeError("rawmodule_intercept_contract")
        return original(path=str(CUBIN))

    cp.RawModule = factory
    try:
        try:
            evidence = r9.r7.run_nvidia(record, input_bytes)
        except r9.r6.NvidiaRunFailure as exc:
            normalize(exc.evidence)
            exc.evidence["rawmodule_intercept_calls"] = call_count
            raise
        normalize(evidence)
        evidence["rawmodule_intercept_calls"] = call_count
        if call_count != 1:
            raise r9.r6.NvidiaRunFailure("rawmodule_intercept_cardinality", evidence)
        return evidence
    finally:
        cp.RawModule = original


def normalize(evidence):
    ledger = evidence.get("ledger", [])
    expected = {"op": "compile", "source_sha256": r9.r5.EXPECTED_CUDA_SOURCE_SHA, "success": True}
    if ledger and ledger[0] == expected:
        ledger[0] = {"op": "cubin_load", "cubin_sha256": EXPECTED[CUBIN], "cubin_bytes": 62319, "success": True}


def validate(ledger):
    expected = {"op": "cubin_load", "cubin_sha256": EXPECTED[CUBIN], "cubin_bytes": 62319, "success": True}
    if not ledger or ledger[0] != expected:
        raise RuntimeError("r12_cubin_ledger")
    surrogate = [{"op": "compile", "source_sha256": r9.r5.EXPECTED_CUDA_SOURCE_SHA, "success": True}] + ledger[1:]
    result = r9.r6.validate_success_ledger(surrogate)
    # The inherited R9 adjudicator consumes the generic load-exact field under
    # its historical name; R12 additionally records the precise cubin meaning.
    result.update({"ptx_load_exact": True, "cubin_load_exact": True, "cubin_sha256": EXPECTED[CUBIN]})
    return result


def main() -> int:
    if RUN.exists():
        raise FileExistsError(RUN)
    inherited, prior = predevice_gate()
    original_gate, original_run, original_result, original_prereg = r9.predevice_gate, r9.RUN, r9.RESULT, r9.PREREG
    original_file, original_runner, original_validator = r9.__file__, r9.run_nvidia, r9.validate_ledger
    original_writer = common.write_atomic_new

    def gate():
        return inherited, prior

    def writer(path, data):
        value = json.loads(data)
        value["kind"] = "het_next_l0_ph0x_r12_direct_noftz_cubin_nvidia"
        value["source_compilation_performed"] = False
        value.setdefault("bindings", {})["r12_prereg_sha256"] = common.file_digest(PREREG)
        value["bindings"]["r12_runner_sha256"] = common.file_digest(Path(__file__))
        value["bindings"]["cubin_sha256"] = EXPECTED[CUBIN]
        original_writer(path, common.canonical(value))

    r9.predevice_gate, r9.RUN, r9.RESULT, r9.PREREG = gate, RUN, RESULT, PREREG
    r9.__file__, r9.run_nvidia, r9.validate_ledger = __file__, cubin_run, validate
    common.write_atomic_new = writer
    try:
        return r9.main()
    finally:
        common.write_atomic_new = original_writer
        r9.__file__, r9.run_nvidia, r9.validate_ledger = original_file, original_runner, original_validator
        r9.predevice_gate, r9.RUN, r9.RESULT, r9.PREREG = original_gate, original_run, original_result, original_prereg


if __name__ == "__main__":
    raise SystemExit(main())
