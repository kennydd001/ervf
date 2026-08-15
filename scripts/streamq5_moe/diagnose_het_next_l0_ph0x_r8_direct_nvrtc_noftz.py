from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import het_next_l0_ph0r3_common as common
import diagnose_het_next_l0_ph0x_r4_cuda_compile_staging as r4


OUT = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r8_direct_nvrtc_noftz_diagnostic.json"
PTX_OUT = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r8_direct_nvrtc_noftz.ptx"
R7_RESULT = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r7_nvidia_only_lifecycle_repair/ph0x_r7_result.json"
EXPECTED_R7_SHA = "314e08fc907965cf13b2af110b6a45424a9ac75ec5ec429b8f7bc7bf99fdba53"
OPTIONS = (
    "--std=c++17",
    "--fmad=true",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
    "--gpu-architecture=compute_120",
    "--device-as-default-execution-space",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if OUT.exists() or PTX_OUT.exists():
        raise FileExistsError("R8 diagnostic output exists")
    result: dict[str, object] = {
        "kind": "ph0x_r8_direct_nvrtc_noftz_compile_diagnostic",
        "r7_result_sha256": common.file_digest(R7_RESULT),
        "cuda_source_sha256": sha(r4.CUDA_SOURCE.encode()),
        "options": list(OPTIONS),
        "kernel_launched": False,
        "device_allocations": 0,
        "h2d_calls": 0,
        "d2h_calls": 0,
    }
    program = None
    error = None
    try:
        if result["r7_result_sha256"] != EXPECTED_R7_SHA:
            raise RuntimeError("r7_result_hash_drift")
        from cupy_backends.cuda.libs import nvrtc

        program = nvrtc.createProgram(r4.CUDA_SOURCE, "ph0x_r8.cu", (), ())
        nvrtc.compileProgram(program, OPTIONS)
        ptx = nvrtc.getPTX(program)
        log = nvrtc.getProgramLog(program)
        text = ptx.decode("utf-8")
        evidence = {
            "ptx_bytes": len(ptx),
            "ptx_sha256": sha(ptx),
            "compile_log": log,
            "compile_log_sha256": sha(log.encode()),
            "ftz_modifier_count": text.count(".ftz"),
            "mul_f32_count": text.count("mul.f32"),
            "fma_rn_f32_count": text.count("fma.rn.f32"),
            "add_rn_f32_count": text.count("add.rn.f32"),
            "shuffle_width8_present": ", 8;" in text and "shfl.sync.down" in text,
        }
        PTX_OUT.write_bytes(ptx)
        evidence["ptx_output_sha256"] = common.file_digest(PTX_OUT)
        result["compile"] = evidence
        result["diagnostic_pass"] = bool(
            evidence["ftz_modifier_count"] == 0
            and evidence["mul_f32_count"] > 0
            and evidence["fma_rn_f32_count"] > 0
            and evidence["add_rn_f32_count"] > 0
            and evidence["shuffle_width8_present"]
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result.update({"diagnostic_pass": False, "error": error, "traceback": traceback.format_exc()})
    finally:
        destroy_error = None
        if program is not None:
            try:
                nvrtc.destroyProgram(program)
            except Exception as exc:
                destroy_error = str(exc)
        result["cleanup"] = {"program_destroy_attempted": program is not None, "error": destroy_error}
    common.write_atomic_new(OUT, common.canonical(result))
    print(json.dumps({"diagnostic_pass": result["diagnostic_pass"], "compile": result.get("compile"), "cleanup": result["cleanup"], "error": error}, indent=2))
    return 0 if result["diagnostic_pass"] and result["cleanup"]["error"] is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
