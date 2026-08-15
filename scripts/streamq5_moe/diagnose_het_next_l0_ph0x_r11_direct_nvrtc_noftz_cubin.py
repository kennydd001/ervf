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


OUT = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r11_direct_nvrtc_noftz_cubin_diagnostic.json"
CUBIN = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r11_direct_nvrtc_noftz.cubin"
R10 = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r10_direct_noftz_ptx_provenance_repair/ph0x_r10_result.json"
EXPECTED_R10_SHA = "14eb1b20b8b3f077fe5bcd73e652fe0aa4b2b6233530b637d33d73388977e51e"
EXPECTED_SOURCE_SHA = "3ede786f3e71b76ee74f2591bde4cbb317a94f05e84bfd3ef5d64c22f6ce8435"
OPTIONS = (
    "--std=c++17",
    "--fmad=true",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
    "--gpu-architecture=sm_120",
    "--device-as-default-execution-space",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if OUT.exists() or CUBIN.exists():
        raise FileExistsError("R11 output exists")
    result: dict[str, object] = {
        "kind": "ph0x_r11_direct_nvrtc_noftz_cubin_diagnostic",
        "r10_sha256": common.file_digest(R10),
        "cuda_source_sha256": sha(r4.CUDA_SOURCE.encode()),
        "options": list(OPTIONS),
        "module_loaded": False,
        "kernel_launched": False,
        "allocations": 0,
        "copy_calls": 0,
    }
    program = None
    error = None
    try:
        if result["r10_sha256"] != EXPECTED_R10_SHA or result["cuda_source_sha256"] != EXPECTED_SOURCE_SHA:
            raise RuntimeError("input_hash_drift")
        from cupy_backends.cuda.libs import nvrtc

        program = nvrtc.createProgram(r4.CUDA_SOURCE, "ph0x_r11.cu", (), ())
        nvrtc.compileProgram(program, OPTIONS)
        cubin = nvrtc.getCUBIN(program)
        log = nvrtc.getProgramLog(program)
        CUBIN.write_bytes(cubin)
        evidence = {
            "bytes": len(cubin),
            "sha256": sha(cubin),
            "elf_magic": cubin[:4].hex(),
            "compile_log": log,
            "compile_log_sha256": sha(log.encode()),
            "output_sha256": common.file_digest(CUBIN),
        }
        result["compile"] = evidence
        result["diagnostic_pass"] = bool(len(cubin) > 0 and cubin[:4] == b"\x7fELF" and evidence["sha256"] == evidence["output_sha256"])
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
