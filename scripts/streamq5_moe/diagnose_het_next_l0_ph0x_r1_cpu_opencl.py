from __future__ import annotations

import ctypes as C
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
for entry in (str(ROOT), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import het_next_l0_ph0r3_common as common
import het_next_l0_ph0r3_intel as ib
from run_het_next_l0_ph0x_exploratory_real_projection import INTEL_SOURCE


OUT = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r1_cpu_opencl_diagnostic.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compile_only() -> dict[str, object]:
    obj = ib.Intel()
    code = None
    log = b""
    cleanup: list[str] = []
    try:
        platform, device, identity = obj.select()
        error = C.c_int()
        props = (C.c_ssize_t * 3)(ib.CL_CONTEXT_PLATFORM, int(platform.value), 0)
        devices = (C.c_void_p * 1)(device.value)
        obj.context = obj.l.clCreateContext(props, 1, devices, None, None, C.byref(error))
        ib._check(error.value, "context")
        source = INTEL_SOURCE.encode()
        strings = (C.c_char_p * 1)(source)
        sizes = (C.c_size_t * 1)(len(source))
        obj.program = obj.l.clCreateProgramWithSource(obj.context, 1, strings, sizes, C.byref(error))
        ib._check(error.value, "program")
        code = int(obj.l.clBuildProgram(obj.program, 1, devices, b"-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt", None, None))
        length = C.c_size_t()
        obj.l.clGetProgramBuildInfo(obj.program, device, ib.CL_PROGRAM_BUILD_LOG, 0, None, C.byref(length))
        buffer = C.create_string_buffer(max(1, length.value))
        obj.l.clGetProgramBuildInfo(obj.program, device, ib.CL_PROGRAM_BUILD_LOG, length.value, buffer, None)
        log = bytes(buffer.raw[: length.value])
        return {"identity": identity, "build_code": code, "source_sha256": sha(source), "build_log": log.decode(errors="replace"), "build_log_sha256": sha(log)}
    finally:
        try:
            obj.close()
        except Exception as exc:
            cleanup.append(f"{type(exc).__name__}: {exc}")


def sensitivity() -> dict[str, object]:
    source = common.read_exact(common.SHARD, common.SOURCE_OFFSET, common.SOURCE_BYTES)
    record, _ = common.build_record(source)
    codes, scales = common.split_record(record)
    fields = common.unpack_fields(codes).reshape(-1)
    index = int(np.flatnonzero(fields != 15)[0])
    stored = int(fields[index])
    step = stored - 1 if stored > 15 else stored + 1
    scale_word = int(np.frombuffer(scales, "<u2")[0])
    scale = common.bf16_to_f32(np.asarray([scale_word], dtype=np.uint16))[0]
    q0, q1 = stored - 15, step - 15
    weight0 = int(common.f32_to_bf16(np.asarray([np.float32(q0) * scale]))[0])
    weight1 = int(common.f32_to_bf16(np.asarray([np.float32(q1) * scale]))[0])
    activation = np.zeros(common.COLS, dtype="<u2")
    activation[0] = 0x3B80
    original = np.zeros(common.ROWS, dtype="<u2")
    mutated = np.zeros(common.ROWS, dtype="<u2")
    original[0] = common.round_f32_bits_to_bf16(common.soft_fma_bits(weight0 << 16, 0x3B800000, 0))
    mutated[0] = common.round_f32_bits_to_bf16(common.soft_fma_bits(weight1 << 16, 0x3B800000, 0))
    return {
        "index": index,
        "row": index // common.COLS,
        "column": index % common.COLS,
        "stored": stored,
        "q": q0,
        "step_stored": step,
        "step_q": q1,
        "scale_word": scale_word,
        "weight_words": [weight0, weight1],
        "output_words": [int(original[0]), int(mutated[0])],
        "changed_words": int(np.count_nonzero(original != mutated)),
        "activation_sha256": sha(activation.tobytes()),
        "original_sha256": sha(original.tobytes()),
        "mutated_sha256": sha(mutated.tobytes()),
        "frozen_expected": {"activation": "2498a04e393ec5eb0ec88b7f098523dd5f3a1cbaf9803fa7ace4b4776c17f561", "original": "ca913e50693d83329869fb61dabb75467df7091d39e9a5dd9e17e8480bbeb9f6", "mutated": "1868bd78f7059362bed974138ae89c4efa7b930fdf3d07db11db6cd94677ee23"},
    }


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    value = {"kind": "ph0x_r1_cpu_opencl_diagnostic", "compile": compile_only(), "sensitivity": sensitivity(), "kernel_launched": False, "nvidia_opened": False}
    OUT.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
