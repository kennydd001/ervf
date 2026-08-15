# PH1 NVIDIA NC2 compile-only preregistration

Date: 2026-08-14  
Status: **design frozen; implementation, preflight, compiler, Driver and device execution closed**.

## Narrow claim and frozen source

NC2 can establish only that one pinned NVRTC 13.3 program accepts one exact CUDA source buffer under one exact option vector and returns independently verified raw log, PTX and CUBIN artifacts. It makes no numerical, performance, device-capability, CUDA-Driver or byte-repeatability claim.

The sole compiler input file is `scripts/streamq5_moe/het_next_l0_ph1_nvidia_n5_kernels.cu`, 6,173 bytes, SHA-256 `9f369ab3621c6d56b2a3597bca59c25be8d15e7ac3a2a150d916d6695623a781`. It has no NUL byte and exactly two CUDA entrypoints, in order: `q5_linear`, `bf16_lut_activation`. Four later physical launch uses—gate, up, activation, down—remain outside NC2.

The exact `nvrtcCreateProgram` operands are:

- `prog`: address of one zero-initialized pointer-width `nvrtcProgram` cell;
- `src`: a stable `ctypes.create_string_buffer(source_bytes + b"\0")`; buffer bytes are the exact 6,173 source bytes followed by exactly one NUL, with no embedded NUL;
- `name`: stable ASCII buffer `b"het_next_l0_ph1_nvidia_nc2_kernels.cu\0"`, exactly one terminal NUL and no embedded NUL;
- `numHeaders`: signed `c_int(0)`;
- `headers`: typed null `POINTER(c_char_p)()`;
- `includeNames`: typed null `POINTER(c_char_p)()`.

Exactly one program is compiled with, in this order, `--std=c++17`, `--fmad=true`, `--prec-div=true`, `--prec-sqrt=true`, `--ftz=false`, `--gpu-architecture=sm_120`, `--device-as-default-execution-space`. No extra/reordered option, fast math or FTZ is permitted.

## Exact cdecl ABI

Aliases are `Program=c_void_p`, `PProgram=POINTER(c_void_p)`, `PInt=POINTER(c_int)`, `PSize=POINTER(c_size_t)`, `PChar=POINTER(c_char)`, `PCharP=POINTER(c_char_p)`. All ten functions have `restype=c_int` and exact `argtypes`:

| sequence | function | argtypes |
|---:|---|---|
| 0 | `nvrtcVersion` | `[PInt,PInt]` |
| 1 | `nvrtcCreateProgram` | `[PProgram,c_char_p,c_char_p,c_int,PCharP,PCharP]` |
| 2 | `nvrtcCompileProgram` | `[Program,c_int,PCharP]` |
| 3 | `nvrtcGetProgramLogSize` | `[Program,PSize]` |
| 4 | `nvrtcGetProgramLog` | `[Program,PChar]` |
| 5 | `nvrtcGetPTXSize` | `[Program,PSize]` |
| 6 | `nvrtcGetPTX` | `[Program,PChar]` |
| 7 | `nvrtcGetCUBINSize` | `[Program,PSize]` |
| 8 | `nvrtcGetCUBIN` | `[Program,PChar]` |
| 9 | `nvrtcDestroyProgram` | `[PProgram]` |

The NC1 ten-row state machine and first-primary/later-secondary semantics remain unchanged and are directly bound. Compile nonzero still retrieves log size/log when possible, skips PTX/CUBIN, and destroys a nonnull program. Each row is present; invoked host exceptions remain `attempted=true`; skipped suffix rows are `attempted=false/code="not_attempted"`; destroy is last.

## Raw artifact canonicalization

The retained `build.log` length equals `nvrtcGetProgramLogSize`. It contains exactly one terminal NUL and no earlier NUL. A one-byte `b"\0"` log is valid and represents empty logical text; a size below one, non-NUL final byte, embedded NUL or second terminal NUL is invalid. Logical log text is UTF-8 decoding of `raw_log[:-1]`; the raw digest includes the NUL.

The retained `ptx.bin` length equals `nvrtcGetPTXSize`, is greater than one, contains exactly one terminal NUL and no earlier NUL. Logical PTX is strict UTF-8 decoding of `raw_ptx[:-1]`; every parser operates only on those logical bytes, while its size/digest/manifest cover all retained raw bytes including NUL. Missing, embedded and duplicate NUL are independently rejected even when the reported size is made consistent.

Logical PTX has exactly one `.version`, `.target sm_120`, `.address_size 64`, and exactly the entrypoint set above. FTZ, approximate/fast-math instructions and unresolved external functions are rejected. CUBIN is greater than one byte, bounded, little-endian ELF64 with valid bounded tables and both named CUDA kernel symbols and no additional kernel-entry symbol. This is an honest “NVRTC returned ELF under the recorded option” claim; no independent architecture or SASS claim is made.

## Toolchain and absolute exclusions

Direct identities remain: `nvrtc64_130_0.dll` 101,385,328 bytes/SHA `c7af6b5d…b05fe`; `nvrtc-builtins64_133.dll` 6,684,784 bytes/SHA `82c70380…b2733a`; `nvrtc.h` 57,749 bytes/SHA `316a1375…72d21`. Loading is absolute cdecl `CDLL(...,winmode=0x1100)` while the exact DLL-directory cookie is alive. Module snapshots prove the exact resolved NVRTC/builtins identities and reject nvcuda, cudart, CuPy or another non-system compiler module.

Official shards, D2, CPU stages/LUT, model/tokenizer, Torch, Transformers, Safetensors, CUDA Driver/runtime/device/context/module/memory/copy/launch/sync calls and physical artifacts are forbidden. Only the frozen CUDA source may be opened after authorization. Exact counters are `payload_open_attempts=0`, `payload_bytes_read=0`, `driver_load_attempts=0`, `driver_calls=0`, `device_calls=0`, `unexpected_filesystem_mutations=0`.

No implementation or execution is authorized by this document.

