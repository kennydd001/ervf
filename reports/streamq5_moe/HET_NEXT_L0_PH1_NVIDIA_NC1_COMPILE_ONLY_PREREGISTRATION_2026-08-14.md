# PH1 NVIDIA NC1 compile-only preregistration

Date: 2026-08-14  
Status: **design frozen; implementation, preflight, compiler and execution closed**.

## Claim and immutable input

NC1 can establish only that the pinned NVRTC 13.3 compiler accepts one exact CUDA translation unit under one exact option vector and returns independently verified build-log, PTX and CUBIN bytes. It makes no numerical, performance, device-capability, CUDA-Driver or heterogeneous-execution claim.

The sole compilation input is `scripts/streamq5_moe/het_next_l0_ph1_nvidia_n5_kernels.cu`, 6,173 bytes, SHA-256 `9f369ab3621c6d56b2a3597bca59c25be8d15e7ac3a2a150d916d6695623a781`. It contains exactly two CUDA entrypoints, in source order:

1. `q5_linear`;
2. `bf16_lut_activation`.

The later physical experiment would issue four launches (`q5_linear:gate`, `q5_linear:up`, `bf16_lut_activation`, `q5_linear:down`). Those launches are outside NC1. NC1 never describes them as four source entrypoints and never launches either entrypoint.

Exactly one NVRTC program is compiled with this ordered option vector and no other option:

1. `--std=c++17`
2. `--fmad=true`
3. `--prec-div=true`
4. `--prec-sqrt=true`
5. `--ftz=false`
6. `--gpu-architecture=sm_120`
7. `--device-as-default-execution-space`

`--use_fast_math`, `--ftz=true`, option reordering and implicit extra options are forbidden.

## Compiler-only toolchain boundary

The only permitted non-system compiler modules are:

- `.venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc64_130_0.dll`, 101,385,328 bytes, SHA-256 `c7af6b5dbd001852d1b4a18effc6fbcfc94787eddadffea629a8333cb25b05fe`;
- `.venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc-builtins64_133.dll`, 6,684,784 bytes, SHA-256 `82c703802846329d3bab3d8df06f8c956516a0eeec568033092d6c0a69b2733a`;
- `.venv/Lib/site-packages/nvidia/cu13/include/nvrtc.h`, 57,749 bytes, SHA-256 `316a1375c18c69c5f1857dfc154c47a58a6795ffe462d2fcb50f5272ea472d21`.

The future runner uses the absolute NVRTC path with cdecl `ctypes.CDLL`, `winmode=0x1100`, while the exact containing directory is held by `os.add_dll_directory`. It records the resolved module path from the loaded handle and process-module snapshots before load, after compile and after destroy. The after-compile snapshot must contain exactly the pinned NVRTC and builtins paths among non-system CUDA/compiler modules, with matching sizes and hashes. `nvcuda.dll`, cudart, CuPy and every CUDA Driver/runtime/device module or API are forbidden and absent. No `nvcuda.dll` binding is part of NC1.

## Exact ten-row state machine

Ledger operations 0 through 9 are exactly:

`nvrtcVersion`, `nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetProgramLogSize`, `nvrtcGetProgramLog`, `nvrtcGetPTXSize`, `nvrtcGetPTX`, `nvrtcGetCUBINSize`, `nvrtcGetCUBIN`, `nvrtcDestroyProgram`.

Every row exists. Before an invoked call it is appended with `attempted=true`; normal return records its exact integer code, and a host exception records `code="ctypes_exception"` plus bounded exception text. Skipped rows have `attempted=false`, `code="not_attempted"`, null sizes and the last known program identity. A nonnull program is registered immediately, before its create status is adjudicated, and row 9 attempts destroy exactly once on every later path. A null program makes row 9 `not_attempted`. Destroy is always last; its post-call handle value must be null on code zero.

The transition rules are frozen:

- version failure: rows 1-9 are `not_attempted`;
- create failure/null/exception: rows 2-8 are `not_attempted`; row 9 is attempted only for a returned nonnull program;
- compile code zero: rows 3-8 proceed in order until the first error or invalid size/content, after which the remaining retrieval rows are `not_attempted`; row 9 is still attempted;
- compile nonzero: rows 3 and 4 are nevertheless attempted to retain the diagnostic log when possible; rows 5-8 are always `not_attempted`; row 9 is attempted;
- log-size failure makes row 4 and rows 5-8 `not_attempted`; log-read failure makes rows 5-8 `not_attempted`;
- PTX-size/read or CUBIN-size/read failure skips only later retrieval rows and never skips row 9.

The earliest operational or semantic failure is `primary_error`. A compile nonzero remains primary even if log retrieval or destroy later fails. Every later failure is an ordered `secondary_errors` row and cannot mask the primary. If no earlier error exists, destroy failure becomes primary. Success is exactly ten attempted rows, ten integer zero codes, one stable nonzero program identity for rows 1-9, and a null post-destroy handle.

The reported log size is at least one and equals retained `build.log` bytes. A one-byte log is valid only when it is exactly NUL; its logical message is empty but the artifact is nonempty. Larger logs end in exactly one retained terminal NUL. PTX and CUBIN sizes are each greater than one and exactly equal the corresponding size rows.

## Artifact gates

PTX must decode as ASCII/UTF-8, contain exactly one `.version`, `.target sm_120`, `.address_size 64`, and exactly the two `.entry` names above with no other entrypoint. `.ftz`, approximate arithmetic, fast-math evidence and unresolved external function declarations are rejected.

CUBIN must be a nonempty little-endian 64-bit ELF with valid bounded section/string/symbol tables. Its named CUDA kernel-symbol set must contain `q5_linear` and `bf16_lut_activation` and no additional kernel-entry symbol. NC1 claims only that NVRTC returned this ELF under the recorded `sm_120` option; it does not infer an architecture from ELF magic alone and performs no disassembly.

## Authorization and exclusions

A later open lock must bind the exact ACK `ACK_HET_NEXT_L0_PH1_NVIDIA_NC1_COMPILE_ONLY_ONCE`, its independent audit token and every implementation/preflight/verifier hash. Authorization, lock drift, interpreter and clean-topology checks happen before source read, recovery, DLL load or filesystem mutation. Wrong ACK, closed state or hash drift exits with no file, directory, quarantine or failure write.

Official shards, D2, CPU-stage/LUT data, models, tokenizers, Torch, Transformers, Safetensors, CuPy and all physical artifacts are forbidden. Counters must prove `payload_bytes_read=0`, `real_driver_calls=0`, `real_device_calls=0`. Only the 6,173-byte CUDA source may be opened after authorization.

No implementation, preflight, compiler or device call is authorized by this preregistration.

