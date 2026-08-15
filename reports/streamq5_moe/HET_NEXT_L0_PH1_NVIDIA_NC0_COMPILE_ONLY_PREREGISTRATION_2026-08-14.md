# PH1 NVIDIA NC0 compile-only preregistration

Status: **design frozen; implementation and execution closed**.

## Objective and boundary

NC0 asks only whether the exact frozen four-kernel CUDA translation unit can be compiled by the pinned NVRTC 13.3 toolchain into nonempty PTX and CUBIN with complete, independently verifiable provenance. It makes no numerical, performance, device-capability, or heterogeneous-execution claim.

The only candidate source is `scripts/streamq5_moe/het_next_l0_ph1_nvidia_n5_kernels.cu`, SHA-256 `9f369ab3621c6d56b2a3597bca59c25be8d15e7ac3a2a150d916d6695623a781`. It is opened once after authorization and retained byte-for-byte as `source.cu`.

Exactly one NVRTC program is created. Exact options, in order:

1. `--std=c++17`
2. `--fmad=true`
3. `--prec-div=true`
4. `--prec-sqrt=true`
5. `--ftz=false`
6. `--gpu-architecture=sm_120`
7. `--device-as-default-execution-space`

`--use_fast_math`, `--ftz=true`, extra options and option reordering are forbidden.

## Frozen ten-operation ledger

The intended order is exactly: `nvrtcVersion`, `nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetProgramLogSize`, `nvrtcGetProgramLog`, `nvrtcGetPTXSize`, `nvrtcGetPTX`, `nvrtcGetCUBINSize`, `nvrtcGetCUBIN`, `nvrtcDestroyProgram`.

Every row records sequence, attempted, return code or `ctypes_exception`/`not_attempted`, program identity, and relevant byte count. A host exception at an invoked API remains `attempted=true`; all later retrieval calls are explicit `not_attempted`. If a non-null program was returned, destroy is attempted exactly once on every path and is the last row. A null program makes destroy explicitly `not_attempted`.

Success requires NVRTC version 13.3; all ten calls attempted with code zero; one nonzero stable program identity through calls 2–10; destroy nulls the handle; nonempty build log artifact (a one-byte NUL-only log is permitted only when the reported log size is exactly one), PTX size greater than one, CUBIN size greater than zero, and CUBIN ELF magic.

## Absolute exclusions

NC0 must not import or open the official shard, D2 raw, CPU freeze, model/tokenizer, Torch, Transformers, Safetensors, CuPy, CUDA runtime, or any PH1 physical artifact. It must not load `nvcuda.dll`, enumerate a device, create/retain/push a context, load a module, allocate memory, create a stream/event, copy, memset, launch, synchronize, or query device memory. Optional `cuobjdump`/`nvdisasm` is outside NC0 and may only be introduced in a later separately authorized revision.

## Outputs and adjudication

Success is a create-new directory containing exactly `result.json`, `source.cu`, `build.log`, `ptx.bin`, `cubin.bin`, `manifest.json`, and commit-last `commit.json`. The manifest binds canonical names, byte counts and SHA-256 values. Independent verification must pass on the temporary bundle before promotion. Any compile, retrieval, verifier, transaction, or cleanup error yields one bounded create-new failure artifact after destroy disposition; it cannot be scientific negative evidence. Existing valid commit returns `already_complete`; stale/corrupt/partial state is quarantined and aborts without retry.

No compiler or device call is authorized by this document.
