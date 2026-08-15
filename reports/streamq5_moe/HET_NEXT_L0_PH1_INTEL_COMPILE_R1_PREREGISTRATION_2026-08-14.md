# HET-NEXT-L0 PH1 Intel compile-only R1 — preregistration

Datum: 2026-08-14  
Status: **execution closed; source-audit required; no preflight/compiler/device call authorized by this document**.

## Narrow objective and claim boundary

R1 asks only whether the already frozen PH1 Intel OpenCL-C source can be compiled once on the exact local Intel Arc device into one nonempty program binary. It does not load checkpoint/D2/Q5 payloads; does not create a queue, kernel, event, USM allocation or OpenCL buffer; and does not launch or time a kernel. A positive result is merely compile eligibility for a separately preregistered physical correctness phase.

This revision supersedes R0 only for the four blockers in the immutable independent audit
`HET_NEXT_L0_PH1_INTEL_COMPILE_R0_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md`, SHA-256
`ad1151b2a0a907e99ab0a99a6ac1b426587a14549fc4282821966f912544a841`.

## Frozen source repair

The kernel source is reconstructed from the immutable R0 backend SHA-256
`1c70d4248bdf64404589916a6be624594e8343442a64c57e926e52926f51ceac` with exactly these textual edits:

1. `cl_intel_required_sub_group_size` becomes canonical `cl_intel_required_subgroup_size`;
2. the entire `cl_khr_int64` extension pragma is removed.

No other kernel byte changes are allowed. The resulting UTF-8 source is 7,909 bytes, SHA-256
`06be3a9ba863d5f01d4025dc8d5e5679cdcc9827c13db5663c149227f3254528`.
The build options remain exactly `-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt`.

## Static preflight gates

Before any physical call, the separately hash-bound preflight must pass every conjunction:

- re-extract R0 `SRC` through Python AST and reproduce the exact R1 source hash/byte count;
- validate canonical pragma presence, forbidden pragma absence, all four entrypoints, width-8 geometry, buffers and exact `multiply_bf16_exact` source body;
- run a Python bit-level emulator of the source routine against the frozen signed-zero, subnormal, normal and overflow-safe test vectors;
- prove targeted pragma, reduction, entrypoint and tie-round source mutations are rejected;
- AST-audit the complete reachable `compile_only` backend surface and require exactly the ten allowed OpenCL functions; forbid queue/kernel/event/memory/USM/copy/payload/launch APIs;
- scan backend and runner imports/constants for model, shard, D2, Q5, safetensors, mmap, torch, transformers and CuPy payload paths;
- verify its own SHA plus backend, runner, preregistration, source, prior-audit and eligibility hashes against the closed lock;
- execute the actual production create-new/recovery functions in a temporary directory, including valid commit, stale temp quarantine, corrupt final quarantine, and immutable failure disposition;
- require the physical output and preflight-result paths to be absent.

Any failed conjunction keeps execution closed.

## Physical compile-only gate

After a new independent source audit and a separately frozen open authorization, the runner may make exactly one attempt. Before loading `OpenCL.dll`, it rehashes the lock, runner, backend, preflight, preregistration, prior audit and inherited CPU evidence.

Allowed device operations are restricted to platform/device enumeration and identity, context creation, program creation from source, program build, build-log query, program-device-count query, program-binary-size query, program-binary read, program release and context release.

The positive conjunction is:

- exactly one matching Intel Arc device: name `Intel(R) Arc(TM) Pro 140T GPU (32GB)`, vendor `Intel(R) Corporation`, driver `32.0.101.8517`, PCI `0000:00:02.0`, and `cl_intel_unified_shared_memory` present;
- build return code zero with exact frozen options;
- `CL_PROGRAM_NUM_DEVICES == 1`;
- exactly one queried binary size, strictly greater than zero;
- read byte length equals that queried size and SHA-256 is computed over exactly those bytes;
- raw source, build log and binary are retained;
- program and context release are both attempted, cleanup has no error;
- all explicit counters for payload, queue, kernel, event, memory object, allocation and launch remain zero.

The runner may set `compile_positive=true` only after all of these gates are independently true.

## Immutable lifecycle

The output path is create-new. A valid existing commit returns `already_complete` and is never changed. Before any device open, any corrupt final bundle or stale `.inprogress` directory is moved with write-through semantics to a unique quarantine directory, gets immutable recovery evidence, and aborts that invocation.

After device open, every exception during capture, source/log/binary/result writes, manifest creation, commit creation, verification or promotion moves the whole partial/final attempt into a unique failed-attempt directory and appends create-new failure evidence. The manifest binds every pre-commit file; commit is written last; the complete bundle is independently reparsed before and after atomic write-through promotion. No retry or outcome-based retuning is allowed.

## Frozen inherited evidence

- CPU R2 commit: `f3677e9610bea03649fec172b97c0c314f2f2e4c0d40bf9d864df0ec88a44f06`;
- CPU independent verification: `1c7f2772fb637485020be00f74b6f9295a18ec3d7d10af0587ea350e8756cbc8`, `pass=true`;
- physical contract: `7097a304eb6cd082367472cbc4c84ff9792414f3dd67e2590ba55b61dac3e981`;
- NVIDIA context addendum: `dde29c369c5218f5cca3ed12248979a8c03c95b51e8b433f65175750d74d695c`.

R1 contains no authorization to execute its preflight or physical compile.
