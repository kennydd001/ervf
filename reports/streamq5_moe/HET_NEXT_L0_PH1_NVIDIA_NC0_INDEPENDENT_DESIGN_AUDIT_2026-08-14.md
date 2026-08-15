# PH1 NVIDIA NC0 compile-only — independent design audit

Date: 2026-08-14  
Mode: design-only/read-only. No implementation import, preflight, payload, compiler, Driver or device call was performed.

## Verdict

**NO-GO for implementation from the frozen NC0 design.**

The compile-only boundary is scientifically appropriate and much smaller than the abandoned full-expert source line. However, the frozen documents contain one direct source contradiction and leave compiler dependency, failure-ledger, bootstrap and artifact semantics open in ways that would permit post-design choices.

## Frozen integrity and topology

The three handed-off hashes match exactly:

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC0 preregistration | 3,239 | `79e85a85704bf319654580a1848a22d12e6c624196641d7f8f26c2b353b6e4dd` |
| static-preflight/verifier design | 2,998 | `e1b01912ce97ed557d310407f1244c6d7e1e0c21539bb5a6dc958dc6a9c48f6b` |
| closed design lock | 2,210 | `f55ea82a445b699a97915befdc6e458b524ab8c5ec821e608aeb5f0dd4f2313a` |

All lock bindings rehash true, 9/9. The NC0 family contains only these three design artifacts; implementation, preflight and output artifacts are absent. The lock is correctly closed for implementation, preflight and compile.

## Blocking findings

### 1. The frozen source has two entrypoints, not four

The bound N5 CUDA source SHA `9f369ab3621c6d56b2a3597bca59c25be8d15e7ac3a2a150d916d6695623a781` contains exactly two `extern "C" __global__` entrypoints:

1. `q5_linear`;
2. `bf16_lut_activation`.

The preregistration calls it a “four-kernel CUDA translation unit,” and the independent-verifier design requires “exactly the four frozen entrypoints.” Four is the later physical **launch count**—gate, up, activation and down—not the compiled entrypoint count. As written, an honest verifier must reject the actual frozen source/PTX or invent two names after freeze.

The design must state exactly two compiled entrypoints and distinguish them from four later launches, which are outside NC0.

### 2. Fake-NVRTC fixtures contradict the no-candidate-import rule

The static-preflight design says it never imports the future compiler runner/backend, while also requiring executable fake-NVRTC fixtures “against the actual future compiler function.” Executing that function requires loading candidate code somewhere. The design does not define a separate pure compiler-contract module, an absolute-path isolated child, dependency bootstrapping, or which candidate top-level code is permitted to execute.

Freeze one non-circular architecture before implementation:

- a standalone compiler core with inert imports, absolute-path/hash-bound loading in a `-I -B` fixture child; or
- a source-extracted pure state machine whose exact AST/body is shared with production and independently proved identical.

The child must reject any real `CDLL`/NVRTC/Driver call and record an exact fake-call ledger. Merely monkeypatching an already imported backend is insufficient.

### 3. Exact NVRTC toolchain provenance omits the builtins dependency and loaded-module identity

The lock binds `nvrtc64_130_0.dll` and `nvrtc.h`, but the installed NVRTC toolchain also contains `nvrtc-builtins64_133.dll` (6,684,784 bytes), which NVRTC compilation depends on. It is neither bound nor covered by a load-time module-evidence contract.

The design also does not freeze safe Windows loading details: absolute `ctypes.CDLL` path, cdecl, `winmode`, the loaded handle’s resolved `GetModuleFileNameW` path/hash, allowed transitive compiler DLLs before/after compile, and the absence of `nvcuda.dll`/cudart/device modules. Binding only the requested DLL path does not prove what compiler modules were actually loaded.

NC0 needs direct SHA/size bindings for every required non-system NVRTC dependency and exact before/after loaded-module evidence. This remains compiler-only; it must not load Driver/runtime/device libraries.

### 4. Compile-error log and ten-row suffix semantics are ambiguous

The preregistration says that after an API failure all later retrieval calls are `not_attempted`. For `nvrtcCompileProgram` failure, however, the build log is the primary diagnostic and the prior compiler implementation deliberately called `nvrtcGetProgramLogSize` and `nvrtcGetProgramLog` before adjudicating the compile return code.

The frozen rule currently permits or implies two incompatible ledgers:

- compile nonzero → log size/log still attempted, then PTX/CUBIN not attempted, destroy attempted;
- compile nonzero → every later retrieval including log marked not attempted, destroy attempted.

Freeze the first behavior explicitly, including precedence when compile fails and log retrieval or destroy also fails. For every boundary, specify exact attempted/not-attempted suffix, stable/null program identity, pointer-after-destroy, primary error, secondary cleanup error and failure disposition. The fake matrix and independent failure verifier must use the same table.

### 5. PTX/CUBIN target proof is incomplete

The independent verifier checks entrypoints, forbidden PTX tokens and CUBIN ELF magic. For the stated `sm_120` compile claim it must also require the exact PTX target contract—at least `.target sm_120`, expected `.address_size 64`, exactly the two entrypoints and no unexpected entrypoints—and bind artifact byte counts to the corresponding NVRTC size rows.

ELF magic alone is too weak for CUBIN identity. Freeze the expected CUDA ELF machine and architecture/header fields that can be checked without disassembly. If an exact architecture field cannot be preregistered independently, narrow the claim to “NVRTC returned a nonempty ELF CUBIN under the recorded option vector” and do not imply independent target verification.

### 6. Failure artifact and transaction schemas are not exact enough

The documents require bounded create-new failure evidence, commit-last publication and several fault cases, but do not freeze:

- failure-root/attempt naming and the exact `failure.json` schema;
- artifact/failure byte caps and oversize-summary fields;
- `device_opened=false`, `driver_loaded=false`, `payload_bytes_read=0` and compiler-opened semantics;
- exact prelink/postlink/fsync recovery/quarantine dispositions;
- primary versus secondary writer/cleanup failure precedence;
- Windows durable-file mode and directory/promotion primitives;
- behavior when the failure writer itself leaves a temporary/orphan attempt.

The success bundle’s seven filenames are clear. The lifecycle needs an equally exact state machine and independent topology verifier before source implementation.

### 7. Authorization and no-payload/no-device proof need executable, phase-local gates

“Authorization before source read” is correct, but the design does not freeze exact ACK, wrong-ACK mutation-free behavior, runtime interpreter identity or how source/lock/implementation/verifier/preflight hashes are directly closed in later locks.

The no-payload/no-Driver assertion needs an actual file/module/call guard in the isolated fixtures, not only forbidden strings. Static tests should inject attempts to open D2/shard/CPU/model paths and to call `CDLL`/`WinDLL`/Driver symbols, and prove zero byte reads and zero real loader calls. This is especially important because the bound kernel source is the sole permitted post-authorization input.

## Sound elements to retain

- The claim boundary is honest: compile provenance only; no numerical, performance, device-capability or heterogeneous-execution claim.
- The exact seven ordered NVRTC options are appropriate and prohibit fast math/FTZ drift.
- Exactly one program, a ten-row ledger, mandatory destroy disposition and create-new seven-file output are good core invariants.
- D2, shard, CPU/model/tokenizer and all physical artifacts are explicitly out of scope.
- Independent precommit verification, valid-repeat handling and abort-on-stale/corrupt state are the right lifecycle shape.
- Disassemblers are correctly deferred to a separately preregistered revision.

## Required NC0-R1 design repairs

Before implementation:

1. correct “four kernels/entrypoints” to two entrypoints and four out-of-scope future launches;
2. freeze the isolated fake-compiler bootstrap and exact production-core equivalence mechanism;
3. bind `nvrtc-builtins64_133.dll` and exact loaded-module/safe-loader evidence;
4. publish the exact ten-row success/failure transition table, especially compile-error log retrieval and primary/secondary precedence;
5. freeze PTX `.target sm_120`/address-size/two-entrypoint checks and an honest CUBIN ELF target boundary;
6. freeze exact failure schema, caps, atomic publication/recovery and fault dispositions;
7. freeze phase-local authorization, runtime, no-payload and no-Driver mutation gates plus direct later-lock closure.

Implementation and execution remain closed.
