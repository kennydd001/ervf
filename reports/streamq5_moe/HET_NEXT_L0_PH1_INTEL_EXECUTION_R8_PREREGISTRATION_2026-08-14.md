# PH1 Intel execution R8P — pinned-runtime preparation gate

Date: 2026-08-14

## Immutable prior outcome

R7D1 remains an infrastructure-negative attempt. Its sole 931-byte failure record has SHA-256 `88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0`, reports `ModuleNotFoundError: psutil`, and proves `device_opened=false`. It is neither deleted nor reclassified. This revision binds the independent runtime diagnosis SHA-256 `a7fcef86b8cee812643593ad38e201798df798f63e2258001e4599135ed719b7`.

## R8P, closed CPU preparation phase

The only eligible command after independent source approval is:

`.venv\Scripts\python.exe -I -B scripts\streamq5_moe\preflight_het_next_l0_ph1_intel_execution_r8.py --ack PH1_INTEL_EXECUTION_R8P_EXACT_VENV_CPU_PREPARATION_CLOSED`

R8P is not a device preflight. It deliberately rereads only the three frozen source slices and one D2 input slice needed to reproduce the existing CPU package. It must not load OpenCL, compile a kernel, allocate a device object, or observe device output.

Hard gates:

1. Exact CPython 3.12.10 venv executable path, 274,424-byte executable SHA, prefix/base-prefix, cache tag and Windows platform.
2. `sys.flags.isolated=no_user_site=dont_write_bytecode=1` and exact original flag order `-I -B`.
3. Exact `pyvenv.cfg`, psutil 7.2.2 Python/native/METADATA/RECORD files, and NumPy 2.2.6 Python/METADATA/RECORD files. Every hashed, non-bytecode row in both wheel RECORDs is reread and verified.
4. Live psutil telemetry succeeds and available RAM is at least 16 GiB before CPU preparation.
5. The independent and production CPU builders produce byte-identical 675,840-byte gate/up/down records, the exact 4,096-byte BF16 input, the exact 131,072-byte LUT, and exactly the same 22 safe predevice controls with zero device counters.
6. Independent width-8 integer-FMA replay produces the exact five frozen BF16 stage arrays and hashes. Those raw bytes must equal the immutable CPU safetensors evidence. Canonical preparation digest is `f5a15db125c7a69357574111bd9549c36ae74b67af12205fc71a99a4c8962a49`.
7. The result is create-new and atomically linked from an fsynced temporary file. The standalone R8P verifier must independently rerun the runtime, wheel, codec, control and numerical checks and reject all ten frozen result mutations.

R8P writes no result until all computations have completed. A failed gate is not authorization for physical execution.

## Future R8A, explicitly not open here

Only after an immutable R8P PASS and independent verification may a fresh R8A namespace be authored. R8A must bind both PASS artifacts, use the same exact `.venv\python -I -B` contract, retain the immutable R7D1 failure, and reconstruct the frozen R7D PASS9/PASS7/PASS18 gates plus the R7A-verifier-absence gate. It must never import or call R7D1 `authorize()`; R7D1 is correctly non-repeatable. It may delegate only to unchanged R7D/R7C2 science under fresh R8 lifecycle paths.

The current lock remains `execution_open=false`, `audit_token=PENDING`. No compiler, OpenCL, payload execution beyond the CPU R8P preparation, or device call is authorized by this document.

## Claim boundary

R8P can establish only runtime and CPU-preparation equivalence under NumPy 2.2.6/psutil 7.2.2. A later physical R8A PASS would retain the prior narrow claim: one real expert/input Intel correctness component only. It would not prove performance, a full expert/layer/model, generalization, or an industrial breakthrough.
