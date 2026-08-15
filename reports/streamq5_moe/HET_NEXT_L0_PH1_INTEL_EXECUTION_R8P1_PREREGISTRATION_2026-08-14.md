# PH1 Intel execution R8P1 — invocation and transaction repair

Date: 2026-08-14

R8P1 supersedes the frozen R8P source package only for the three blockers in independent audit SHA-256 `60518c8999e35e9a09e873c398c4cb18c15a9b5b6e6f0e4972c034f36b3e5a37`. It changes no runtime version, source tensor, input, codec, control, numerical oracle, threshold, physical backend, or claim.

## Exact closed command

After a new independent source GO, the only eligible preflight invocation is the exact Windows command line represented by this six-element vector:

1. `C:\Users\de_do\Documents\ChatGPT\New project\.venv\Scripts\python.exe`
2. `-I`
3. `-B`
4. `C:\Users\de_do\Documents\ChatGPT\New project\scripts\streamq5_moe\preflight_het_next_l0_ph1_intel_execution_r8p1.py`
5. `--ack`
6. `PH1_INTEL_EXECUTION_R8P1_EXACT_FULL_INVOCATION_CPU_PREPARATION_CLOSED`

The preflight must retain and exactly validate `GetCommandLineW`, the independent `CommandLineToArgvW` parse, `sys.orig_argv`, `sys.argv`, resolved executable/script paths, absolute direct-script entry, and absence of every extra interpreter or application argument. It rejects `-c`, `-m`, import trampolines, reordered flags, alternate scripts/tokens and extras.

The independent verifier has its own exact four-element command vector: the same interpreter, `-I`, `-B`, and its absolute verifier path, with no application arguments.

## Unchanged preparation gates

R8P1 retains the exact CPython/pyvenv/psutil 7.2.2/NumPy 2.2.6 identities, 17+899 hashed non-cache RECORD-file checks, 16-GiB start gate, three records, D2 input, LUT, 22 controls, five BF16 stage arrays, and canonical preparation SHA-256 `f5a15db125c7a69357574111bd9549c36ae74b67af12205fc71a99a4c8962a49` from R8P.

The immutable R7D1 failure tree must contain exactly one attempt directory and exactly one 931-byte `failure.json`, SHA-256 `88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0`. Its eight-key schema, covered-stage list, kind, status, stage, error, `device_opened=false`, disposition and traceback markers are independently checked. Missing, extra, wrong-size/hash, non-Boolean, wrong-disposition and wrong-stage TEMP fixtures must all fail.

## Result transaction

R8P1 publishes exactly `result`, `manifest`, then `commit` using exclusive fsynced temporary files and create-new hard-link promotion. The manifest freezes the result byte count and SHA; the commit freezes both result and manifest SHAs. This is the non-tautological stored-result binding.

Before execution, all R8/R8P1 output, verification, failure, quarantine and temporary paths must be absent except the required immutable R7D1 failure. The actual transaction helpers are TEMP-tested for existing destination, stale temp, hard-link failure, post-link cleanup interruption, partial quarantine and repeated verifier execution. The verifier independently requires the exact three-file committed topology and zero temp/failure/quarantine/unexpected R8 sidecars.

## Authorization and claim boundary

The R8P1 lock is closed (`execution_open=false`, `audit_token=PENDING`). This document authorizes no preflight until source audit GO and no compiler, OpenCL or device action under any circumstance. R8P1 remains CPU preparation/runtime evidence only. A later fresh R8A still requires independently verified R8P1 PASS evidence and a separate authorization audit. The narrow possible physical claim remains one real expert/input Intel correctness component only.
