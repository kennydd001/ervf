# PH1 Intel execution R8P3 — independent frozen source audit

Date: 2026-08-14  
Scope: static/read-only audit before execution. No candidate import, preflight, payload read, compiler, OpenCL, or device call was performed.

## Verdict

**NO-GO for the final R8P3 freeze.**

The three intended R8P2 repairs are present and the corrected verifier no longer has the nested-reference runtime error. One new lifecycle regression remains: the frozen preregistration claims the production early-failure writer is TEMP-tested, but it is neither tested nor failure-safe in the candidate. This requires a minimal immutable R8P4/R8P3P repair; the dual identity, runtime, preparation and all scientific fields can remain unchanged.

## Final frozen identities

The earlier verifier `6f865d81…` and lock `11a9acbd…` were superseded before this verdict. This audit uses only the final set:

| Artifact | SHA-256 | Status |
|---|---|---|
| preflight | `152531edcdda542bd3fa94767e4dc3618e4d2559e4b2b767dd9b4cf8b320e7ac` | exact final freeze |
| independent verifier | `286f648e32901ed7cad3f37bddaf2725932266cf22dc919c2086bde9e41bdf61` | exact final freeze |
| preregistration | `eda72f9d8b83279d2be2d077f6f82f059c1d55a2b64f9d8e86043e10c6816493` | exact final freeze |
| closed lock | `5456c50d71073a1403fc1c2fecdc54dedb5da81a9c1ed8c97992cdc3aa26f8e8` | closed/PENDING |

All 27 chain hashes match the 49-key lock. The current lower-case R8 family contains only the four R8/R8P1/R8P2/R8P3 locks; every result, manifest, commit, verification, failure, quarantine and temp path is absent.

## Correct repairs

1. The preflight TEMP simulator calls the exact current `atomic_create`, `publish`, `verify_bundle`, `quarantine_core`, and cleanup helpers. Its seven exact nonempty keys cover clean commit, preserved repeat rejection, stale cleanup, pre-link failure, post-link interruption/recovery, partial-publication quarantine and immutable committed bytes (`preflight`, lines 66–128 and 175–180).
2. The independent verifier exercises its own writer for clean create, repeat preservation, stale cleanup, pre-link failure and post-link recovery, plus exact topology mutations (`verifier`, lines 68–109 and 148–152). Stored-result adjudication requires exact transaction keys and rejects empty, missing, extra and false transaction dictionaries (`verifier`, lines 131–141).
3. The candidate and verifier restore an AST/callgraph boundary over the current and inherited CPU bootstrap. The current source has no model/compiler/OpenCL/CUDA/device entrypoint; only `kernel32` and `shell32` command-line evidence calls are allowed (`preflight`, lines 130–145; `verifier`, lines 111–124).
4. The verifier constructs and validates its own live dual-identity record and executes 12 mutations through its current validator (`verifier`, lines 48–66 and 145–152).
5. The preregistration now correctly permits only CPU reading of the already frozen preparation slice and forbids model forward, compiler, OpenCL, CUDA and device actions. The result schema carries those exact Boolean boundaries.
6. The final verifier correctly uses `prior.prior.base` for the inherited independent runtime/preparation implementation; the superseded source would have raised on the missing `prior.base` attribute.

## Blocking lifecycle regression

The preregistration states that the production R8P3 “failure writer [is] exercised through the actual functions in TEMP.” It is not.

- `atomic_failure()` is defined at preflight lines 155–159 and uses the real global `FAILED` path; it has no injectable TEMP root.
- `transaction_simulation()` at lines 102–128 never calls `atomic_failure()`.
- R8P2's prior actual failure simulation and its `failure_simulation` result/check fields were removed from R8P3.
- The new writer creates the attempt directory and then calls `atomic_create()` without cleanup on writer failure. A pre-link error can leave an empty attempt directory; a post-link unlink error can leave a promoted `failure.json` plus `.inprogress.*`. The outer handler swallows the secondary exception, and future attempts skip evidence whenever `FAILED` already exists (`preflight`, lines 183–186).

Consequently the correct-token early-failure promise is not non-vacuously gated, and its implementation regressed from R8P2's cleanup behavior. This is a lifecycle/provenance blocker even though the nominal R8P3 success path is otherwise coherent.

## Minimal repair

A fresh revision should make `atomic_failure(stage, exc, identity, root=FAILED)` TEMP-injectable and use a `try` cleanup that removes an empty attempt after pre-link failure while retaining and explicitly adjudicating any promoted evidence after post-link cleanup failure. Its actual-function TEMP suite must require:

- exact bounded schema and cap;
- two distinct attempts with no overwrite;
- pre-link failure leaves no attempt/temp;
- post-link interruption is detected and leaves an explicitly valid/recoverable disposition;
- no bare, empty, multiple-unadjudicated or oversized evidence can pass;
- production and verifier result schemas bind exact failure-simulation keys and reject empty/missing/extra/false cases.

No dual-identity, runtime, wheel, RAM, source slice, control, stage digest, numerical or claim threshold should change.

