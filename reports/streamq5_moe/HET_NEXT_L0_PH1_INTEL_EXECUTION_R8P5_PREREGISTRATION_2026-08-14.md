# PH1 Intel execution R8P5 — CPU-slice failure-stage provenance

Date: 2026-08-14  
Status: immutable closed preregistration; no execution authorization.

R8P5 supersedes R8P4 and binds its independent audit SHA-256 `a02e2e619c155d36d5c21fb010504e77fbe4526ca9e0cd846918c1156a9d9815`. It changes only CPU-slice stage provenance in result and failure evidence. All R8P4 dual-identity, runtime, RECORD, RAM, CPU preparation, control, stage-hash, transaction, failure-lifecycle, static-boundary and no-device gates remain unchanged.

The immutable state enum is exactly `not_started`, `started_not_completed`, or `completed`. It is initialized before the main try block, set to `started_not_completed` immediately before `preparation_summary()`, and set to `completed` immediately after it returns. Every failure path passes that exact state through `preserve_primary`, `atomic_failure`, and `failure_row`.

Two Booleans are derived, never supplied independently:

- `cpu_frozen_slice_read_started` is false only for `not_started` and true otherwise;
- `cpu_frozen_slice_read_completed` is true only for `completed`.

The actual production writer TEMP suite retains all five R8P4 lifecycle outcomes and adds a sixth exact outcome covering valid evidence for all three states and rejection of unknown state, wrong started Boolean, wrong completed Boolean, missing state, and extra state. The independent verifier reimplements the state/writer logic, repeats all states and mutations, requires the exact six-key failure-simulation dictionary, and rejects empty/missing/extra/false result variants.

Invalid application argv/token returns before filesystem mutation. A later source GO may authorize CPU-only reading of the frozen preparation slice. Model forward, compiler, OpenCL, CUDA and device actions remain forbidden. R8P5 is closed/PENDING and may not be executed before independent source audit.
