# PH1 Intel execution R5 — test-coverage-only lifecycle revision

Status: closed/PENDING. No preflight, payload, compiler, OpenCL load, or device call is authorized.

R5 binds the immutable R4 independent source-audit SHA-256 `29d418b30ceef6cae37292d5d00b368c533c94069ac39a80638be36dee9a7136`. It changes no Q5 codec rule, source record, input, LUT, kernel source/binary, launch geometry, buffer size, threshold, identity gate, or scientific claim. The claim remains one real expert/input Intel correctness component only.

The lifecycle delta is explicit and test-only:

- pre-call ownership placeholders and immediate non-null ownership are exercised for context, queue, program, each of four kernel positions, and host-USM;
- TEMP simulations inject post-return status failure for every create position and host allocation, and type/base/size attestation failure after non-null USM return; each applicable object must be released and the cleanup ledger must end with zero live resources;
- each of the seven create returns, fourteen allocation returns, and fourteen free rows is crosslinked by exact pointer/name to the promoted main-ledger object and release identity;
- the runner and independent verifier require exactly 95 ordered ownership rows, alignment 4096, `event_requested=false`, and `owned_before=true`, with mutations for missing/duplicate/aliased ownership, returned pointer, pending flag, alignment, event request, release ownership/order/status, arguments, launches, reads, resources, provenance, controls, outputs, and bundle files;
- production-path TEMP simulations cover authorization failure with no filesystem mutation, start-RAM failure, payload failure, ordinary post-device failure, primary device failure plus secondary telemetry error, oversized attempt quarantine with a bounded FAILED summary, and immutable already-complete semantics.

The 16 MiB cap applies normatively to `OUT` and `FAILED`. `QUAR` is explicitly forensic/development evidence outside the scientific/failure bundle and outside that cap. An oversized temp is measured before movement; its byte count and digest are retained in a capped FAILED summary. The physical temp may be quarantined only when the disposition says `oversized_temp_quarantined_not_retained_failure_bundle`; no positive or scientific claim depends on quarantine retention.

Resource evidence remains exactly twelve backend samples in the frozen stage order plus four runner boundary peaks. Telemetry errors are secondary evidence and cannot replace the primary error. A positive result requires every sample present, finite integer fields, null telemetry error, available memory gates, peak cap, strict QPC order, and exact maximum-summary equality.
