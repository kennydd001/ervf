# PH1 Intel execution R4 — lifecycle-only preregistration

Status: closed implementation. No preflight, payload, compiler, OpenCL load, or device call is authorized.

R4 supersedes immutable R3 after independent audit SHA-256 `204e8a5c4c33b228c524f34abcdb77f745a7893356dcc8961b52413980fed88b`. It makes no numerical, codec, kernel, launch, buffer, device-identity, threshold, or claim change. The claim remains one real expert/input Intel correctness component only.

Every create/allocate wrapper appends a pre-call ledger placeholder. Any non-null context, queue, program, kernel, or host-USM result becomes owned immediately, before the inherited error/status or attestation check can raise. Exceptions and non-null results are retained. Cleanup attempts every owned resource even after an earlier release error. Static fault tests cover thrown create calls and non-null results that are followed by an injected error or failed attestation.

Authorization is strictly read-only: a wrong ACK or any lock/provenance failure returns negative without creating, recovering, quarantining, or modifying any filesystem path. Only after successful authorization may recovery and evidence creation begin. A valid committed result remains read-only and is never polluted by a failure artifact.

An in-progress bundle is measured before movement. At most 16 MiB is ever retained below the failed-attempts root. An oversized attempt is moved to quarantine and the failed-attempts bundle contains only a bounded hash/byte-count/disposition summary. Failure generation itself is bounded before its atomic create-new write.

Resource sampling is non-throwing. A telemetry exception is retained as a secondary error and cannot replace the primary device/lifecycle exception. A positive result requires exactly twelve samples in this order: backend entry; pre/post each of four launches; pre/post finish; post-cleanup. Each sample retains QPC, available memory, RSS, peak working set, and a null telemetry error. The summary peak must equal the maximum of all twelve sample peaks and the four runner boundary peaks (start, post-payload, final, post-serialization).

The independent verifier reconstructs the scientific outputs and checks exact provenance, identity, fourteen distinct non-null aligned host-USM pointers with type/base/size attestations, eighteen pointer arguments, four launches, finish-before-nine-reads, twenty-one successful ordered releases, all controls, and exact output hashes. Static preflight mutates actual verifier inputs for pointer alias/type/base, argument pointer, launch geometry, read ordering, release ordering/status, resource order/peak/summary, identity, output, provenance, controls, forbidden calls, and bundle manifest/commit/file set. No R4 output or preflight result may exist at freeze.
