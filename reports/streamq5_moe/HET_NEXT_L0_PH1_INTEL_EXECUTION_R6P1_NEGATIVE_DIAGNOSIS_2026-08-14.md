# PH1 Intel execution R6P1 — 15/16 verifier diagnosis

Datum: 2026-08-14. Read-only diagnosis of immutable result SHA-256
`0b368d3ed9c823405a821f96b59f58cbb8cb4b2c48fa1d19431ca3f88db0742f`.
No payload, compiler, OpenCL load, or device call occurred.

R6P1 passed 15/16 checks. Its retained baseline map localizes the sole false
conjunct to `oracle_outputs`; every surrounding provenance, record, control,
ledger, ownership, resource, buffer, counter and authorization conjunct passed.

The independent verifier's `linear()` has `out[row]=rb(lanes[0])` indented one
space outside `for row in range(r)`. It therefore writes only the last row and
leaves preceding `np.empty` rows uninitialized. This affects the verifier CPU
oracle only. It does not modify or execute the Intel kernel/backend.

R7 must move that assignment inside the row loop and add deterministic full-row
sentinels for both supported production widths: 512 rows × 2048 columns and
2048 rows × 512 columns. Every row must equal its predeclared alternating
nonzero BF16 word, two independent evaluations must be byte-identical, and the
two run digests must equal the digest of the constructed expected byte array.
No kernel, codec, reduction tree, threshold, backend or claim change follows.
