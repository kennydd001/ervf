# PH1 Intel execution R7C1 — delegated-return adjudication revision

Date: 2026-08-14

R7C1 is the immutable single-repair successor to R7C. It preserves the exact R7A physical computation, R7B/R7C authorization-result gate, resources, numerical outputs, and claim. It is closed pending source audit and static-preflight authorization.

The sole repair is lifecycle adjudication after the delegated immutable R7A call:

- Snapshot existing R7A failure artifacts before delegation.
- A valid committed R7A positive or negative bundle is authoritative and is never polluted by R7C1 failure evidence.
- On delegated nonzero without a valid commit, require exactly one newly created R7A `failure.json`. Validate its kind/status/error/device flag/disposition, directory cardinality, total bytes at or below 16 MiB, every retained file hash, failure hash, and deterministic bundle digest. Write a bounded atomic R7C1 summary referencing that evidence.
- Missing, multiple, malformed, or oversized inherited artifacts and bare nonzero returns remain validly reported negative protocol outcomes, never passes. Their R7C1 summary is bounded and create-new.
- Raised early exceptions retain the R7C outer boundary. Stale R7C1 temporaries are quarantined and abort.

The closed no-device preflight must exercise actual R7C1 functions for early raised exceptions, structured inherited failure plus return 3, bare return 3, oversized inherited evidence, stale quarantine, success-without-commit, and valid positive and valid negative commits. No physical payload/device action is allowed before a separate authorization revision.

Claim remains limited to one real expert/input Intel correctness component only.
