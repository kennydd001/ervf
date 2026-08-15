# NC19I1 fixture erratum

Date: 2026-08-14. Scope: source/static fixtures only.

This immutable erratum binds the NC19 design and its GO audit, plus the NC19I0 source-audit SHA `0d15c5d594e8398e91bdaf8b59117483e89539e03b29dcdcba01de56920b9fd2`.

The corrected fixture manifest preserves all 1,106 NC19 cases and the exact 100-absent/57-required/zero-intersection source-lock repair. In the implementation-freeze positive and its nine derived absent-set mutations, the two nominal NC19 preflight/verifier lock observations are re-encoded from their declared NC19 typed schemas; their raw Base64, byte counts, SHA-256 values, case totals, and ordered-tree digests are recomputed. Intentional negative corruption remains case-local and is evaluated by the production classifier; it is not treated as a global manifest-integrity failure.

No compiler, payload, Driver, runtime, or device phase is authorized by this erratum.
