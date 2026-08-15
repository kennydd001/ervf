# PH1 Intel execution R6P — 12/15 preflight diagnosis

Datum: 2026-08-14. Read-only analysis of immutable result SHA-256
`7542e0ffe248176d2d571941f9fae3f12b54faeb9d83cbdef74d4d471437043b`.
No payload, compiler, OpenCL load, or device call occurred.

R6P completed without exception and returned 12/15. The three false checks are
synthetic-fixture defects only:

1. `no_device_static`: the source uses a literal substring scan that sees its
   own forbidden literal. Replace this with AST import/call inspection of the
   actual no-device modules; do not weaken the forbidden API surface gate.
2. `codec_fma_width8_full_shapes`: NumPy ties-to-even produces the exact first
   eight q values `[-15,-7,-4,0,4,7,11,15]`, not the stale expected
   `[-15,-8,-4,0,4,8,11,15]`. The production quantizer remains unchanged.
3. `actual_verifier_mutations_full_shapes`: the synthetic records encode q=1
   with BF16 scale 1, hence decoded weights are BF16 1.0. The prepared verifier
   weights were zero, so the independently rebuilt record graph and supplied
   graph disagreed. R6P1 must derive weights independently from the packed
   record bytes (or use exact BF16-one weights), retain zero input and outputs,
   run the positive baseline through the real verifier, then require every
   frozen mutation to fail it.

The full production buffer/output/counter shape repair remains valid. This
result is not a scientific or Intel-device negative.
