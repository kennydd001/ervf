# PORT80B T0Q5 S0-R5 + C1-R2A combined adjudication

## Outcome

**Formal overall result: negative.** S0-R5 remains verifier-negative because its frozen natural `p0/n8` shared-down `q=6 -> 5` mutation changes **0 BF16 words** in both the raw and gated output. C1-R2A is a separate synthetic sensitivity control and does not repair or reinterpret that frozen result.

Two narrower results are positive:

- **R5 numerical-quality arm:** all **96/96** measurements pass `relL2 <= 0.08`: routed 32/32 (max `0.068281357623850403`), shared-raw 32/32 (max `0.0750221523726263`), and shared-gated 32/32 (max `0.075311459958494031`).
- **C1-R2A synthetic integrity control:** its completed independent verifier is **11/11 positive**. The preregistered one-field mutation changes exactly one BF16 output word (`14520 -> 14489`) and the safe checker rejects the digest mismatch before unsafe decode/linear calls.

For context, the same R5 mutation at frozen natural `p0/n15` changes one BF16 word in both raw and gated outputs. That neighbor witness does not override the failed `p0/n8` conjunct.

## Evidence and provenance

The machine-readable adjudication binds the immutable raw, result, commit, runner, runner-lock, verifier, verifier-lock and preregistration artifacts for both experiments by byte count and SHA-256. Both commits validate, both result-to-raw hashes validate, and both frozen verifier source hashes agree with their locks and result records. The adjudicator opened only the two small committed raw bundles and JSON/source evidence; it did not open the official checkpoint or run a model.

Machine-readable adjudication: `reports/streamq5_moe/PORT80B_T0Q5_S0R5_C1R2A_COMBINED_ADJUDICATION_2026-08-13.json`.

## Claim boundary

This is validation-only and non-heldout. It is not evidence for a complete layer, complete model, generation quality, throughput, heterogeneous-device performance, industrial superiority, novelty, or a breakthrough. R5 remains formally negative; only its numerical-quality sub-arm and C1's separate synthetic integrity-control are positive.
