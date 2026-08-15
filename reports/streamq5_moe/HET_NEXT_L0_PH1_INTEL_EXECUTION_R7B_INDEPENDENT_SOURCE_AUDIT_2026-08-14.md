# PH1 Intel execution R7B — independent frozen physical source audit

Date: 2026-08-14  
Scope: static/read-only audit. No payload, compiler, OpenCL, physical execution, or verifier execution was performed.

## Verdict

**NO-GO for the frozen R7B physical attempt.**

R7B correctly closes the prior authorization-result bypass and leaves the inherited numerical/device path unchanged. Two new blockers remain: pre-attempt exceptions after authorization bypass R7A's structured failure handler, and the R7B authorization-chain verifier is not independent because it imports and trusts the candidate R7B runner's chain, constants, and validator.

## Frozen identities and absence

Observed SHA-256 values match the handoff:

- runner: `7022b0ff1369d3b1b08920bcc26698341ac38314dd2ad7a0a598caca150d7dd6`
- verifier: `7c5d3fc9c79f4d4728cdfeb471144057c100d65eccef6617513b4b859dd5ad54`
- preregistration: `9a9c4f8e42188c1d0ed78b19f1ae5e31c7c02c213c2912ceefa9dfe89ebc335c`
- open lock: `7d787f48be7850fcf6fe48093cd0c7e77837d320784ad74ae1c0ccd06c8fc055`
- R7A authorization result: `a5b8e70cd40e241e16a250347cf06258a6540100f40423bc7216cb3639191265`
- R7A audit: `cbcbd1a861fc54e0dd529de22eb8fd3658a7fa81292e2c0ae0b188366055a5cd`

The physical R7A output directory, R7B verification result, R7B failure directory, and R7B quarantine directory are absent. No physical attempt has occurred.

## Checks that pass

1. **Exact authorization result.** `validate_auth_result()` first hashes the result against the frozen `a5b8e70c...`, requires the exact eight-key schema, exact kind, `pass=true`, `passed=total=7`, `no_payload_compiler_device=true`, exact R7A ACK, exact R7P SHA, exact seven check names, and every check exactly `True`.

2. **Complete R7B lock binding.** `authorize()` hashes the runner, verifier, preregistration, R7A runner/verifier/preflight/prereg/lock, authorization result, R7A audit, R7P chain, corrected R7 runner/verifier, R6 backend/common, and R0 backend/runner. It requires the exact lock key set, open state, token, physical output, verifier name, and every digest.

3. **Ordering/no bypass.** `main()` checks the exact R7B ACK, completes `authorize()`, validates the 7/7 result and lock chain, and calls the inherited `physical.authorize()` before delegating to `physical.execute_authorized()`. No R7A configure/recovery, payload read, backend construction, OpenCL load, allocation, or launch occurs before this gate. Importing R7A/backend modules does not instantiate the backend or load OpenCL.

4. **Inherited science.** The physical call is the unchanged R7A `execute_authorized()` using the unchanged R6 backend/common, corrected R7A numerical verifier, exact stage hashes, controls, buffers, launches, resource gates, and claim. The R7B extension is attached to the inherited authorization dict and would therefore be serialized with the R7A physical result.

## Blocker 1 — pre-attempt failures lose structured evidence

R7B calls `physical.execute_authorized(authorization)` directly at runner line 137. It does not wrap that call in R7A's outer `main()` failure handler.

Within R7A `execute_authorized()`, configuration/recovery, `psutil` import, starting-RAM gate, payload construction, post-payload telemetry, and the pre-device resource gate occur before the inner `try` that begins with attempt creation/backend execution. Exceptions from these stages are normally caught by R7A `main()` and archived as an atomic structured predevice failure. Under the R7B wrapper they propagate uncaught, producing no required failure bundle or disposition.

This changes lifecycle/evidence semantics despite unchanged numerical science. A low-RAM condition, payload read/hash error, telemetry/import failure, or pre-device resource rejection could consume the sole attempt without immutable evidence.

Required repair: a fresh R7C/R7B1 wrapper must retain the R7B authorization gate but invoke the inherited physical path through an exact structured outer failure boundary equivalent to R7A `main()`. It must preserve R7B authorization evidence and test, in TEMP only, authorization rejection/no-write plus RAM, payload, resource-predevice, ordinary device failure, telemetry failure, oversize quarantine, and valid-complete paths.

## Blocker 2 — authorization verifier trusts the candidate runner

The R7B verifier imports `run_het_next_l0_ph1_intel_execution_r7b as r7b`. It then:

- derives the chain from `r7b.CHAIN`;
- reads `r7b.AUTH_RESULT`;
- compares against `r7b.AUTH_RESULT_SHA`, `r7b.ACK`, and `r7b.R7P_SHA`;
- calls `r7b.validate_auth_result()` for the pass-7 adjudication.

That is not an independent verification of the new authorization extension. A coordinated error in candidate paths, constants, omitted chain entries, or validation logic is inherited by the verifier. The frozen R7A numerical verifier remains independently authored and is suitable for numerical replay, but the new R7B chain layer is circular.

Required repair: the next verifier may reuse the immutable R7A numerical verifier, but must not import the new candidate runner. It must define its own exact paths, expected digests, token, result schema, seven check names, lock schema, and R7P/R7A identities; independently hash/read them; and independently validate the serialized `r7b_authorization` extension. Add negative fixtures for a missing/extra lock key, altered authorization-result field/check, changed token, omitted extension, altered observed digest, and changed result SHA.

## Next step

Freeze a minimal R7C/R7B1 authorization/lifecycle revision. Do not change backend/common, numerical arithmetic, stage hashes, controls, buffers, launches, thresholds, resource limits, or claim. After independent source audit, one physical attempt may be considered, followed by the repaired independent verifier.

## Claim boundary

No physical Intel result exists. This audit confirms the authorization-result bypass is closed in source but does not establish PH1 correctness, performance, or a model-level result.
