# PH1 Intel execution R8P7 independent frozen-source audit — 2026-08-14

## Verdict

**NO-GO** for the exact R8P7 no-device preflight. The local-entry repair is correct, but the production topology enumerator retained R8P6's ancestor depths after adding a new wrapper level. The first deterministic failure is an `AttributeError` while constructing topology; even after repairing that accessor alone, a separate six-path omission would make the topology gate false. Both occur before runtime collection or the frozen CPU-slice read.

No preflight, import, payload, compiler, OpenCL, model, or device action was performed by this audit.

## Frozen inputs

- Preflight SHA-256: `ffb2dcf95a9e5ee0937447571836106db11363dc5169dc861b6d3f4e78ff8782`
- Verifier SHA-256: `773bd69df92ca8d5bf50c9fc9d089378b31f29d8eac68c035f7506827b601808`
- Preregistration SHA-256: `711f62fab5fff0dc4ba2cb73cc8f2d6fc6241ad4e908217d50b17714c6fdc853`
- Lock SHA-256: `8d8b2da029bf6a4faa1d9e46d87bf3e2bf2cb9f8f0ffb8ee878dc3d69dabb465`
- R8P6 failure SHA-256: `03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372`, exactly one 2,986-byte file.
- R8P6 diagnosis SHA-256: `85d59b75a4940dd01df15d5072a0c9a1f4e9faf62260c6f8df07ed6fbfc0cba5`.

All four delivered hashes match. The closed lock has exactly 71 keys, is `execution_open:false` / `audit_token:PENDING`, matches the three current files and the R8P6 lock/diagnosis/audit/failure files, and preserves all 61 non-metadata R8P6 lock values (renaming only the inherited R8P6 preflight/verifier/prereg keys as designed).

## Local-entry repair — PASS

- The preflight defines its own `identity()` and `main()` calls that local function at line 191; there is no attribute/ancestor `.identity()` call.
- The collector retains current-module `entry_name`, `entry_spec_is_none`, `entry_package`, and resolved `entry_file`. `direct_entry` is their exact conjunction.
- The validator recomputes the conjunction, requires it and the retained boolean to be true, and retains the exact raw/parsed/native/original/application vectors plus venv/base identities and hashes.
- The exact 16 mutation names cover native/raw/orig/argv, venv/base identity, flag/trampoline/script changes, all four local-entry primitives, and the derived boolean.
- The static AST gate requires the local call and all four primitive globals and rejects ancestor identity calls.
- The verifier separately implements live identity capture and validation for its own direct invocation and independently checks the stored preflight identity.

## Fatal topology blocker

In `preflight_het_next_l0_ph1_intel_execution_r8p7.py`:

- R8P7 binds `prior` to R8P6. Therefore `prior.prior.prior.prior.prior` resolves to R8P2, not R8P1.
- Line 177 dereferences `prior.prior.prior.prior.prior.BASE_R8_PATHS`. R8P2 does not define `BASE_R8_PATHS`; that constant exists in R8P1. `topology()` therefore deterministically raises `AttributeError` before returning a snapshot.
- Independently, line 173 enumerates R8P2, R8P3, R8P4, R8P5, and R8P5 again. R8P1 is absent.
- Lines 175–176 enumerate failure roots for R8P2, R8P3, R8P4, R8P5, and R8P6, then exclude R8P6. R8P1 is again absent.
- The duplicate R8P5 entries collapse in the `absent` dictionary and add no keys.
- If line 177 alone were corrected to reach R8P1, `topology()` would still produce only 41 unique `absent` keys: 6 base-R8 paths + 20 non-failure paths for four unique R8P ancestors + 5 R8P6 non-failure paths + 4 ancestor failure roots + 6 current R8P7 paths.
- Line 181 requires `len(x["absent"]) == 47`. The missing R8P1 `CORE` (3), `VERIFY_RESULT` (1), `QUARANTINE` (1), and `FAILED` (1) are exactly the six-key deficit.

This is deterministic and independent of filesystem state. First the invalid `BASE_R8_PATHS` owner raises; correcting only that owner still cannot make 41 equal 47.

### Minimal repair

Use the five distinct R8P1-through-R8P5 modules in both ancestor loops and resolve `BASE_R8_PATHS` from R8P1. From R8P7's `prior=R8P6`, these are attribute depths 5, 4, 3, 2, and 1 respectively; R8P1 is `prior.prior.prior.prior.prior.prior`, while R8P5 is `prior.prior`.

A clearer and safer repair is to assign explicit named module variables or one frozen `ANCESTORS = (R8P1, R8P2, R8P3, R8P4, R8P5)` tuple and reuse it for the non-failure and failure loops. Obtain `BASE_R8_PATHS` from the explicit R8P1 member. The repaired TEMP/static gate must assert five distinct module identities, that the base-path owner is R8P1, and the exact 47-key set—not merely its length.

## Other audited gates

- The exact R8P6 failure root is permitted while its result/manifest/commit/verifier/quarantine remain required absent. File count, directory count, size, SHA, schema, stage, error, state, device/compiler flags, and false `direct_entry` are checked; the full-file SHA closes the remaining immutable content.
- Current R8P7 result, manifest, commit, verifier, failure, quarantine, and temp targets are absent.
- The production success writer and transaction simulation use current R8P7 kinds/helpers. The production bounded failure writer uses the current R8P7 failure kind and typed CPU-state schema.
- The independent verifier owns its atomic output writer and bounded failure simulation; it does not trust the candidate identity validator.
- Runtime, wheel, RAM, preparation, state, R7D1, and no-device boundaries remain inherited and hash-bound. This audit found no second deterministic blocker in those inspected paths.

## Authorization boundary

Do not execute the frozen R8P7 command. Freeze and independently audit a topology-only successor, then—if it passes—authorize exactly one no-device preflight. The immutable R8P6 negative remains unchanged; no physical Intel/OpenCL action is opened.
