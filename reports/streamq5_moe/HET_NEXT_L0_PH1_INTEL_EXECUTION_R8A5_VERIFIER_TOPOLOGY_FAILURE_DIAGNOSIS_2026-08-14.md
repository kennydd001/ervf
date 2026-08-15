# R8A5 verifier topology failure diagnosis

Date: 2026-08-14. Read-only diagnosis; no rerun, edit, payload, OpenCL or device action.

## Exact failure

The immutable R8A5 physical bundle is complete and internally positive. The first frozen independent verifier wrote `het_next_l0_ph1_intel_execution_r8a5_independent_verification.json` (1,724 bytes, SHA-256 `d6b630658c59e1c6913ba099bb8d617fe1b451e14e31ee38b68d351fb9fde917`) and returned negative only because `checks.topology=false`; consequently `terminal_contract=false`, `terminal_state=invalid`, `terminal_valid=false`, and `pass=false`. Its other 27 checks are true, including authorization, historical provenance, lock, live invocation and mutations, the full 31-case production matrix, the committed adjudicator mutations, and every independent numerical check.

The deterministic bug is in the verifier's live `topology()` enumeration. On Windows, `Path.glob("het_next_l0_ph1_intel_execution_r8a5*")` is case-insensitive. It therefore returns five report entries:

1. `het_next_l0_ph1_intel_execution_r8a5` (committed bundle);
2. `het_next_l0_ph1_intel_execution_r8a5_independent_verification.json`;
3. `het_next_l0_ph1_intel_execution_r8a5_lock.json`;
4. `HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A5_PREREGISTRATION_2026-08-14.md`;
5. `HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A5_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md`.

The allowed set contains only the first three. The uppercase preregistration and source-audit paths are thus unexpected solely because the prefix glob ignores case. They are mandatory frozen provenance, not terminal artifacts. No failure or quarantine tree and no in-progress artifact exists. This is a verifier topology bug, not a physical, numerical, authorization, ownership, lifecycle, resource or cleanup failure.

## Immutable physical evidence

- `result.json`: 99,483 bytes, SHA-256 `9d1ac21f4fdd9657160e877f267369b5e831ff9f7a65e998f27895947c9cad50`.
- `manifest.json`: 167 bytes, SHA-256 `2d13137f143ff183be3ffe89a3b85754cb2f35b52f92885580f49676e5fcfb7b`.
- `commit.json`: 210 bytes, SHA-256 `07d9f03e8907a029d8bc31e40da6298de080b6bc0f0914769f8d52517b2dd965`.
- Result kind/status: `ph1_intel_execution_r7a` / `intel_execution_positive`; `positive=true`.
- All 18 physical result gates are true: allocations, args, compile identity, controls, counters, extensions, finish/read ordering, forbidden-call contract, device identity, initialization, launch, ledger order, ownership, release, resource samples, resources, stages and writes.
- Exact stage hashes: gate `e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867`; up `f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08`; SiLU `a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8`; activation `762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f`; down `142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc`.
- Peak retained working set: 154,890,240 bytes; final available RAM: 48,636,030,976 bytes. All 12 backend resource samples are present and telemetry-error-free.

The bundle manifest and commit bind the result hashes exactly. The existing physical bundle and failed verifier artifact must remain immutable. There is no authorization for a physical rerun.

## Superseding verifier requirement

A fresh verifier-only R8V1 must bind the immutable R8A5 bundle, first verifier artifact, this diagnosis and source audit. It must enumerate a fixed, explicit, casefold-aware set of terminal/provenance paths rather than a broad prefix glob. The verifier must retain the same numerical verifier and 31-case production terminal adjudicator. Its TEMP matrix must add uppercase provenance/collision cases and prove exact-set rejection for every extra, missing, case-colliding, temporary, failure or quarantine path. R8V1 writes a new create-only verification path and performs no payload, compiler, OpenCL or device action.
