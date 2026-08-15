# PH1 Intel execution R8P5 — independent frozen source audit

Date: 2026-08-14  
Scope: static/read-only audit before execution. No candidate import, preflight, payload read, compiler, OpenCL, or device call was performed.

## Verdict

**NO-GO for the frozen R8P5 command.**

The requested three-state CPU-slice provenance repair is correct from `main()` through canonical failure evidence and through the independent verifier. However, R8P5 reintroduces the previously rejected prior-helper transaction gap: it defines a new R8P5 result/manifest/commit publisher but gates the R8P4 publisher instead. A small current-helper revision is required. A secondary validator hardening is also needed so a wrong-type state is rejected rather than raising unexpectedly.

## Frozen identities and clean state

| Artifact | SHA-256 | Status |
|---|---|---|
| preflight | `cec0603d6fdcc6934a5ae831a49e1cc243d2be117318d2c231127582d7648243` | exact handoff/lock |
| independent verifier | `f17f82fb36d064fe3a9e305b017dfee6e0c95127a1aa9f6dab00fe1a92ca3dc7` | exact handoff/lock |
| preregistration | `be5e0ce857370369b7600b89ab2337170cbd119fff8bd98a6a83b8d1723fff60` | exact handoff/lock |
| closed lock | `8efea981606f6b3a70d66226f514db7ae50bf08070980690449b0b41299d28a9` | closed/PENDING |

The lock has 59 keys and binds the complete 48-field inherited chain including R8P4 audit SHA `a02e2e619c155d36d5c21fb010504e77fbe4526ca9e0cd846918c1156a9d9815`. The lower-case R8 family contains only the six R8 through R8P5 locks; every result, manifest, commit, independent verification, failure, quarantine and temporary path is absent.

## Correct three-state repair

- `cpu_state` is initialized to `not_started` before the main try block, set to `started_not_completed` immediately before `preparation_summary()`, and changed to `completed` only after the call returns (`preflight`, lines 127–139).
- The exact invariants are derived in one function: `not_started → false/false`, `started_not_completed → true/false`, and `completed → true/true` (`preflight`, lines 27–34).
- The current state flows without replacement through `preserve_primary → atomic_failure → failure_row` and is bound into the canonical evidence together with the two derived Booleans (`preflight`, lines 63–86 and 143–145).
- The actual production writer TEMP suite covers all three states/stages, the five retained R8P4 lifecycle outcomes, and five state/Boolean/schema mutations for every state (`preflight`, lines 88–113).
- The independent verifier separately reimplements the state mapping and writer, repeats all three states and mutations, requires the exact six-key failure simulation, and rejects result-level state/Boolean mismatches (`verifier`, lines 30–37, 51–95 and 117–133).
- Dual launcher/base identities, CPU preparation digest, runtime/RECORD/RAM gates, failure lifecycle, static no-device boundary and clean topology remain source-bound.

## Blocking current-transaction regression

R8P5 defines its own `verify_bundle()` and `publish()` with the new R8P5 `KIND` at preflight lines 50–61. The production success path uses that R8P5 publisher at line 142.

But the gate at line 139 executes `prior.transaction_simulation()`, which is R8P4's simulator and exercises R8P4's `publish`, `verify_bundle`, `atomic_create`, and quarantine path. It does not call the new R8P5 publisher. Therefore a defect in the exact R8P5 manifest kind, default paths, commit binding, exception quarantine, or current wrapper wiring is invisible to `current_transactions=true`.

The R8P5 independent verifier likewise runs no own current output-writer transaction suite. It validates the stored inherited transaction dictionary but does not exercise its own `atomic_create(OUTPUT, ...)` for clean create, repeat preservation, pre-link failure, post-link recovery and stale-temp handling.

This is the same class of non-vacuous coverage problem rejected at R8P2. The fact that the current code is textually simple does not make a prior namespace's PASS evidence evidence for the new production helpers.

## Secondary exact-enum validator gap

`state_bits()` evaluates `state not in CPU_STATES` before proving that `state` is a string. A JSON list or object therefore raises `TypeError`, while `state_valid()` catches only `ValueError` (`preflight`, lines 27–34; verifier, lines 30–37). The current mutations cover unknown strings and Boolean contradictions but not wrong-type enum values.

Required repair: require `isinstance(state, str)` or catch `(ValueError, TypeError)`, and add list/integer/null mutations that must return false without crashing.

## Minimal next revision

R8P6/R8P5P should retain the complete state repair unchanged and:

1. run a TEMP transaction suite through the exact R8P6/R8P5P production `publish`, `verify_bundle`, `atomic_create`, cleanup and quarantine helpers;
2. make the independent verifier exercise its exact output writer and require exact nonempty transaction keys plus empty/missing/extra/false mutations;
3. harden the state validator and add wrong-type mutations.

No runtime identity, payload, preparation, control, stage digest, numerical, resource, no-device or claim field should change.

