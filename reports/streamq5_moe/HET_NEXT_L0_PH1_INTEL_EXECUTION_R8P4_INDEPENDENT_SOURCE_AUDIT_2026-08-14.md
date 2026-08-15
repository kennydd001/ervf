# PH1 Intel execution R8P4 — independent frozen source audit

Date: 2026-08-14  
Scope: static/read-only audit before execution. No candidate import, preflight, CPU payload read, compiler, OpenCL, or device call was performed.

## Verdict

**NO-GO for the final R8P4 freeze.**

The R8P3 failure-writer cleanup blocker is mechanically repaired, and the final verifier restores its own current/inherited static boundary. One failure-provenance defect remains: every production failure records `cpu_frozen_slice_read=false`, including failures that occur after the frozen CPU preparation slice was read. A minimal state-tracking revision is required; no identity, runtime, source slice, numerical, transaction, device or threshold field should change.

## Final frozen identities

The earlier verifier `5e3c072b…` and lock `18b59a53…` were superseded before this verdict. This audit uses only:

| Artifact | SHA-256 | Status |
|---|---|---|
| preflight | `e640e904954e435fbb62666a58c018c25658a02c1e3486a1d9455be8731e5672` | exact final freeze |
| independent verifier | `ea6b61c949d3b4c0102755a8c0fb2133a76d9788c4c4a7c80304d81f73a54408` | exact final freeze |
| preregistration | `cc36485766a9705c6e3200c5b603550dbb91e5147b9829a4c3b8b80266809af6` | exact final freeze |
| closed lock | `521af55027b97aad5d24014cb074da58280d132c550fca06ec15a27460a9be60` | closed/PENDING |

The 54-key lock binds the complete frozen chain, including the 43 inherited bindings and R8P3 audit SHA `9aec9abc77a790f2eb4ef4685d84891b79c28cf00fc5c082e91d90099587fb85`. The lower-case R8 family contains only its five lock files. Every R8P4 result, manifest, commit, verifier output, failure, quarantine and temporary path is absent.

## Correct repair behavior

- The production writer accepts an injectable root/create function, enforces the 64-KiB cap, uses create-new attempt directories, and removes an empty attempt after pre-link failure (`preflight`, lines 118–136).
- A post-link temp-unlink failure is recovered only when the promoted target exactly equals the canonical bytes; remaining temps are removed and the recovery is explicitly marked (`preflight`, lines 121–136).
- Two baseline calls produce distinct attempt paths and preserve the first bytes. The current production functions are TEMP-tested for baseline schema/cap, uniqueness/no overwrite, pre-link cleanup, post-link canonical recovery and primary-versus-secondary handling (`preflight`, lines 138–161).
- `preserve_primary()` catches a secondary writer exception and the enclosing production boundary re-raises the original primary exception (`preflight`, lines 138–140 and 190–192). Thus a writer error cannot replace the real preflight failure.
- The verifier independently reimplements the writer and five-outcome TEMP suite, requires the exact nonempty failure dictionary, and rejects empty, missing, extra and false dictionaries (`verifier`, lines 62–98 and 114–138).
- R8P3's current transaction suite, exact dual venv/base identity, runtime/preparation digest, clean topology and CPU-only/no-model/no-compiler/no-OpenCL/no-device boundary remain bound. The final verifier now independently calls its current/inherited static-boundary gate.

## Blocking provenance defect

`failure_row()` hardcodes `cpu_frozen_slice_read: False` (`preflight`, lines 118–119). The same writer is used for every exception caught by `main()` (`preflight`, lines 174–192):

- before runtime and preparation, `false` is correct;
- while `preparation_summary()` is reading the slice at line 185, the state may be partial/unknown;
- after preparation returns, failures in `failure_simulation()`, transaction/static/hash checks, result construction, or `publish()` have definitely occurred after the CPU slice read, so `false` is incorrect.

Both the production and independent TEMP suites validate only the hardcoded false variant. They therefore certify internally canonical evidence whose resource/provenance statement can be factually wrong. The kind `early_failure` is also misleading for simulation/publication failures after preparation.

This does not affect the nominal PASS arithmetic, but it defeats the purpose of the newly formalized failure evidence and is therefore blocking in a revision dedicated specifically to failure provenance.

## Minimal repair

A fresh R8P5/R8P4P revision should introduce explicit immutable CPU-read state, for example `not_started`, `started_not_completed`, and `completed`, initialized before the try block and updated immediately before and after `preparation_summary()`. Pass that state through `preserve_primary()` and `atomic_failure()` into the canonical evidence.

The production and independent TEMP suites must cover at least:

- pre-preparation failure → `not_started`;
- preparation-body failure → `started_not_completed`;
- simulation/publication failure after successful preparation → `completed`;
- Boolean summary, if retained, derived only from this state (`true` exactly for `completed`, never hardcoded);
- wrong/missing/extra state and contradictory Boolean mutations rejected.

All five existing writer lifecycle outcomes and exact result/verifier key gates must remain unchanged.

