# PH1 Intel execution R8A2 — independent frozen-source audit

Date: 2026-08-14  
Scope: read-only audit of the frozen runner, standalone verifier, preregistration, lock, inherited bindings, and fresh topology. No preflight, payload, compiler, OpenCL, or device call was made.

## Verdict

**NO-GO for the physical R8A2 command.**

R8A2 closes the important R8A1 classification defects: the committed-negative false-gate set is now exactly `{stages, counters}`, every other physical/protocol gate must be true, the numerical false-set must match the physical false-set exactly, early infrastructure evidence is terminal-invalid, and correlated delegated negatives require `device_opened=true`.

Two preregistered gates remain unimplemented: backend failure-root exhaustion has a concrete orphan-file acceptance path, and the advertised production terminal/topology mutation matrix does not exercise the production failure-state function or filesystem topology. The physical science is unchanged, but a one-shot attempt must remain closed until those verifier-only defects are repaired and re-audited.

## Frozen package verified

- runner: `scripts/streamq5_moe/run_het_next_l0_ph1_intel_execution_r8a2.py`, SHA-256 `e46199021ba33db8cd0bb6a56bdb08bd41eb68dac99a8d07c67aa3c56b48b654`, 7,635 bytes;
- verifier: `scripts/streamq5_moe/verify_het_next_l0_ph1_intel_execution_r8a2.py`, SHA-256 `cb06467801858b3f1b7b56d88b7c0d9706cf908af7e7b3b75e228a5f468860d8`, 19,877 bytes;
- preregistration: `reports/streamq5_moe/HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A2_PREREGISTRATION_2026-08-14.md`, SHA-256 `b3f697837ecc99e3685d7a26efc85c524fe1f413f2b3f8315e36d972a41c6ba7`, 3,112 bytes;
- open lock: `reports/streamq5_moe/het_next_l0_ph1_intel_execution_r8a2_lock.json`, SHA-256 `8c4e332dffed0a08eae728169d8b016607d3d14571d9296223656e85eb642a89`, 3,329 bytes;
- token: `PH1_INTEL_EXECUTION_R8A2_AFTER_R8P8_PASS_AND_TERMINAL_AUDIT_GO`;
- all six R8A2 result/failure/quarantine/verifier paths were absent at audit time.

## Gates that pass

1. **Runner/science preservation.** R8A2 is a namespace and authorization wrapper around R8A1/R7A. It changes no payload preparation, Q5 codec, arithmetic, kernels, thresholds, resource gates, launch sequence, or frozen R7A physical lifecycle. It authorizes before mutation/payload/OpenCL, never calls R7D1 authorization, remaps all fresh physical paths, and performs exactly one delegated execution.
2. **Invocation and provenance.** The runner validates the exact `.venv\Scripts\python.exe -I -B` command through native and Python identities, direct current-module entry primitives, launcher/base interpreter hashes, the R8P8/R7D/R7A chain, both immutable historical failures, the R8A1 audit, exact lock, token, and fresh topology.
3. **Committed result adjudication.** `adjudicate_committed()` requires exact gate and numerical schemas. A positive requires every gate/check true. An allowed negative requires a nonempty false set contained in `{stages,counters}`, all 16 protected gates true, and the exact corresponding numerical false set:
   - common: `positive_schema`, `runner_gates`;
   - add `oracle_outputs` iff `stages` is false;
   - add `counters` iff `counters` is false.
   Authorization, provenance, identity, ledger, ownership, allocation, write, initialization, arguments, launch/read, cleanup/release, resource, extension, and forbidden-call failures therefore remain invalid.
4. **Terminal validity.** Only `positive`, `allowed_device_negative`, and `correlated_device_negative` set `terminal_valid=true`. Early infrastructure returns `early_invalid`. Correlated delegated evidence requires nonzero return, exact wrapper/inherited equality, exact status/disposition, one inherited failure, and `device_opened=true` on both sides.
5. **Standalone boundary.** The verifier imports no R8A2 candidate runner. It independently defines the current paths, lock chain, invocation contract, adjudicator, topology functions, writer, and verification output. Historical/numerical imports are frozen ancestors and hash-gated.

## Blocking defects

### 1. Backend failure topology accepts an orphan root file

In `failure_terminal()`, `exact_backend` requires one descendant directory, one descendant `failure.json`, that file's parent equal to the sole directory, and between one and four descendant files (`verify...r8a2.py:76-80`). It does **not** require every descendant file to be inside that sole attempt directory.

Concrete accepted shape:

```text
het_next_l0_ph1_intel_execution_r8a2_backend_failed_attempts/
├── orphan.bin
└── attempt_x/
    └── failure.json
```

Here `len(bdirs)==1`, `len(bf)==1`, and `len(bfiles)==2`, so `exact_backend` remains true. `reconstruct()` scans only `failure.json`'s parent, so `orphan.bin` is omitted from the correlated evidence and does not invalidate an otherwise valid wrapper. This contradicts the preregistered rejection of extra/orphan files and the requested exhaustive topology.

Required repair: require the backend root's complete descendants to equal the one declared attempt directory plus exactly that directory's bounded declared files. Equivalently, assert `set(bfiles)==set(files_used_by_reconstruct)` and reject any additional directory, file, symlink/reparse entry, or in-progress artifact. Keep the current one-wrapper rule.

### 2. The mutation matrix does not test production failure/topology adjudication

`adjudicator_mutations()` correctly exercises committed positives, all three allowed `{stages,counters}` subsets, and all 16 protected false gates. Its failure fixtures, however, call only the pure `correlated(w, actual, count)` dictionary predicate (`verify...r8a2.py:81-96`). They never call production `failure_terminal()`, `one_wrapper()`, `topology_roots()`, or `reconstruct()` against a temporary filesystem.

Therefore the claimed missing/multiple/extra/orphan/topology-collision coverage is not present. The `count=0/2` arguments are not filesystem missing/multiple tests, `error="success_without_commit"` is only a dictionary mutation, and the early fixture proves merely that it is not correlated—not that production classifies it as terminal-invalid. The orphan acceptance above is consequently undetected.

Required repair: add a bounded temporary-root production fixture that redirects all current R8A2 terminal paths and runs the actual terminal functions for, at minimum:

- committed positive;
- `stages`, `counters`, and combined allowed negatives;
- every protected false gate;
- valid early infrastructure evidence with `terminal_valid=false`;
- valid correlated post-device failure;
- missing wrapper/backend, multiple wrappers/backends, root-level orphan file, nested extra directory/file, temp/quarantine collision, wrong kind/status/disposition/hash/device flag/correlation;
- bare nonzero and success-without-commit states.

Require exactly one terminal class for every fixture and restore every redirected global in `finally`.

## Minimal next revision

Keep the R8A2 runner, token discipline, frozen R7A physical science, allowlist, and numerical mapping unchanged. Revise only the standalone verifier plus preregistration/lock bindings to:

1. enforce exact backend-root descendant equality; and
2. execute the production terminal/topology state machine over deterministic temporary fixtures.

Re-freeze, confirm all six physical R8A3 paths absent, and obtain independent source GO before the single physical command. **Do not execute R8A2.**
