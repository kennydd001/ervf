# PH1 Intel execution R8A3 — independent frozen-source audit

Date: 2026-08-14  
Scope: read-only frozen-source, provenance, topology, and terminal-state audit. No preflight, payload, compiler, OpenCL, or device call was made.

## Verdict

**NO-GO for physical execution.**

The production backend-tree repair itself is correct and closes the concrete R8A2 root-orphan acceptance bug. The physical runner remains a fresh authorization/namespace wrapper around the unchanged frozen R7A science. However, the preregistered production mutation gate is still not implemented as claimed: the TEMP suite never invokes the production topology scanner and omits several required wrapper, temp, quarantine, status, and device-flag terminal states.

Because the package explicitly makes that nonvacuous state-machine test a precondition for its single irreversible attempt, the current open token must not be used.

## Frozen package verified

- runner: `scripts/streamq5_moe/run_het_next_l0_ph1_intel_execution_r8a3.py`, SHA-256 `1f9e8a64a834287de05d495ba01ae95257f330fbd5efa2b29876a54af020789d`, 7,486 bytes;
- verifier: `scripts/streamq5_moe/verify_het_next_l0_ph1_intel_execution_r8a3.py`, SHA-256 `bd89bf2de4e871576a303641c92843be8485068c155efe94a0fb02f2d6650f66`, 16,022 bytes;
- preregistration: `reports/streamq5_moe/HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A3_PREREGISTRATION_2026-08-14.md`, SHA-256 `f25a82604f7f4656c9f80f9bd4be6f9a0634bfe88d031e6fe2926532f60a4ac6`, 2,002 bytes;
- open lock: `reports/streamq5_moe/het_next_l0_ph1_intel_execution_r8a3_lock.json`, SHA-256 `7b61a746e07514bad11b4e85ec1b76a66a4b5c2c5e83117546b5a5562ce64aa9`, 3,947 bytes;
- token: `PH1_INTEL_EXECUTION_R8A3_AFTER_R8P8_PASS_AND_TREE_AUDIT_GO`;
- all six fresh R8A3 physical/verifier paths were absent at audit time.

## Gates that pass

1. **Physical science and delegation.** The runner changes only current names, bindings, authorization extension, and wrapper writer. It retains the exact R8A2/R8A1/R7A physical delegate, Q5 package, arithmetic, kernels, thresholds, resources, launch sequence, and lifecycle. Authorization precedes mutation/payload/OpenCL; R7D1 authorization is not called.
2. **Invocation/provenance.** Exact `.venv\Scripts\python.exe -I -B`, native/base identities, direct-entry primitives, token, one-attempt lock, R8A2 audit SHA, R8P8/R7D/R7A chain, historical failures, and clean current/ancestor topology remain bound.
3. **Committed adjudication.** R8A3 reuses the frozen R8A2 adjudicator: only `stages` and/or `counters` may be false; every protected physical/protocol/resource gate must be true; the numerical false-set must exactly correspond. Early infrastructure remains terminal-invalid; a correlated valid negative requires `device_opened=true`.
4. **Production exact tree.** `exact_tree()` recursively enumerates the entire root and requires exactly two descendants: one direct attempt directory and its single canonical `failure.json`. A root file, extra/nested directory, extra/hidden file, or in-progress entry makes the tree invalid (`verify...r8a3.py:41-45`). This closes the R8A2 orphan path.
5. **TEMP redirection safety.** The existing suite redirects `OUT`, `FAILED`, `BACKEND_FAILED`, `QUAR`, and `BACKEND_QUAR` to a temporary root and restores all five in `finally` (`verify...r8a3.py:63-75`). It calls the actual `failure_terminal()`, which in turn calls actual `exact_tree()` and `reconstruct()`. The verifier never imports the R8A3 candidate runner.

## Blocking defect: claimed production topology matrix is incomplete

The preregistration states that the independent verifier tests its actual production topology scanner as well as failure/reconstruction logic. It does not. `failure_fs_mutations()` never calls `topology()`; `topology()` is called only once against the live filesystem in `main()` (`verify...r8a3.py:61,63-75,86`). Consequently there is no injected proof that the current family scanner rejects unknown current-family roots or `*.inprogress*` entries.

The TEMP suite has exactly these 11 cases:

- valid correlated baseline;
- missing backend;
- multiple backend attempts;
- extra file inside the backend attempt;
- root-level backend orphan;
- extra backend directory;
- wrong inherited kind;
- wrong correlation hash;
- wrong inherited disposition;
- early infrastructure;
- zero return labeled success-without-commit.

It does **not** cover the complete requested terminal/topology matrix:

- missing or multiple wrapper attempts;
- wrapper-root orphan or extra/nested wrapper entry;
- an `inprogress`/temp current-family path through production `topology()`;
- wrapper or backend quarantine collision through production `failure_terminal()`;
- wrong inherited status;
- inherited or wrapper `device_opened=false`;
- mismatched wrapper/inherited device flag;
- wrong wrapper kind/status/disposition;
- nonzero bare return without a canonical wrapper;
- a mixed committed bundle plus failure tree.

Some of these are rejected by inspection of the production predicates, but that is not the frozen nonvacuous mutation gate the preregistration promises. Most importantly, its reported `failure_filesystem_mutations=true` cannot certify `topology()` because that function was never invoked by the suite.

## Minimal next immutable revision

Keep the R8A3 runner, exact-tree implementation, committed allowlist, and physical science unchanged. Revise only the standalone verifier/preregistration/lock:

1. make the family-root used by `topology()` injectable or redirectable;
2. call the actual production `topology()`, `failure_terminal()`, `exact_tree()`, and `reconstruct()` for every fixture;
3. add the omitted wrapper, temp, quarantine, wrong-status/device, bare-return, and mixed-terminal cases;
4. require the exact fixture-name set, all expected classifications, exactly one terminal class per case, and full global restoration in `finally`;
5. re-freeze and independently audit before a single physical command.

The current R8A3 physical command remains closed.
