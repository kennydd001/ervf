# PH1 Intel execution R8A1 — independent frozen-source audit

Date: 2026-08-14  
Scope: read-only source, lock, provenance, invocation, topology, and terminal-state audit. No preflight, payload read, compiler, OpenCL, or device call was performed.

## Verdict

**NO-GO for the physical R8A1 command.**

The runner-side repair is sound: the frozen R7A physical science is unchanged, live authorization is before mutation/payload/OpenCL, the `.venv -I -B` identity is checked from current-module primitives, the exact R8P8/R7D/R7A history is bound, R7D1 authorization is never called, and R7A `authorize()` has no hidden clean-topology rejection of the allowed historical R8P8/R7D1/R8P6 evidence. Fresh R8A1 paths are absent.

The standalone verifier is not yet an implementation of the preregistered mutually exclusive terminal-state contract. It can label infrastructure or protocol-invalid evidence as a valid negative. Therefore a one-shot physical attempt is not yet defensible: a negative result could be misclassified after the fact.

## Frozen set verified

- runner: `scripts/streamq5_moe/run_het_next_l0_ph1_intel_execution_r8a1.py`, SHA-256 `dcacbac4eca3e5e799852495401528245f601827d7ce59dfb4e6cbf6619b22b9`, 10,559 bytes;
- verifier: `scripts/streamq5_moe/verify_het_next_l0_ph1_intel_execution_r8a1.py`, SHA-256 `a60060a33540f7f61fb5fa2b85e9f8ef452b7d9b0c652ae0f12dbef5c6cebf72`, 18,489 bytes;
- preregistration: `reports/streamq5_moe/HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A1_PREREGISTRATION_2026-08-14.md`, SHA-256 `b9dbf24540c48e2c81ed4a4a7a336d9459f5a5877e94c3fee2b67b70df757670`, 3,029 bytes;
- open lock: `reports/streamq5_moe/het_next_l0_ph1_intel_execution_r8a1_lock.json`, SHA-256 `72a8d2008a7f619c54f5e1d3030184f0216f527bbcb54603e5a3b1e0ec484f84`, 2,737 bytes;
- exact authorization token: `PH1_INTEL_EXECUTION_R8A1_AFTER_R8P8_PASS_AND_TERMINAL_AUDIT_GO`;
- lock schema: exactly 32 keys; every direct and inherited file hash resolves to the frozen value;
- all six fresh R8A1 result/failure/quarantine/verifier targets were absent at audit time.

## Gates that pass

1. **Runner delegation and science preservation.** The only physical authorization/execution is the frozen R7A `physical.authorize()` followed by `physical.execute_authorized()`. R8A1 remaps the R7A output/failure/quarantine paths first. It does not call R7D1 authorization and does not alter payload preparation, Q5 arithmetic, kernels, launch sequence, resource gates, or thresholds.
2. **No hidden old-topology blocker.** The frozen R7A authorization validates its lock and evidence chain but has no old-output clean gate. The explicitly retained R8P8 bundle and immutable R7D1/R8P6 failures therefore do not cause a concealed pre-device rejection.
3. **Live and stored invocation.** The runner derives native `GetCommandLineW`/`CommandLineToArgvW`, `sys.orig_argv`, `sys.argv`, `.venv` executable/prefix and WindowsApps base executable/prefix, `-I`, `-B`, hashes, and current `__name__`, `__spec__`, `__package__`, and `__file__`. The standalone verifier independently checks its own invocation and the stored runner invocation and includes direct invocation mutations.
4. **Correlation mechanics.** For a delegated nonzero, the verifier binds the wrapper to an inherited failure path, SHA-256, byte count, file list, bundle digest data, return code, disposition, and `device_opened` equality. Missing/multiple `failure.json`, wrong kind, wrong disposition, wrong hash, correlation mismatch, and malformed inherited dictionaries are rejected by the exercised correlated-wrapper mutation set.
5. **Standalone boundary.** The verifier does not import the R8A1 candidate runner. Frozen historical and numerical verifiers are loaded only behind the relevant contract gates.

## Blocking defects

### 1. Early infrastructure evidence is marked terminal-valid

`failure_state()` returns `("early_failure", ok, ...)` for a bounded outer infrastructure failure, and `main()` assigns that `ok` directly to `terminal_valid` (`verify...r8a1.py:70-75,108-109`). A correctly shaped early failure therefore produces `terminal_valid: true`, contradicting the preregistration: early infrastructure evidence may be retained but is never a valid component terminal.

Required repair: separate `evidence_valid` from `scientific_terminal_valid`. A valid early failure must have `evidence_valid=true`, `terminal_state="early_failure"`, `terminal_valid=false`, `valid_negative=false`, and nonzero verifier exit.

### 2. The committed-negative allowlist is absent

The committed-negative predicate requires only exact gate keys, at least one false gate, authorization, and three numerical provenance checks (`authorization`, `compile_package`, `records_input_lut`). It then checks `false_gates <= GATE_KEYS`, which is tautological because exact key equality was already required (`verify...r8a1.py:104-106`). `committed_negative_stage_allowlist` is merely assigned `positive or negative`; it does not enforce an allowlist.

Consequently a committed bundle with a false precondition/protocol gate—such as identity, ledger order, ownership, allocations, writes, initialization, args, launch, finish/read, release, resource samples/resources, or forbidden-runtime checks—can be labeled `committed_negative` and `terminal_valid=true` as long as the three named provenance checks remain true. That violates the frozen rule that authorization, provenance, precheck, lifecycle, and evidence-integrity failures are invalid committed states.

Required repair: freeze an explicit set of outcome/mechanism gates allowed to be false, require at least one of those false, and require every authorization, provenance, input, identity, protocol, lifecycle, resource, cardinality, and evidence-integrity gate true. The exact allowed set must be specified before execution; it must not be inferred from a result.

### 3. Terminal mutation coverage is nonvacuous only for correlated failure

`terminal_mutations()` tests the correlated delegated-wrapper dictionary only (`verify...r8a1.py:79-84`). It does not exercise:

- a valid committed positive;
- a valid allowlisted scientific/post-device negative;
- a committed authorization/provenance/precheck/lifecycle false gate that must be invalid;
- early infrastructure evidence that is evidence-valid but terminal-invalid;
- success-without-commit, invalid committed schema, or terminal-class exclusivity.

Required repair: factor terminal adjudication into a pure function and exercise all terminal classes plus one mutation for every forbidden false gate. Require exactly one class predicate true per fixture and repeat deterministically.

### 4. Backend failure-root topology is not exact

The verifier enumerates only `BACKEND_FAILED.rglob("failure.json")` (`verify...r8a1.py:70-71`). With exactly one valid failure JSON, unrelated non-`failure.json` files or orphan directories elsewhere in `BACKEND_FAILED` are ignored. The inherited bundle file scan is confined to that failure file's parent. Thus extra backend artifacts outside the selected parent do not invalidate the terminal topology.

Required repair: require the backend failure root to contain exactly one attempt directory, with all and only the bounded files declared by the inherited bundle; reject every other file, directory, link/reparse point, or temporary artifact. Apply the same exact-root rule to the wrapper failure root.

### 5. A correlated scientific/post-device negative does not require device-opened true

The correlation check requires the wrapper and inherited `device_opened` booleans to agree, but accepts either value (`verify...r8a1.py:66-69`). If this class is intended to be the allowed scientific/post-device failure terminal, equality alone is insufficient: a correlated pre-device inherited failure can be accepted as terminal-valid.

Required repair: require `device_opened is true` for the valid correlated scientific/post-device negative class. A correlated inherited failure with `device_opened=false` may remain bounded evidence but must be terminal-invalid/infrastructure-negative.

## Minimal next immutable revision

Keep the R8A1 runner, physical delegate, numerical science, token discipline, and fresh namespace unchanged. Revise only the independent terminal adjudicator and its preregistration/lock bindings:

1. separate evidence validity from scientific terminal validity;
2. freeze an explicit false-gate allowlist and require all structural/precondition gates true;
3. enforce exact failure-root topology and post-device `device_opened=true` for the valid correlated-negative class;
4. add deterministic pure adjudicator fixtures for every terminal class and every forbidden mutation;
5. re-freeze hashes, confirm the six current output targets remain absent, and obtain a new independent source GO before the single physical command.

Until those repairs are frozen and audited, **do not execute R8A1**.
