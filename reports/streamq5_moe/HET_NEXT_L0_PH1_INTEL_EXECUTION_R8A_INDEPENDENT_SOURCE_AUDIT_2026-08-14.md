# PH1 Intel execution R8A independent frozen-source audit — 2026-08-14

## Verdict

**NO-GO for the one physical R8A attempt.** The runner's preauthorization and physical delegation are coherent, but the frozen standalone verifier is not fail-closed for the new authorization evidence or for negative/failure adjudication. Because this is a one-attempt physical gate, those verifier gaps must be repaired and frozen before device execution.

No runner, verifier, preflight, payload, compiler, OpenCL, model, or device call was made during this audit.

## Frozen inputs and observed state

- Runner SHA-256: `552a7f08f83f2ba2ce3da29581029dfdd79e86fbb75faeb71356965073228f15`
- Verifier SHA-256: `c1125ef9ff47600f608f2b163311381bdc05097a35b2313b429f61cc5271c4c1`
- Preregistration SHA-256: `bdd71591d461e83165034a4391ace10872265153b75dffc00893f8960109328a`
- Open lock SHA-256: `92416d5a8d2f546b4711f0c9141f7a213a6c1bde58ae48e4b0b135902ff06819`
- Token: `PH1_INTEL_EXECUTION_R8A_AFTER_R8P8_PASS_AND_SOURCE_AUDIT_GO`; `one_attempt:true`.

All 23 lock-bound file hashes match. The R8P8 result/manifest/commit/verifier hashes are respectively `5e77ef9f…`, `b6b70284…`, `4431a491…`, and `577881b1…`; their content is exact 18/18 and 14/14, no-device, with preparation digest `f5a15db…`. R7C2/R7A/R7P are exact PASS9/PASS7/PASS18. The R7A verification output is absent. R7D1 and R8P6 each retain exactly their frozen failure file (`88335dc0…`, 931 bytes; `03e48ed7…`, 2,986 bytes). All six R8A output/failure/quarantine/verifier targets and matching root temps are absent.

## Runner and delegation — PASS

- `authorize()` captures and validates the live R8A invocation before `clean_now()`, chain parsing, payload, compiler, OpenCL, or device use. It requires exact native/orig/application argv, the venv and base interpreter identities, `-I`, `-B`, launcher/config hashes, and local direct-entry evidence.
- R8P8's committed bundle and verification, the complete R7D lock/PASS9/PASS7/PASS18 chain, R7A-verifier absence, and both historical exact failures are checked before delegation.
- R7D1 `authorize()` is never imported or called. The only physical authorization call is the frozen R7A `physical.authorize()`.
- `physical.authorize()` does **not** contain a clean-topology gate; it checks the R7A lock and frozen compile/CPU packages. Therefore the permitted R8P8, R7D1, and R8P6 evidence cannot cause a hidden stale-topology rejection.
- After authorization, `configure()` redirects the frozen R7A `OUT`, `FAILED`, and `QUAR` globals to fresh R8A paths. `physical.execute_authorized()` then calls its own `configure()` and `recover()` using those redirected paths.
- The new outer boundary handles an exception, a committed result, or a returned nonzero separately. Existing R8A state makes future attempts fail `clean_now()`, preserving the one-attempt boundary.

No deterministic runner-side execution blocker was found in this scope.

## Fatal verifier gaps

### 1. New invocation authorization is not independently verified

Runner lines 44–48 require an exact 13-field invocation record. The committed result retains that record in `authorization.r8a_authorization.invocation`.

Verifier line 35 checks only:

```python
isinstance(ext.get("invocation"), dict)
```

It does not require the exact key set, raw/parsed/native/orig/application vectors, venv/base identities, isolation flags, launcher/config hashes, or `direct_entry:true`. An empty or arbitrarily mutated dictionary passes this part of the independent authorization gate. There is no invocation mutation suite.

### 2. Invalid delegated outcomes can be certified as valid negatives

Runner line 111 deliberately records `inherited_evidence_valid`, `new_inherited_failure_count`, `delegated_return`, stage, disposition, and the digest/size/count of inherited R7A evidence. In particular, `success_without_commit` and nonzero-without-one-valid-inherited-failure are protocol-invalid outcomes.

Verifier lines 50–58 ignore those adjudication fields. They accept any one or two recursively found `failure.json` files when each merely has:

- size at most 16 MiB,
- an error string,
- `status == valid_negative_failure`, and
- a boolean `device_opened`.

The verifier does not require the correct kind per root, exact root/directory/file cardinality, exact disposition/stage, total bundle cap, an outer-to-backend digest match, one valid inherited failure, or correlation between the two rows. Consequently the runner's own `success_without_commit` summary or `delegated_nonzero` summary with `inherited_evidence_valid:false` can be emitted as `valid_committed_negative_failure` with `valid_negative:true`.

### 3. Any committed precheck failure is mislabeled a valid negative

For a committed bundle, verifier lines 64–69 construct authorization/provenance checks. The frozen numerical verifier is called only if those prechecks all pass, which is good. But regardless of why a check fails, line 69 labels the result `committed_negative` and sets `valid_negative:not passed`.

Thus a bad R8P8 chain, malformed authorization, wrong historical failure, stale failure path, or other provenance/integrity failure is classified as a valid negative experiment rather than invalid/incomplete evidence. A genuine committed physical negative must be distinguished from a verifier/provenance failure.

## Required repair before a physical attempt

Freeze a verifier-focused successor while keeping the runner science and delegation unchanged:

1. Independently validate the retained R8A invocation with the exact runner schema and expected vectors/identities/hashes/direct-entry facts. Add nonvacuous mutations for every field family and require all to reject.
2. Define exact mutually exclusive terminal states:
   - committed positive;
   - committed physical negative with all authorization, provenance, structure, resources, cleanup, and non-outcome numerical gates valid;
   - outer-only predevice failure with its exact schema/disposition and no backend bundle;
   - delegated backend negative with exactly one valid backend attempt plus exactly one correlated outer summary whose digest, bytes, count, `device_opened`, return code, stage, and `inherited_evidence_valid:true` match;
   - otherwise invalid/incomplete, with `valid_negative:false`.
3. Explicitly reject `success_without_commit`, delegated nonzero without exactly one valid inherited failure, malformed/extra/missing failure files, correlation mismatch, and any failed authorization/provenance precheck.
4. Only after all authorization/provenance checks pass may the frozen numerical verifier decide physical positive versus physical negative. A precheck failure must never become `valid_negative:true`.
5. Add TEMP-only fixtures/mutations covering all terminal states and the exact create-new/one-attempt topology. The verifier must not import the candidate runner.

## Authorization boundary

Do not execute R8A. A verifier-only or fresh-namespace revision may reuse the unchanged physical runner/delegate logic, but its hashes, open lock, output absence, and exact one-attempt command require another independent source audit. No retry, compiler, OpenCL, or device action is authorized by this report.
