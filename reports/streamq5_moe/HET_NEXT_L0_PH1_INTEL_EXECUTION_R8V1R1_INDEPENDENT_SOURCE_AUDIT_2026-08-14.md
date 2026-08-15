# HET-NEXT-L0 PH1 Intel execution R8V1-R1 — independent closed-source audit

Date: 2026-08-14  
Scope: read-only audit of the frozen verifier, preregistration, lock, immutable R8A5 evidence, and closed-phase topology. No verifier execution, payload/model forward, compiler, OpenCL, or device call was performed.

## Verdict

**GO to construct a separate authorization-only successor.** Do not execute this closed R8V1-R1 source. Its lock is deliberately closed and its invocation token is pending independent audit. A later auth revision must bind this audit, add its exact provenance paths to the literal family set, retain fresh output absence, and receive its own final authorization audit.

The three prior R8V1 NO-GO findings are closed nonvacuously:

1. the immutable failed R8A5 verifier is validated as the exact 27/29 protocol-negative artifact;
2. the new result explicitly separates that prior protocol-negative from the positive immutable-bundle adjudication;
3. the canonical output is created only after every current check is true.

## Frozen package

- verifier: `b7d762a15e7ba4a4adf2264daf1f46fe3194cc01bc110d08395b2a6dfa360fdb`, 15,196 bytes;
- preregistration: `58b78061944eeaff605261a894689126213ed5cc32a7a65e9e31855ff77c4e23`, 3,268 bytes;
- closed lock: `97ab375270e0d3410f03116d93d6c21bf510673f18c16f9d459bc62b74b863b2`, 1,307 bytes.

All 13 hash bindings in the closed lock match: current verifier/preregistration; frozen R8V1 verifier/preregistration/lock/independent audit; immutable R8A5 result/manifest/commit/failed verifier/topology diagnosis/post-run audit/post-run JSON. The lock is `execution_open=false`, token `PENDING_INDEPENDENT_R8V1R1_SOURCE_AUDIT`, and `one_attempt=true`.

Before this audit file was written, the literal case-preserving family contained exactly the preregistered 13 entries. The canonical R8V1-R1 output, failure root, quarantine root, and in-progress artifacts were absent.

## Exact prior-verifier adjudication

`prior_verifier_contract()` at lines 88–91 requires:

- SHA-256 `d6b630658c59e1c6913ba099bb8d617fe1b451e14e31ee38b68d351fb9fde917`;
- exactly nine top-level fields and the frozen kind/claim;
- `pass=false`, `passed=27`, `total=29`;
- `terminal_state=invalid`, `terminal_valid=false`;
- exactly 29 frozen check names;
- exactly `{topology, terminal_contract}` false, all other 27 checks true, and every check value a real boolean;
- exactly 31 frozen mutation names, all true.

The immutable artifact matches this contract exactly. It remains `verifier_protocol_negative`; R8V1-R1 does not reclassify it as positive.

Lines 93–108 implement 17 independent in-memory negative cases: wrong kind, claim, pass, passed, total, terminal state and validity; extra and missing top-level fields; missing/extra check; a third false check; removal of one required false check; non-boolean check; and missing/extra/false matrix entries. Every structural class is rejected by the same production contract.

## Topology and provenance

The closed expected set is exactly:

- the eight frozen R8A5 bundle/runtime/provenance entries;
- old R8V1 preregistration, lock, and independent audit;
- R8V1-R1 preregistration and closed lock.

Literal directory enumeration is authoritative and case-preserving. Both Windows glob observations are compared against separately frozen expected diagnostic sets. The mutation suite has `13 + 8 = 21` required rejections: each missing entry plus uppercase extra, lowercase extra, orphan, temporary, failure, quarantine, case-only change, and casefold collision.

Writing this audit intentionally ends the 13-entry pre-audit topology. The auth-only successor must explicitly include this audit and its own new preregistration/lock paths. It must not reuse the closed set unchanged.

## Retained scientific and physical checks

R8V1-R1 reuses the hash-pinned frozen R8V1/R7A logic without changing scientific arithmetic. It requires:

- the exact three-file result/manifest/commit bundle;
- exact R8A5 authorization and invocation evidence;
- all 15 direct-physical aggregate checks true, covering all 18 physical result gates;
- exact Intel identity, 102 ledger rows, 95 ownership rows, 14 host-USM allocations, 18 pointer arguments, four launches, one finish, nine reads, 21 releases, cleanup with zero live resources, 22 controls, extension/forbidden counters, resources, and five stage hashes;
- exactly 20 frozen independent CPU numerical checks, all true;
- the independently replayed 31-case terminal matrix, all true.

The numerical verifier rebuilds the official source-Q5/BF16 graph on CPU and does not call a compiler, OpenCL, or a device. The terminal-matrix import reaches only device-free module definitions and calls the pure filesystem mutation harness; it does not open a device API.

## Output and lifecycle

Lines 171–179 enforce exact invocation and the closed/open lock transition. In the future open successor, any false current check returns nonzero at line 177 before `write_positive()` is called. Therefore no canonical negative or partial output is published.

Only an all-true adjudication may create the canonical output. Its schema explicitly records:

- `prior_verifier_outcome="verifier_protocol_negative"`;
- `bundle_adjudication="positive"`;
- `terminal_state="positive"`;
- `terminal_valid=true`;
- `pass=true` and all current checks passed.

The create-new writer uses an exclusive temporary file, flush plus `fsync`, a no-overwrite hard link to the canonical path, and finally removes the temporary path.

## Authorization boundary

Authorized next action: construct and freeze an auth-only successor that changes only namespace/path bindings, the exact audit-inclusive family set, the open token/lock, and the expected invocation vector. Then perform one final independent source audit before any CPU-only verifier execution.

Not authorized: execution of the closed R8V1-R1 source, physical R8A5 rerun, compiler, OpenCL, device use, or any broader claim.

Even after a future successful verifier run, the maximum claim remains one official real expert/input Intel correctness component. It is not a performance, full-layer, full-model, heterogeneous, industrial-readiness, or breakthrough result.

