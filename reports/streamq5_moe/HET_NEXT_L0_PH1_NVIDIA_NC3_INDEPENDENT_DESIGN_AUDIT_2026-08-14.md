# PH1 NVIDIA NC3 compile-only - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, compiler, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC3 as frozen.**

NC3 corrects the deterministic NC2 buffer error and substantially improves phase separation, schemas and resource ownership. Its claimed 294-fixture suite is count-consistent, but it freezes fixture names rather than the actual injection/expected-evidence matrix. Several negative terminal and crash states also remain contradictory. Those gaps permit implementation-time choices that can change adjudication.

## Integrity checks

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC3 preregistration | 3,574 | `1dfc062f5d68aaae6ad0f6d17153e8d962d8e8c9f7fc20af2ff2e180ad6d522c` |
| NC3 exhaustive design | 13,467 | `f005b54f7dbd9d2a0d989a75278d23b45266d9105ac784c7d36d0a8ad26641c8` |
| NC3 closed design lock | 6,153 | `3c57574b03435738b01772380d096a25c63137f3d3e5e3cba7c3423bdb8d0d93` |

All 19 direct bindings independently match size and SHA-256. All 16 expected source, lock, result, failure and quarantine paths are absent. All execution flags are false.

The source/name repairs independently recompute correctly:

- source has 6,173 bytes and no NUL; source plus one NUL has 6,174 bytes and SHA `34f8f67c033061fc82866b5fe72c88d80c121b5b994dc4ce38d27aa4a0cc3c47`;
- program name has 37 ASCII bytes; its one-NUL buffer has 38 bytes and SHA `78416327c270a471f60289892d406e5e7f145d44e8e7288eb50759dfb1e2c890`;
- fake PTX logical text has 129 bytes; with its single NUL it has 130 bytes and SHA `3b4cde8b9803cd2dd6131ac2776730915a5f2b3c5f17c9b690c08db6143f4336`.

The group arithmetic is also correct: `1+10+10+8+10+7+28+12+23+9+19+17+12+128 = 294`.

## NC2 closures to retain

- Correct one-NUL source and name construction with fixed hashes.
- Separate noncircular artifact rows and truthful `pre_result_serialize` label.
- Separate compile and verifier automata that preserve compile bytes during verifier recovery.
- Exact phase roots, caps, top-level schemas and phase-local allowlists.
- Direct Win32 cdecl loading without an owning `ctypes.CDLL` wrapper.
- Explicit program/wrapper/library/cookie cleanup ordering and after-release module checks.
- A much broader named negative-test inventory and exact narrow claim boundary.

## Blocking findings

### 1. The 294-row matrix is not actually frozen

The design freezes 294 names and generic row fields, but not the material values needed to construct or independently replay those rows. It omits, per applicable fixture:

- the injected NVRTC integer status code;
- exception class and bounded message;
- nonnull fake program sentinel and pointer-before/after values;
- exact requested/returned sizes and raw buffers;
- exact ABI mutation applied;
- exact corrupted byte/field/value for parser and nested-schema mutations;
- the literal ten-row `expected_ledger` content;
- exact `expected_primary` and ordered `expected_secondary` objects;
- exact filesystem starting tree, injected fault index and final tree digest;
- exact result digest or canonical manifest digest.

Saying that the future preflight will publish these values leaves the scientific negative oracle to implementation time. Freeze a literal design-time JSON manifest now, or a complete deterministic generator plus its source hash, output byte count and output SHA. The independent source audit must be able to recompute all 294 rows before any preflight runs.

### 2. The frozen fake ELF cannot be reconstructed from the prose

NC3 gives a 536-byte SHA and several offsets, but the described builder does not specify all bytes. Missing values include most ELF-ident bytes, ELF flags and header fields, every section header's complete name/type/flags/address/offset/size/alignment fields, symbol name offsets/other/section/value/size fields, and all padding bytes. Many different 536-byte ELFs satisfy the prose, while only one can match the frozen digest.

Store and bind the literal 536-byte fixture, publish its complete hex/base64, or freeze complete packing pseudocode with every integer and padding range plus a recomputed digest. A future implementer must not reverse-engineer an unpublished SHA preimage.

### 3. Compiler-negative evidence has no valid immutable topology state

The compile automaton has only `compile_fresh`, `compile_valid`, `compile_recoverable` and `compile_invalid`. Yet terminal classes include a valid bounded compiler failure. On a later invocation, an exact prior compiler-negative attempt is neither fresh nor a positive commit nor stale debris. It therefore collapses into generic invalid state.

Add `compile_valid_failure`: exactly one bounded, schema-valid compiler-negative attempt, no success/staging/quarantine conflicts, immutable return of the same negative classification/exit without compile, deletion or quarantine. Add distinct fixtures for first compiler-negative publication and valid-negative repeat.

### 4. Verifier-negative evidence is incorrectly recoverable and retryable

Given a valid compile bundle, NC3 places a “bounded verifier failure” in `verifier_recoverable`, quarantines it, aborts and permits a later verifier retry. That can erase and reclassify a genuine independent protocol negative.

Separate at least:

- recoverable verifier transaction debris, before a valid terminal verifier decision;
- `verifier_protocol_negative`, a valid immutable verification result/failure proving a check false, never quarantined merely to retry;
- verifier writer/infrastructure invalid evidence, which does not establish either positive or protocol-negative science.

Only stale/corrupt uncommitted debris may be quarantined and retried. A deterministic negative against an unchanged compile bundle must remain terminal.

### 5. The exact compile-result schema still conflicts across revisions

NC2 inherits an exact top-level `artifact_sizes` field. NC3 introduces `result.json.artifacts` and says NC2 nested lists remain “except `artifacts`,” although NC2 had no field by that name. It never explicitly publishes the complete revised top-level result key set or says `artifact_sizes` is removed. An honest implementation/verifier can therefore disagree about one versus both fields.

Freeze the entire NC3 result top-level key list, explicitly remove `artifact_sizes`, and bind the exact `artifacts` map/row order and disassembly sentinel. Apply the same complete-key treatment to every terminal result rather than relying on ambiguous inheritance.

### 6. Postlink failure evidence is not representable under the frozen topology

The preflight failure attempt is exactly one `failure.json`; compile failure schema is defined as that schema plus fields. Yet all phase matrices require `postlink` and writer-secondary cases. Once a create-new canonical `failure.json` has been linked and flushed, a later directory-flush/cleanup failure cannot be appended to it without overwrite. NC2's optional `postlink_secondary.json` mechanism is not clearly retained, and allowing it would contradict the NC3 one-file attempt topology.

Freeze phase-specific one-file versus two-file attempt shapes, exact sidecar fields/cap/hash linkage, and what happens if the sidecar writer also fails. The topology verifier must distinguish a valid bounded postlink-secondary attempt from orphan debris and must never reinterpret a writer failure as a compiler negative.

### 7. The ownership-safe “compiler child” has no process protocol

The Win32 handle design is safer than NC2, but NC3 newly relies on a compiler child without freezing whether that means the top-level one-shot runner or a spawned subprocess. If spawned, no exact child script, argv/token, environment, stdin/stdout framing, timeout, raw-artifact transport, maximum message size, parent/child failure precedence or process/job cleanup is specified. Transferring up to 52 MiB through unspecified stdout would also conflict with the one-line verifier protocol and bounded evidence.

Either state that the directly invoked runner itself is the one-shot compiler process and remove parent/child ambiguity, or freeze a separate hash-bound child and exact IPC/artifact transaction. In both cases publish the Win32 function signature table for `AddDllDirectory`, `LoadLibraryExW`, `GetProcAddress`, `FreeLibrary`, `RemoveDllDirectory` and `GetModuleHandleW`, including `use_last_error`, handle widths and secure flags.

### 8. Compiler-cache feasibility is unresolved

The I/O contract says cache presence is diagnostic but any cache mutation is invalid. It does not name the cache paths/environment or say how all relevant directories are found before load. The pinned header exposes `--no-cache` specifically to disable PTX/CUBIN cache use, but NC3 freezes the seven-option vector without it. Thus a legitimate NVRTC cache write can deterministically invalidate an otherwise correct compile, and an unenumerated cache write can evade the snapshots.

Freeze an explicit policy before implementation: either add and bind `--no-cache` (with the corresponding supersession of the inherited option vector), or run with a complete named disposable/read-only cache environment and classify its exact mutations. Do not claim `unexpected_filesystem_mutations=0` from a partial directory list.

### 9. Phase failure/recovery schemas still need exact valid-negative dispositions

NC3 improves paths and caps, but it does not fully freeze:

- static-preflight valid-negative repeat versus recoverable corrupt result;
- postcommit verifier protocol-negative artifact path/schema;
- failure-attempt manifest or digest binding when a second sidecar exists;
- exit behavior when cleanup succeeds but positive publication fails;
- whether quarantine disposition itself belongs inside or outside the moved payload;
- exact handling of more than one invalid bounded entry without partially mutating before discovering a later collision.

Classify the entire family read-only before the first move, validate all collision targets, then apply a preregistered all-or-bounded-partial disposition policy. TEMP tests must use the actual production classifiers and writers, not schema-only stand-ins.

## Required NC4 repair

Before implementation:

1. freeze a literal/recomputable 294-row fixture manifest with all injection and expected-evidence values;
2. bind literal fake ELF bytes or a complete byte-for-byte builder;
3. add immutable compiler-negative and verifier-protocol-negative states with no retry reclassification;
4. publish the full revised compile result key set;
5. make postlink secondary evidence representable and independently verifiable;
6. freeze the actual one-shot/compiler-child process and Win32 ABI contract;
7. resolve compiler-cache paths and mutation policy;
8. close phase-specific valid-negative, sidecar, quarantine and multi-entry failure semantics.

NC3 remains closed. No implementation, preflight, compiler, Driver or device action is authorized.
