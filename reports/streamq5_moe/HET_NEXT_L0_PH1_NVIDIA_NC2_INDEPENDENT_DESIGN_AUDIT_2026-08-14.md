# PH1 NVIDIA NC2 compile-only - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only audit. No source import, preflight, compiler, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC2 as frozen.**

NC2 closes most NC1 design gaps, but it contains one deterministic ctypes buffer error and several contradictions between its exact fixture list, topology classifier and evidence schema. These are preregistration defects, not empirical failures, and should be repaired before any implementation is written.

## Integrity and topology

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC2 preregistration | 5,140 | `d82db72d3e1600d313addfe18bda979a576f1536cf73bceaf5d42f409a32648b` |
| NC2 design | 13,076 | `cb1e6cfec5240382b94a33cca00d931bc6830c5cf26926104e0edac5e204fb95` |
| NC2 closed design lock | 6,912 | `cb432fbd4408f0d6f5f030dc4834e746352da639d2b3042041e375771595b6ca` |

All 18 direct lock bindings independently match size and SHA-256. All 14 expected implementation, lock, preflight, compile, failure, quarantine and verifier paths are absent. The closed flags and pending token are coherent. The 84 listed fixture names sum correctly and are unique by construction of the ten documented groups.

## Correct NC1 repairs to retain

NC2 now freezes the cdecl type aliases and ten-function table, exact runtime/base/venv identities, native-versus-launcher argv semantics, three phase-local ACKs, raw PTX/log NUL rules, an isolated verifier command, durable postcommit verification, a four-state topology concept, resource/cleanup rows and bounded success/failure transactions. The toolchain and two-entrypoint/four-launch distinction remain correct. The claim explicitly excludes numerical, device and byte-repeatability conclusions.

## Blocking findings

### 1. Frozen source-buffer construction creates two NULs, not one

Preregistration line 15 specifies:

`ctypes.create_string_buffer(source_bytes + b"\0")`

and simultaneously requires a 6,174-byte buffer containing 6,173 source bytes plus exactly one NUL. In the pinned CPython 3.12 standard library, `create_string_buffer(init)` sets `size=len(init)+1` when `init` is bytes. Passing the already NUL-terminated 6,174-byte value therefore creates a **6,175-byte buffer with two terminal NUL bytes**. This contradicts the preregistration, `source_buffer_bytes=6174` in the lock, and the proposed source-buffer digest.

Freeze either `ctypes.create_string_buffer(source_bytes)` with `sizeof(buffer)==6174` and exact `.raw == source_bytes+b"\0"`, or pass the explicit size 6,174 with the already terminated initializer. Apply the same explicit `sizeof`/`.raw` rule to the program-name buffer so its construction cannot acquire a second NUL.

### 2. The exact 84-fixture manifest regresses inherited NC1 coverage

The NC2 manifest is exact, so omitted negative cases cannot be assumed to exist under another name. It omits cases that NC1 directly required:

- PTX reported size zero;
- CUBIN reported size zero and one;
- returned-buffer versus size-row disagreement;
- separate over-cap log, PTX, CUBIN and JSON cases;
- distinct shard, D2, CPU-stage/LUT and model/tokenizer reads;
- cudart load, CuPy import, Driver-symbol call, context/device query and preauthorization output mutation;
- mixed/multiple topology, quarantine collision, commit-write/promotion failure and failure-writer primary failure;
- mutations of every nested success/failure/verifier schema field.

The generic names `payload_open`, `driver_load`, `oversize` and `topology` do not freeze multiple independent inputs and expected dispositions. Likewise the ten generic production-verifier mutations cannot establish the design's claim that every schema field and terminal class is nonvacuously rejected.

NC3 must publish one literal mutation manifest with unique names, exact baseline fixture hashes, injected boundary, expected primary/secondary ledger and expected terminal classification. Grouped cases may share helper code but not one undifferentiated Boolean.

### 3. A postcommit-verifier crash state is unreachable in the classifier

The postcommit verifier can create a staging tree only after the seven-file compile bundle is valid. Yet `recoverable_stale` permits a verifier-staging tree only when there is **no valid commit**. `valid_committed` rejects any temp/orphan. Consequently the realistic state “valid compile commit plus one stale verifier staging tree” falls into `invalid` rather than a verifier-recovery state.

That can either strand verification permanently or cause the generic invalid handler to quarantine good compile evidence. Add explicit mutually exclusive states for:

- valid compile bundle with verifier absent;
- valid compile bundle with valid verifier;
- valid compile bundle with exactly one recoverable verifier temp/corrupt uncommitted verifier tree;
- valid compile bundle with valid bounded verifier-negative evidence;
- invalid mixed/multiple verifier states.

Recovery must preserve the valid compile bundle, quarantine only verifier debris once, abort that invocation, and permit the separately authorized verifier on a later clean invocation. Compile must never rerun.

### 4. Success evidence contains circular and temporally impossible fields

The proposed `result.json` contains `artifact_sizes` for `result`, `manifest`, `commit` and `total`. The byte size of `result.json` depends on the decimal values stored inside itself; manifest and commit sizes are not known until later serializations whose content depends on result/manifest hashes. No canonical fixed-point algorithm or proof of convergence is frozen.

The same result also contains a resource sample named `post_serialize`. A final sample cannot both occur after serialization and be included in that already serialized immutable result. Serializing again makes the recorded event merely post-provisional-serialization.

Remove self/future sizes from result and let manifest/commit carry them, or freeze and executable-test an exact fixed-point constructor. Rename the final in-result sample to a truthful `pre_result_serialize`/`post_raw_artifacts` stage; any genuine postcommit sample belongs in a later independent evidence artifact.

### 5. Preflight and verifier recovery paths/schemas remain incomplete

The preflight says a stale/corrupt result is quarantined, but neither document nor lock names a preflight quarantine root or exact disposition schema. The postcommit verifier similarly defines a positive tree and failure root but no complete stale/corrupt/orphan recovery transaction or quarantine path. Its `verification_manifest.json` and `verification_commit.json` are described relationally, not by exact field sets, canonical JSON rules and caps. Preflight and verifier failure-attempt schemas are also not closed.

Freeze every path in the source/preflight/verifier locks, exact one-file failure and quarantine disposition schemas, valid-repeat behavior, prelink/postlink/fsync/writer-secondary states, and independent TEMP tests using the actual phase-local functions.

### 6. Several nested contracts are still not exact enough to verify independently

The compiler result's nested field names are improved, but the following remain open:

- exact row fields for the ordered `runtime_modules.rows` and how system/non-system classification is derived;
- normative expected `after_release` state for NVRTC and builtins;
- equality between ledger destroy row and cleanup destroy row;
- exact `filesystem_observation` fields and permitted read/write set;
- exact compiler cache observation semantics;
- exact `mutation_results`, preflight `fixture_results` and `transaction_results` schemas;
- exact positive/negative verifier manifest and failure field sets;
- process exit codes for compile-positive, compiler-negative, infrastructure-invalid, writer-invalid and already-complete outcomes.

Preregistration line 54 also says only the CUDA source may be opened after authorization, while authorized operation necessarily reads locks, implementation files and toolchain bytes and writes its evidence. Replace this literal contradiction with an exact allowlisted read/write policy distinguishing scientific payload from implementation/toolchain/evidence paths. Counters must be incremented at the common guarded boundary before the attempted operation.

### 7. Compiler handle cleanup needs an exact ownership-safe mechanism

NC2 requires `_ctypes.FreeLibrary` on a handle owned by a `ctypes.CDLL` object and then an after-release module snapshot. Freeze how the CDLL wrapper is made non-owning or otherwise prevented from a later duplicate release, keep every exported function wrapper alive only through the final call, and define the expected loaded/unloaded status of both NVRTC and builtins after release. The DLL-directory cookie must still close once after compiler release even if destroy or FreeLibrary fails.

An isolated one-shot compiler child whose process exit is the ultimate loader cleanup boundary is safer, but if chosen it must be frozen now together with its parent/child evidence and failure protocol.

## Required NC3 repair

Before implementation:

1. correct and assert the exact source/name buffer construction;
2. expand the fixture manifest to every inherited raw-size, cap, forbidden-I/O/call, transaction and nested-verifier mutation;
3. add valid-bundle-plus-verifier-debris recovery states that preserve compile evidence;
4. eliminate result-size/resource-sample circularity;
5. close preflight/verifier failure, quarantine, manifest and commit transactions;
6. freeze every nested result and terminal exit schema plus an exact allowed filesystem policy;
7. freeze ownership-safe NVRTC unload and after-release module expectations.

NC2 remains closed. No implementation, preflight, compiler, Driver or device action is authorized.
