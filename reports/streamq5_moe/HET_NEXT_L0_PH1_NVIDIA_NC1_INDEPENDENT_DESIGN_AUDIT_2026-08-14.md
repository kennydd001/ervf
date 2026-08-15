# PH1 NVIDIA NC1 compile-only - independent design audit

Date: 2026-08-14  
Mode: frozen design-only, read-only audit. No candidate import, preflight, compiler, payload, Driver or device call was made.

## Verdict

**NO-GO for source implementation from NC1 as frozen.**

NC1 materially repairs all seven NC0 finding clusters, and its narrow compile-only claim remains appropriate. The remaining blockers are evidence-protocol details, not objections to a one-program NVRTC experiment. They must nevertheless be frozen before implementation because they affect compiler inputs, raw artifact identity, authorization, independent adjudication and crash-state interpretation.

## Frozen integrity and absence

The handed-off files match exactly:

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC1 preregistration | 6,461 | `9a823c1c061a91321d6256c79c203f3bbf25c6956b7f51eddf336f13a969bef9` |
| NC1 static-preflight/verifier/lifecycle design | 8,444 | `c8e781ca3744c53cd99010f12b2775937dc553e13bf4325863110f48847fa25a` |
| NC1 closed design lock | 5,012 | `f0db311f4630c766cbfea8affd30afeebd0ba72ade7236886afaa69376293635` |

All 14 lock bindings independently rehash true. All eight `expected_absent` implementation, preflight and output targets are absent. Before this report, the NC1 family contained only the preregistration, design and closed lock. The lock correctly has `implementation_open=false`, `preflight_open=false`, `compile_open=false` and a pending token.

The bound CUDA source is 6,173 bytes with SHA-256 `9f369ab3621c6d56b2a3597bca59c25be8d15e7ac3a2a150d916d6695623a781`. Direct source inspection confirms exactly two `extern "C" __global__` definitions, `q5_linear` and `bf16_lut_activation`. The four later labels are correctly classified as out-of-scope launch uses, not four compiled entrypoints.

The bound NVRTC DLL, builtins DLL and header sizes and hashes also match. The installed header explicitly lists `sm_120` and all seven frozen options as supported. It documents that `nvrtcGetPTXSize` includes the terminal NUL and that CUBIN size is nonzero for an actual rather than virtual architecture.

## NC0 repair assessment

NC1 correctly supplies:

1. two source entrypoints versus four future launch uses;
2. an import-inert compiler core and an isolated `python -I -B` fake-library child;
3. direct NVRTC, builtins and header identities plus absolute cdecl loading and module snapshots;
4. a coherent ten-row transition policy, including diagnostic log retrieval after compile failure and primary/secondary precedence;
5. PTX target/address/entry checks and an honestly narrowed CUBIN claim;
6. named success/failure roots, caps, commit-last publication, quarantine and fault categories;
7. an exact future ACK, pre-source authorization order, forbidden-payload/device counters and direct future-lock closure.

Those repairs should be retained.

## Remaining blockers

### 1. The complete `nvrtcCreateProgram` input is not frozen

The preregistration freezes source bytes and compile options, and the design says the ABI will be exact, but neither document freezes the remaining material create operands. The installed header's signature is:

`nvrtcCreateProgram(nvrtcProgram*, const char* src, const char* name, int numHeaders, const char* const* headers, const char* const* includeNames)`.

NC1 must state the exact UTF-8 source-buffer construction and terminal-NUL rule, the exact program-name bytes (or exact NULL choice), and exactly `numHeaders=0`, `headers=NULL`, `includeNames=NULL`. Program name can appear in diagnostics and generated text, so it is part of the replayable compiler input. It must also publish the full ten-function ctypes signature table (`nvrtcProgram`, `size_t`, pointer types, output-buffer types and `restype=c_int`) rather than leaving the expected ABI to implementation.

### 2. PTX raw-byte versus logical-text semantics are incomplete

The local pinned `nvrtc.h` says the PTX size includes the trailing NUL. NC1 defines exact terminal-NUL behavior for `build.log`, but not for `ptx.bin`. Requiring only UTF-8/ASCII decoding is insufficient because NUL is valid in both and an embedded or duplicated NUL could pass textual directive checks.

Freeze all of the following:

- retained `ptx.bin` is exactly the reported raw byte count;
- it has exactly one terminal NUL and no earlier NUL;
- textual parsing operates only on the bytes before that NUL;
- PTX digest and manifest cover the retained raw bytes including the terminal NUL;
- fake and verifier mutations cover missing, embedded and duplicate NULs independently of size disagreement.

Apply the analogous no-embedded-NUL rule to multi-byte `build.log` while retaining the already correct one-byte-NUL empty-log case.

### 3. Static-preflight publication is not designed

The lock anticipates `het_next_l0_ph1_nvidia_nc1_static_preflight_result.json`, but the design gives it no exact kind/schema, gate names and count, pass criterion, no-device counters, command/identity record, byte cap, manifest/hash binding, create-new writer, failure disposition or repeat behavior. It also does not freeze a distinct closed-to-open preflight authorization token and exact command vector.

The fake categories are broad but not an exact executable fixture manifest. Freeze baseline fixture byte digests, exact mutation names/cardinality and expected rejection for every row transition, ABI mutation, authorization mutation and artifact mutation. Otherwise an implementation can satisfy the prose with a vacuous or selectively incomplete suite.

### 4. Independent verification has no exact process or postcommit evidence contract

The proposed verifier is structurally independent, but NC1 does not freeze:

- the exact isolated verifier command, interpreter and argv;
- how the runner passes a staging path without importing verifier code;
- verifier stdout/exit schema and timeout;
- the canonical postcommit independent-verification path and exact result schema;
- create-new/atomic failure behavior for that verification result;
- how a separately invoked verifier binds the committed three-file transaction and records positive versus compiler-negative outcomes.

Precommit validation alone, invoked by the candidate runner, is not durable evidence that the committed bundle was independently adjudicated. The claim may say "independently verified" only after a separately bound verifier result exists. The current lock's absence list contains no such target.

### 5. Physical runtime and authorization identity remain placeholders

The preregistration says an interpreter check occurs before source read, but freezes no expected interpreter path/hash, base/venv identity, version, `sys.executable`, direct-entry predicates, `sys.orig_argv`/`sys.argv`, or exact `-I -B` physical invocation. Only the fake child gets an explicit `-I -B` contract.

Freeze the actual compile runner and verifier interpreter identities and exact direct command vectors before their authorization revisions. The exact ACK is already sound. Wrong ACK, wrong interpreter, extra argument, imported entry and hash drift must all return nonzero without creating output, failure, quarantine or temporary paths.

### 6. Clean topology and recovery order are ambiguous

The preregistration places a clean-topology check before recovery. The lifecycle design also requires a valid committed bundle to return `already_complete`, and corrupt/stale states to be quarantined by recovery. A strict "all paths absent" clean check would make both recovery branches unreachable.

Freeze an exact pre-mutation topology classifier with mutually exclusive states: fresh, one exact valid commit, recoverable nonclean, and invalid/multiple/oversize. Authorization and hash checks must precede it; correct authorization may inspect and then either return `already_complete`, perform one bounded quarantine and abort, or proceed from fresh. Wrong authorization must never recover or write. TEMP tests must call the actual classifier/recovery functions for each state.

### 7. Nested evidence/resource/cleanup schemas are still open

The top-level success and failure keys are named, but the exact nested field sets and types for `authorization`, `source_identity`, `toolchain_identity`, all three `runtime_modules` snapshots, each ledger row, `artifact_sizes`, `exclusion_counters`, `resource_samples`, `cleanup` and `dispositions` are not frozen. In particular:

- the `os.add_dll_directory` cookie and compiler-library lifetime have no cleanup/disposition record;
- `resource_samples` has no labels, sampler, threshold or failure policy;
- postlink directory-flush or cleanup failure cannot be appended to an already create-new canonical `failure.json`, yet its exact secondary-evidence location is unspecified;
- success, compiler-negative, infrastructure-invalid and writer-failure terminal classes are not given one exact mutually exclusive adjudicator;
- fake/preflight/verifier mutation names and counts are not fixed for these nested objects.

Freeze exact nested schemas, ownership rows and the canonical location of post-publication secondary evidence. If resource values remain diagnostic, state that explicitly and still freeze their types and nonfinite/error representation. The compiler process should also record whether any compiler cache or unexpected non-output filesystem mutation occurred; NC1 makes no byte-replayability claim and should not imply one without a deterministic-repeat contract.

## Required NC2 design repair

Before source implementation:

1. freeze the complete create-program operands and ten-function ctypes ABI table;
2. freeze PTX/log raw-NUL canonicalization and mutations;
3. define exact static-preflight artifact, authorization, transaction and fixture manifest;
4. define isolated precommit invocation plus a separately committed postcommit verifier result;
5. freeze physical interpreter/direct-entry/argv identities;
6. define the reachable fresh/valid/recoverable/invalid topology state machine;
7. close every nested evidence, resource, ownership, cleanup and terminal-adjudication schema.

NC1 remains closed. No preflight, compiler, Driver or device action is authorized.
