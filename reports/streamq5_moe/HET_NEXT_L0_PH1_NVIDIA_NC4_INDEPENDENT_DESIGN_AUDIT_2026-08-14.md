# PH1 NVIDIA NC4 compile-only - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, compiler, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC4 as frozen.**

NC4 resolves the major NC3 architectural blockers, including same-process ownership and literal fixture publication. However, the normative fixture manifest is internally inconsistent in at least three deterministic ways. The negative-evidence and postlink topology contracts also remain insufficient for immutable independent adjudication.

## Integrity and independently recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC4 preregistration | 3,853 | `8092ae9deb294c68d169ad019a1f8fc201f58a79e501f1218a240ab5001b6993` |
| NC4 design | 6,954 | `e7580e5c1a81e734d4ae2984ba9b275d29e3457f9055d5d70efef7b1dab1478a` |
| NC4 fixture manifest | 2,258,998 | `9c35f2520c9c5cfa1c4e2e0e10d31815a9857a124f65a7ca27594797a0c1597e` |
| NC4 closed design lock | 5,557 | `fdefa777f159e46be7569a9bc10cc2f63dd5dc9f416836b73b15d68e2ee731eb` |

All 19 direct bindings match size and SHA-256. All 18 expected implementation, phase-lock, result, failure and quarantine paths are absent. The manifest parses as JSON, declares 294 cases, contains 294 cases and has 294 unique names. Every case has the same exact top-level, injection and disposition key sets; every expected ledger has ten ordered nine-field rows.

The source and name buffers independently match the frozen sizes and hashes. The embedded ELF decodes to exactly 536 bytes with SHA `93abe3a2a7c4f7b4e6b6b9ce202ecc9440a02c3d37a9b9e8f476939d102cf2c8`. Independent parsing confirms ELF64 little-endian ET_REL, EM_CUDA 190, section table offset 216, five 64-byte section headers, the specified string tables, `.text`, `.symtab`, and the three exact symbols. The literal ELF repair is sound.

## Sound NC3 repairs to retain

- Direct one-shot compile process; no compiler subprocess or IPC.
- Exact source/name buffer construction and explicit complete result key set.
- Literal fixture-manifest binding and complete literal ELF bytes.
- Separate positive, compiler-negative, incidental and verifier-negative roots.
- Valid negative transactions are immutable and are not recovery debris.
- Same-process non-owning Win32 loader shape and cleanup ordering.
- Private cache/temp containment intent and narrow no-device/no-repeatability claim.

## Blocking findings

### 1. The normative PTX literal is corrupt and contradicts its own metadata

The manifest declares baseline PTX size 130 and SHA `3b4cde8b9803cd2dd6131ac2776730915a5f2b3c5f17c9b690c08db6143f4336`, matching the design's two `.visible .entry` lines. But the embedded base64 decodes to **128 bytes**, SHA `9205f4f9787ce986fe511a9a86a2632eee087ebad66d3f549bbb09892dc7cb08`, and contains `.isible` twice rather than `.visible`.

The baseline therefore fails its own raw-byte equality, size row, digest and PTX parser. Replace only the literal with the exact canonical 130-byte sequence and independently assert decoded bytes, one terminal NUL, logical text and SHA before freezing the successor manifest.

### 2. All 292 attempted create rows have the wrong pre-call handle

The manifest defines ledger tuple fields as `index,operation,attempted,code,handle_before,handle_after,requested,returned,error`. The program cell is explicitly zero-initialized. Therefore every attempted `nvrtcCreateProgram` row must have `handle_before=0`.

Instead, all 292 cases that attempt create record `handle_before="H1"`. Only the two version-failure cases skip create and retain `0 -> 0`. Baseline is recorded as `H1 -> H1`; null-create cases are `H1 -> 0`; nonnull failure cases are `H1 -> H1`. This defeats the ownership transition the suite is meant to prove.

Regenerate every expected create row as `0 -> H1` or `0 -> 0` according to the exact fake outcome. Keep rows after successful/non-null create at `H1 -> H1`, and destroy success at `H1 -> 0`. Rehash the entire manifest and test this invariant independently rather than patching selected cases.

### 3. `vtx_valid_repeat` contradicts immutable already-complete behavior

The case `vtx_valid_repeat` has terminal `already_complete`, exit 0 and retry false, but also `publish="failure_attempt"` and nonnull `expected_primary="verifier_transaction:valid_repeat"`. A valid repeat must be mutation-free, must publish nothing and has no primary failure.

The compile and preflight valid-repeat cases also carry nonnull `expected_primary` strings, while all three fresh positive cases do the same. This conflicts with the design statement that the manifest records primary failures. Freeze the semantic type of `expected_primary`: if it is an error, it must be null for every positive/already-complete case; if it is merely an exercised boundary, rename it and add a distinct nullable primary-error field. `vtx_valid_repeat.publish` must be `none`.

### 4. A valid compiler-negative transaction discards the raw evidence needed to verify it

The compile-negative transaction contains only `negative.json`, `negative_manifest.json` and `negative_commit.json`. Its `artifacts` object contains sizes and hashes, not raw bytes. For compile failure the diagnostic build log is not retained; for log/PTX/CUBIN retrieval or parser failure, the rejected raw artifact is not retained. An independent verifier cannot reproduce the stated negative from hashes alone.

Freeze stage-dependent negative bundle files. Retain the exact source and every available bounded raw log/PTX/CUBIN artifact, or retain a preregistered bounded prefix plus full size/SHA when an over-cap artifact cannot be stored. The negative manifest must hash every retained evidence file. A compiler-negative may be called valid only if its primary condition and cleanup can be independently recomputed from immutable bytes.

### 5. Postlink incident plus committed terminal can be reclassified as success

NC4 represents a postlink error as a new incidental attempt correlated to an already immutable commit. The topology order then checks “exact committed terminal” before other states and explicitly permits phase failure histories beside a terminal. On a subsequent invocation, a commit plus its correlated postlink durability incident can therefore return `already_complete` even though the originating attempt was classified incidental-invalid.

Freeze a higher-priority correlated-incident rule. A valid commit accompanied by an exact postlink/durability incident is not an unqualified positive/already-complete terminal. It remains immutable infrastructure-invalid unless a separately preregistered adjudication proves durability without rewriting or rerunning compilation. The verifier must reject unexplained or mismatched correlated incidents.

### 6. Static-preflight result size/schema cannot safely carry the literal manifest as inherited

The normative fixture manifest alone is 2,258,998 bytes. The inherited static-preflight positive result cap is 1 MiB and its exact fields include `fixture_manifest` and `fixture_results`. NC4 does not state whether `fixture_manifest` means the full object or a compact identity, nor how 294 full result ledgers fit under the cap.

Freeze it as a compact exact identity such as `{path,bytes,sha256,count}` and define compact per-case result rows/digests, or raise the cap with a preregistered byte bound. The preflight output must not silently omit fields, exceed its cap or embed an implementation-chosen summary.

### 7. Win32 loader ABI remains named but not typed

The ownership sequence is now coherent, but the design still does not publish the exact ctypes argtypes/restypes and `use_last_error` policy for `AddDllDirectory`, `LoadLibraryExW`, `GetProcAddress`, `FreeLibrary`, `RemoveDllDirectory` and `GetModuleHandleW`. Pointer-width errors here can truncate HMODULE/cookie/function addresses on Win64 before any NVRTC row runs.

Freeze the full Win32 signature table, constants and last-error capture semantics. Preflight mutations must alter each argument/result type, flag and null/non-null transition and require rejection before a physical library load.

### 8. The cache gate is not proven feasible or complete

NC4 retains the seven compiler options and relies on `CUDA_CACHE_DISABLE`, `CUDA_CACHE_MAXSIZE`, `CUDA_CACHE_PATH` and `NVRTC_CACHE_PATH`, yet the directly bound NVRTC header documents `--no-cache` as the explicit mechanism for disabling PTX/CUBIN cache use. No bound source establishes that the named environment variables disable NVRTC 13.3's compiler cache.

Moreover, snapshots after API calls cannot detect a transient file created and removed inside a call. Resolve this before implementation: bind authoritative semantics for the chosen variables, add the explicit option with a documented supersession, or permit and fully contain recorded temporary/cache writes inside the private directory while forbidding mutations elsewhere. The current “any child entry/file mutation is invalid” may make a legitimate compile deterministically negative and cannot prove that no transient mutation occurred.

### 9. Recoverable-debris retry semantics are ambiguous in the literal rows

The design permits recovery of one exact `.inprogress` tree and a later clean phase invocation, while every manifest row has `retry=false`, including all stale-stage recoveries. Freeze whether `retry` means no retry in the same process or no future authorized phase attempt. Use separate fields such as `same_invocation_retry=false` and `future_clean_invocation_allowed=true/false`; valid negative/protocol-negative must set the latter false, while recovered transaction debris may set it true.

## Required NC5 repair

Before implementation:

1. replace the PTX base64 with the exact 130-byte literal and assert its digest;
2. regenerate all 292 attempted create rows with pre-call handle zero;
3. correct valid-repeat publication and define primary-error semantics for positive cases;
4. retain stage-appropriate raw compiler-negative evidence;
5. prevent correlated postlink incidents from becoming already-complete positives;
6. freeze compact preflight manifest/result identities within an explicit cap;
7. publish the exact Win32 loader ABI and negative mutations;
8. make cache containment executable and evidence-backed;
9. distinguish same-invocation retry from future clean recovery.

NC4 remains closed. No implementation, preflight, compiler, Driver or device action is authorized.
