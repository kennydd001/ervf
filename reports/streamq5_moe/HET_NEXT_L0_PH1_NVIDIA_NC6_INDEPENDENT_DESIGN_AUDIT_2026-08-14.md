# PH1 NVIDIA NC6 compile-only - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, compiler, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC6 as frozen.**

NC6 soundly repairs the NC5 postcommit classification, debris taxonomy and shared production-function architecture. The new cache and embedded-prefix specifications nevertheless contain deterministic evidence gaps: the retained-file fixture cannot be materialized from its normative bytes, its tree omits a real parent directory, the promised cache mutations are absent, and over-cap evidence no longer binds the full artifact.

## Integrity and recomputed invariants

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC6 preregistration | 3,056 | `205f029b79d6e53c27c502f9c88dcfd76e5f2ffe16cac19f1ecd6914b1d72352` |
| NC6 design | 4,689 | `f223f3f650a82819c7047dfe229739b95a21b6540adbc2fa8f7f5e6ab4aea9e6` |
| NC6 fixture manifest | 2,657,451 | `3d2ff77232005199da87024d6518e0fcce46a38c562b0e40ed123b959f2a0609` |
| NC6 closed design lock | 5,846 | `b6272b33c22f167a8cdccb123037a5ad0e207f917c1da327df96a6cd0b0ece40` |

All 19 bindings match byte count and SHA-256. All 19 expected implementation, lock and output paths are absent; all execution flags remain false.

The manifest parses with 297 cases, 297 unique names and one uniform case keyset. Independent checks confirm 295 attempted create rows with zero prehandle, no invalid primary union, seven and only seven next-allowed `transaction_debris` rows, 217 nonretryable `incidental_failure` rows, three nonretryable `postcommit_incident` rows, three correct repeat rows, three 4,096-byte embedded prefixes with matching prefix SHA, and three cache cases. The canonical 130-byte PTX and 536-byte ELF remain correct.

## Sound NC5 repairs to retain

- Postlink errors are now `postcommit_incident`, exit 3, higher priority than the immutable terminal and not directly retryable. The original compile is not upgraded until a separately authorized CPU durability adjudication succeeds.
- Recoverable states are now exactly `transaction_debris`; incidental terminals never allow a later compile.
- The import-inert shared contract has exact exports and runner/preflight function-object/code-identity gates.
- Cache files have a declared retained `private_tree/` bundle location and a canonical observation schema.
- Over-cap prefixes are embedded rather than path-inferred.

## Blocking findings

### 1. The retained-cache fixture has no normative file bytes

`cache_private_file_retained` specifies `nvrtc/cache.bin`, size 18 and SHA `bc83d5765f33126ec5098d12a88a855dcefc94d8a4070a9187c0adeec02cdc3d` (manifest around lines 45401 and 45554-45558), but provides no hex/base64 bytes or deterministic byte derivation. That digest is not a reversible fixture definition. An implementation cannot create the required TEMP file and independently hash it without introducing an unregistered literal outside the sole normative matrix.

Embed the exact 18 bytes plus encoding and SHA in the manifest. Require preflight to materialize those bytes on disk before invoking the actual shared classifier.

### 2. The cache snapshot omits the parent directory and conflicts with the frozen enumeration rule

Creating `private_tree/nvrtc/cache.bin` necessarily creates `private_tree/nvrtc/`. The design says the terminal manifest enumerates every directory/file under `private_tree/` and the observation schema includes `path,type,size,mtime_ns,sha256`. Yet `cache_private_file_retained.after_entries` contains only the file. Its published tree digest is the digest of that one file row; no `nvrtc` directory row is included.

Freeze directory semantics explicitly. If directories are normative, include the `nvrtc` row and regenerate the digest. If observations are file-only, remove `type` ambiguity and the “every directory/file” claim, and separately prove reparse/symlink/directory topology rejection.

### 3. The manifest does not contain the cache mutation coverage claimed by the design

Design line 21 promises mutations for entry order, path, type, size, mtime, SHA, tree digest, missing/extra retained file, traversal and external write. The 297-case manifest contains only three direct cache cases plus one inherited coarse `nested_filesystem_compiler_cache` field mutation. It has no distinct executable cases for the promised entry-level mutations.

Because the manifest is declared the sole normative matrix, later code cannot invent these mutations. Add literal named cases for every promised mutation and require each to execute the actual shared cache/tree validator on a TEMP filesystem.

### 4. The cache observation schema cannot retain the mandated per-call history

The design requires snapshots before load, after every NVRTC call, after unload and precommit. The exact result cache object is frozen as only `private_root,before,after,tree_digest`; a snapshot is only `qpc,entries,tree_digest`. There is no ordered `snapshots` collection or stage/operation identity for the intermediate observations.

Add an exact ordered history schema with stage names and one snapshot per required boundary, or narrow the claim to the two actually retained snapshots. Add missing/duplicate/reordered-stage mutations.

### 5. Embedded over-cap evidence no longer binds the full raw artifact

Each `bounded_prefix_embedded` object contains only `source,offset,length,base64,sha256,derivation`. It omits the observed full byte count and full SHA-256. The synthetic fixture can infer “cap plus one” from its injection, but a physical over-cap NVRTC output can be arbitrary. A 4,096-byte prefix cannot independently identify or verify the discarded remainder.

Restore exact `full_bytes` and `full_sha256` fields for the physically observed artifact, retain the embedded prefix SHA separately, and mutate both full fields. This is required for the valid-negative claim; otherwise many different raw outputs produce the same retained evidence.

### 6. Exact redirected subdirectory identities remain unspecified

The design says `CUDA_CACHE_PATH`, `TMP`, `TEMP` and `NVRTC_CACHE_PATH` point to exact subdirectories, but never names the four relative paths or states whether any may alias. The fixture only uses generic `private_root="cache"` and a single `nvrtc/` child. Filesystem topology and environment provenance therefore remain implementation choices.

Freeze four exact normalized relative paths, require them to be distinct or explicitly alias as intended, record their resolved paths and environment before/after values, and add swap/alias/absolute/traversal/restore-failure mutations.

## Required successor repair

Before implementation:

1. embed the exact retained cache-file bytes;
2. resolve directory-versus-file snapshot semantics and regenerate the tree digest;
3. add the promised executable cache mutation cases;
4. freeze the complete ordered per-call snapshot history;
5. restore full byte count/SHA for over-cap raw artifacts;
6. name and validate every redirected cache/temp subdirectory.

NC6 remains closed. No implementation, preflight, NVRTC, Driver or device action is authorized.
