# NC1 static preflight, verifier and lifecycle design

## Future source architecture

Implementation is intentionally absent at design freeze. A later source revision must split into four frozen files: `compile_core.py`, `compile_runner.py`, `compile_verifier.py` and `static_preflight.py`.

`compile_core.py` is stdlib-only and import-inert. It may import only `ctypes`, `hashlib`, `json`, `os`, `pathlib`, `struct` and typing/dataclass helpers. It accepts immutable source bytes, the exact option tuple and an injected NVRTC object; it returns immutable artifacts plus the ten-row state-machine evidence. It has no repository imports, CLI, output paths, ambient recovery, `CDLL` call or top-level action. The production runner alone constructs the real cdecl library object after authorization and passes it to this same core function.

The static preflight first rehashes the absolute core path and AST-checks its imports/body. It then starts an isolated `python -I -B` child using a frozen, stdlib-only bootstrap. The bootstrap uses `importlib.util.spec_from_file_location` on that already hash-checked absolute core path and injects a fake NVRTC object before calling the exact production core function. It never imports the runner, backend or verifier. The child rejects any attempt to invoke a real `CDLL`/`WinDLL`, open a non-source/payload path, import a repository module or create a file. Exact child argv, empty stdin, bounded timeout, JSON stdout schema and exit zero are required.

## Executable static gates

AST and fake-library mutation suites require:

- exactly two source entrypoints and their exact signatures; a mutation asserting four entrypoints fails, while a separate constant records four future out-of-scope launch uses;
- exact cdecl ABI argtypes/restype for all ten named NVRTC functions, pointer widths, call cardinality/order and the exact seven options;
- the full success ledger and every transition in the preregistered state machine, including compile-error log retrieval, nonnull-plus-error create, null create, host exceptions at each call, invalid sizes/content and destroy failure;
- correct first-error/secondary-error precedence, exact `not_attempted` suffixes, stable handle identity and one reverse destroy;
- loader-source structure fixed to absolute `CDLL(..., winmode=0x1100)` plus `add_dll_directory`, and exact before/after module evidence for NVRTC and builtins;
- executable forbidden attempts for a shard, D2, CPU data, model path, `nvcuda.dll`, cudart, CuPy, Driver symbol, context/device query and output mutation before authorization; each must be blocked, with zero payload bytes and zero real loader/Driver/device calls;
- wrong ACK, closed lock and one-byte drift each exit mutation-free before source read/recovery/load.

The fake success emits a one-byte NUL log plus valid synthetic PTX and ELF CUBIN containing exactly the two symbols. Mutations cover zero-byte/unterminated logs, PTX/CUBIN sizes 0/1, row-size disagreement, PTX target/address/entry drift, extra entry, FTZ/approx/unresolved tokens, bad ELF bounds and missing/extra CUBIN kernel symbols.

## Success schema and durable publication

The canonical output directory is `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc1_compile_only`. On success it contains exactly seven files: `result.json`, `source.cu`, `build.log`, `ptx.bin`, `cubin.bin`, `manifest.json`, `commit.json`.

`result.json` has exactly: `kind`, `revision`, `status`, `claim`, `authorization`, `source_identity`, `toolchain_identity`, `runtime_modules`, `options`, `entrypoints`, `future_launch_uses`, `ledger`, `primary_error`, `secondary_errors`, `artifact_sizes`, `exclusion_counters`, `resource_samples`. Fixed values are kind `het_next_l0_ph1_nvidia_nc1_compile_only_result`, revision `NC1`, status `positive`, claim `compile_only`, null primary error, empty secondary errors, two entrypoints and four explicitly out-of-scope launch labels.

`manifest.json` has exactly `kind`, `revision`, `canonical_json`, and ordered `files`; its five entries are result, source, log, PTX and CUBIN, each with exact name/size/SHA-256. Canonical JSON is UTF-8, sorted keys, compact separators and one terminal LF. `commit.json` has exactly `kind`, `revision`, `state="complete"`, `manifest_sha256`, and the exact six promoted file hashes: those five files plus `manifest.json`.

All six precommit files are written create-new in sibling `.het_next_l0_ph1_nvidia_nc1.inprogress.<pid>.<nonce16>`, opened nontruncating writable for `FlushFileBuffers`, and the staging directory is flushed through a Windows directory handle. The independent verifier validates staging. `MoveFileExW` with `MOVEFILE_WRITE_THROUGH` and without replace promotes the absent directory. Only then is `commit.json.inprogress.<nonce16>` written/fsynced, promoted without replacement to `commit.json`, and the final directory flushed. Commit is always last.

Recovery first verifies a valid seven-file committed bundle and returns `already_complete` without mutation. Any staging glob, uncommitted final, corrupt commit, extra file or partial promotion is moved create-new to `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc1_compile_only_quarantine/<utc>-<pid>-<nonce16>` with an exact disposition JSON, directory flush, then the invocation aborts without compile retry. Colliding quarantine targets abort without overwrite.

Caps are source 64 KiB, log 4 MiB, PTX 16 MiB, CUBIN 32 MiB, each JSON 1 MiB and total success tree 56 MiB. Every cap is checked before and after serialization and before promotion.

## Failure schema

The failure root is `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc1_compile_only_failures`. An attempt directory name matches `attempt-YYYYMMDDTHHMMSSZ-<pid>-<nonce16>` and contains exactly one `failure.json`. Its exact fields are `kind`, `revision`, `status="invalid_compile_failure"`, `stage`, `error_type`, `error`, `traceback_sha256`, `primary_operation`, `primary_error`, `secondary_errors`, `device_opened=false`, `driver_loaded=false`, `compiler_loaded`, `payload_bytes_read=0`, `ledger`, `runtime_modules`, `cleanup`, `dispositions`, `artifact_bytes`, `oversize_source_sha256`.

`failure.json` is capped at 1 MiB; error text is 8 KiB and secondary rows 64 KiB. If raw evidence would exceed the cap, only canonical counts, bounded prefixes and SHA-256 digests are retained and `oversize_source_sha256` is nonnull. Primary execution error always wins; log/destroy/writer/cleanup faults are ordered secondary evidence.

The failure writer uses a sibling create-new temporary attempt, file and directory flush, then write-through no-replace promotion. Prelink failure leaves no canonical attempt and quarantines/removes its temp with disposition. Postlink/temp-unlink or writer-cleanup failure preserves at most one bounded canonical attempt and records/quarantines the orphan; it never overwrites or pollutes a valid success. Preflight exercises actual future publication/recovery/failure functions in TEMP for clean success, valid repeat, corrupt/stale/orphan, verifier rejection, file/directory fsync fault, pre/post-promotion fault, commit fault, failure-writer primary+secondary fault, oversize and collision.

## Independent verification

The future verifier imports neither candidate runner nor core. It independently parses the CUDA source, PTX and ELF, rehashes the full directly bound small-file/toolchain chain, validates the exact schemas/topology/caps and reconstructs all ten ledger transitions. It checks resolved compiler module paths/hashes and rejects Driver/runtime/device modules or counters. It verifies source bytes, exact options, two entrypoints, four out-of-scope labels, log rules, PTX target/address/two-entry contract, CUBIN ELF bounds/symbols, manifest, commit and commit-last evidence.

A valid-shaped synthetic bundle must pass the production verifier. Mutations of every schema field, row status/order/code/identity, primary/secondary precedence, module path/hash, source/artifact byte/size/hash, PTX directive/entry, ELF/symbol table, manifest/commit, topology, cap and forbidden counter must each fail. Failure verification requires the exact one-file attempt topology and schema and can never return a compile pass.

The closed NC1 design lock authorizes no import or call. A later implementation lock must directly bind every source, verifier, preflight, document, audit and toolchain identity; indirect trust is forbidden.
