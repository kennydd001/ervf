# PH1 NVIDIA NC19I0 compile-only implementation - independent source audit

Date: 2026-08-14  
Mode: frozen source-only/read-only. No candidate module was imported or executed. No candidate preflight, NVRTC/compiler, scientific payload, nvcuda/CUDA Driver, runtime, device or model call was made.

## Verdict

**NO-GO for the NC19I0 static preflight as frozen.**

The package is hash-clean, closed and free of an obvious hidden payload/Driver/device call. Its static preflight nevertheless has three independent deterministic failure paths, while the production topology, NVRTC lifecycle, authorization/failure protocol and standalone verifier do not implement material inherited NC19/NC1 requirements. A fresh source revision is required, and one newly exposed inconsistency lives in the frozen NC19 fixture manifest itself and therefore requires a separately frozen design/fixture erratum rather than a code-only exception.

## Frozen package integrity

| artifact | bytes | SHA-256 |
|---|---:|---|
| shared compile contract | 21,027 | `3cacb807728f1379f8f244dc8d0f8baffcbd162ceaf414b9681b9a2a79b816ee` |
| physical runner | 22,971 | `490368b570a2e59216e9f0eee454d3ddf15bd6e5f1f5532d1ceb906a2540c365` |
| static preflight | 20,600 | `830713c92342b0b76c427cb93eca27d638db2092cf72eb87920bc70e3225b346` |
| standalone verifier | 15,237 | `61b9e0c7eddd45a8da402363b549a28d514ca86738aba514d7b82ec555a7218d` |
| implementation preregistration | 1,824 | `e3bc521b6f02584e84eea1640a66bbc3384d8061875655a48509142514c29caf` |
| source design | 1,217 | `8d56de6e4537dd0abdc7b3211032e88a5ce9b9c961c539a466f0495056cc8889` |
| source lock | 14,426 | `9bbc3f65fcd4145f84567f1399fb5c1423c92e60556385b9627389cfbba86a4d` |
| preflight lock | 6,482 | `2400120c5fa037a618457fade6d22669ae177642ff0a899b91ae8d50285ff788` |
| verifier lock | 6,871 | `a78c73bdd0a3a99650eaf5092f3b2737fa5a2a4aaef50864be0ae4b3b8339fe7` |
| authorization bootstrap lock | 3,226 | `ca42f9defe6142f4c16eb8566afe6c954bb90066148a0be3006a976878608d3a` |

The source lock's 22 direct source/provenance rows, preflight lock's 10 rows, verifier lock's 11 rows and bootstrap lock's 3 rows all rehash exactly. All four locks are closed. Their twelve unique NC19I0 runtime targets are absent, and all 157 immutable NC19 design paths remain absent. No NC19I0 pycache, generator, preflight result, output, failure, quarantine or verification artifact was found.

Positive source findings: the shared contract is import-inert and stdlib-only at module scope; the runner performs its allowed 6,173-byte CUDA-source read after authorization; the exact seven NVRTC options, program name/NUL construction and ten ABI declarations are present; the verifier imports no candidate source; and no direct nvcuda, cudart, CuPy, model, shard or D2 access was found.

## Blocking findings

### 1. The exact static preflight is deterministically negative in three separate places

1. `check_manifest()` in `preflight_het_next_l0_ph1_nvidia_nc19i0_compile_only.py:86-103` requires every observed file's decoded bytes to match its retained size/SHA/schema, including deliberately invalid fixture rows. Recomputing its exact boolean yields false:
   - the inherited intended negative `nc17_source_lock_mismatch` has the deliberate raw/SHA mismatch; and
   - twenty `valid_json` rows have decoded JSON unequal to `schema_key_values`. In the nominal `nc19_source_lock_absent_set_positive` case, the raw NC19 preflight- and verifier-lock bytes still say kind/revision `NC18`, while the retained schema says `NC19`. The same two stale raw documents recur in all nine NC19 absent-set mutations.

   The NC19 absent-set repair itself remains correct, but a case declared implementation-freeze-valid cannot simultaneously retain stale NC18 raw locks. Because the NC19 manifest is immutable and bound, this needs a fixture/design erratum. The preflight must evaluate each case through the production classifier and compare with `expected_result`; it must not require intentionally corrupted negative inputs to be internally valid.

2. `static_ast()` at preflight lines 106-120 is always false for the frozen contract. It checks `ImportFrom` aliases rather than the source module, so `from pathlib import Path` contributes alias `Path`, which is not the allowlisted string `pathlib`.

3. `transaction_matrix()` reaches `atomic_create()` in the shared contract. Contract lines 319-320 reopen the hard-linked destination with `rb` and call `os.fsync()` on that read-only Windows descriptor. This is the same Windows `_commit` failure mechanism already established in the S0 lifecycle evidence; the handle must be opened writable (`r+b`) before fsync. The preflight's first nominal publish therefore cannot be relied on to complete.

Any one of these is sufficient for NO-GO.

### 2. The shared topology/source-lock implementation is not the NC19 classifier

The production `classify_topology()` at contract lines 112-147 does not parse or authenticate observed raw/Base64 file bytes, verify content hashes or identity policies, inspect in-progress patterns, or emit the manifest's terminal/recovery dispositions. It looks for nonexistent root key `schema_rule`; every NC19 root instead carries `required_schema_spec`, so typed schema checking is skipped. Its helper also expects type names `bool/int/str`, while the manifest uses `boolean/integer/string`.

The terminal composition is impossible: at runtime a valid terminal observation contains the 57 required inputs plus one selected terminal, but line 145 requires `set(seen) == required`. Thus every valid positive or negative terminal is rejected. Unknown terminal IDs can conversely obtain an empty absence set. Pattern/debris cases cannot be classified as designed.

The preflight never calls `classify_topology()` for any of the 1,106 cases and never reads any `expected_result`; its `topology_matrix` only checks that `paths_for_revision()` returns 157 paths. `source_lock_contract()` uses the retained decoded `document` object rather than decoding and bootstrap-authenticating the raw source-lock bytes. Consequently the claimed sole byte authority and the 1,106-case production mutation gate are vacuous.

### 3. The exact NVRTC state/resource machine is incomplete

The basic Win64 kernel32/cdecl ABI table is plausible, but the evidence and failure state machine are not exact:

- size-return operations are not retained in ledger rows, and log/PTX/CUBIN size values are not bounded before `create_string_buffer` or array allocation (`runner:210-227`); an untrusted compiler size can allocate before the declared caps are checked;
- a nonzero `nvrtcDestroyProgram` result is appended only to `secondary` (`contract:260-278`), leaving `primary.state == none`; program release is absent from `adapter.cleanup`, so `cleanup_ok` can remain true even though the program was not destroyed;
- the fixed twelve-snapshot requirement conflicts with negative paths: snapshots are taken only for attempted calls, while skipped ledger rows receive none. A compile-error/log path therefore has fewer than twelve snapshots and is converted into an incidental `cache_history_count` exception instead of the designed valid compiler-negative terminal;
- module evidence has pre-load and post-release basename handles but no during-load resolved-path/hash proof that the pinned NVRTC and builtins modules were the modules actually used; and
- the fake fault suite omits destroy failure, invalid/oversize returned sizes, malformed terminal-NUL data, exact size/artifact equality, module-resolution faults and snapshot failures.

The preflight hand-builds its success cache history rather than deriving it from the production runner path. These gaps prevent exact replay of the ten-row lifecycle and compiler-negative branch.

### 4. Authorization, topology and failure durability are not fail-closed

The preflight authorization checks only its open flag/token; it does not authenticate its frozen lock bindings or the source-lock hash. A coherently modified source lock plus source files can pass `source_bindings()` without an immutable anchor. The physical runner checks only a subset of the future chain: it does not bind the preflight-result SHA from the authorization lock, exact 18/18 names/checks/total, verifier-lock/source identity, or the precommit verifier hash before launching it.

The runner's clean-state check covers only positive, negative, compile-failure and compile-quarantine roots (`runner:377-380`). It ignores the other declared NC19I0 roots, all in-progress patterns and stale `compile_work` directories. `recover_inprogress()` and `adjudicate_terminal()` are never called by production. An already-valid positive can return success while unrelated failure/quarantine/verifier debris exists.

Work-directory creation, private-directory creation, environment capture/application and adapter construction occur before the cleanup-protected region (`runner:289-302`). An exception there can leave a work tree or partially changed process environment. Later failure evidence always claims `compiler_loaded=false` and `device_opened=false` regardless of the actual reached stage (`runner:383-391`) and discards the retained compile/cache/cleanup evidence.

`write_incidental_failure()` creates the canonical attempt directory before its file is durably complete, has no staged-attempt promotion, and has no protected secondary writer path. `recover_inprogress()` renames without flushing either directory. Publication does not exercise the inherited prelink, postlink, fsync, verifier-reject, commit, failure-writer, oversize, collision and multi-debris matrix.

### 5. The standalone verifier is independent but not sufficiently adjudicative

The verifier's lack of candidate imports is good. Its production checks nevertheless admit or miss material mutations:

- candidate files are read fully before the 64 MiB cap is checked;
- PTX checks require the two named entries but do not reject an additional entry or require exact directive cardinalities;
- CUBIN validation is only ELF magic plus raw substring presence, not bounded ELF section/string/symbol parsing with exactly the permitted kernel-symbol set;
- source/toolchain paths, sizes and the complete direct lock/provenance chain are not reconstructed; provenance only checks the verifier's own row in a mutable source lock;
- postcommit `--candidate` is not confined to exactly one canonical positive/negative root;
- authorization checks only `execution_open` and ACK from candidate-controlled result data;
- cache entries' per-snapshot tree digests are not recomputed, applied private environment values/containment are not validated, and the five exact directories are not required once each per snapshot; and
- the preflight tests one positive baseline plus only eleven mutations, no compiler-negative baseline, no per-check rejection mapping, no failure artifact and no exhaustive nested-schema/ledger/ABI/artifact/topology mutation set.

Verifier `_atomic_output()` has no directory fsync/write-through promotion and no bounded verifier-failure evidence despite frozen failure/quarantine roots.

## Required next immutable step

1. Freeze a minimal NC19 fixture erratum that corrects the two stale raw NC18 lock documents in the nominal NC19 composition and its nine derived absent-set cases, while preserving the valid 100/57/0 source-lock repair. Recompute all affected Base64, sizes, hashes, schema projections, bootstrap identities and tree digests.
2. Build a fresh implementation namespace. Make the shared classifier consume authenticated raw observations and execute all expected cases; repair the schema vocabulary, runtime-terminal composition and pattern/debris semantics.
3. Repair the Windows writer and implement the complete transaction/failure topology with actual TEMP fault injection. Anchor every preflight/physical/verifier dependency before mutation or source/compiler access.
4. Complete the NVRTC size/ledger/destroy/module/cache lifecycle and use the same production functions for success and every fault suffix.
5. Strengthen the standalone verifier and mutation suite, including a valid compiler-negative baseline, structural PTX/ELF checks and exact canonical candidate topology.

Only the fresh source package may then receive another source audit. A no-device preflight remains unauthorized until that audit returns GO. NVRTC/compiler, payload, Driver and device phases remain separately closed.

## Claim boundary

This is a source-executability verdict, not an execution result. It does not invalidate the independently verified NC19 100-path absent-set repair or the frozen N5 CUDA arithmetic. It makes no compile-success, numerical, performance, Driver or device claim.
