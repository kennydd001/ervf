# PH1 NVIDIA NC19I1 compile-only implementation - independent source audit

Date: 2026-08-14  
Mode: frozen source-only/read-only. No candidate module was imported or executed. No candidate preflight, NVRTC/compiler, scientific payload, nvcuda/CUDA Driver, CUDA Runtime, device or model call was made.

## Verdict

**NO-GO for the exact NC19I1 no-device static preflight.**

The fixture erratum genuinely repairs the twenty stale NC18 raw lock observations, the direct artifact/lock bindings are clean, all declared outputs are absent, and several NC19I0 ABI/lifecycle defects are repaired. The frozen static preflight nevertheless has multiple independent deterministic false gates. The production runner also does not use the frozen topology/terminal classifier, the durability protocol can leave canonical post-link/post-rename evidence while reporting failure, and the standalone verifier omits material evidence cross-checks. A fresh source/fixture revision is required; opening the present closed preflight lock would not make this candidate executable.

## Frozen package integrity

| artifact | bytes | SHA-256 |
|---|---:|---|
| shared compile contract | 36,112 | `59f52ff1e331889354f4fbea8ae8675e4925529111f87c00b4f555dd392d1b7e` |
| physical runner | 28,627 | `68bfdb4246d8b482fc231ec4b3e35bd190e376b685d8e79a21ad3e65fe266b41` |
| static preflight | 28,925 | `a9086bc996c0e87c3d371459720de53ec64aa1bfd29f3e8f0597feb079542b32` |
| standalone verifier | 21,267 | `0cd9ed273704cdd891b8e92437fd8188ab91b8b8f93bf31eb5b098a6805390d6` |
| fixture erratum | 921 | `2e03baac72f5b09031687e40f45cdb11e85973cc7dee2ab992fb0eef2d6b7925` |
| implementation preregistration | 2,343 | `f526541cd1ea1f6bbf4d22b27a7f6ce01d03acd6aecaf24d836ffd558bf2e345` |
| source design | 1,627 | `832e6250c84a0d84e2513732229699f8e13e410ff7efeafeec1576b85c2ea100` |
| corrected 1,106-case manifest | 5,716,095 | `e5254c911e5e5997427977df02a5afba3435931e6e8caa7168fb7e2d641a4a90` |
| source lock | 6,837 | `5e35ff5f8341e2753d0b787ffb59f12ffec526a835f7444456ba7d6332ba6513` |
| preflight lock | 3,023 | `1d6242e5e8b6b58624d7ad941428f2902e9f01113337f974dcce842a8711b2ca` |
| verifier lock | 3,189 | `dff2dc135dbd28ed5a0531b0f6217b9c6c7c4e58076bd8e13ed74b2296aff160` |
| authorization bootstrap lock | 2,263 | `cd35e2e9ff5cd511b33106fc145502bbc72112950dd75fdb7979f929d218c933` |

The four direct binding sets rehash exactly: source 31/31, preflight 10/10, verifier 11/11 and authorization 5/5, without duplicate paths. The source/preflight/verifier/authorization phases are all closed. Their common twelve declared result/failure/quarantine paths are absent. The source and preflight locks bind the immutable NC19I0 audit `0d15c5d594e8398e91bdaf8b59117483e89539e03b29dcdcba01de56920b9fd2`.

Positive repair evidence:

- all twenty corrected raw rows (two NC19 lock documents across the nominal case and nine derived absent-set cases) decode strictly, match retained byte count and SHA-256, equal their schema projections and carry the intended NC19 kinds/revision;
- the raw NC19 source-lock authority, exact 100 absent paths, 57 required paths, zero intersection and 32 source identities are retained;
- `atomic_create()` now reopens the linked destination with `r+b` before `fsync`, closing the NC19I0 read-only Windows descriptor defect;
- exact NVRTC size replies are retained, caps precede artifact allocation, skipped operations receive snapshots, destroy failure can become primary, PTX is checked for exact directive/two-entry cardinality, and CUBIN uses a bounded ELF symbol-table parser; and
- no source path introduces a direct payload, nvcuda/Driver, CUDA Runtime or device call into the static phase.

## Blocking findings

### 1. The exact preflight is deterministically negative

There are at least four independent source-level false gates.

1. `check_manifest()` (`preflight...nc19i1...py:108-122`) and `evaluate_fixture_case()` (`compile_contract.py:266-275`) hash each parsed `observed_entries` list using the dictionaries' parsed insertion order. The manifest explicitly defines the digest order as `observed_entry_fields = [path,node_type,size,sha256_or_null,children,schema_key_values,parse_status,content_base64_or_null]`, while its serialized row objects are alphabetically keyed. Independent recomputation of the exact candidate expression matches only 32/1,106 retained tree digests; **1,074/1,106 fail**. The second case alone hashes to `39af39b7...` under the candidate expression but retains the correct declared-order digest `3d74c5e3...`. Intentional case-local mutations also deliberately retain a baseline tree digest, so a global all-case digest assertion contradicts the erratum's own semantics.

2. The new evaluator still feeds unit/root fixtures to the full-topology classifier. For example `nc16_typed_valid_root_031531d5be646709` contains the one intended root observation; the NC16 runtime descriptor requires 49 roots. The frozen evaluator therefore returns `missing_required_present`, while the case requires `root_file_valid`. Mixed/multiple-terminal and stage-intersection fixtures likewise cannot yield their frozen expected classifications through the implemented path. Descriptor mutation cases are declared invalid merely by `mutated != original`, rather than being adjudicated through the production descriptor/classifier contract.

3. `static_ast()` (`preflight:125-141`) walks imports inside function bodies. The contract imports `copy` in `evaluate_fixture_case()` (`contract:288`), but `copy` is absent from the allowlist. The exact `set(observed) <= allowed` predicate is therefore false.

4. `no_payload_driver_device` (`preflight:382-383`) searches the concatenated runner/contract/verifier source for the forbidden literals `cudart` and `cupy`. The verifier itself contains those literals in its negative scan (`verifier:197`), so this check self-matches and is always false.

`topology_matrix` is additionally defined as the path-count check AND the already-false 1,106-case result. The current preflight lock is closed, so the command presently returns authorization failure before these checks; after a hypothetical open-only revision it would still write a negative result.

### 2. Production topology and terminal adjudication are not the fixture-tested contract

The runner has zero calls to `classify_topology()`, `evaluate_fixture_case()` or `adjudicate_terminal()`. Its live clean-state rule is a hand-written tuple of eleven paths plus one broad in-progress glob (`runner:417-430`). Thus the claimed shared production topology is used by preflight fixtures only, not before the physical source/compiler path.

`recover_inprogress()` authenticates only name cardinality/regex. It does not require a directory, cap bytes/entries, validate a bundle/tree schema, or reject multiple current-family debris prefixes as a single global invalid state. Arbitrary matching debris can therefore become retryable. The current runner also ignores undeclared family paths outside its tuple/glob.

There is a concrete namespace split: all four locks and the runner reserve/check `het_next_l0_ph1_nvidia_nc19i1_independent_verification.json`, but the verifier's live `OUT` is the directory `het_next_l0_ph1_nvidia_nc19i1_independent_verification` (`verifier:24,215-277`). The actual verifier output is consequently outside the frozen expected-absent topology and runner terminal set.

### 3. Transaction and failure durability remain non-atomic after link/promotion

`atomic_create()` links the canonical destination before destination fsync and two parent flushes (`contract:549-567`). If destination fsync or the first parent flush fails, its exception path removes only the temporary hard link and leaves the canonical destination present while reporting failure. The preflight injects only a pre-link failure (`preflight:224-231`), so this production failure mode is untested.

`publish_transaction()` renames the staging directory to the canonical output before the parent flush (`contract:631-649`). A post-rename flush failure leaves the canonical bundle, raises, and the runner then writes an incidental failure, producing mixed terminal evidence. The declared durability-adjudication path is never written. `write_incidental_failure()` similarly lacks injected post-promotion/root-flush coverage; `failure_durability` is merely assigned the nominal `transaction_matrix` boolean (`preflight:376-377`). The static suite does not cover post-link, post-rename, fsync/flush, multi-debris, oversize or secondary-writer terminal composition.

The preflight itself has no bounded exception writer: if manifest parsing fails before the contract local is defined, the catch constructs a result but line 389 still calls undefined `contract.atomic_create()`. Its authorization gate also does not require a direct `.venv -I -B` invocation or a globally clean twelve-path topology.

### 4. NVRTC/cache evidence is improved but still not exact

The direct Win64 ABI declarations, cdecl `CFUNCTYPE` wrapping, seven frozen options, source/name terminal-NUL operands, ten-row ledger and destroy-primary repair are materially better. Remaining gaps are:

- `WinNvrtcAdapter.load()` requires both the NVRTC and builtins module handles immediately after `LoadLibraryExW`, before even `nvrtcVersion` or compilation (`runner:208-240`). Independent PE import inspection finds no normal or delay import of `nvrtc-builtins64_133.dll`; the DLL name appears in NVRTC's dynamic builtins-load error strings. This ordering can reject the legitimate lazy builtins load before the one allowed compile. At minimum, the source does not prove that this pre-operation requirement is feasible.
- `cache_entries()` reads every compiler-created file fully before applying any per-file or aggregate cap (`contract:350-364`). A compiler-created oversized private file can therefore consume unbounded memory during each snapshot before becoming an incidental failure.
- `validate_compile_evidence()` accepts any nonnegative returned size and never correlates the three size rows with the retained log/PTX/CUBIN byte counts (`contract:478-505`). The static mutations do not exercise size/artifact mismatch, module timing/path failure, postrelease ownership, or snapshot failure.
- The live terminal decision does not call `adjudicate_terminal()`. `valid_negative` is composed independently in the runner (`runner:393-398`), while the exported adjudicator itself checks only a status string plus one cleanup boolean and is not exercised by preflight.

These are physical-source blockers; no compiler call was made in this audit.

### 5. The standalone verifier is independent but materially under-checks evidence

The verifier imports no candidate runner/contract and now performs bounded ELF symbol parsing and exact two-entry PTX checks. It still does not independently adjudicate the full retained record:

- `verify()` reads and parses `result.json` once to choose the terminal kind before `verify_bundle()` performs its size cap (`verifier:135-145`), so the claimed cap-before-read rule is false for the first candidate read.
- `compile_checks` is accepted as an arbitrary result field; it is never recomputed or compared. Ledger returned sizes are not checked/correlated, middle-row handle continuity is not enforced, and `primary`/`secondary` schemas are incomplete.
- loader validation ignores the full `modules.before` map, the builtins during-load handle/path/hash, returned module/cookie identities and the postrelease zero-handle map. Cleanup validates only five resource names plus `attempted/code`, not ownership identities, `owned_before`, program `identity_after`, or module release correlation.
- toolchain validation does not bind Python bytes/SHA. Invocation validation omits raw/native argv0, executable/base-executable and prefix/base-prefix identities. Environment validation does not require exact disable/maxsize values, exact four suffixes or mutual non-aliasing; its containment uses a string-prefix test.
- the live verifier lock is not independently rehashed by `verify()`, and the actual verifier output namespace is absent from the locks as noted above.

The preflight's `verifier_negative` check is vacuous: `verifier_positive`, `verifier_negative` and `verifier_mutations` are all assigned the same one-positive-baseline/eleven-mutation boolean (`preflight:378-381`). No valid `compile_valid_negative` bundle is constructed, and there is no per-check negative/mutation matrix for the omitted fields above. Verifier publication has its own post-replace/flush orphan risk and no bounded failure handling for publication exceptions.

## Required fresh revision

1. Define one canonical row-ordering/digest helper and either recompute intentional-mutant digests or explicitly route declared integrity mutations without a contradictory global gate. Execute all 1,106 cases with fixture-kind-aware production evaluators, including root-unit, descriptor, mixed-terminal and stage-intersection semantics.
2. Repair the AST allowlist and use AST/callgraph checks rather than forbidden-literal self-matching.
3. Make the live runner call the same topology and terminal adjudicators; freeze one exact verifier output namespace and validate all family roots/debris before source/compiler access.
4. Make post-link/post-rename failures explicitly terminal and recoverable only by a complete bounded schema. Exercise the actual production writers with pre/post-link, fsync/flush, collision, oversize, multi-debris and secondary-writer faults.
5. Move/define builtins module proof at a feasible lifecycle point, cap cache reads before allocation, correlate returned sizes to artifacts, and expand the ten-row/cache/module/cleanup mutation matrix.
6. Strengthen the independent verifier for every retained field and add a genuine valid-negative baseline plus targeted rejection mutations.

Only a newly frozen closed source package should receive another source audit. The present source/preflight/verifier/authorization locks must remain closed; there is no authorized or defensible NC19I1 preflight command.

## Claim boundary

This is a source-executability verdict, not a preflight or compiler result. It does not invalidate the corrected twenty raw NC19 lock observations, the NC19 100/57/0 absent-set design result, or the frozen N5 CUDA arithmetic. It makes no compile-success, numerical, performance, Driver or device claim.
