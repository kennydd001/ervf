# PH1 NVIDIA NC5 compile-only - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, compiler, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC5 as frozen.**

NC5 correctly repairs the literal PTX, create-handle ledger, primary-error union, repeat publication, raw negative evidence, compact preflight evidence, Win64 ABI table and most retry terminology from NC4. It does not close the correlated-postlink blocker, its normative retry rows contradict the preregistration, and the private compiler-tree evidence cannot be reconciled with the frozen exact bundle topologies. Those are specification blockers, not implementation details that may be chosen later.

## Integrity and independently recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC5 preregistration | 3,680 | `2f330a6b9f51e54eb3ddcf2e0bc65e9d37d733fc643856a8d500adba8b9fbe8f` |
| NC5 design | 5,299 | `ce6b07efdd0131374f4061a7fbf50ec01ac81b5a5bef66c34cc42394af500e1b` |
| NC5 fixture manifest | 2,566,009 | `c8b222991fe00a76dedc3f9a02c596880b6a0834260a84c6d4133e0a0309bdaf` |
| NC5 closed design lock | 6,024 | `53be8e43d66880c4b72b779cd6cea1bdeab607ad23439ffcbfd2ff356b82d2a0` |

All 20 lock bindings independently match exact byte count and SHA-256. All 17 expected implementation, lock, result, failure and quarantine paths are absent. The lock remains closed for implementation, preflight, compile and verification.

The manifest parses and contains 294 cases with 294 unique names. Independent checks confirm:

- 292 cases attempt `nvrtcCreateProgram`; every attempted create row has `handle_before=0` and `handle_after` exactly zero or `H1`.
- All 294 primary unions are well typed: `none/null` or `failure/nonempty-string`.
- All three valid-repeat cases have no primary failure, publish `no_write_existing_terminal`, and disallow retry.
- All 45 compile-valid-negative cases carry source plus stage-appropriate retained/missing/bounded raw-evidence declarations.
- The PTX literal is exactly 130 bytes, SHA `3b4cde8b9803cd2dd6131ac2776730915a5f2b3c5f17c9b690c08db6143f4336`, has one final NUL, no embedded NUL, two `.visible` entries and one occurrence of each required entrypoint.
- The CUBIN fixture is exactly 536 bytes, SHA `93abe3a2a7c4f7b4e6b6b9ce202ecc9440a02c3d37a9b9e8f476939d102cf2c8`, and independently parses as little-endian ELF64 ET_REL, EM_CUDA 190, five 64-byte section headers at offset 216, with the frozen section/string/symbol layout.
- The 2,566,009-byte fixture manifest is below the 4 MiB manifest cap; compact preflight result rows are specified rather than embedding full ledgers.

## Sound repairs to retain

- Exact 6,174-byte source buffer and 38-byte program-name buffer construction and digests.
- Literal 130-byte PTX and 536-byte ELF fixture bytes.
- Correct zero-before-create ownership transitions and cleanup ordering.
- Tagged nullable primary-error union and mutation-free valid-repeat rows.
- Stage-specific compile-negative raw evidence and commit-last manifest binding.
- Exact Win64 loader signatures, search flags, pointer-width requirements and immediate last-error capture.
- Direct one-shot runner process, separately terminated CPU verifier and narrow compile-only claim.
- Separate same-invocation and future-invocation retry fields.

## Blocking findings

### 1. The correlated postlink repair deliberately preserves the forbidden success reclassification

The NC4 audit required a correlated postlink/durability incident not to become an unqualified positive/already-complete terminal absent a separately preregistered durability adjudication. NC5 instead says an exact correlated incident preserves `already_complete` (`NC5` preregistration line 17; design line 34). The normative `ctx_postlink`, `vtx_postlink` and `ptx_postlink` rows do exactly that: each records a primary postlink failure and exit 3 while assigning terminal `already_complete` and a positive/preflight/verifier correlated terminal (manifest around lines 20828-20969, 22886 and 25238).

No independent durability proof is specified. A write-through rename followed by a failed directory flush is immutable evidence of an infrastructure incident; byte immutability alone does not establish the durability property that failed. The successor must either classify the combined state as infrastructure-invalid, or define and preregister a separate read-only durability adjudication whose positive result is not the original compile attempt being silently upgraded.

### 2. Seven normative rows contradict the frozen retry/terminal contract

The preregistration says only exact `.inprogress` debris is recoverable and every incidental terminal has `next_invocation_allowed=false` (line 19). The manifest nevertheless labels seven cases `terminal="incidental_failure"` while setting `attempt_consumed=false` and `next_invocation_allowed=true`:

`ctx_stale_stage`, `ctx_prelink`, `ctx_orphan_failure_temp`, `vtx_stale_stage_preserve_compile`, `vtx_prelink`, `ptx_stale`, and `ptx_prelink`.

For example, `ctx_stale_stage` has those values at manifest lines 19064 and 19195-19203, while `ctx_prelink` repeats them at lines 20681 and 20812-20820. The same rows publish `failure_attempt`, so they are not represented as merely quarantined `.inprogress` debris.

Freeze one unambiguous taxonomy. Recoverable rows should have a distinct nonterminal/debris disposition and an exact quarantine topology; a successfully published incidental failure is a terminal and must not authorize another compile. Preflight and the later verifier must mutate both directions.

### 3. Private compiler-tree retention conflicts with the exact bundle schemas

The design permits retained files in a private cache/temp tree and requires final private-tree contents to be retained inside the evidence or negative bundle and hashed by its manifest (line 30). But the successful bundle is frozen to exactly five immutable data/result files plus manifest and commit, and the exact successful `result.json.artifacts` map contains only `source,build_log,ptx,cubin,disassembly` (preregistration line 13). The negative manifest rows likewise enumerate exact raw/terminal files and contain no private-tree artifact.

Therefore a nonempty private tree has no frozen destination or schema:

- leaving it beneath the promoted staging tree creates extra directories/files outside the exact seven-file positive topology;
- deleting it violates the requirement to retain final contents;
- embedding it in `result.json` has no exact field/byte representation or per-file content rule;
- storing only the canonical path/hash snapshot does not retain the compiler bytes claimed by line 30.

Freeze exact relative subdirectory names, the final tree serialization (including whether file bytes or only metadata are normative), its result/negative schema, manifest entries and caps. Add positive/nonempty, empty, over-cap, traversal/symlink/reparse, external-write and cleanup-failure cases to the normative matrix. The current 294 cases contain no direct cache/private-tree cases; nested verifier field mutations are not an executable filesystem fixture.

### 4. Bounded-prefix evidence lacks a self-contained artifact path field

The three over-cap rows enumerate files such as `build.log.prefix`, `ptx.bin.prefix` and `cubin.bin.prefix`, but the corresponding `negative_artifacts.<artifact>` object sets `state="bounded_prefix"` and `file=null`, retaining only `prefix_bytes` and `metadata_file` (manifest lines 8517-8521, 9924-9928 and 11161-11165). The top-level policy names the metadata fields but does not freeze a `prefix_file` field or a normative derivation rule.

An independent verifier should not infer the evidence filename from an implementation convention. Add an exact nonnull `prefix_file` member (or define a single explicit derivation in the schema), require it to occur exactly once in `expected_bundle_files`, and mutate missing, swapped, duplicate and path-escaping names.

### 5. “Actual future functions” is not yet an executable preflight architecture

Design line 36 requires static preflight to run fake APIs through the actual future transaction/classifier functions. NC5 simultaneously freezes only runner, verifier and preflight implementation paths and specifies a direct, non-imported runner. It does not freeze whether the preflight imports an inert runner, extracts functions by AST, or uses a shared import-safe core, nor the callgraph gate that proves the fake path reaches the same functions later used by the physical runner.

This can be repaired without adding a compiler process: freeze an import-safe shared transaction/classifier module, or explicitly permit an inert runner import and define AST/runtime identity checks. Require mutations proving preflight rejects a copied/toy classifier and that importing the shared code performs no authorization, filesystem, NVRTC, Driver, payload or device action.

## Required successor repair

Before source implementation:

1. resolve postlink/durability incidents without reclassifying the failed attempt as an unqualified positive;
2. make recoverable debris a distinct nonterminal disposition and remove `next_invocation_allowed=true` from incidental terminals;
3. freeze an exact private-tree evidence/topology schema and executable cache/filesystem fixtures;
4. give every bounded prefix an explicit immutable artifact-path binding;
5. freeze how preflight executes the exact production transaction/classifier functions without side effects or toy substitution.

NC5 remains closed. No implementation, preflight, compiler, Driver or device action is authorized.
