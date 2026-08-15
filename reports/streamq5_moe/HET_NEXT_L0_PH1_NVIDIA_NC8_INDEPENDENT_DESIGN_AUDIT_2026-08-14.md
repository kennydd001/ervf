# PH1 NVIDIA NC8 compile-only - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, NVRTC, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC8 as frozen.**

The three NC7 repairs are internally sound, but NC8 introduces one deterministic cap violation and leaves two inherited lifecycle/production-identity gaps.

## Integrity and recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC8 preregistration | 1,938 | `eef4535bd1f0f9634289cac9f651d2409ef73bff245b9395f32176052234d47c` |
| NC8 erratum | 1,790 | `e33131258ac6d3f244fa884f30f3be46319fbfe464a452ebab3ea68d1023a497` |
| NC8 fixture manifest | 6,984,712 | `ca3b279f676463cc8d9953d0b50f5677a9605ed3a729ab383025fc6cc796fb67` |
| NC8 closed design lock | 4,623 | `189cd733907ce971e539ba3b1e76eddee83a62525cb4f979958ffe3ed230b327` |

All 13 lock bindings match exact size and SHA-256. All 18 declared implementation/output paths are absent and every execution flag is false.

The manifest has 368 cases and 368 unique names. It binds the NC8 shared-contract path consistently. It contains 74 cache/environment fixtures and 50 environment-protocol cases. Independent canonical JSON recomputation of every one of the 74 stored histories produced zero digest mismatches; the nominal digest is exactly `e013aac00b3e20eb0e7f058c0669f9c4cccb0e7be14fda88c4d3e322c8ef90df`.

## Sound NC7 repairs to retain

- One exact NC8 shared-contract path is used by preregistration, manifest and lock.
- The six-variable capture/set/reverse-restore protocol is explicit, including empty original strings and continued restoration after failure.
- Fifty literal environment cases cover per-variable absent/present/wrong/set/restore states and global alias/swap/outside/missing/extra/preauthorization/partial/secondary states.
- `history_digest` has an exact serializer, schema, expected value and missing/wrong/type/extra mutations.

## Blocking findings

### 1. The normative manifest violates its own immutable cap

The NC8 fixture manifest is **6,984,712 bytes**. Its own `caps.fixture_manifest` remains **4,194,304 bytes**, and NC8 explicitly says all NC7/NC6 caps are unchanged. A conforming preflight must therefore reject the exact frozen normative input before evaluating any of its 368 cases.

This is not an implementation choice. Either raise the frozen fixture-manifest cap above the exact file size with bounded headroom, or compact the manifest below 4 MiB without weakening literal evidence. Rehash the successor manifest/lock and add exact `size==bound`, `size==bound+1` negative checks.

### 2. The inherited durability-adjudication topology disappeared from the lock

NC8 says every NC7/NC6 lifecycle and postcommit rule is unchanged. NC7's expected topology included `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc7_durability_adjudication`. NC8's 18-entry `expected_absent` list omits the corresponding NC8 durability-adjudication path, even though postcommit rows still permit a later CPU verifier adjudication.

Freeze the exact NC8 durability-adjudication root and require it absent in the implementation/source phase. The topology classifier and independent verifier must reject unregistered, mixed, stale or multiple adjudication roots.

### 3. Environment lifecycle functions are not part of the shared production identity contract

The shared module still exports exactly only `classify_topology`, `recover_inprogress`, `publish_transaction`, `write_incidental_failure`, and `adjudicate_terminal`. None is the exact environment capture/apply/restore helper required before compilation and in `finally`. Yet the preflight is required to execute the actual production environment protocol rather than a copied fixture predicate.

Add exact import-inert shared exports such as `capture_environment`, `apply_environment`, and `restore_environment` (or one exact lifecycle object covering all three). Bind their module path/SHA/qualified names/code digests and prove runner/preflight use the same function objects. Mutations must reject runner-local copies, monkeypatches, reordered restore, early publication and skipped continuation after failure.

## Required successor repair

Before implementation:

1. make the fixture-manifest cap consistent with the exact frozen file;
2. restore the NC8 durability-adjudication topology binding;
3. place environment capture/apply/restore in the shared production identity contract.

NC8 remains closed. No implementation, preflight, compiler, Driver or device action is authorized.
