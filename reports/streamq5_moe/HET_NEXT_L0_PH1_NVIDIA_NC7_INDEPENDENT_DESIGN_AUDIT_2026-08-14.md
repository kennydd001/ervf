# PH1 NVIDIA NC7 compile-only - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, compiler, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC7 as frozen.**

NC7 repairs the literal cache bytes, directory topology, per-operation history, entry-level cache mutations, full over-cap identity and redirected path names. Three exact contract gaps remain: the shared-module namespace is contradictory, environment/restore failure mutations are absent, and the required history digest is not represented in the normative manifest schema or cases.

## Integrity and recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC7 preregistration | 2,547 | `74b0ec6e808da3e5f36c5bda28653c5d50ac4ffe550a064ac2bf4d11aa2259b1` |
| NC7 erratum | 2,330 | `0f1d30a4888b3dbf0392313d7ec453f4f3a8e71052992fc6b8e9cf50bd774966` |
| NC7 fixture manifest | 3,791,692 | `b5227030dcd868ad507c6a13842c10fd9dea52bd6f498823710d4a7cc86c977a` |
| NC7 closed design lock | 6,092 | `7ce7aa2a6bcbeb91e82b00acd4abc36a86ae9792ccef5bd8a8c3a12bebc113b9` |

All 20 bindings match byte count and SHA-256. All 19 expected implementation/output paths are absent and every execution flag is false.

The manifest has 314 cases and 314 unique names. Independent checks confirm 312 attempted create rows with zero prehandle; seven and only seven retryable `transaction_debris` cases; 235 nonretryable incidental failures; 20 cache cases; exact 12-row stage/index history in every cache case; and the literal 19-byte sentinel with terminal NUL and SHA `5d7bfc7021fa3b29532e4cb32c29eb6fc5f6ad165d602eb76afda433c29d916f`. The five-entry nominal directory tree digest independently recomputes to `f8d7776392bea26cea4d4027b9c69fc7f46de5066095c5731ffbb5576a73148b`.

## Sound NC6 repairs to retain

- Exact distinct `private_tree/{cuda_cache,tmp,temp,nvrtc_cache}` destinations and environment restoration semantics.
- Root plus four directory entries, with files forbidden nominally.
- Exact 12-stage history and literal entry mutations.
- Reproducible sentinel bytes/path/base64/SHA.
- Embedded over-cap prefix plus full byte count and full SHA.
- NC6 postcommit, debris and import-inert shared-function semantics.

## Blocking findings

### 1. The shared production module has two incompatible frozen identities

The NC7 design lock reserves and requires absence of future `scripts/streamq5_moe/het_next_l0_ph1_nvidia_nc7_compile_contract.py`. The normative NC7 manifest instead freezes `shared_contract.module` as `scripts/streamq5_moe/het_next_l0_ph1_nvidia_nc6_compile_contract.py`. Neither exists.

The runner/preflight identity requirement cannot simultaneously bind the NC7 path promised by the lock and the NC6 path required by the manifest. Freeze one module path, list that same path in the implementation topology, and mutate an ancestor-revision import. Prefer a fresh NC7 module whose source can inherit the NC6 contract while binding the NC7 cache schema.

### 2. The required environment and restoration mutations are absent

NC7 freezes exact mapping, no aliases, original-value capture and terminal restore failure. But the sole 314-case normative matrix has no cases for swapped destinations, aliased destinations, absolute/outside paths, missing/extra environment keys, mutation before authorization, absent-versus-present restoration, or injected restoration failure. The 20 cache cases cover directory/tree/history mutations only.

This leaves a production-critical `finally` boundary untested and does not close the NC6 requirement for swap/alias/absolute/traversal/restore-failure mutations. Add literal TEMP cases for all six environment variables, both original-presence states, partial restore and secondary restore errors; each must exercise the current production environment helper and prove exact restoration before publication.

### 3. `history_digest` is mandatory in prose but absent from the normative manifest

The erratum says `filesystem_observation.cache_tree` has exactly `private_root,environment,environment_original,environment_restored,history,history_digest`, and defines `history_digest` as SHA-256 of canonical compact UTF-8 JSON. Yet the manifest's `cache_observation_schema` contains no top-level cache-object field list or history-digest serialization rule, and none of the 20 cache fixtures contains an expected `history_digest`. There is also no mutation case for a wrong/missing/extra history digest.

Additionally, “canonical compact JSON” does not freeze key order, ASCII escaping, separators or newline policy. Freeze the exact serializer (for example, UTF-8 of `json.dumps(..., sort_keys=True, separators=(",",":"), ensure_ascii=False)` with no final newline), embed the nominal expected digest, and add wrong/missing/type/extra/reordered-history digest mutations.

## Required successor repair

Before implementation:

1. use one exact NC7 shared-contract path everywhere;
2. add executable environment mapping/capture/restore mutation cases;
3. add an exact history-digest schema, serializer, expected value and negative mutations.

NC7 remains closed. No implementation, preflight, NVRTC, Driver or device action is authorized.
