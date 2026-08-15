# PH1 NVIDIA NC16 typed schema and freeze - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, compiler, payload, Driver, device or model call was performed.

## Verdict

**NO-GO for source implementation from NC16 as frozen.**

NC16 closes the NC15 shallow typed-schema blocker and adds explicit three-stage cases. The frozen source-lock, however, contradicts its own valid implementation/runtime topologies and is represented by two independently mutable authorities.

## Integrity and recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC16 preregistration | 1,643 | `8493600d5bb77cc2902a4648a93a8950932ec16325b13279c44807be029a7ab1` |
| NC16 typed-value/source-lock design | 711 | `f9b7134f6274ca80a442babee9c79073f6b93e61766f6560510b3ee8bd64aae8` |
| NC16 typed-schema/freeze manifest | 3,597,165 | `c1fbbb2ff2369d70b1f9001eb9bd6d4658b404c3758cf81bd9a5060d7d8634c1` |
| NC16 closed design lock | 19,967 | `415a0503aae471127a3f12611324d6b31d3c1bc96ff2861b932ae6c18959e211` |

All twelve bindings rehash exactly. The 137 expected paths are unique and absent, and their sorted newline-terminated digest is the declared `17ff8884da82f4a0cadc11c01f08ba8bc372d5a490fcaa24c9d0f18aa36bdbd9`.

I independently verified all 1,093 unique case names, all compact ordered-JSON tree digests, all top-level byte totals, all 137 unique NC16 roots, all 28 typed roots, and all 28 unique source-lock entries. Every source entry has a normalized descriptor path, positive integer byte count and lowercase 64-hex digest, and the entry path set exactly equals the 28 `bound_size_sha256` roots.

For every one of the 203 required JSON fields, the manifest contains exactly one missing, wrong-type and wrong-value mutation. Exact revision-specific kinds and revisions, false authorization booleans, nonempty object/array predicates, and boolean preflight fields fix the core NC15 schema problem. The case stages are now explicit: design 200, implementation freeze 8 and runtime 885.

## Blocking findings

### 1. The NC16 source-lock's `expected_absent` set contradicts both positive lifecycle cases

`synthetic_nc16_source_lock.expected_absent` is the complete 137-path current design-absence set. It includes all 28 source paths, all 21 provenance-lock paths, and the NC16 source-lock path itself.

But the case declared `implementation_freeze_valid` observes 49 paths, all 49 of which are in that lock's `expected_absent` set. The declared valid runtime case observes 50 paths, all 50 of which are in the same set. The source-lock therefore asserts its own absence and the absence of every source/provenance file whose presence it is supposed to authorize.

An implementation cannot simultaneously honor this immutable lock and accept either frozen positive case. The typed schema only checks that `expected_absent` is a nonempty normalized array; it never checks its semantic consistency with the selected stage.

Required repair: freeze a phase-appropriate source-lock absence set containing only paths that must actually be absent after source freeze, explicitly exclude the source/provenance inputs and the lock itself, and add intersection-empty plus missing/extra/wrong-stage absence-set mutations.

### 2. The source-lock document has two unbound authorities in the negative freeze fixtures

Each freeze case carries both:

1. the observed NC16 source-lock root, including parsed `schema_key_values`; and
2. a separate top-level `source_lock_input` consumed for identity resolution.

They are equal only in `nc16_implementation_freeze_complete`. In all seven negative freeze cases, `source_lock_input` is mutated while the observed source-lock record, its parsed values, byte count, SHA-256 and tree digest remain the unchanged positive baseline. For example, `nc16_freeze_wrong_hash` changes an entry only in the side input while the observed file still contains the good 28-entry bindings.

Consequently, the matrix can prove rejection of an injected side object without proving that the production collector parses and adjudicates the actual on-disk source-lock exactly once. It also does not test disagreement between the two representations.

Required repair: make the parsed observed source-lock the sole resolution input, or bind `source_lock_input` by exact equality to it and to the observed file identity. Mutate the actual observed document consistently in every negative fixture and add explicit side-input mismatch rejection if two arguments remain.

## Claim boundary

This verdict is limited to design executability. It does not challenge the recomputed fixture hashes or typed per-key coverage and makes no CUDA, numerical, performance or device claim. NC16 remains closed; no source implementation, static preflight, compiler, payload, Driver or device action is authorized from this freeze.
