# PH1 NVIDIA NC18 authority cleanup - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, compiler, payload, Driver, device or model call was performed.

## Verdict

**NO-GO for source implementation from NC18 as frozen.**

NC18 exactly closes both final NC17 findings: the normative top level contains no legacy authority object, and the observed-entry declaration is the exact ordered eight-name vector. One lifecycle contradiction remains inside the sole NC18 source-lock document itself.

## Integrity and recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC18 preregistration | 1,431 | `487b921d5a7e20e18ec94962e1265ecbb555881db58be690d356f31ee6b0a434` |
| NC18 field-set/authority erratum | 842 | `8c600655f1d7078c6592fb7de1e773ca165759e220b784857e3c5b2c73445095` |
| NC18 cleanup manifest | 7,447,855 | `97694be6ee73146c3dd3c1beeb792bf10ba8a01eb0e16803290fbb9c456d3561` |
| NC18 closed design lock | 16,849 | `0ea597d70aeb28860edb8bfbb3f61edd70135f945a2d14fab82e208c83e88059` |

All fourteen bindings rehash exactly. All 157 unique expected paths are absent. The manifest has 157 unique NC18 roots and 1,119 unique cases; all compact ordered-JSON tree digests and top-level byte totals recompute exactly. The NC18 sorted newline-terminated path-set digest is the declared `c7b267502e1657d932f6c9bbea7c0f24127a4518c130150c88a8b14bfe2268fd`.

The three forbidden legacy top-level keys are absent. The sole NC18 authority has exactly 32 unique rows and its path set equals the 32 `bound_size_sha256` roots. `observed_entry_fields` is exactly eight ordered unique names. The five authority and four field-declaration rejection cases are concrete and negative. All non-intentionally corrupted Base64 records reproduce their byte count/SHA and all valid JSON records reparse to the retained schema values.

## Blocking finding

### The sole NC18 source-lock still asserts absence of every required freeze/runtime input

The byte-frozen `nc18_observed_source_lock_authority.document.expected_absent` contains all 157 current paths. It therefore contains:

- all 32 source paths;
- all provenance/bootstrap lock paths;
- the NC18 source-lock path itself.

The NC18 descriptor simultaneously freezes 57 paths as `required_present_by_stage` for both `implementation_freeze` and `runtime`, and correctly excludes those 57 from the descriptor's stage-specific 100-path absence set. But all 57 required-present paths remain members of the immutable source-lock document's own `expected_absent` array. The source-lock is thus present while asserting its own absence, and its 57/57 intersection with each required-present set is nonempty.

This is not resolved by the correct descriptor stage sets: the source-lock is the declared sole authority and its typed schema retains `expected_absent` as provenance evidence. Either that field is enforced, making both positive compositions invalid, or it is ignored, making the lock field semantically vacuous.

Required repair: freeze the source-lock's `expected_absent` to the phase-appropriate 100 runtime/output roots (and applicable in-progress patterns), exclude the 57 source/provenance/bootstrap inputs and the lock itself, require exact equality with the implementation-freeze absence projection, and add missing/extra/self/intersection mutations against the actual byte-backed document.

## Claim boundary

This is a design executability verdict only. It does not invalidate the NC18 legacy cleanup, field-set repair or recomputed fixture evidence and makes no CUDA, numerical, performance or device claim. NC18 remains closed; no source implementation, static preflight, compiler, payload, Driver or device action is authorized from this freeze.
