# PH1 NVIDIA NC15 semantic identity - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, compiler, payload, Driver, device or model call was performed.

## Verdict

**NO-GO for source implementation from NC15 as frozen.**

NC15 fixes the NC14 representation-level absence of root schema fields and adds a coherent synthetic size/SHA binding table. It does not yet define the claimed JSON value-type contract, and its matrix does not exercise the three-stage source lifecycle that the design says must be proven.

## Integrity and independently recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC15 preregistration | 1,641 | `a4c130a30cd3febe65c1122f0ffb6539df3ff75d95919884bd5c6726d854bce7` |
| NC15 file-schema/source-identity design | 698 | `7e0e4c99571c5d66a6b18e23488f7ba39993301e85d9c407c8b207954c6a1407` |
| NC15 semantic-identity manifest | 1,048,176 | `7e7738589d43d72735b4382499735b29deb2a973dd13d603fb560c5b7b906d26` |
| NC15 closed design lock | 17,593 | `c814dd1e56faebdaada57653be47f4517b06f0e52def388bb2bceb2170de4451` |

All eleven lock bindings rehash exactly. The lock contains 118 unique expected-absent paths and all 118 are absent. The manifest has 118 unique NC15 roots and 415 unique cases. I independently verified all 415 compact ordered-JSON tree digests, all 415 top-level byte totals, all 32 Base64 source records, their exact byte counts and SHA-256 values, and their literal `NC15_SOURCE_BINDING::` derivations. Every observed root has exactly the declared seven fields. The NC15 path-set digest recomputes as `4626e8f758fe36be5b2ba159efaf590c1b0e354a7f837eff2b61688ba1744e0b` using the inherited sorted newline-terminated convention.

These facts close the mechanical portions of the two NC14 findings: a root file can now carry parse/schema observations, and a bound source can carry exact size/SHA evidence.

## Blocking findings

### 1. The claimed JSON value-type/schema contract is absent and the positive fixtures are not valid lock/result schemas

The preregistration says that key set, kind, status **and value types** must exactly match the descriptor. The frozen `root_fields`, however, contain only `required_schema_keyset`, `required_schema_kind` and `required_schema_status`; there is no per-key type map or required value map.

The 24 `schema_valid` fixtures expose the consequence:

- every historical NC10-NC14 lock/result fixture sets `revision` to the string `NC15`, while no descriptor rule checks the revision value;
- lock fixtures set `bindings="fixture"` and `expected_absent="fixture"` instead of object/list values;
- preflight-result fixtures set `pass="pass"`, `checks="fixture"` and `device_opened="fixture"` instead of boolean/object/boolean values;
- each `schema_wrong_type` case changes only `kind` to an integer, and each `schema_value` case changes only `kind`; neither suite tests the unbound types/values above.

Thus a semantically malformed lock or result is frozen as `root_file_valid`. `parse_status="valid_json"` cannot repair this because the descriptor has no type/value contract against which to adjudicate the parsed object.

Required repair: add exact per-key types and required values (at least revision, pass/status and device-opened semantics, with bindings/checks/expected-absent container types and nested minimum schemas), generate genuinely valid revision-specific positives, and mutate every protected field class rather than only `kind`.

### 2. The promised design / implementation-freeze / runtime source policy is not represented by the fixture matrix

The 415 cases contain 199 `design` and 216 `runtime` cases, but **zero** `implementation_freeze` cases. There are also zero NC15-targeted design-stage presence cases: all 198 design presence violations belong to the inherited NC14 descriptor. All NC15 roots retain multiplicity `0..1` in every stage.

The source portion has 24 isolated runtime-positive records and five single-target negative mutations, but no case proves:

- unresolved source identities reject at implementation freeze;
- an exact source-lock mapping resolves them atomically;
- design-stage presence of any of the 19 NC15-added roots rejects;
- runtime omission of a required current source or lock rejects;
- the full runtime source/lock set is accepted together.

This directly contradicts the design instruction that static preflight must prove design absence, implementation resolution and runtime validation as distinct states. In particular, the descriptors keep `expected_bytes` and `expected_sha256` null, while the design does not freeze the exact source-lock schema/input through which immutable runtime values replace those nulls without mutating the manifest.

Required repair: define a stage-specific required/optional path projection, an exact immutable source-lock mapping schema and classifier inputs, then add full positive/negative cases for all three stages, including unresolved, missing, extra, wrong-size, wrong-hash and complete-runtime composition.

## Claim boundary

This is a design failure only. It does not invalidate the recomputed manifest hashes or the mechanical fixture digests, and it makes no statement about CUDA compilation, device execution, numerical quality, performance or the PH1 Intel result. NC15 remains closed; no source implementation, static preflight, compiler, payload, Driver or device action is authorized from this freeze.
