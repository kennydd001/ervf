# PH1 NVIDIA NC14 semantic topology - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, compiler, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC14 as frozen.**

NC14 closes the NC13 loss-of-role, label-only fixture, exact-mutation and pattern-test blockers. Two representation contradictions remain: schemas cannot be attached to file roots, and immutable source roots have no expected digest despite cases that require them to fail.

## Integrity and recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC14 preregistration | 1,799 | `125f6c197f8c317f4fa57c7e50102658fd12a8545fe0f0e25a720d1a0e5c2490` |
| NC14 semantic classifier design | 925 | `d85a26aa1da68ae278783ceab384af5a355755156e9c133c52d498d54eee2ff3` |
| NC14 semantic fixture manifest | 520,555 | `810b07bb3f14e8f10ae28e4e9c5ec0b712a9fbe7344a5bba02c8ae74b9ecb2ac` |
| NC14 closed design lock | 15,262 | `63795fb1c56e68f59fad2e073e3549e6c930919fff2223b75bf6e834ff3fe0fa` |

All ten bindings rehash exactly. The 99 current paths are unique and absent; all phases remain closed. The manifest has 242 unique cases with one uniform case field set.

I independently verified:

- all 242 `observed_tree_digest` values as SHA-256 of compact ordered JSON;
- all 242 `observed_total_bytes` values as the top-level observed-entry byte sum;
- exact NC8-NC13 lockset equality and canonical digests for all six historical descriptors;
- exact 99/99 equality and digest `583248eb603c82d0229a1bfa627f1c987346703851532b0e19c11bdc2100f225` for NC14;
- 99 file plus 99 directory root cases, six exact/drop/duplicate/wrong-revision/wrong-path descriptor groups, and five concrete pattern mutations.

Typed phase, role, node type, cap, multiplicity, recovery and disposition metadata now survive descriptor expansion. This materially closes the NC13 design findings.

## Blocking findings

### 1. Root-file schemas cannot be represented by `observed_entries`

`observed_entry_fields` is exactly `path,node_type,size,sha256_or_null,children`. Schema values exist only in `child_schema_fields`, which are nested under a directory's children. There is no root-level `schema_key_values` field.

NC14 nevertheless has twenty allowed-file roots with nonempty required schemas:

- fifteen source/preflight/verifier lock files requiring `kind,revision,bindings,expected_absent`;
- five static-preflight result files requiring `kind,revision,pass,checks,device_opened`.

No literal observed record can carry the values needed to validate those roots. Their file fixtures consequently report `invalid_schema`, but a valid file-root fixture is impossible under the frozen schema. Add root-level schema values (or a single node schema shared by roots and children), plus valid and missing/extra/wrong-key cases for both lock and preflight-result files.

### 2. Immutable source provenance has no expected hash and contradicts its cases

The current descriptor has twenty script roots with:

- `allowed_node_type=file`;
- empty required schema/key/kind/status;
- `cap_bytes=65536`;
- multiplicity 0..1;
- `immutable=true` and `disposition=provenance_source`.

But `root_fields` has no expected SHA-256 or exact byte count. Each corresponding `root_file` fixture supplies a one-byte, under-cap file with a syntactically valid 64-hex digest and no required schema, yet expects `classification=invalid_schema`. Every declared root predicate is satisfied, so the expected result cannot be derived from the descriptor.

This also creates a live-stage contradiction: after implementation, the NC14 contract, runner, verifier and preflight sources necessarily exist. The descriptor either rejects them as its cases demand or accepts them without any immutable provenance check. Freeze expected byte count and SHA-256 for immutable file roots, set stage-appropriate required multiplicity, and include exact-positive plus wrong-size/wrong-hash cases. If design-lock absence paths and runtime provenance paths are different states, make the state explicit rather than treating an empty observation as universally `fresh`.

## Required successor repair

Before implementation:

1. give every file node, including a root, an exact schema-value representation;
2. bind bytes and SHA-256 for immutable source/lock roots;
3. add positive and negative file-root fixtures;
4. freeze the design-stage versus implemented-runtime presence policy.

NC14 remains closed. No source implementation, preflight, compiler, Driver, payload or device action is authorized.
