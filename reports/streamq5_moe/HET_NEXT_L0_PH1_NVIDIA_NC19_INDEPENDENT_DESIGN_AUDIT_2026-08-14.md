# PH1 NVIDIA NC19 source-lock absent-set - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, compiler, payload, CUDA Driver, device or model call was performed.

## Verdict

**GO for source implementation only.**

NC19 closes the sole NC18 blocker without reopening the inherited topology, schema or authority contracts. This verdict does not authorize a static preflight, compiler invocation, payload read, CUDA Driver call, device call or execution attempt.

## Frozen artifact integrity

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC19 preregistration | 1,207 | `e399ed05c130229f45ef849c3a48a5b69d02c8583eb5bc9845fa00b04c36384a` |
| NC19 exact-100-path erratum | 642 | `bd36a33488b28ee22ade187cfdb0ebc1809e859c5db55f05f45f1e4aef42b690` |
| NC19 fixture manifest | 7,657,913 | `481914c80b5dc8970c2217d8e08c783dc97aabe6a07d1d073fc45d8851709018` |
| NC19 closed design lock | 16,838 | `2d74d16e595e1eaad0fa6503bbe76977f04a70c5dd8a32d3988f59aaede2c104` |

All fifteen design-lock bindings rehash exactly. The lock contains 157 unique expected-absent paths and all 157 are absent. The lock remains closed: implementation, preflight, compile and verifier flags are false and the authorization token is pending.

## NC18 blocker closure

The sole byte-backed NC19 source-lock authority is internally coherent:

- its raw Base64 decodes to 13,461 bytes with SHA-256 `28d0a1805fbf6da6459d8b69aaf1e2d2df95df579a8a5797bf94806b53283df8`;
- the decoded JSON equals the retained document exactly;
- `expected_absent` is sorted, unique and has exactly 100 paths;
- that array equals the NC19 descriptor's `implementation_freeze.paths` array exactly;
- its intersection with all 57 required implementation-freeze/runtime inputs is empty;
- the source-lock's own path is excluded;
- the 32 unique `{path,bytes,sha256}` mapping rows equal the 32 `bound_size_sha256` source roots exactly, with no self row, duplicate or extra;
- the three forbidden parallel/legacy authority keys remain absent; and
- the observed-entry declaration is the exact ordered eight-name vector.

Thus the NC18 self-absence and 57/57 required-input contradiction is removed. At runtime, each of the 32 permitted positive/negative terminal choices has an exact 99-path absence set produced by removing only the selected terminal root from the 100 pre-run roots. The immutable source-lock's 100-path field is freeze/pre-run authorization evidence; it does not assert that a selected terminal remains absent after a valid run commits.

## Fixture and negative-control recomputation

The manifest has 157 unique NC19 roots and 1,106 uniquely named cases. Every compact ordered-JSON observed-tree digest and every observed byte total recomputes exactly.

The positive source-lock case has the exact 100-path equality, zero required-input intersection and a bootstrap document that binds the observed source-lock path, byte count and digest. Each of the nine negative cases mutates the actual source-lock bytes and independently refreshes the observed file metadata, tree digest and bootstrap binding:

1. missing root;
2. extra required path;
3. source-lock self path;
4. duplicate path;
5. unsorted array;
6. wrong path;
7. 99-path count;
8. 101-path count; and
9. required-input intersection.

All nine are classified invalid. Every non-intentionally corrupted Base64 record reproduces its declared size and SHA-256, all `json_valid` records reparse to their retained schema values, and no valid case has a raw-content mismatch. The one retained raw/hash mismatch belongs only to the inherited deliberately negative NC17 SHA-mutation fixture.

The declared current path-set digest also recomputes as SHA-256 over compact ordered JSON of the sorted path array: `7bc55fb878388e08624a593323ef6adba3a1c53774c5aeb8a06f132951fad5aa`.

## Claim boundary and next gate

This is an implementation-executability judgment for the frozen design, not evidence that an implementation, verifier or static preflight exists or passes. A successor may implement the shared stdlib-only, import-inert contract and the source/preflight/runner/verifier packages, then freeze actual source identities and lifecycle locks under a new closed source package. That package requires a separate source audit before any no-device static preflight; later compiler, payload, Driver and device phases remain separately gated.
