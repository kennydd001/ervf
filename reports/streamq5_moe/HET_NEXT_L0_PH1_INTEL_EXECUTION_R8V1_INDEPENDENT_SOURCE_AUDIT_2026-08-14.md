# HET-NEXT-L0 PH1 Intel execution R8V1 — independent closed-source audit

Date: 2026-08-14  
Scope: read-only source, lock, immutable R8A5 bundle/evidence, and protocol audit. No R8V1 execution, payload/model forward, compiler, OpenCL, or device call was performed.

## Verdict

**NO-GO for constructing the open/auth-only successor from this exact source.** The closed package is correctly non-executable, and its bundle, topology, numerical, physical, lifecycle, resource, and provenance checks are largely sound. However, it does not independently preserve the exact negative state of the immutable first R8A5 verifier. It also omits the terminal fields required by the frozen post-run contract and would publish an output even when verification checks are false.

The immutable R8A5 physical bundle remains **substantively positive with verifier-protocol adjudication pending**. This audit does not reclassify the existing R8A5 verifier result: that artifact remains `verifier_protocol_negative` / `terminal_state=invalid` because exactly `topology` and `terminal_contract` are false. No physical rerun is authorized.

## Frozen inputs and phase state

The three handed-off hashes match the current files exactly:

- verifier: `f638906df71918452f4b0a7dbf3fb58d05e6d9388c71c8b37920232e2e39fe36`, 23,045 bytes;
- preregistration: `9ef4c756a2b05b440749bdc01679a5c185f9ec967504de6bef62eb42037b7009`, 3,984 bytes;
- closed lock: `a4c180e5cc48e2f7f2974a3095d914b1c11cd8de09f3dddd77a70f51ccbdbe91`, 1,580 bytes.

All 17 hash fields in the lock match the verifier, preregistration, immutable three-file R8A5 bundle, first verifier, topology diagnosis, post-run audit and JSON, R8A5 lock/prereg/source audit/runner/verifier, R8A runner, and frozen R7A runner/verifier. The R8V1 verification output is absent. The lock is deliberately `execution_open=false` with token `PENDING_INDEPENDENT_R8V1_SOURCE_AUDIT`.

The current file cannot be opened merely by flipping the lock: `lock_contract()` requires `execution_open is False` at line 181, while `main()` simultaneously requires `execution_open is True` at line 214. This is safe for the closed phase and means a separately frozen successor source/lock is mandatory.

## Checks that are correctly implemented

### Exact topology

Lines 64–69 freeze the exact case-preserving family names. The live set is exactly 10 entries: eight R8A5 entries plus the closed R8V1 preregistration and lock. It contains the mandatory uppercase provenance files and the lowercase runtime files, with no casefold collision. The R8V1 output, R8A5 failure roots, quarantine roots, and in-progress paths are absent.

Lines 105–120 use literal directory enumeration as authority. The Windows glob is retained only as a diagnostic consistency observation. Lines 122–130 non-destructively reject every missing member plus uppercase/lowercase extras, an R8V1 orphan, in-progress, failure, quarantine, a case-only replacement, and a duplicate-casefold collision. The expected mutation cardinality is 18 and is enforced.

An open successor must not reuse this 10-name set unchanged: it must add its newly created independent audit and its own new preregistration/lock names explicitly, while still requiring its output absent before execution.

### Immutable bundle and provenance

Lines 132–138 require exactly `result.json`, `manifest.json`, and `commit.json`, rehash every frozen input, rebuild the one-row manifest contract, and verify the commit’s manifest/result hashes. The exact bundle hashes are:

- result: `9d1ac21f4fdd9657160e877f267369b5e831ff9f7a65e998f27895947c9cad50`;
- manifest: `2d13137f143ff183be3ffe89a3b85754cb2f35b52f92885580f49676e5fcfb7b`;
- commit: `07d9f03e8907a029d8bc31e40da6298de080b6bc0f0914769f8d52517b2dd965`.

Lines 140–152 independently validate the R8A5 authorization extension, exact lock hash map, explicit historical/physical module identities, four frozen predicates, direct-entry evidence, and the dual venv/base-interpreter identity.

### Physical and numerical evidence

Lines 154–177 independently require the exact set of 18 result gates, all true, and directly verify the retained physical evidence:

- Intel Arc Pro 140T, driver `32.0.101.8517`, PCI `0000:00:02.0`;
- 102 main-ledger rows with the exact operation counts;
- 95 ownership rows;
- 14 unique aligned host-USM allocations;
- 18 pointer arguments, four eventless launches, one finish, and nine post-finish direct reads;
- 21 successful release attempts and one cleanup row with zero live resources/errors;
- 22 passing predevice controls;
- extension counts `14/14/18/42` and all six forbidden API counters zero;
- 12 ordered resource stages, no telemetry errors, start RAM at least 16 GiB, every retained sample at least 2 GiB available, and retained peak working set at most 12 GiB;
- the five exact gate/up/SiLU/activation/down hashes.

Lines 198–201 invoke the hash-pinned, device-free-at-import R7A CPU verifier. Inspection of its `verify_dict()` confirms that it independently rereads the three official source ranges, D2 input and frozen LUT, requantizes the three records, rebuilds the integer BF16 graph, compares all output bytes/stage hashes, and validates exactly 20 named checks. It also enforces exact buffer sizes/order, writes, initialization, argument maps, launch shapes, finish/read order, ownership links, counters, release order, resources, controls, compile package, and authorization. This is CPU reconstruction only; it performs no model forward or device call.

The retained R8A5 terminal-matrix import transitively imports physical module definitions, but those modules perform no OpenCL library load or device discovery at import. The audited source calls only the pure 31-case filesystem mutation harness. No device API is invoked by this path.

## Blocking findings

### 1. Immutable failed verifier is hash-bound but not semantically adjudicated

Lines 206–207 parse the first verifier but assert only:

- equality of its retained 31-case mutation matrix;
- `committed_adjudicator_mutations=true`;
- `production_matrix=true`.

They do **not** require its exact kind, exact 29-check name set, `passed=27`, `total=29`, `pass=false`, `terminal_state=invalid`, `terminal_valid=false`, or exact false set `{topology, terminal_contract}` with every other check true. This violates the previously frozen R8V1 contract and leaves the central “verifier failed solely because of topology” statement unproved by the new verifier.

The existing immutable verifier artifact itself is unambiguous: it has 29 checks, exactly 27 true, only `topology` and `terminal_contract` false, `terminal_state=invalid`, `terminal_valid=false`, and `pass=false`. R8V1 must retain that result as `verifier_protocol_negative`, not overwrite or retroactively label it positive.

### 2. Result schema does not separate physical adjudication from verifier history

Line 216 emits only a generic `pass` and the one-expert/input claim. It has no explicit immutable-first-verifier classification and no separate fields such as:

- `immutable_first_verifier_state: verifier_protocol_negative`;
- `bundle_adjudication: positive`;
- `terminal_state: positive` and `terminal_valid: true` for the new R8V1 adjudication.

Without this separation, a successful future R8V1 JSON can be misread as retroactively changing the immutable R8A5 verifier result rather than independently validating the bundle.

### 3. Publication is unconditional after verification

Lines 215–217 call `write_result(row)` even when one or more checks are false. The preregistration says the create-new verification JSON is written only after every topology, provenance, numerical, and direct-physical check succeeds. The successor must either fail without publishing the success artifact or publish a separately named bounded failure artifact; it must not create the canonical positive verification path on a failed adjudication.

## Required bounded repair

Before any auth-only/open successor is constructed:

1. Add an independent `failed_verifier_contract()` that enforces exact kind, exact 29 check names, exact false set `{topology, terminal_contract}`, all other checks true, `27/29`, `pass=false`, `terminal_state=invalid`, `terminal_valid=false`, the exact 31-case matrix, and the two retained mutation gates.
2. Add pure in-memory negative mutations for every protected field and for each false/true check-state boundary so the new check is nonvacuous.
3. Emit separate immutable-history and new-adjudication fields. The old verifier must remain `verifier_protocol_negative`; only the exact immutable bundle may be adjudicated positive by R8V1.
4. Publish the canonical R8V1 success JSON only when every check is true, with explicit `terminal_state=positive` and `terminal_valid=true`. Preserve a bounded, separate failure path if failure evidence is required.
5. Freeze a new closed R8V1-R1 source/lock, update the exact case-preserving family set to include this audit and the new revision provenance, and obtain a new independent source audit before any open authorization revision.

## Claim boundary

Even after the repair and a successful verifier-only run, the maximum claim is: one official real expert/input Intel correctness component passed the frozen source/Q5/device oracle and lifecycle checks. It is not a performance, full-layer, full-model, heterogeneous, industrial-readiness, or breakthrough result.

