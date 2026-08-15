# PH1 NVIDIA NC10 compile-only - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, NVRTC, compiler, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC10 as frozen.**

The three NC9 repair subjects are materially present: the two missing historical preflight roots are restored, the four manifest-size boundary records and their hashes are exact, and the NC9 durability root now has a schema plus positive/negative topology records. The successor nevertheless reopens topology gaps and leaves the loader's real-input encoding and pre-execution ordering contradictory.

## Integrity and independently recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC10 preregistration | 1,464 | `d53d4cee4cf036aa3d1dcc5ff6c193c848f934dfee71286630734298f8b0cbe6` |
| NC10 topology/manifest erratum | 918 | `cd0ff74de5daa1192d76e93273aa0c0e8bf203a600ec4d925c05eb761ea323c3` |
| NC10 fixture manifest | 7,174,867 | `5b5ee5f493dd81a0519236c43a290c5fcd447ccbd56a9e17d284f00c391f57b0` |
| NC10 closed design lock | 4,168 | `518e445a4a322e2f44221e3837034e1f5b30a2a14690a88b520cfa16e977947c` |

All 11 lock bindings rehash exactly. All 19 paths declared by the lock are absent, and all execution flags are false. The manifest is below its 8,388,608-byte cap and parses as 389 cases with 389 unique names.

For the literal prefix `{"kind":"fixture_size_sentinel"}` (32 bytes), independently extending with ASCII spaces gives:

- 8,388,607 bytes: `9a78d75de6c19459842697e733fdbc2a14f916d02914c86d3bf2c73605952333`;
- 8,388,608 bytes: `ffeeddb5cb6d2b40de00730fd5a1ba0523de870c51f459177e0e96fca912e0dd`.

These exactly match the manifest. The empty and cap-plus-one records freeze zero bytes read and zero parse calls. The manifest also contains four NC9 static-preflight failure/quarantine file/directory cases, the retained three NC8 durability cases, and five NC9 durability cases.

## Blocking findings

### 1. The closed topology loses four inherited/current paths and does not freeze current NC10 durability semantics

The preregistration says all NC9 output requirements remain unchanged and historical NC8 durability absence remains frozen. NC9's lock had four independent-verifier roots and the historical NC8 durability root. NC10's 19-entry `expected_absent` set omits:

- `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc10_independent_verification_negative`;
- `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc10_independent_verification_failures`;
- `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc10_independent_verification_quarantine`;
- `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc8_durability_adjudication`.

All four happen to be absent now, but that fact is not frozen by the NC10 lock. The cardinality hides the regression because new paths were added while inherited paths were dropped.

There is a second revision asymmetry: the lock lists `het_next_l0_ph1_nvidia_nc10_durability_adjudication`, but the normative manifest has no NC10 durability object, schema or mutation. It specifies only NC8 and NC9 durability. A future NC10 postcommit incident therefore has no current-revision schema or executable file/directory/orphan/over-cap rejection matrix. This is the same current-path defect the NC9 audit required NC10 to close, shifted forward one namespace.

### 2. The actual manifest's BOM is outside the frozen loader fixture contract

The normative NC10 manifest begins with bytes `EF BB BF 7B` (UTF-8 BOM followed by `{`). The boundary fixtures instead begin directly with ASCII `{`, and `manifest_loader_contract` says only "UTF-8 JSON parse". With Python's standard library, `json.loads(raw_bytes)` and UTF-8-sig decoding accept the real file, while strict `raw.decode("utf-8")` followed by `json.loads(text)` deterministically rejects it as an unexpected BOM.

The future implementation is therefore not uniquely determined and can pass all four ASCII boundary fixtures yet fail on the frozen manifest it must load. Freeze one exact operation (for example `json.loads(raw_bytes)`, or explicit UTF-8-sig decoding) and add an exact self-load record for the real BOM-bearing manifest, including byte count, digest, bytes-read count and parse count. Alternatively remove the BOM and re-freeze all dependent hashes.

### 3. Rejecting loader fixtures contradict the required pre-case/no-device ordering

`manifest_size_empty0` and `manifest_size_cap_plus_1` correctly say zero bytes read/zero parses and a loader failure, but each also freezes a complete ten-row successful NVRTC ledger, `attempt_consumed=true`, and `next_invocation_allowed=false`. A manifest rejected before open/parse and before any case execution cannot also have completed the simulated compile ledger or consumed the physical compile attempt. The accepted cap-minus-one/cap records likewise jump directly to `compile_positive` with no frozen separation between loader acceptance and later fixture execution.

This makes an implementation unable to satisfy both the no-device static-preflight boundary and the normative case disposition. Freeze loader-only outcomes separately: loader reject must have no compile ledger, no physical attempt consumption and no compile terminal; loader accept should authorize only subsequent static fixture processing, not itself assert a physical positive bundle.

## Required successor repair

Before implementation:

1. restore all four omitted absent paths and bind their current absence;
2. add exact NC10 durability schema and mutations, while retaining both NC8 and NC9 historical roots;
3. pin BOM handling and execute a real-manifest self-load fixture;
4. separate loader-only boundary outcomes from compile ledgers and physical-attempt dispositions.

NC10 remains closed. No source implementation, preflight, compiler, Driver, payload or device action is authorized.
