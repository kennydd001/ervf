# PH1 NVIDIA NC11 compile-only - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, NVRTC, compiler, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC11 as frozen.**

NC11 closes the raw-BOM ambiguity, all six rejecting loader paths, and the lock-level output topology. Two contradictions remain: the two accepted boundary cases still authorize a compile, and the historical NC10 topology is named but not given the schema and executable mutations required by the NC10 audit.

## Integrity and recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC11 preregistration | 1,554 | `49066549d904713786070d7e4a0be57b8d9443baee6459bd2cd3aa11d2564948` |
| NC11 BOM/pre-case/topology erratum | 902 | `cffc7333b0695121a837a8ee93f91450dabc8fd7bfc8569dcdb316086ed90110` |
| NC11 fixture manifest | 7,297,714 | `596032128b8f603c0c65828ede4c156817f15abe4e66e312bc9a377b2a0bf65c` |
| NC11 closed design lock | 5,881 | `2a471fb13345f9a2eaaf87ae023fd7d207c956909f319ed803ec851bc454ce4c` |

All ten lock bindings rehash exactly. The 42 declared absent paths are unique and all absent. All execution flags remain false. The manifest contains 402 cases with 402 unique names.

The raw manifest begins exactly `EF BB BF 7B` and has one leading BOM. Independently rebuilding BOM + the 32-byte sentinel JSON + ASCII-space padding gives the frozen hashes:

- 8,388,607 bytes: `25d35fd8f88c98b1df6922ee4e1132e7be19d5c00e43ce20e7f4d5d4e9ce77c4`;
- 8,388,608 bytes: `bf1caa9799be539146161d42d1cee0b2de282e707b5f3bc1b2c2bbd0959d5eba`.

All six rejecting loader cases (`empty0`, `cap_plus_1`, missing/double/wrong BOM and bad schema) have ten ordered rows with `attempted=false`, `error=not_attempted:manifest_precheck`, `compiler_loaded=false`, `attempt_consumed=false`, no publication, and a later corrected invocation allowed. These six are sound.

The lock now explicitly includes historical NC8 durability; the NC10 verifier positive/negative/failure/quarantine quartet; NC9/NC10 preflight and durability roots; and the complete current NC11 family. This fixes the NC10 lock omission.

## Blocking findings

### 1. Cap-minus-one and cap contradict the loader-only erratum

The erratum states that accepted size fixtures remain loader-only evidence and do not authorize compilation. Both `manifest_size_cap_minus_1` and `manifest_size_cap` instead freeze all of the following simultaneously:

- `loader_observation.compiler_loaded=false`;
- ten NVRTC ledger rows with `attempted=true` and successful results;
- `terminal=compile_positive`;
- `publish=positive_bundle`;
- `attempt_consumed=true` and `next_invocation_allowed=false`.

Thus the only two accepted loader fixtures violate their own phase boundary. Change them to an explicit loader-accepted/preflight-continue outcome with ten not-attempted rows, no publication and no physical-attempt consumption. Loader acceptance may allow later fixture processing, but must not itself be a compile terminal.

### 2. Inherited NC10 topology is not executable or schema-symmetric

The NC10 audit required an exact NC10 durability schema and current-revision mutations. NC11's `inherited_topology.nc10_durability_adjudication` has no `schema` at all and contains only `file_present`, `dir_present`, and `temp_present`. It has no explicit baseline, orphan or over-cap case. By contrast, NC9 has a schema plus baseline/file/directory/orphan/over-cap, and current NC11 has a schema plus six cases.

Likewise, the NC10 static-preflight failure and quarantine roots appear in `inherited_topology` and the lock but have no associated manifest cases. Their NC9 equivalents have two cases each. The manifest's 21 topology cases are therefore not revisionally symmetric: it carries 2+2 NC9 preflight cases but zero NC10 preflight cases, and three underspecified NC10 durability cases.

Lock-level current absence is necessary but does not replace executable classifier/preflight mutations. Add the missing NC10 schema and baseline/file/directory/orphan/over-cap/collision-or-temp policy, plus exact NC10 static-preflight failure/quarantine presence cases.

## Required successor repair

Before implementation:

1. make both accepted boundary cases genuinely loader-only and no-compile;
2. freeze the full NC10 durability schema/matrix required by the NC10 audit;
3. add executable NC10 preflight-failure and quarantine mutations while retaining all 42 absence bindings.

NC11 remains closed. No source implementation, preflight, compiler, Driver, payload or device action is authorized.
