# PH1 NVIDIA NC9 compile-only - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, NVRTC, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC9 as frozen.**

The three NC8 repairs are substantively correct: the manifest is below its new cap, the NC8 durability root is bound and mutated, and the shared NC9 contract now owns the environment lifecycle. Three smaller topology/fixture omissions remain.

## Integrity and recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC9 preregistration | 1,907 | `4b9b6cc283704d5faf6f2f009585d3df9b2c5a368e685357ec86a7aa5f30e88f` |
| NC9 erratum | 1,081 | `7ce6b984d4afc948281bc41a83252d87a0614239f5907da50c4f69f991fc4752` |
| NC9 fixture manifest | 7,055,538 | `61f9795853995aac7c510def6997527b7d179bd20720b5606f9a44bb33f3c8c9` |
| NC9 closed design lock | 4,299 | `914a22a68548b32f8fe62f5092d6c0b39776a16192bbeb1a388aed2d7a062f89` |

All 12 bindings match exact byte count and SHA-256. All 18 declared absent paths are absent and execution flags remain false. The 7,055,538-byte manifest is nonempty and below 8,388,608 bytes.

The manifest parses with 376 unique cases. It freezes the exact NC9 contract path and eight exports. It contains five shared-contract cases, three NC8 durability-topology cases and the inherited fifty environment cases.

## Sound NC8 repairs to retain

- `caps.fixture_manifest=8388608` and the observed manifest is below it.
- NC8 durability file/directory/temp appearances are literal rejection cases.
- `capture_environment`, `apply_private_environment`, and `restore_environment` join the five transaction/classifier exports.
- Runner/preflight object and code-object identity, copy, monkeypatch, order and early-publication cases are explicit.

## Blocking findings

### 1. Static-preflight failure and quarantine roots disappeared from the current topology

NC8's exact absent set included `het_next_l0_ph1_nvidia_nc8_static_preflight_failures` and `..._static_preflight_quarantine`. NC9 changes only the three audited defects, but its 18-entry absent set omits both NC9 equivalents. It compensates numerically by adding two durability paths, masking the regression in the cardinality.

Restore both NC9 preflight roots. The classifier/preflight must reject stale, mixed, multiple, oversized and unexpected entries and preserve valid immutable failure evidence separately from quarantine debris.

### 2. The new 8 MiB cap has no executable boundary fixtures

The manifest freezes the scalar cap but contains no fixture for empty manifest, exact 8,388,608-byte acceptance or 8,388,609-byte rejection. The only name containing `fixture_manifest` is an inherited nested-result field mutation, not a size-boundary test.

Add exact nonempty, `cap-1`, `cap`, and `cap+1` fixtures through the actual preflight manifest loader before any case execution. Require bounded failure evidence for rejection.

### 3. Only the historical NC8 durability path is normative in the manifest

The lock lists both NC8 and NC9 durability-adjudication roots as absent. The manifest's `inherited_topology` freezes and mutates only the NC8 path; it contains no NC9 durability path or file/directory/temp mutations. Yet any later adjudication of an NC9 postcommit incident must use the current NC9 root.

Add an exact current `nc9_durability_adjudication` topology object and the same three presence mutations. Both historical NC8 and current NC9 roots must be absent during compile authorization; later verifier authorization may open only the current root under its own immutable lock.

## Required successor repair

Before implementation:

1. restore NC9 static-preflight failure/quarantine roots;
2. add exact fixture-manifest cap boundary cases;
3. add current NC9 durability-root schema and mutations.

NC9 remains closed. No implementation, preflight, compiler, Driver or device action is authorized.
