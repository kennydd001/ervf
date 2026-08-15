# PH1 NVIDIA NC12 compile-only - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, NVRTC, compiler, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC12 as frozen.**

NC12 closes both explicit NC11 audit clusters. One bounded topology-symmetry defect remains: lock-level absence covers NC11 and NC12, but the normative topology contract and executable cases stop at NC10 static-preflight roots and NC11 durability.

## Integrity and recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC12 preregistration | 1,217 | `d62e944df7532ceb550ea22e62da591299febc27d61a1d79cc074dbed813dd15` |
| NC12 loader-only/NC10-symmetry erratum | 565 | `5f73b5d31323b155e22bd8efb557ee3bb0288cf82d777d17ad2e575160b5390f` |
| NC12 fixture manifest | 7,413,583 | `e697936ffa58349330d5f6940dcdd138d4d3b558fb7cabce461a8b1f415831d4` |
| NC12 closed design lock | 13,171 | `b186cb71b6c2019fdb2a1e554889db477351ed0523cae3b483e217aa1bc32b6a` |

All 15 bindings rehash exactly. The 61 declared absent paths are unique and absent; the set is the exact 42-path NC11 set plus 19 NC12 paths. All phase flags remain false. The one-BOM manifest is below 8 MiB and contains 415 cases with 415 unique names.

## Confirmed NC11 repairs

- `manifest_size_cap_minus_1` and `manifest_size_cap` now terminate as `loader_accepted_no_compile`, publish nothing, consume no attempt, keep `compiler_loaded=false`, and have ten ordered `attempted=false` ledger rows. The other six loader rejects remain pre-case/no-compiler.
- NC10 durability now has a literal six-key schema, 4 MiB cap and six cases: absent, file, directory, orphan, over-cap and collision.
- NC10 static-preflight failure and quarantine roots each have a schema and five named cases. Their 6+5+5 cardinalities match the preregistration.
- Historical NC8/NC9/NC10 and current NC11/NC12 absent paths remain in the lock.

## Blocking finding: manifest topology remains one revision behind the lock

The NC12 lock names all NC11 and NC12 static-preflight failure/quarantine and durability paths, but the normative manifest contains no occurrence at all of:

- `nc11_static_preflight_failures`;
- `nc11_static_preflight_quarantine`;
- `nc12_static_preflight_failures`;
- `nc12_static_preflight_quarantine`;
- `nc12_durability_adjudication`.

Consequently `inherited_topology` ends with NC10 static-preflight schemas and NC11 durability. There are zero executable cases for both NC11 static-preflight roots and all three current NC12 roots. This is not full inherited/current symmetry: the 61-entry lock proves only the present filesystem snapshot, while the future shared NC12 `classify_topology` contract has no schema or mutation matrix for those paths.

The source implementation cannot nonvacuously prove that current NC12 failure, quarantine and postcommit debris are rejected before compilation. Add literal schemas and executable absence/presence/corrupt/collision/over-cap cases for NC11 static-preflight roots and current NC12 static-preflight/durability roots. Preserve the 61-path union.

NC12 remains closed. No source implementation, preflight, compiler, Driver, payload or device action is authorized.
