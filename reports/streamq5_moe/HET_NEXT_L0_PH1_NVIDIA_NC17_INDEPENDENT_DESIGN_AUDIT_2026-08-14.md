# PH1 NVIDIA NC17 source-lock authority - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, compiler, payload, Driver, device or model call was performed.

## Verdict

**NO-GO for source implementation from NC17 as frozen.**

NC17 corrects the NC16 stage-set intersection and makes the new NC17 lock fixtures byte-backed. The normative manifest nevertheless retains two older source-identity authorities plus a stale active NC16 contract, contradicting the stated sole-authority rule. Its observed-node field declaration also contains an impossible duplicate field.

## Integrity and recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC17 preregistration | 1,723 | `6c0ebd2db2731b0dbd53e49c08735d2777a50a9255faeeb4d1f1d227cd0e6ef0` |
| NC17 stage-set/authority design | 644 | `7968326af56fc7489e38f8eda26ada32043d5cde483ec05638a24864b3a697f7` |
| NC17 source-lock-authority manifest | 5,631,872 | `a27f655ca19197954684e00e8546efe3f58987ddc8ab56659ba2d9b37f9fe2b6` |
| NC17 closed design lock | 22,554 | `1221a928c76ec54471f161589d44fdd821f6f4477912dec98349158f05925e8d` |

All thirteen bindings rehash exactly. The 157 expected paths are unique and absent. The manifest has 157 unique NC17 roots and 1,096 unique case names. I independently recomputed every compact ordered-JSON tree digest and every top-level byte total.

The new stage projection is coherent in isolation: design requires 0 and excludes 157 roots; implementation freeze requires 57 and excludes 100; each runtime terminal choice requires the same 57 and excludes the other 99 runtime roots. All required/absent intersections are empty. The valid freeze case observes exactly 57 roots, and the valid runtime case observes the 57 provenance/source roots plus one terminal.

All non-mutated Base64 contents decode to their declared byte count and SHA-256, and every JSON content record reparses exactly to its retained `schema_key_values`. The one metadata mismatch in `nc17_source_lock_mismatch` is intentional and adjudicated negative. The eight new source-lock mutation classes operate on observed content/tree evidence rather than a `source_lock_input` case field.

## Blocking findings

### 1. The normative manifest still contains three competing source-identity contracts

The design declares the parsed observed NC17 source-lock to be the sole authority and sets `parallel_authority_forbidden=true`. But the same normative manifest still contains:

- `synthetic_source_bindings`, the 32-entry NC15 synthetic source table;
- `synthetic_nc16_source_lock`, the 28-entry NC16 synthetic lock document; and
- `implementation_freeze_source_lock_contract.root = reports/streamq5_moe/het_next_l0_ph1_nvidia_nc16_source_lock.json`.

The new NC17 authority instead targets the observed `nc17_source_lock.json` and resolves 32 roots. The frozen shared-contract design does not state that the three legacy fields are inert, forbidden inputs or excluded from the production data model. In particular, the field still named `implementation_freeze_source_lock_contract` is an active-looking contract pointing at the wrong revision and cardinality.

This directly contradicts the preregistered “no parallel content authority” invariant and leaves an implementation with two different lock roots and 28-versus-32 resolution sets.

Required repair: remove the legacy authority fields from the current normative schema, or relocate them into explicitly historical fixture payloads that production cannot consume. Freeze one NC17 contract/root/cardinality and add static rejection of any legacy authority field or NC16 lock selection.

### 2. `observed_entry_fields` contains a duplicate required field

The frozen array has nine elements but only eight unique names: `content_base64_or_null` appears twice. Every actual observed entry correctly has one JSON property of that name, because a JSON object cannot carry two independently addressable equal keys.

A strict ordered-field validator therefore cannot make the actual records equal the declared field list; a set-based validator would silently weaken the exact-field contract. The lock's claimed raw-field schema is consequently not implementable as written.

Required repair: freeze the eight unique fields exactly once and add a manifest-schema gate requiring uniqueness of every declared field-name list.

## Claim boundary

This is a design executability verdict only. It does not invalidate the recomputed stage-set, tree, Base64, schema or mutation evidence and makes no CUDA, numerical, performance or device claim. NC17 remains closed; no source implementation, static preflight, compiler, payload, Driver or device action is authorized from this freeze.
