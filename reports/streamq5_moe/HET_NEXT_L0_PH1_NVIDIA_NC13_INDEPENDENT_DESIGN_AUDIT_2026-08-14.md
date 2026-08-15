# PH1 NVIDIA NC13 generic topology - independent design audit

Date: 2026-08-14  
Mode: frozen design-only/read-only. No candidate import, preflight, compiler, payload, Driver or device call was performed.

## Verdict

**NO-GO for source implementation from NC13 as frozen.**

The generic descriptor correctly eliminates the per-revision path-enumeration lag identified in NC12. The proposed classifier is not yet implementable or nonvacuously testable, because descriptor flattening discards phase semantics and the fixtures do not contain concrete observed-entry records or terminal schemas.

## Integrity and recomputed facts

| artifact | bytes | SHA-256 |
|---|---:|---|
| NC13 preregistration | 2,012 | `f53dadcdfe4618612fd48305482bf3b71c9cdfbf0f1cda0df6f614dcdfe610c3` |
| NC13 descriptor/classifier design | 1,022 | `679b360340d9c746185539ab7b326db7298a726f66c842f053923b59b1364c13` |
| NC13 generic fixture manifest | 138,255 | `3e2586aec93ea68f7819bcd7d5d62ca6d588c7a25e60abc81435d064a7d786a0` |
| NC13 closed design lock | 12,841 | `8c2500f547190b809b4ba187fa98feac2c3b7c905f146459e80cc5b930b93d63` |

All nine bindings rehash exactly. The 80 expected paths are unique and absent, and all phase flags remain closed. The manifest contains six descriptors and 198 unique case names.

I independently expanded the seven support templates and flattened the phase arrays. The generated sets exactly equal every frozen lock:

| revision | paths | symmetric difference | canonical newline-final digest |
|---|---:|---:|---|
| NC8 | 18 | 0 | `a65b4cd317bb8a0477558cced66af3232af73998f841e8ee7d4a1f9d7c5d8748` |
| NC9 | 18 | 0 | `f2b7164f238ea5bce632bbb38bec6b8e3afa95cb4700dd4a3c69ac601fbfe1ee` |
| NC10 | 19 | 0 | `82da115ca1332e0b6560d1f338700bce41fe7a2d610f49e059da64543a1b68cc` |
| NC11 | 42 | 0 | `6e662cd9baccb14604f785a8b9dcf4cb7de57da9bb5bd4625a92620d222ae26e` |
| NC12 | 61 | 0 | `f9d2f8c39b8f015f1fab7223567d65d48d716e1cebbe5962a754fe50b2458818` |
| NC13 | 80 | 0 | `27c8229d1faae359c6eb91fae963fd73452e85fa219275fdefa7fb0a502949eb` |

Thus the NC12 path-coverage blocker itself is closed.

## Blocking findings

### 1. Flattening removes information required by `classify_topology`

The only descriptor fields are `revision,prefix,phase_roots,inprogress_patterns,caps`. `paths_for_revision` is specified to flatten the twelve phase arrays and return only an ordinal path set plus in-progress patterns. The design then says `classify_topology` consumes only that return value, the three global caps and observed records.

That input no longer identifies which path is preflight failure, quarantine, durability, compile positive/negative/failure, or verifier terminal. It also contains no per-root schema, allowed entries, required filenames, record keys, collision policy or phase-specific cap. Therefore the classifier cannot distinguish an exact valid compile terminal or recoverable debris from the same root appearing as invalid file/directory topology without parsing names or adding hidden revision/phase rules.

Preserve typed root descriptors through expansion. Each generated root needs at least role, allowed state, exact tree/record schema, cap and disposition policy. The classifier must consume that typed structure, not a flattened untyped set.

### 2. The 198-case manifest is not an executable observation matrix

The 160 per-root cases provide only `mutation=root_file/root_directory` and a target path. The remaining composition cases (`valid_terminal`, `single_debris`, `missing`, `orphan`, `collision`, `mixed`, `multiple`, `oversize`) have `target=null` and no `observed_entries`, file tree, byte payload, digest, cap value or expected retained evidence.

This creates a direct ambiguity: a file or directory at every generated path is declared invalid, while `valid_terminal` and `single_debris` require some generated paths to be present and valid, but no shape distinguishes those cases. A static preflight could satisfy the labels with a toy mutation switch rather than call the production filesystem classifier.

Freeze exact observed-entry arrays and bounded contents for every composition case. Require the production classifier to consume those arrays, and independently reconstruct them in the verifier. Add per-pattern in-progress matches and near-miss/outside/traversal cases; the current 160 path cases cover only exact roots, not the returned patterns.

### 3. Descriptor-negative mutations are named but not defined

Each descriptor has `drop_path`, `duplicate_path`, `wrong_revision` and `wrong_path`, but the cases specify no field/index/value being changed. Exact historical equality proves the baseline generator, not that all malformed descriptor paths are rejected. Freeze the precise mutation operand and expected validation stage, including dot segment, traversal, slash, wrong prefix/revision relationship and duplicate-before-dedup failures.

## Required successor repair

Before implementation:

1. retain typed phase/root metadata through `paths_for_revision`;
2. add exact per-root schemas, caps, allowed states and dispositions to descriptors;
3. replace label-only cases with concrete observed-entry and descriptor-mutation records;
4. exercise in-progress patterns and malformed/near-miss paths through the same production functions.

NC13 remains closed. No source implementation, preflight, compiler, Driver, payload or device action is authorized.
