# HET-NEXT-L0 PH1 NVIDIA NC14 semantic-topology preregistration

Status: immutable design-only; implementation and execution remain closed. NC14 supersedes only the topology semantics found incomplete by NC13 independent audit SHA `247212354eb9f8eca229771f9cfdd68e3066e87f31d491feac2b6a542ef306d0`.

The future shared stdlib-only, import-inert NC14 contract exports only `paths_for_revision` and `classify_topology`. No revision-specific branch is allowed. Each immutable descriptor has `revision,prefix,roots,patterns,caps`. Every root contains exactly `id,path,phase,role,allowed_node_type,required_schema_keyset,required_schema_kind,required_schema_status,cap_bytes,multiplicity,immutable,recoverable,disposition`. Every in-progress pattern freezes regex, glob, phase, role, cap, multiplicity, immutability, recoverability and disposition.

`classify_topology(descriptor, observed_entries)` receives literal records with exactly `path,node_type,size,sha256_or_null,children`; parsed child schemas have their own exact fields. It returns exact classification, validity, terminal/recoverable flags and disposition. It may not infer roles from revision strings.

NC14 freezes 99 current roots and seven in-progress patterns. The 242-case matrix contains a full literal observed tree and byte total in every case: fresh; both file and directory shapes at all 99 roots; exact valid terminal; one debris; missing schema; orphan, collision, mixed, multiple and over-cap compositions; exact descriptor operands for drop, duplicate, wrong revision and wrong path for NC8-NC13; and concrete pattern match, nonmatch, traversal, case-fold and collision cases.

Descriptor root paths still reproduce the exact NC8-NC13 locksets. All NC13/NC12 non-topology requirements remain unchanged. No run is authorized.
