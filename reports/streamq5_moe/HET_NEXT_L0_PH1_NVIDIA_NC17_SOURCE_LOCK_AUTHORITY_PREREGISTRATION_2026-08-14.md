# HET-NEXT-L0 PH1 NVIDIA NC17 source-lock authority preregistration

Status: immutable design-only; implementation and execution remain closed. NC17 supersedes only the two source-lifecycle defects in NC16 independent audit SHA `1fbaf581b14f3e466a75696c1407f20c5908453aed773ca2fd00fd5fef365fa9`.

Every descriptor now freezes exact required-present and expected-absent sets by stage. Design observes no future implementation path. Implementation-freeze requires all source files plus source/preflight/verifier and authorization-bootstrap locks, while all 100 runtime roots and every in-progress pattern are absent. Runtime requires the same 57 provenance/source paths and selects a terminal-specific absence set. Every classification enforces empty `observed ∩ expected_absent` and the exact required-present set before identity or terminal adjudication.

There is no `source_lock_input` field or parallel content authority. The sole mapping authority is the strict JSON parsed from the observed NC17 source-lock bytes. Before parsing, its exact path, raw byte count and SHA-256 are verified against the observed authorization-bootstrap lock, whose identity is frozen by the future authorization chain. The parsed mapping must contain exactly one normalized `{path,bytes,sha256}` row for every one of 32 source roots, with no self-entry, duplicate, unresolved row or extra.

Negative fixtures mutate the actual observed source-lock content, metadata, bootstrap-bound identity and tree: mismatch, duplicate, missing, extra, self, path, size and SHA. Stage compositions cover design fresh/presence, implementation-freeze complete/output intersection, runtime full terminal and runtime missing source. No run is authorized.
