# HET-NEXT-L0 PH1 NVIDIA NC16 typed-schema and freeze preregistration

Status: immutable design-only; implementation and execution remain closed. NC16 supersedes only the typed-value and source-lifecycle gaps in NC15 independent audit SHA `04c1ebf4fe991332cb2bd4edd676dbc355515787d97132c92bf3d904ba9dc43f`.

Every required JSON key now has an exact `{type,constraint,value}` rule. Lock documents require the exact kind/revision, four false authorization booleans, a nonempty bindings object and a nonempty unique normalized expected-absence array. Preflight results require correct revision/kind, boolean pass/device-opened and a nonempty all-true boolean checks object. Other schema keys use explicit exact, enum or named predicates. For all 28 current lock/result roots, fixtures include a revision-correct positive plus missing, extra, wrong-type and wrong-value mutation for every key.

The source lifecycle has three distinct stages. Design requires every future source absent. During `implementation_freeze`, all 28 current source files and provenance locks are present, no runtime output exists, and the exact NC16 source-lock document maps every source through `{path,bytes,sha256}` with normalized unique path, positive integer bytes and lowercase 64-hex digest. Only this document populates descriptor identity values. Runtime requires the complete resolved set.

Freeze fixtures cover complete success, unresolved, extra, duplicate, wrong path/hash/size and malformed source lock. Composition fixtures cover design presence, complete implementation freeze, runtime terminal success and runtime missing source. No run is authorized.
