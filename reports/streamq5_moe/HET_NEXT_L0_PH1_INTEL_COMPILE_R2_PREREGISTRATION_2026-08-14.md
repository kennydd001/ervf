# PH1 Intel compile-only R2 — preregistration

Status: **closed static package; no preflight/compiler/device call authorized**.

R2 supersedes the R1B source after its single valid negative compile evidence SHA-256 `62107b4cee0809fd744bacfe5d6890c7e09ec9002b0b029a6e84c98359f95fbb`. It applies exactly three textual replacements to the frozen R1 source: remove the one warning-producing required-subgroup extension pragma; rename the one `ulong half` declaration to `ulong halfway`; rename its two predicate uses. This is no arithmetic change. The resulting source is 7,852 bytes, SHA-256 `f1b3ccdae6d202ed210810e3cd419f726ea89ffa8fba0c84df5c2bfca3a84d21`.

The closed preflight must independently rederive R1 then R2, require exact three replacement sites, run the BF16 integer emulator and reject mutations restoring the reserved identifier, warning pragma, wrong tie predicate, wrong entrypoint or reduction geometry. It must bind and adjudicate the R1B build log, error `-11`, exact device identity, program/context release code zero, final cleanup complete/live-zero, and all queue/kernel/event/memory/allocation/launch counters zero. It must AST-audit the inherited compile-only callgraph, exact-one/nonempty binary gate, runner transaction and output absence, plus self-hash all frozen R2 files against the closed lock.

After independent source audit only the closed static preflight may be run. A later authorization-only revision is required before one physical compile-only attempt. A positive result still requires exactly one program device and one strictly nonempty binary whose queried/read sizes and byte digest agree, retained raw source/log/binary, clean program/context releases, and zero forbidden counters. Any failure is an immutable negative with stale/corrupt recovery before device open and write-through quarantine after device open.

Claim boundary: compile eligibility only. No payload, kernel creation/launch, correctness, timing or performance claim.
